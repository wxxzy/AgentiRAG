# -*- coding: utf-8 -*-
"""
@desc: 数据注入脚本（已升级为并行与批处理模式）

本脚本使用多进程并行处理文档，并通过批处理方式存入数据库，以提升注入效率。
"""

import os
import shutil
import multiprocessing
from tqdm import tqdm
import pandas as pd
import chromadb
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 在加载其他模块前，先加载配置，确保环境变量等设置生效
import config
from agentic_rag.chains import get_embedding_function, get_summarizer_chain
from config import EXCEL_METADATA_COLUMNS

# --- 配置 ---
DATA_PATH = "data"
PERSIST_PATH = "chroma_db"
SUMMARY_COLLECTION_NAME = "doc_summaries"
CHUNK_COLLECTION_NAME = "doc_chunks"

# --- 工作函数：用于并行处理 ---
def process_document_worker(doc):
    """
    对单个文档进行摘要生成和文本切分的工作函数。
    注意：为了避免多进程中的序列化问题，此函数内部会自行初始化所需的链和分割器。
    """
    summarizer_chain = get_summarizer_chain()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    doc_content = doc.page_content
    doc_source = doc.metadata.get('source', 'unknown_source')
    if 'row_index' in doc.metadata:
        doc_source = f"{doc_source}_row_{doc.metadata['row_index']}"

    try:
        # 1. 根据数据类型智能生成摘要
        doc_type = doc.metadata.get('data_type', 'narrative') # 默认为叙事型
        summary = ""
        if doc_type == 'narrative':
            # 对叙事型文档，调用LLM生成摘要
            summary = summarizer_chain.invoke({"document_content": doc_content}).content
        elif doc_type == 'tabular':
            # 对表格型数据，直接使用原文作为摘要，免除LLM调用
            summary = doc_content
        
        summary_metadata = {"source": doc_source}

        # 2. 切分区块（对于表格行，通常只切分出它自身）
        splits = text_splitter.split_documents([doc])
        chunk_docs = [split.page_content for split in splits]
        chunk_metadatas = [split.metadata for split in splits]
        chunk_ids = [f"{doc_source}_chunk_{i}" for i in range(len(splits))]

        return (doc_source, summary, summary_metadata, chunk_ids, chunk_docs, chunk_metadatas)
    except Exception as e:
        print(f"处理文档 {doc_source} 时出错: {e}")
        return None

# --- 主逻辑 ---
def main():
    """
    主函数：执行并行化和批处理的数据注入流程。
    """
    print("---" + " 开始并行化数据注入流程" + " ---")

    if os.path.exists(PERSIST_PATH):
        print(f"正在删除旧的数据库 '{PERSIST_PATH}'...")
        shutil.rmtree(PERSIST_PATH)

    # 1. 加载所有文档
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        print(f"错误：数据目录 '{DATA_PATH}' 不存在或为空。")
        return
    documents = load_documents_from_directory(DATA_PATH)
    if not documents:
        print("未能成功加载任何文档。" )
        return
    print(f"\n成功加载 {len(documents)} 份原始文档/数据行。" )

    # 2. 并行处理所有文档
    all_summaries, all_summary_metadatas, all_summary_ids = [], [], []
    all_chunks, all_chunk_metadatas, all_chunk_ids = [], [], []

    # 创建进程池
    num_processes = max(1, os.cpu_count() - 1) # 留一个核心给主进程
    print(f"---" + " 使用 " + f"{num_processes}" + " 个进程并行处理文档" + " ---")
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 使用imap_unordered来获取进度条
        results = list(tqdm(pool.imap_unordered(process_document_worker, documents), total=len(documents), desc="摘要与切分"))

    # 3. 收集处理结果
    for result in results:
        if result:
            doc_source, summary, summary_metadata, chunk_ids, chunk_docs, chunk_metadatas = result
            all_summary_ids.append(doc_source)
            all_summaries.append(summary)
            all_summary_metadatas.append(summary_metadata)
            all_chunk_ids.extend(chunk_ids)
            all_chunks.extend(chunk_docs)
            all_chunk_metadatas.extend(chunk_metadatas)

    if not all_summary_ids or not all_chunk_ids:
        print("未能成功处理任何文档，注入中止。" )
        return

    # 4. 批量存入数据库（分批次）
    print("--- 开始批量存入向量数据库 ---")
    embedding_function = get_embedding_function()
    client = chromadb.PersistentClient(path=PERSIST_PATH)
    
    # 定义一个合理的批次大小
    CHROMA_BATCH_SIZE = 4096

    # 批量存入摘要
    summary_collection = client.get_or_create_collection(SUMMARY_COLLECTION_NAME, embedding_function=embedding_function)
    total_summaries = len(all_summary_ids)
    print(f"正在分批存入 {total_summaries} 条摘要...")
    for i in tqdm(range(0, total_summaries, CHROMA_BATCH_SIZE), desc="存入摘要"):
        end_i = min(i + CHROMA_BATCH_SIZE, total_summaries)
        summary_collection.add(
            ids=all_summary_ids[i:end_i],
            documents=all_summaries[i:end_i],
            metadatas=all_summary_metadatas[i:end_i]
        )

    # 批量存入区块
    chunk_collection = client.get_or_create_collection(CHUNK_COLLECTION_NAME, embedding_function=embedding_function)
    total_chunks = len(all_chunk_ids)
    print(f"正在分批存入 {total_chunks} 个区块...")
    for i in tqdm(range(0, total_chunks, CHROMA_BATCH_SIZE), desc="存入区块"):
        end_i = min(i + CHROMA_BATCH_SIZE, total_chunks)
        chunk_collection.add(
            ids=all_chunk_ids[i:end_i],
            documents=all_chunks[i:end_i],
            metadatas=all_chunk_metadatas[i:end_i]
        )

    print("\n--- 并行化数据注入完成 ---")
    print(f"知识库已成功构建在 '{PERSIST_PATH}' 中。" )

# --- 辅助函数定义 ---
def load_documents_from_directory(directory_path):
    """逐个加载目录中的文档。"""
    # ... (此处省略与之前版本相同的完整代码)
    documents = []
    loader_map = {'.pdf': PyPDFLoader, '.txt': TextLoader, '.md': UnstructuredMarkdownLoader, '.docx': UnstructuredWordDocumentLoader, '.doc': UnstructuredWordDocumentLoader}
    supported_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in loader_map or ext in ['.xlsx', '.xls']:
                supported_files.append(os.path.join(root, file))

    for file_path in tqdm(supported_files, desc="加载文档"):
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
                for index, row in df.iterrows():
                    content_parts = []
                    metadata = {"source": file_path, "row_index": index, "data_type": "tabular"}
                    for col_name in df.columns:
                        value = row[col_name]
                        value_str = str(value) if not pd.isna(value) else ""
                        content_parts.append(f"{col_name}: {value_str}")
                        if col_name in EXCEL_METADATA_COLUMNS:
                            metadata[col_name] = value_str
                    doc = Document(page_content="\n".join(content_parts), metadata=metadata)
                    documents.append(doc)
            elif ext in loader_map:
                loader = loader_map[ext](file_path, encoding='utf-8') if ext == ".txt" else loader_map[ext](file_path)
                loaded_docs = loader.load()
                # 为叙事型文档打上标签
                for doc in loaded_docs:
                    doc.metadata["data_type"] = "narrative"
                documents.extend(loaded_docs)
        except Exception as e:
            print(f"加载文件 {file_path} 失败: {e}")
    return documents

if __name__ == "__main__":
    # 在Windows上使用多进程时，必须将主逻辑放在 if __name__ == '__main__': 下
    multiprocessing.freeze_support() 
    main()