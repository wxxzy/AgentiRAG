# -*- coding: utf-8 -*-
"""
@desc: 数据注入脚本

本脚本负责加载 'data' 目录下的所有文档，将其切分、嵌入，
并存储到一个持久化的ChromaDB向量数据库中。
"""

import os
import nltk
from tqdm import tqdm
import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredExcelLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# 在加载其他模块前，先加载配置，确保环境变量等设置生效
import config
from agentic_rag.chains import get_embedding_function
from config import EXCEL_METADATA_COLUMNS

# --- 配置 ---
DATA_PATH = "data"
PERSIST_PATH = "chroma_db"

def download_nltk_data():
    """
    检查并下载unstructured库所需的NLTK数据包。
    """
    required_packages = ['punkt', 'averaged_perceptron_tagger']
    print("--- 检查并下载NLTK数据包 ---")
    for package in required_packages:
        try:
            # 检查数据包是否存在
            nltk.data.find(f"tokenizers/{package}" if package == 'punkt' else f"taggers/{package}")
            print(f"[NLTK] 数据包 '{package}' 已存在。")
        except LookupError:
            print(f"[NLTK] 正在下载数据包: '{package}'...")
            try:
                nltk.download(package, quiet=True)
                print(f"[NLTK] 数据包 '{package}' 下载成功。")
            except Exception as e:
                print(f"\n--- NLTK下载失败 ---")
                print(f"错误：无法自动下载NLTK数据包 '{package}'。")
                print("这通常是由于网络问题（如防火墙、代理）导致的。")
                print(r"请尝试手动下载，或在网络环境良好的机器上运行此脚本以下载数据，")
                print(r"然后将 C:\Users\<你的用户名>\AppData\Roaming\nltk_data 目录复制到当前机器的相同位置。")
                print(f"原始错误: {e}")
                exit(1) # 下载失败则退出，因为这是后续步骤的先决条件
    print("--- NLTK数据包检查完成 ---")

def load_documents_from_directory(directory_path):
    """逐个加载目录中的文档，并对每个加载过程进行错误处理。"""
    documents = []
    print(f"扫描目录: {directory_path}")
    
    loader_map = {
        ".pdf": PyPDFLoader,
        ".xlsx": None, # Handled separately by pandas
        ".xls": None,  # Handled separately by pandas
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
    }

    # 首先，统计所有支持的文件
    supported_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in loader_map or ext in ['.xlsx', '.xls']:
                supported_files.append(os.path.join(root, file))

    # 使用tqdm显示文件加载进度
    for file_path in tqdm(supported_files, desc="加载文档"):
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext in ['.xlsx', '.xls']:
                print(f"正在使用 pandas 加载并处理 Excel 文件: {file_path}")
                df = pd.read_excel(file_path)
                
                # Get column names for metadata and content construction
                column_names = df.columns.tolist()
                
                for index, row in df.iterrows():
                    # Construct page_content for the document
                    content_parts = []
                    metadata = {"source": file_path, "row_index": index} # Basic metadata
                    
                    for col_name in column_names:
                        value = row[col_name]
                        # Convert non-string values to string for content
                        if pd.isna(value): # Handle NaN values
                            value_str = ""
                        else:
                            value_str = str(value)
                        
                        content_parts.append(f"{col_name}: {value_str}")
                        
                        # Add specific columns as metadata (customize as needed)
                        # Example: if '药品名称' is a column, add it to metadata
                        if col_name in EXCEL_METADATA_COLUMNS:
                            metadata[col_name] = value_str
                            
                    page_content = "\n".join(content_parts)
                    
                    doc = Document(page_content=page_content, metadata=metadata)
                    documents.append(doc)
                print(f"成功从 {file_path} 加载 {len(df)} 行数据。")

            elif ext in loader_map:
                loader_cls = loader_map[ext]
                print(f"正在使用 {loader_cls.__name__} 加载: {file_path}")
                if loader_cls is TextLoader:
                    loader = loader_cls(file_path, encoding='utf-8')
                else:
                    loader = loader_cls(file_path)
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
            else:
                # This case should ideally not be reached due to supported_files filtering
                print(f"跳过不支持的文件类型: {file_path}")

        except Exception as e:
            print(f"\n--- 加载文件失败: {os.path.basename(file_path)} ---")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误详情: {e}")
            print("----------------------------------------\n")

    return documents

def main():
    """
    主函数：加载、切分并嵌入文档，然后将其存储到持久化的ChromaDB中。
    """
    print("--- 开始数据初始化流程 ---")
    download_nltk_data()

    # 1. 加载文档
    if not os.path.exists(DATA_PATH) or not os.listdir(DATA_PATH):
        print(f"错误：数据目录 '{DATA_PATH}' 不存在或为空。")
        print("请创建该目录并将您的知识库文档放入其中。")
        return
        
    documents = load_documents_from_directory(DATA_PATH)

    if not documents:
        print("未能成功加载任何文档。请检查data目录下的文件或查看上面的错误信息。")
        return
    print(f"\n成功加载 {len(documents)} 篇文档。")

    # 2. 切分文档
    print("正在将文档切分为小块...")
    # 对于Excel，我们已经按行处理，所以这里不需要再进行文本切分，直接使用documents
    # 但为了兼容其他文档类型，我们仍然保留text_splitter，它只会处理非Excel文档
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    print(f"文档被切分为 {len(splits)} 个小块。")

    # 3. 获取嵌入模型
    print("正在加载嵌入模型... (如果使用本地模型，此过程可能需要一些时间)")
    embedding_function = get_embedding_function()

    # 4. 创建并持久化向量数据库
    print(f"正在创建向量数据库并将其持久化到 '{PERSIST_PATH}'...")
    # 初始化一个空的ChromaDB
    db = Chroma(
        persist_directory=PERSIST_PATH,
        embedding_function=embedding_function
    )

    # 定义批处理大小
    batch_size = 32
    print(f"将以 {batch_size} 的批大小处理 {len(splits)} 个文本块...")

    # 使用tqdm显示嵌入和存储的进度条
    for i in tqdm(range(0, len(splits), batch_size), desc="嵌入并存储文档"):
        batch = splits[i:i + batch_size]
        db.add_documents(batch)

    print("--- 数据初始化完成 ---")
    print(f"您的数据已成功处理并存储在 '{PERSIST_PATH}' 目录中。")
    print("现在您可以运行 'python main.py' 来查询您的数据了。")

if __name__ == "__main__":
    main()