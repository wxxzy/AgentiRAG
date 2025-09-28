# -*- coding: utf-8 -*-
"""
@desc: 分层检索模块

实现了先检索摘要，再从相关文档中检索具体区块的两步检索策略。
"""

import chromadb
from langchain_core.documents import Document

from agentic_rag.chains import get_embedding_function

# --- 配置 ---
PERSIST_PATH = "chroma_db"
SUMMARY_COLLECTION_NAME = "doc_summaries"
CHUNK_COLLECTION_NAME = "doc_chunks"

# --- 初始化ChromaDB客户端 ---
# 建议在应用启动时初始化一次，避免重复加载
client = chromadb.PersistentClient(path=PERSIST_PATH)
embedding_function = get_embedding_function()
summary_collection = client.get_collection(SUMMARY_COLLECTION_NAME, embedding_function=embedding_function)
chunk_collection = client.get_collection(CHUNK_COLLECTION_NAME, embedding_function=embedding_function)


def hierarchical_retriever(query: str, n_docs=3, n_chunks=5) -> list[Document]:
    """
    执行分层检索。
    1. 在摘要集合中检索，找到最相关的文档。
    2. 在区块集合中，仅从这些相关文档里检索出具体的文本块。
    """
    print("--- 执行分层检索 ---")
    
    # 步骤1: 在摘要层检索，找到最相关的n_docs个文档
    print("--- 步骤1: 检索摘要层 ---")
    summary_results = summary_collection.query(
        query_texts=[query],
        n_results=n_docs,
    )
    
    if not summary_results or not summary_results.get('metadatas') or not summary_results['metadatas'][0]:
        print("未在摘要层找到相关文档。")
        return []

    relevant_doc_sources = [meta['source'] for meta in summary_results['metadatas'][0]]
    if not relevant_doc_sources:
        print("未在摘要层找到相关文档源。")
        return []
    
    print(f"找到相关文档源: {relevant_doc_sources}")

    # 步骤2: 在区块层中，使用元数据过滤器，仅在相关文档中检索
    print("--- 步骤2: 在区块层进行过滤检索 ---")
    
    where_filter = {
        "source": {
            "$in": relevant_doc_sources
        }
    }
    
    chunk_results = chunk_collection.query(
        query_texts=[query],
        n_results=n_chunks,
        where=where_filter
    )

    if not chunk_results or not chunk_results.get('documents') or not chunk_results['documents'][0]:
        print("在相关文档的区块中未找到匹配项。")
        return []

    # 将检索结果格式化为LangChain的Document对象
    final_chunks = []
    for i, doc_text in enumerate(chunk_results['documents'][0]):
        final_chunks.append(
            Document(page_content=doc_text, metadata=chunk_results['metadatas'][0][i])
        )
        
    return final_chunks
