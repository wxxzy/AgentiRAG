# -*- coding: utf-8 -*-
"""
@desc: 查询本地ChromaDB向量库的脚本（已升级为多集合支持）。

本脚本用于连接到由 ingest.py 创建的持久化ChromaDB，
并允许用户通过命令行参数选择要查询的集合（摘要或区块），
然后输入查询，返回相关的文档。
"""

import os
import argparse
import chromadb
from agentic_rag.chains import get_embedding_function
from dotenv import load_dotenv

# 加载环境变量和配置
load_dotenv()
import config

# --- 配置 ---
PERSIST_PATH = "chroma_db"
SUMMARY_COLLECTION_NAME = "doc_summaries"
CHUNK_COLLECTION_NAME = "doc_chunks"

def main():
    """
    主函数：连接到ChromaDB并允许用户查询指定的集合。
    """
    # 1. 设置命令行参数解析
    parser = argparse.ArgumentParser(description="查询ChromaDB向量库中的特定集合。")
    parser.add_argument(
        "-c", "--collection",
        type=str,
        choices=['summaries', 'chunks'],
        default='chunks',
        help="要查询的集合名称: 'summaries' (摘要) 或 'chunks' (区块)。默认为 'chunks'。"
    )
    parser.add_argument(
        "-k", "--top_k",
        type=int,
        default=5,
        help="要返回的相关文档数量。默认为 5。"
    )
    args = parser.parse_args()

    collection_name_map = {
        "summaries": SUMMARY_COLLECTION_NAME,
        "chunks": CHUNK_COLLECTION_NAME
    }
    target_collection_name = collection_name_map[args.collection]

    print(f"--- 准备查询集合: '{target_collection_name}' ---")

    if not os.path.exists(PERSIST_PATH):
        print(f"错误：向量数据库目录 '{PERSIST_PATH}' 不存在。")
        return

    try:
        # 2. 连接到数据库并获取指定集合
        embedding_function = get_embedding_function()
        client = chromadb.PersistentClient(path=PERSIST_PATH)
        collection = client.get_collection(name=target_collection_name, embedding_function=embedding_function)

        print(f"\n集合 '{target_collection_name}' 已加载。请输入您的问题，输入 'exit' 退出。")

        # 3. 进入查询循环
        while True:
            query = input("\n您的查询: ")
            if query.lower() == 'exit':
                break
            if not query.strip():
                continue

            print(f"正在查询: '{query}'...")
            
            # 4. 执行相似度搜索
            results = collection.query(
                query_texts=[query],
                n_results=args.top_k
            )

            if not results or not results.get('documents') or not results['documents'][0]:
                print("未找到相关文档。")
            else:
                print(f"\n找到 {len(results['documents'][0])} 篇相关文档:")
                for i, (doc, meta, dist) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0])):
                    print(f"--- 文档 {i+1} (距离: {dist:.4f}) ---")
                    print(f"来源: {meta.get('source', '未知')}")
                    print(f"内容: {doc[:500]}...")
                    print("-" * 30)

    except Exception as e:
        print(f"\n查询向量库时发生错误: {e}")

    print("\n--- 查询向量库结束 ---")

if __name__ == "__main__":
    main()