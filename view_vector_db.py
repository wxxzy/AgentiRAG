# -*- coding: utf-8 -*-
"""
@desc: 查看本地ChromaDB向量库内容的脚本（已升级为多集合支持）。

本脚本用于连接到由 ingest.py 创建的持久化ChromaDB，
并允许用户通过命令行参数选择要查看的集合。
"""

import os
import argparse
import chromadb
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
    主函数：连接到ChromaDB并显示指定集合的内容。
    """
    # 1. 设置命令行参数解析
    parser = argparse.ArgumentParser(description="查看ChromaDB向量库中特定集合的内容。 সন")
    parser.add_argument(
        "-c", "--collection",
        type=str,
        choices=['summaries', 'chunks'],
        default='chunks',
        help="要查看的集合名称: 'summaries' (摘要) 或 'chunks' (区块)。默认为 'chunks'。"
    )
    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=5,
        help="要显示的文档数量。默认为 5。"
    )
    args = parser.parse_args()

    collection_name_map = {
        "summaries": SUMMARY_COLLECTION_NAME,
        "chunks": CHUNK_COLLECTION_NAME
    }
    target_collection_name = collection_name_map[args.collection]

    print(f"--- 准备查看集合: '{target_collection_name}' ---")

    if not os.path.exists(PERSIST_PATH):
        print(f"错误：向量数据库目录 '{PERSIST_PATH}' 不存在。 সন")
        return

    try:
        # 2. 连接到数据库并获取指定集合
        client = chromadb.PersistentClient(path=PERSIST_PATH)
        collection = client.get_collection(name=target_collection_name)

        count = collection.count()
        print(f"\n集合 '{target_collection_name}' 中总共包含 {count} 个条目。 সন")

        if count == 0:
            return

        # 3. 获取并显示内容
        limit = min(args.limit, count)
        print(f"\n--- 显示前 {limit} 个条目 ---")
        results = collection.get(limit=limit)

        for i, (id, meta, doc) in enumerate(zip(results['ids'], results['metadatas'], results['documents'])):
            print(f"--- 条目 {i+1} ---")
            print(f"ID: {id}")
            print(f"元数据: {meta}")
            print(f"内容: {doc[:500]}...")
            print("-" * 20)

    except Exception as e:
        print(f"\n查看向量库时发生错误: {e}")

    print(f"\n--- 查看集合 '{target_collection_name}' 结束 ---")

if __name__ == "__main__":
    main()