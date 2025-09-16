# -*- coding: utf-8 -*-
"""
@desc: 查看本地ChromaDB向量库内容的脚本。

本脚本用于连接到由 ingest.py 创建的持久化ChromaDB，
并显示其包含的文档数量和部分文档内容。
"""

import os
from langchain_chroma import Chroma
from agentic_rag.chains import get_embedding_function
from dotenv import load_dotenv

# 加载环境变量和配置
load_dotenv()
import config # Ensure config is loaded for embedding function

# --- 配置 ---
PERSIST_PATH = "chroma_db"
COLLECTION_NAME = "langchain" # Chroma.from_documents 默认使用的集合名称

def main():
    """
    主函数：连接到ChromaDB并显示其内容。
    """
    print("--- 开始查看向量库内容 ---")

    if not os.path.exists(PERSIST_PATH):
        print(f"错误：向量数据库目录 '{PERSIST_PATH}' 不存在。")
        print("请确保您已经运行过 'python ingest.py' 来初始化数据库。")
        return

    try:
        # 获取嵌入模型函数
        embedding_function = get_embedding_function()

        # 连接到持久化数据库
        # 注意：这里我们直接使用chromadb的PersistentClient API来获取更底层的控制
        # 否则，langchain_chroma.Chroma的.get()方法可能不直接暴露所有元数据
        import chromadb
        client = chromadb.PersistentClient(path=PERSIST_PATH)
        
        # 获取集合
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
        except Exception as e:
            print(f"错误：无法获取集合 '{COLLECTION_NAME}'。可能数据库为空或集合名称不正确。")
            print(f"详细错误: {e}")
            return

        count = collection.count()
        print(f"\n向量库 '{COLLECTION_NAME}' 中包含 {count} 个文档块。")

        if count == 0:
            print("向量库为空，没有内容可显示。")
            return

        # 获取并显示部分文档内容
        print("\n--- 部分文档内容 (前5个) ---")
        results = collection.peek(limit=5) # 获取前5个项目

        for i, (id, embedding, metadata, document) in enumerate(zip(results['ids'], results['embeddings'], results['metadatas'], results['documents'])):
            print(f"--- 文档块 {i+1} ---")
            print(f"result: {results}")
            print(f"ID: {id}")
            print(f"元数据: {metadata}")
            print(f"内容: {document[:200]}...") # 打印前200字符
            print("-" * 20)

    except Exception as e:
        print(f"\n查看向量库内容时发生错误: {e}")
        print("请确保您的嵌入模型配置正确，并且数据库文件没有损坏。")

    print("\n--- 查看向量库内容结束 ---")

if __name__ == "__main__":
    main()
