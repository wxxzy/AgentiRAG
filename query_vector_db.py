# -*- coding: utf-8 -*-
"""
@desc: 查询本地ChromaDB向量库的脚本。

本脚本用于连接到由 ingest.py 创建的持久化ChromaDB，
并允许用户输入查询，然后返回相关的文档。
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

def main():
    """
    主函数：连接到ChromaDB并允许用户查询。
    """
    print("--- 开始查询向量库 ---")

    if not os.path.exists(PERSIST_PATH):
        print(f"错误：向量数据库目录 '{PERSIST_PATH}' 不存在。")
        print("请确保您已经运行过 'python ingest.py' 来初始化数据库。")
        return

    try:
        # 获取嵌入模型函数
        embedding_function = get_embedding_function()

        # 连接到持久化数据库
        vectorstore = Chroma(
            persist_directory=PERSIST_PATH,
            embedding_function=embedding_function
        )
        retriever = vectorstore.as_retriever()

        print("\n向量库已加载。请输入您的问题，输入 'exit' 退出。")

        while True:
            query = input("\n您的查询: ")
            if query.lower() == 'exit':
                break

            if not query.strip():
                print("查询不能为空，请重新输入。")
                continue

            print(f"正在查询: '{query}'...")
            
            # 执行相似度搜索
            results = retriever.invoke(query)

            if not results:
                print("未找到相关文档。")
            else:
                print(f"\n找到 {len(results)} 篇相关文档:")
                for i, doc in enumerate(results):
                    print(f"--- 文档 {i+1} ---")
                    print(f"来源: {doc.metadata.get('source', '未知')}")
                    print(f"内容: {doc.page_content[:500]}...") # 打印前500字符
                    print("-" * 30)

    except Exception as e:
        print(f"\n查询向量库时发生错误: {e}")
        print("请确保您的嵌入模型配置正确，并且数据库文件没有损坏。")

    print("\n--- 查询向量库结束 ---")

if __name__ == "__main__":
    main()
