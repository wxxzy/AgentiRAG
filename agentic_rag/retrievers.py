# -*- coding: utf-8 -*-
"""
@desc: 知识检索模块

负责从不同的知识源（向量数据库、网络等）获取信息。
"""

from langchain_tavily import TavilySearch
from langchain_chroma import Chroma
from agentic_rag.chains import get_embedding_function
import os

# --- 配置 ---
PERSIST_PATH = "chroma_db"

# --- 知识源初始化 ---

# 网络搜索工具
# TavilySearch 提供了对 Tavily 搜索引擎的封装
web_search_tool = TavilySearch(max_results=3)

# 向量数据库
# 连接到由 ingest.py 创建的持久化数据库
if not os.path.exists(PERSIST_PATH):
    raise FileNotFoundError(
        f"向量数据库持久化目录 '{PERSIST_PATH}' 不存在。\n"
        f"请先运行 'python ingest.py' 来创建数据库。"
    )

embedding_function = get_embedding_function()
vectorstore = Chroma(
    persist_directory=PERSIST_PATH,
    embedding_function=embedding_function
)
retriever = vectorstore.as_retriever()
retriever = vectorstore.as_retriever()

# --- 检索函数 ---

def get_retriever():
    """获取向量数据库的检索器。"""
    return retriever

def get_web_search_tool():
    """获取网络搜索工具。"""
    return web_search_tool
