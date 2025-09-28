# -*- coding: utf-8 -*-
"""
@desc: 知识检索模块

负责从不同的知识源（目前主要是网络）获取信息。
本地知识库的检索已移至 hierarchical_retriever.py
"""

from langchain_tavily import TavilySearch

# --- 知识源初始化 ---

# 网络搜索工具
web_search_tool = TavilySearch(max_results=3)

def get_web_search_tool():
    """获取网络搜索工具。"""
    return web_search_tool