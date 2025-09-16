# -*- coding: utf-8 -*-
"""
@desc: 定义Agentic RAG工作流的状态。

该状态在图的节点之间传递，并随着每个节点的执行而更新。
"""
from typing import List, TypedDict, Optional

class AgentState(TypedDict):
    """
    Agentic RAG的状态表示

    Attributes:
        query (str): 用户的原始问题。
        updated_query (str): 经过优化的查询。
        documents (List[str]): 从知识源检索到的文档列表。
        response (str): LLM生成的中间或最终答案。
        route (str): 查询路由的结果（例如，'web_search', 'vectorstore', 'direct'）。
        is_relevant (bool): 答案是否与查询相关。
        error (Optional[str]): 工作流中发生的任何错误。
    """
    query: str
    updated_query: str
    documents: List[str]
    response: str
    route: str
    is_relevant: bool
    error: Optional[str]
