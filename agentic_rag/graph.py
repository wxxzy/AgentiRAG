# -*- coding: utf-8 -*-
"""
@desc: 构建并编译Agentic RAG的工作流图
"""

from langgraph.graph import StateGraph, END

from agentic_rag.state import AgentState
from agentic_rag.nodes import (
    route_query_node,
    rewrite_query_node,
    web_search_node,
    generate_response_node,
    grade_relevance_node,
    direct_response_node
)

def build_graph():
    """构建并返回编译好的LangGraph图。"""
    workflow = StateGraph(AgentState)

    # --- 添加节点 ---
    workflow.add_node("route_query", route_query_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("grade_relevance", grade_relevance_node)
    workflow.add_node("direct_response", direct_response_node)

    # --- 定义边 ---

    # 1. 从路由查询开始
    workflow.set_entry_point("route_query")

    # 2. 根据路由结果，决定是直接回答、网络搜索还是向量检索
    workflow.add_conditional_edges(
        "route_query",
        lambda state: state["route"],
        {
            "web_search": "rewrite_query",
            "hierarchical_search": "rewrite_query",
            "direct_chunk_search": "rewrite_query",
            "direct": "direct_response"
        }
    )
    
    # 3. 直接回答后，对答案进行评估
    workflow.add_edge("direct_response", "grade_relevance")

    # 4. 重写查询后，根据之前的路由决策进行相应的检索或直接生成
    workflow.add_conditional_edges(
        "rewrite_query",
        lambda state: state["route"],
        {
            "web_search": "web_search",
            "hierarchical_search": "generate_response", # 检索已在路由节点完成
            "direct_chunk_search": "generate_response"  # 检索已在路由节点完成
        }
    )

    # 5. 网络搜索后，生成答案
    workflow.add_edge("web_search", "generate_response")

    # 6. 生成答案后，评估相关性
    workflow.add_edge("generate_response", "grade_relevance")

    # 7. 根据相关性评估结果，决定是结束还是重新开始
    workflow.add_conditional_edges(
        "grade_relevance",
        lambda state: state["is_relevant"],
        {
            True: END,           # 答案相关，流程结束
            False: "rewrite_query" # 答案不相关，重写查询
        }
    )

    # 编译图
    graph = workflow.compile()
    return graph
