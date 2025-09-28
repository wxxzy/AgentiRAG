# -*- coding: utf-8 -*-
"""
@desc: 构建并编译Agentic RAG的工作流图
"""

from langgraph.graph import StateGraph, END

from agentic_rag.state import AgentState
from agentic_rag.nodes import (
    retrieve_memory_node,
    consolidate_memory_node,
    route_query_node,
    rewrite_query_node,
    web_search_node,
    generate_response_node,
    grade_relevance_node,
    direct_response_node
)

def build_graph():
    """构建并返回集成了长期记忆的编译好的LangGraph图。"""
    workflow = StateGraph(AgentState)

    # --- 添加所有节点 ---
    workflow.add_node("retrieve_memory", retrieve_memory_node)
    workflow.add_node("route_query", route_query_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("direct_response", direct_response_node)
    workflow.add_node("grade_relevance", grade_relevance_node)
    workflow.add_node("consolidate_memory", consolidate_memory_node)

    # --- 定义边 ---

    # 1. 从“回忆”开始
    workflow.set_entry_point("retrieve_memory")
    workflow.add_edge("retrieve_memory", "route_query")

    # 2. 从“路由”分发
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
    
    # 3. “直接回答”后进行评估
    workflow.add_edge("direct_response", "grade_relevance")

    # 4. “重写查询”后根据路由决策分发
    workflow.add_conditional_edges(
        "rewrite_query",
        lambda state: state["route"],
        {
            "web_search": "web_search",
            "hierarchical_search": "generate_response",
            "direct_chunk_search": "generate_response"
        }
    )

    # 5. “网络搜索”后生成答案
    workflow.add_edge("web_search", "generate_response")

    # 6. “生成答案”后评估相关性
    workflow.add_edge("generate_response", "grade_relevance")

    # 7. 根据“相关性评估”结果，决定是“复盘记忆”还是“重写”
    workflow.add_conditional_edges(
        "grade_relevance",
        lambda state: state["is_relevant"],
        {
            True: "consolidate_memory",    # 答案相关，去“复盘”并形成记忆
            False: "rewrite_query"        # 答案不相关，重写查询
        }
    )
    
    # 8. “复盘记忆”后，流程结束
    workflow.add_edge("consolidate_memory", END)

    # 编译图
    graph = workflow.compile()
    return graph
