# -*- coding: utf-8 -*-
"""
@desc: LangGraph工作流的节点

每个函数代表图中的一个节点，负责执行特定的操作，如路由、检索、生成等。
"""

from langchain_core.prompts import ChatPromptTemplate

from agentic_rag.chains import (
    get_query_router_chain, get_initial_rewriter_chain, get_correctional_rewriter_chain, 
    get_relevance_grader_chain, llm
)
from agentic_rag.retrievers import get_retriever, get_web_search_tool
from agentic_rag.state import AgentState

# --- 节点函数定义 ---

def route_query_node(state: AgentState) -> dict:
    """路由节点：预先从向量库检索，然后根据上下文决定下一步操作。"""
    print("--- 路由查询 ---")
    query = state["query"]
    
    # 1. 预检索
    print("--- 预检索文档 ---")
    retriever = get_retriever()
    documents = retriever.invoke(query)
    
    # 2. 带上下文的路由决策
    router_chain = get_query_router_chain()
    result = router_chain.invoke({"query": query, "context": documents})
    print(f"路由决策: {result['datasource']}")
    
    # 返回决策和已检索的文档
    return {
        "route": result['datasource'], 
        "documents": documents, 
        "query": query
    }

def web_search_node(state: AgentState) -> dict:
    """网络搜索节点：从互联网检索信息。"""
    print("--- 网络搜索 ---")
    updated_query = state["updated_query"]
    web_search = get_web_search_tool()
    documents = web_search.invoke({"query": updated_query})
    return {"documents": documents, "updated_query": updated_query}

def rewrite_query_node(state: AgentState) -> dict:
    """查询重写节点：根据上下文中是否存在失败的答案，智能选择重写策略。"""
    print("--- 重写查询 ---")
    query = state["query"]
    last_response = state.get("response")

    # 如果有上一次失败的答案，则调用修正性重写链
    if last_response:
        print("--- 检测到失败的答案，使用修正性重写 ---")
        rewriter_chain = get_correctional_rewriter_chain()
        result = rewriter_chain.invoke({"query": query, "response": last_response})
    # 否则，调用初始重写链
    else:
        print("--- 首次查询，使用初始重写 ---")
        rewriter_chain = get_initial_rewriter_chain()
        result = rewriter_chain.invoke({"query": query})
    
    print(f"重写后的查询: {result['rewritten_query']}")
    return {"updated_query": result['rewritten_query']}

def generate_response_node(state: AgentState) -> dict:
    """答案生成节点：根据上下文和查询生成答案。"""
    print("--- 生成答案 ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个问答机器人。请根据以下上下文信息来回答用户的问题。\n\n上下文:\n{context}"),
        ("human", "问题: {query}")
    ])
    chain = prompt | llm
    response = chain.invoke({"context": state["documents"], "query": state["updated_query"]})
    return {"response": response.content}

def direct_response_node(state: AgentState) -> dict:
    """直接回答节点：不经过检索，直接由LLM回答。"""
    print("--- 直接回答 ---")
    response = llm.invoke(state["query"])
    return {"response": response.content, "documents": []} # 保证documents字段存在

def grade_relevance_node(state: AgentState) -> dict:
    """相关性评估节点：判断答案是否切题。"""
    print("--- 评估答案相关性 ---")
    grader_chain = get_relevance_grader_chain()
    result = grader_chain.invoke({"query": state["query"], "response": state["response"]})
    print(f"答案是否相关: {result['is_relevant']}")
    if result['is_relevant']:
        return {"is_relevant": True}
    else:
        print("--- 答案不相关，将触发重写 ---")
        return {"is_relevant": False}
