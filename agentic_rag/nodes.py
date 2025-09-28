# -*- coding: utf-8 -*-
"""
@desc: LangGraph工作流的节点（已集成长期记忆）
"""

from langchain_core.prompts import ChatPromptTemplate

from agentic_rag.chains import (
    get_query_router_chain, get_initial_rewriter_chain, get_correctional_rewriter_chain, 
    get_relevance_grader_chain, get_memory_consolidation_chain, llm
)
from agentic_rag.hierarchical_retriever import hierarchical_retriever, direct_chunk_retriever
from agentic_rag.retrievers import get_web_search_tool
from agentic_rag.state import AgentState
from agentic_rag import memory

# --- 新增：记忆相关节点 ---

def retrieve_memory_node(state: AgentState) -> dict:
    """在流程开始时，根据用户问题检索长期记忆。"""
    print("--- 检索长期记忆 ---")
    query = state["query"]
    retrieved_memories = memory.retrieve_memories(query)
    # 将记忆格式化为字符串，以便注入Prompt
    memories_text = "\n".join([mem['text'] for mem in retrieved_memories])
    if not memories_text:
        memories_text = "无相关历史记忆。"
    print(f"检索到的记忆: {memories_text}")
    return {
        "retrieved_memories": memories_text,
        "conversation_history": [] # 初始化对话历史
    }

def consolidate_memory_node(state: AgentState) -> dict:
    """在流程结束时，提炼并存储本次对话的关键信息。"""
    print("--- 复盘并巩固记忆 ---")
    # 将最终的问答对加入历史
    history = state["conversation_history"]
    history.append(("Human", state["query"]))
    history.append(("AI", state["response"]))

    # 格式化历史以供LLM分析
    history_text = "\n".join([f"{role}: {text}" for role, text in history])
    
    consolidation_chain = get_memory_consolidation_chain()
    try:
        result = consolidation_chain.invoke({"conversation_history": history_text})
        if result and result.text and "No valuable information" not in result.text:
            memory.add_memory(text=result.text, type=result.type, importance=result.importance)
    except Exception as e:
        # 如果记忆提炼失败，不影响主流程
        print(f"记忆提炼失败: {e}")
    
    return {}

# --- 现有节点改造 ---

def route_query_node(state: AgentState) -> dict:
    """智能路由节点：现在会利用检索到的记忆来辅助决策。"""
    print("--- 智能路由与调度 ---")
    query = state["query"]
    memories = state["retrieved_memories"]
    
    # 1. 调用已升级的、能接收记忆的路由链
    router_chain = get_query_router_chain()
    result = router_chain.invoke({"query": query, "memories": memories})
    route = result['datasource']
    print(f"路由决策: {route}")

    # 2. 根据决策执行操作
    documents = []
    if route == 'hierarchical_search':
        documents = hierarchical_retriever(query)
    elif route == 'direct_chunk_search':
        documents = direct_chunk_retriever(query)
    
    # 记录到对话历史
    history = state.get("conversation_history", [])
    history.append(("Human", query))

    return {"route": route, "documents": documents, "conversation_history": history}

def web_search_node(state: AgentState) -> dict:
    """网络搜索节点"""
    print("--- 网络搜索 ---")
    updated_query = state["updated_query"]
    web_search = get_web_search_tool()
    documents = web_search.invoke({"query": updated_query})
    return {"documents": documents}

def rewrite_query_node(state: AgentState) -> dict:
    """查询重写节点"""
    print("--- 重写查询 ---")
    query = state["query"]
    last_response = state.get("response")

    if last_response:
        rewriter_chain = get_correctional_rewriter_chain()
        result = rewriter_chain.invoke({"query": query, "response": last_response})
    else:
        rewriter_chain = get_initial_rewriter_chain()
        result = rewriter_chain.invoke({"query": query})
    
    print(f"重写后的查询: {result['rewritten_query']}")
    return {"updated_query": result['rewritten_query']}

def generate_response_node(state: AgentState) -> dict:
    """答案生成节点"""
    print("--- 生成答案 ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个问答机器人。请根据以下上下文信息来回答用户的问题。\n\n上下文:\n{context}"),
        ("human", "问题: {query}")
    ])
    chain = prompt | llm
    response = chain.invoke({"context": state["documents"], "query": state["updated_query"]})
    
    # 记录到对话历史
    history = state.get("conversation_history", [])
    history.append(("AI", response.content))

    return {"response": response.content, "conversation_history": history}

def direct_response_node(state: AgentState) -> dict:
    """直接回答节点"""
    print("--- 直接回答 ---")
    response = llm.invoke(state["query"])
    
    # 记录到对话历史
    history = state.get("conversation_history", [])
    history.append(("AI", response.content))

    return {"response": response.content, "documents": [], "conversation_history": history}

def grade_relevance_node(state: AgentState) -> dict:
    """相关性评估节点"""
    print("--- 评估答案相关性 ---")
    grader_chain = get_relevance_grader_chain()
    result = grader_chain.invoke({"query": state["query"], "response": state["response"]})
    if result['is_relevant']:
        print("答案相关，流程结束。")
        return {"is_relevant": True}
    else:
        print("--- 答案不相关，将触发重写 ---")
        return {"is_relevant": False}