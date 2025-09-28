# -*- coding: utf-8 -*-
"""
@desc: LLM链模块

定义了系统中使用的各种LLM链，例如查询路由、查询重写和答案评估。
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.utils import embedding_functions

from config import (
    LLM_MODEL_NAME, OPENAI_API_BASE,
    EMBEDDING_PROVIDER, EMBEDDING_API_BASE, EMBEDDING_MODEL_NAME, LOCAL_EMBEDDING_MODEL_PATH
)

# --- LLM 初始化 ---
# 构造LLM参数
llm_params = {
    "model": LLM_MODEL_NAME,
    "temperature": 0
}
# 如果配置了自定义API地址，则使用它
if OPENAI_API_BASE:
    llm_params["base_url"] = OPENAI_API_BASE

# 使用 config.py 中定义的模型和可选的自定义API地址
llm = ChatOpenAI(**llm_params)

def get_embedding_function():
    """根据配置获取嵌入模型函数。"""
    if EMBEDDING_PROVIDER == 'openai':
        print("--- 使用OpenAI嵌入模型 ---")
        embedding_params = {
            "model": EMBEDDING_MODEL_NAME
        }
        # 优先使用独立的嵌入模型API地址，否则回退到主API地址
        api_base = EMBEDDING_API_BASE or OPENAI_API_BASE
        if api_base:
            embedding_params["base_url"] = api_base
        return OpenAIEmbeddings(**embedding_params)
    
    elif EMBEDDING_PROVIDER == 'local':
        print(f"--- 使用ChromaDB原生本地嵌入模型: {LOCAL_EMBEDDING_MODEL_PATH} ---")
        return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=LOCAL_EMBEDDING_MODEL_PATH)
        
    else:
        raise ValueError(f"未知的嵌入模型提供商: {EMBEDDING_PROVIDER}。请选择 'openai' 或 'local'。")

# --- 输出数据结构定义 ---

class RouteQuery(BaseModel):
    """根据用户问题决定路由策略。"""
    datasource: str = Field(description="根据问题，选择‘vectorstore’或‘web_search’或‘direct’中的一种。")

class RewriteQuery(BaseModel):
    """一个经过优化的、更适合检索的用户问题版本。"""
    rewritten_query: str = Field(description="对原始问题的改写，使其更适合搜索引擎或向量数据库。")

class RelevanceGrade(BaseModel):
    """评估答案是否与原始问题相关。"""
    is_relevant: bool = Field(description="布尔值，表示答案是否相关。")

# --- LLM 链定义 ---

def get_query_router_chain():
    """获取查询路由链"""
    parser = JsonOutputParser(pydantic_object=RouteQuery)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个查询路由专家。下面提供了一些从本地知识库检索到的、可能与用户问题相关的文档。请根据用户问题和这些文档，判断最合适的知识源。\n\n--- 上下文文档 ---\n{context}\n--- 上下文文档结束 ---\n\n决策指南：\n1. 如果上下文文档内容与用户问题高度相关，能够直接用于回答问题，请选择‘vectorstore’。\n2. 如果上下文文档不相关，但问题看起来需要最新的信息或广泛的通用知识，请选择‘web_search’。\n3. 如果问题非常简单，模型自身的知识就足够回答（例如“你好”），请选择‘direct’。\n\n{format_instructions}"),
        ("human", "问题: {query}")
    ]).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser

def get_initial_rewriter_chain():
    """获取初始查询重写链"""
    parser = JsonOutputParser(pydantic_object=RewriteQuery)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位查询优化专家。请将给定的问题改写成一个更适合在网络搜索引擎或向量数据库中检索的版本，使其更清晰、更具体。\n{format_instructions}"),
        ("human", "原始问题: {query}")
    ]).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser

def get_correctional_rewriter_chain():
    """获取修正性查询重写链"""
    parser = JsonOutputParser(pydantic_object=RewriteQuery)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位查询优化专家。用户之前的查询未能得到相关的答案。请分析原始问题和这个不满意的答案，然后将问题改写得更清晰、更具体，以便更好地检索。\n{format_instructions}"),
        ("human", "原始问题: {query}\n不满意的答案: {response}")
    ]).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser

def get_relevance_grader_chain():

    """获取相关性评估链"""
    parser = JsonOutputParser(pydantic_object=RelevanceGrade)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一位信息相关性评估专家。请根据用户问题，判断提供的答案是否相关。只需回答‘True’或‘False’。\n{format_instructions}"),
        ("human", "问题: {query}\n答案: {response}")
    ]).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm | parser

def get_summarizer_chain():
    """获取文档摘要链"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个文档摘要专家。请为以下文档生成一个简洁但全面的摘要，摘要应捕获所有核心主题、关键实体和结论，以便后续能通过摘要判断文档与用户问题的相关性。"),
        ("human", "文档内容:\n\n{document_content}")
    ])
    return prompt | llm