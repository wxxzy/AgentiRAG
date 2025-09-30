# -*- coding: utf-8 -*-
"""
@desc: 配置模块，用于加载环境变量和管理配置。
"""
import os
from dotenv import load_dotenv

# 从 .env 文件加载环境变量
load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# 如果使用Qwen，请取消下面的注释
# DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# --- LLM --- 
# 使用的模型名称 (例如 "gpt-4o", "deepseek-v3-1-terminus")
LLM_MODEL_NAME = "doubao-seed-1-6-250615"
# LLM_MODEL_NAME = "qwen-turbo"

# 如果您使用自定义的、兼容OpenAI API的端点（例如Ollama, LocalAI等），请在此处设置其URL
# 例如: "http://localhost:11434/v1"
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", None)

# --- Embedding ---

# 选择嵌入模型的提供商: 'openai' 或 'local'
# 'openai': 使用兼容OpenAI API的嵌入模型 (包括OpenAI官方、Azure、Ollama等)。
# 'local': 使用本地句向量模型 (SentenceTransformers/HuggingFace)。
EMBEDDING_PROVIDER = "local" # 可选 'openai' 或 'local'

# -- OpenAI 嵌入模型配置 (当 EMBEDDING_PROVIDER = 'openai') --
# 如果嵌入模型的API地址与主模型不同，请在此处设置
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", None)
# 使用的嵌入模型名称。
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

# -- 本地嵌入模型配置 (当 EMBEDDING_PROVIDER = 'local') --
# 指定本地模型的路径或HuggingFace模型库的ID
# 例如: 'sentence-transformers/all-MiniLM-L6-v2'
LOCAL_EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"

# --- Excel 数据加载配置 ---
# 在加载Excel文件时，指定哪些列应该被提取为文档的元数据。
# 这些列的值将作为键值对存储在向量库中，用于后续的过滤或更精确的检索。
EXCEL_METADATA_COLUMNS = ["药品名称", "生产企业", "批准文号", "药品编码", "本位码"]
