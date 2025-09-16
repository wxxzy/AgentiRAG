# -*- coding: utf-8 -*-
"""
@desc: 测试自定义 OpenAI API Base 模型是否可用。

本脚本会加载 .env 和 config.py 中的配置，
尝试连接您指定的自定义模型并获取一个简单的回复。
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

print("--- 开始测试自定义模型连通性 ---")

# 1. 加载配置
try:
    # 从 .env 文件加载环境变量
    load_dotenv()
    # 从 config.py 加载配置
    from config import LLM_MODEL_NAME, OPENAI_API_BASE, OPENAI_API_KEY
    print("配置加载成功。")
except ImportError:
    print("\n错误：无法从 config.py 导入配置。请确保该文件存在且无语法错误。")
    exit(1)

# 2. 检查关键配置是否存在
if not OPENAI_API_BASE:
    print("\n错误：OPENAI_API_BASE 未在 .env 或 config.py 中设置。")
    print("请在 .env 文件中添加 OPENAI_API_BASE=\"http://your-api-url/v1\"")
    exit(1)

if not LLM_MODEL_NAME:
    print("\n错误：LLM_MODEL_NAME 未在 config.py 中设置。")
    exit(1)

print(f"测试目标API Base: {OPENAI_API_BASE}")
print(f"测试目标模型: {LLM_MODEL_NAME}")
# 注意：出于安全考虑，不打印API Key

# 3. 初始化模型
try:
    llm = ChatOpenAI(
        model=LLM_MODEL_NAME,
        base_url=OPENAI_API_BASE,
        api_key=OPENAI_API_KEY or "dummy-key", # 很多本地模型不需要key，但sdk要求非空
        temperature=0.1,
        request_timeout=15 # 设置15秒超时
    )
    print("ChatOpenAI 客户端初始化成功。")
except Exception as e:
    print(f"\n错误：初始化 ChatOpenAI 客户端时发生异常。\n")
    print(f"异常类型: {type(e).__name__}\n")
    print(f"异常信息: {e}")
    exit(1)

# 4. 发送测试请求
print("\n--- 正在发送测试请求... ---")
try:
    prompt = "你好，请用一句话介绍你自己。"
    response = llm.invoke(prompt)
    
    print("\n✅ 测试成功！模型连接正常。")
    print(f"模型回复: {response.content}")

except Exception as e:
    print(f"\n❌ 测试失败！无法从模型获取回复。")
    print(f"异常类型: {type(e).__name__}\n")
    
    # 提供针对性的错误建议
    if "Connection refused" in str(e) or "timed out" in str(e):
        print("建议: 请检查您的API地址是否正确，以及模型服务是否正在运行。")
    elif "401" in str(e):
        print("建议: 认证失败。请检查您的 OPENAI_API_KEY 是否正确有效。")
    elif "404" in str(e):
        print("建议: 找不到模型。请检查您在 config.py 中设置的 LLM_MODEL_NAME 是否正确，以及您的服务端点是否支持该模型。")
    else:
        print("建议: 请检查错误信息详情，确认服务端状态和网络连接。")
        
    print(f"\n详细错误信息:\n{e}")
