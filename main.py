# -*- coding: utf-8 -*-
"""
@desc: Agentic RAG系统主入口
"""

import uuid
from agentic_rag.graph import build_graph

# 线程ID，用于LangGraph的持久化，这里我们用一个简单的UUID
thread_id = str(uuid.uuid4())

config = {"configurable": {"thread_id": thread_id}}

def main():
    """主函数，运行Agentic RAG流程。"""
    # 构建图
    graph = build_graph()

    print("欢迎使用Agentic RAG系统！输入 'exit' 退出程序。")
    
    while True:
        # 获取用户输入
        query = input("\n请输入您的问题: ")
        if query.lower() == 'exit':
            break
        
        # 定义图的输入
        inputs = {"query": query}

        # 执行图
        print("\n--- 系统开始处理 ---")
        # 增加 recursion_limit 作为安全限制，防止无限循环
        graph_config = {"recursion_limit": 10, **config}
        final_state = graph.invoke(inputs, config=graph_config)
        print("--- 系统处理结束 ---")

        # 打印最终答案
        print("\n最终答案:")
        print(final_state["response"])

if __name__ == "__main__":
    main()
