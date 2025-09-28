# -*- coding: utf-8 -*-
"""
@desc: Agentic RAG系统主入口（已集成记忆管理指令）
"""

import uuid
from agentic_rag.graph import build_graph
from agentic_rag import memory

# 线程ID，用于LangGraph的持久化，这里我们用一个简单的UUID
thread_id = str(uuid.uuid4())

config = {"configurable": {"thread_id": thread_id}}

def handle_memory_commands(query: str) -> bool:
    """处理用户输入的记忆管理指令，如果处理了指令则返回True。"""
    if query.strip() == '!show_memories':
        print("--- 当前的长期记忆 (最近10条) ---")
        mems = memory.view_memories(limit=10)
        if not mems:
            print("记忆库为空。")
        else:
            for i, mem in enumerate(mems):
                print(f"{i+1}. [ID: {mem['id']}, 类型: {mem['type']}, 重要性: {mem['importance']}] - {mem['text']}")
        return True

    elif query.startswith('!forget'):
        topic = query.replace('!forget', '').strip()
        if not topic:
            print("用法错误: 请提供要忘记的主题。例如: !forget 我的项目ID")
            return True
        
        print(f"--- 查找与 '{topic}' 相关的记忆 ---")
        retrieved_mems = memory.retrieve_memories(topic, top_k=5)
        if not retrieved_mems:
            print("未找到相关记忆。")
            return True

        print("找到以下相关记忆：")
        for i, mem in enumerate(retrieved_mems):
            print(f"{i+1}. [ID: {mem['id']}] - {mem['text']}")
        
        confirm = input("您确定要删除以上所有记忆吗? (y/n): ")
        if confirm.lower() == 'y':
            for mem in retrieved_mems:
                memory.delete_memory(mem['id'])
            print("相关记忆已删除。")
        else:
            print("操作已取消。")
        return True
        
    return False

def main():
    """主函数，运行Agentic RAG流程。"""
    # 在启动时确保记忆库已初始化
    memory.initialize_memory_db()
    
    graph = build_graph()

    print("欢迎使用Agentic RAG系统！")
    print("  - 输入问题与Agent对话。  ")
    print("  - 输入 '!show_memories' 查看记忆。  ")
    print("  - 输入 '!forget [主题]' 删除记忆。  ")
    print("  - 输入 'exit' 退出程序。  ")
    
    while True:
        query = input("\n请输入您的问题或指令: ")
        if query.lower() == 'exit':
            break
        if not query.strip():
            continue

        # 优先处理记忆管理指令
        if handle_memory_commands(query):
            continue
        
        # 如果不是指令，则正常执行Agent工作流
        inputs = {"query": query}
        print("\n--- 系统开始处理 ---")
        graph_config = {"recursion_limit": 10, **config}
        final_state = graph.invoke(inputs, config=graph_config)
        print("--- 系统处理结束 ---")

        print("\n最终答案:")
        print(final_state["response"])

if __name__ == "__main__":
    main()