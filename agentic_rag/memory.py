# -*- coding: utf-8 -*-
"""
@desc: 长期记忆模块

负责Agent长期记忆的存储、检索和管理。
采用 SQLite + ChromaDB 的混合存储方案：
- SQLite: 存储记忆的结构化文本和元数据。
- ChromaDB: 存储记忆的向量嵌入，用于语义检索。
"""

import os
import sqlite3
import datetime
import math
import chromadb

# 动态地将根目录加入sys.path，以便能导入项目内的模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_rag.chains import get_embedding_function

# --- 配置 ---
DB_PATH = "long_term_memory.sqlite"
PERSIST_PATH = "chroma_db"
MEMORY_COLLECTION_NAME = "long_term_memory"

# --- 数据库初始化与连接 ---

def get_db_connection():
    """获取SQLite数据库连接。"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_memory_db():
    """初始化记忆库，如果不存在则创建表和集合。"""
    print("--- 初始化长期记忆库 ---")
    # 1. 初始化SQLite数据库
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                type TEXT NOT NULL DEFAULT 'fact',
                importance INTEGER NOT NULL DEFAULT 5,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_accessed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        print(f"SQLite数据库 '{DB_PATH}' 已确保存在。")

    # 2. 初始化ChromaDB集合
    client = chromadb.PersistentClient(path=PERSIST_PATH)
    # 在创建时就指定嵌入函数
    collection_embedding_function = get_embedding_function()
    client.get_or_create_collection(name=MEMORY_COLLECTION_NAME, embedding_function=collection_embedding_function)
    print(f"ChromaDB集合 '{MEMORY_COLLECTION_NAME}' 已确保存在。")
    print("--- 长期记忆库初始化完成 ---")

# --- 核心功能：增、删、查、改 ---

def add_memory(text: str, type: str = 'fact', importance: int = 5):
    """添加一条新的记忆。"""
    print(f"--- 添加新记忆 (类型: {type}, 重要性: {importance}) ---")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        now = datetime.datetime.now()
        cursor.execute(
            "INSERT INTO memories (text, type, importance, created_at, last_accessed_at) VALUES (?, ?, ?, ?, ?)",
            (text, type, importance, now, now)
        )
        memory_id = cursor.lastrowid
        conn.commit()

    # 将向量存入ChromaDB
    client = chromadb.PersistentClient(path=PERSIST_PATH)
    collection = client.get_collection(name=MEMORY_COLLECTION_NAME)
    collection.add(
        ids=[str(memory_id)],
        documents=[text],
        metadatas=[{"type": type, "importance": importance, "sqlite_id": memory_id}]
    )
    print(f"记忆已存入，ID: {memory_id}")

def retrieve_memories(query_text: str, top_k: int = 3) -> list[dict]:
    """根据查询，使用混合加权算法检索最相关的记忆。"""
    print(f"--- 检索与 '{query_text[:20]}...' 相关的长期记忆 ---")
    client = chromadb.PersistentClient(path=PERSIST_PATH)
    collection = client.get_collection(name=MEMORY_COLLECTION_NAME)

    # 1. 语义检索 (获取比top_k更多的候选，以便重排)
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k * 3, 
    )

    if not results or not results.get('ids') or not results['ids'][0]:
        return []

    # 2. 混合评分与重排序
    ranked_memories = []
    now = datetime.datetime.now()
    with get_db_connection() as conn:
        for id_str, distance, meta in zip(results['ids'][0], results['distances'][0], results['metadatas'][0]):
            cursor = conn.cursor()
            res = cursor.execute("SELECT * FROM memories WHERE id = ?", (int(id_str),)).fetchone()
            if not res:
                continue

            # a. 语义分 (Chroma的distance是L2距离，转换为0-1的相似度)
            semantic_score = 1.0 / (1.0 + distance)

            # b. 重要性分
            importance_score = res['importance']

            # c. 热度（新近度）分
            last_accessed = datetime.datetime.fromisoformat(res['last_accessed_at'])
            hours_since_accessed = (now - last_accessed).total_seconds() / 3600
            recency_score = 1.0 / (1.0 + math.log1p(hours_since_accessed))

            # d. 最终加权得分
            # 权重可以根据经验调整
            final_score = semantic_score * (1 + 0.1 * importance_score) * (1 + 0.5 * recency_score)
            
            ranked_memories.append({
                "id": res['id'],
                "text": res['text'],
                "type": res['type'],
                "score": final_score
            })

    # 按最终得分降序排序
    ranked_memories.sort(key=lambda x: x['score'], reverse=True)

    # 3. 更新被访问记忆的时间戳并返回top_k
    top_memories = ranked_memories[:top_k]
    retrieved_ids = [mem['id'] for mem in top_memories]
    if retrieved_ids:
        print(f"检索到的Top-{len(retrieved_ids)} 记忆ID: {retrieved_ids}")
        cursor = conn.cursor()
        cursor.execute(f"UPDATE memories SET last_accessed_at = ? WHERE id IN ({','.join('?'*len(retrieved_ids))})", (now, *retrieved_ids))
        conn.commit()

    return top_memories

def delete_memory(memory_id: int):
    """根据ID删除一条记忆。"""
    print(f"--- 删除记忆 ID: {memory_id} ---")
    # 从SQLite删除
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.commit()
        if cursor.rowcount == 0:
            print(f"警告：在SQLite中未找到ID为 {memory_id} 的记忆。")

    # 从ChromaDB删除
    client = chromadb.PersistentClient(path=PERSIST_PATH)
    collection = client.get_collection(name=MEMORY_COLLECTION_NAME)
    collection.delete(ids=[str(memory_id)])
    print("记忆已从数据库中删除。")

def view_memories(limit: int = 10):
    """查看最近的N条记忆。"""
    print(f"--- 查看最近的 {limit} 条记忆 ---")
    with get_db_connection() as conn:
        cursor = conn.cursor()
        rows = cursor.execute("SELECT * FROM memories ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

# --- 首次运行时可以执行初始化 ---
if __name__ == '__main__':
    initialize_memory_db()
    # 添加一些示例记忆
    print("\n--- 添加示例记忆 ---")
    add_memory("用户的项目ID是 'Project-Gemini'", type='fact', importance=9)
    add_memory("用户偏好使用中文进行交流", type='preference', importance=7)
    print("\n--- 查看记忆 ---")
    print(view_memories())
    print("\n--- 检索示例 ---")
    retrieved = retrieve_memories("我的项目是什么")
    print(retrieved)
