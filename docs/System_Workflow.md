# Agentic RAG 系统工作流程详解

## 1. 概述

本文档详细描述了 Agentic RAG 问答系统的内部工作流程。系统基于 LangGraph 构建，其核心是一个状态图（State Graph），通过一系列功能节点（Node）和条件边（Conditional Edge）的组合，实现对用户问题的智能分析、检索、生成和自我修正。

整个流程可以概括为：**接收问题 -> 思考策略 -> 检索信息 -> 生成答案 -> 评估修正 -> 给出答复**。

## 2. 核心节点 (Nodes)

节点是工作流中的基本处理单元，每个节点负责一项具体任务。

| 节点名称                 | 功能描述                                                                                             |
| ------------------------ | ---------------------------------------------------------------------------------------------------- |
| `retrieve_memory_node`   | **检索长期记忆**：流程的入口。根据当前问题，从长期记忆库中检索相关的历史对话或结论，为后续决策提供背景信息。 |
| `route_query_node`       | **智能路由与本地检索**：系统的“大脑”。它决定处理问题的最佳路径，并内置了**回退逻辑**。                     |
| `rewrite_query_node`     | **查询重写**：优化用户的原始问题，使其更适合作为搜索引擎或数据库的输入。分为“初始重写”和“纠错性重写”两种模式。 |
| `web_search_node`        | **网络搜索**：当需要获取最新信息或本地知识库没有相关内容时，调用此节点，使用 Tavily API 进行网络搜索。       |
| `generate_response_node` | **生成答案**：根据检索到的上下文信息（来自本地或网络）和重写后的问题，调用 LLM 生成最终答案。             |
| `direct_response_node`   | **直接回答**：当问题不需要任何外部知识（如“你好”）时，直接调用 LLM 进行回答。                               |
| `grade_relevance_node`   | **相关性评估**：对 `generate_response_node` 或 `direct_response_node` 生成的答案进行评估，判断其是否解决了用户问题。 |
| `consolidate_memory_node`| **复盘并巩固记忆**：在流程成功结束后，将本次问答的核心内容提炼成一条新的记忆，存入长期记忆库，用于未来的学习。 |

## 3. 工作流详解 (Edges & Conditions)

系统的工作流程由节点间的“边”连接而成，其中“条件边”负责根据当前状态（`AgentState`）进行决策。

---

### **第一步: 记忆检索 (`retrieve_memory`)**

- **触发**: 系统接收到用户问题，流程开始。
- **动作**: 调用 `retrieve_memory_node`，从 `long_term_memory.sqlite` 数据库中寻找与当前问题相关的记忆。同时，初始化对话历史和重试计数器 `correction_attempts = 0`。
- **流向**: 无条件流向 `route_query`。

---

### **第二步: 智能路由与回退决策 (`route_query`)**

- **触发**: 上一步完成。
- **动作**: 这是系统的核心决策点。
    1.  **路由判断**: 节点内的 `router_chain` 会根据用户问题和上一步检索到的记忆，决定一个初步的策略（`route`），可能的值为：
        - `direct`: 无需检索，直接回答。
        - `hierarchical_search`: 需要进行分层检索（适用于开放性、概念性问题）。
        - `direct_chunk_search`: 需要进行直接区块检索（适用于精确查找特定实体）。
        - `web_search`: 需要进行网络搜索。
    2.  **本地检索**: 如果初步策略是 `hierarchical_search` 或 `direct_chunk_search`，则立即执行相应的本地 ChromaDB 检索，并将结果存入 `documents` 列表。
    3.  **回退逻辑 (Fallback)**: 在本地检索执行完毕后，**在节点内部进行检查**：
        - **如果 `documents` 列表不为空**，说明本地检索成功，流程继续。
        - **如果 `documents` 列表为空**，说明本地知识库没有相关信息。此时，节点会**强制将 `route` 的值修改为 `web_search`**，并打印提示“本地检索无结果，转为网络搜索”。
- **流向**: 无条件流向下一步的“路由分发”条件边。

---

### **第三步: 路由分发 (Conditional Edge)**

- **触发**: `route_query` 节点完成。
- **决策**: 系统检查当前状态中的 `route` 值（此值可能已被上一步的回退逻辑修改）。
    - **如果 `route` 是 `direct`**: 流程走向 `direct_response_node`。
    - **如果 `route` 是任何一种搜索类型** (`web_search`, `hierarchical_search`, `direct_chunk_search`): 流程走向 `rewrite_query_node`。

---

### **第四步: 查询重写 (`rewrite_query`)**

- **触发**: 上一步的路由决策指向了搜索。
- **动作**:
    - **初始重写**: 如果这是第一次进入该节点，它会将用户问题改写得更适合机器检索。
    - **纠错性重写**: 如果是因为后续“答案不相关”而循环回到此节点，它会结合上一次失败的答案，对问题进行修正，尝试新的检索角度。
- **流向**: 无条件流向下一步的“检索执行”条件边。

---

### **第五步: 检索执行 (Conditional Edge)**

- **触发**: `rewrite_query` 节点完成。
- **决策**: 系统再次检查当前状态中的 `route` 值。
    - **如果 `route` 是 `web_search`**: 流程走向 `web_search_node`，去执行真正的网络搜索。
    - **如果 `route` 是 `hierarchical_search` 或 `direct_chunk_search`**: **流程直接走向 `generate_response_node`**。因为在第二步的 `route_query_node` 中，本地文档（`documents`）已经检索并加载到状态中了，无需重复检索。

---

### **第六步: 答案生成**

- **触发**:
    - `direct_response_node`: 由第三步的 `direct` 路由触发。
    - `web_search_node`: 完成网络搜索后，流向 `generate_response_node`。
    - `rewrite_query_node`: 对于本地检索，由第五步的条件决策触发，流向 `generate_response_node`。
- **动作**:
    - `direct_response_node`: 直接让 LLM 回答。
    - `generate_response_node`: 结合 `documents`（无论来自本地还是网络）和 `updated_query`，让 LLM 生成答案。
- **流向**: 所有答案生成后，统一流向 `grade_relevance_node` 进行评估。

---

### **第七步: 相关性评估与熔断 (Conditional Edge)**

- **触发**: `grade_relevance_node` 完成评估。
- **决策**: 这是系统的“自我修正”与“熔断”机制，由 `decide_after_grading` 函数控制。
    1.  **检查相关性**:
        - **如果答案被判定为 `is_relevant: True`**: 说明答案质量合格。决策为 `continue`，流程走向 `consolidate_memory_node`。
    2.  **检查重试次数**:
        - **如果答案 `is_relevant: False`**: 系统会检查重试计数器 `correction_attempts`。
        - **如果 `correction_attempts < 2`**: 说明还可以再试。决策为 `retry`，流程**循环回到 `rewrite_query_node`**，开始新一轮的纠错尝试。
        - **如果 `correction_attempts >= 2`**: 说明已经尝试了2次但仍失败。为避免无限循环，触发**熔断机制**。决策为 `end`，**流程直接终止**，并打印“已达到最大重试次数”的提示。

---

### **第八步: 记忆巩固与结束**

- **触发**: 上一步的决策为 `continue`。
- **动作**: `consolidate_memory_node` 会总结本次成功的问答，并存入长期记忆。
- **流向**:
    - 从 `consolidate_memory_node` -> `END` (流程正常结束)。
    - 从第七步的熔断机制 -> `END` (流程异常结束)。

## 4. 示例追踪：查询 “999感冒灵的商品详情”

为了更具体地理解上述流程，我们以一个实际查询为例，追踪其在系统内部的完整生命周期。

**场景**: 首次查询，此时本地知识库中没有任何关于“999感冒灵”的信息。

1.  **`retrieve_memory_node`**: 系统接收到问题 “999感冒灵的商品详情”。此时长期记忆库为空，节点返回 “无相关历史记忆”。
    - **状态更新**: `correction_attempts = 0`

2.  **`route_query_node`**: 
    - **路由判断**: 链分析问题，认为这是一个精确的实体查找，初步决策 `route = "direct_chunk_search"`。
    - **本地检索**: 系统使用 `direct_chunk_retriever` 在 ChromaDB 中搜索，但一无所获。
    - **回退逻辑**: 节点检查到 `documents` 列表为空，**触发回退机制**。它在内部将 `route` 的值从 `"direct_chunk_search"` 修改为 `"web_search"`。
    - **状态更新**: `route = "web_search"`, `documents = []`

3.  **路由分发 (条件边)**: 检测到 `state["route"]` 是 `"web_search"`，于是将流程导向 `rewrite_query_node`。

4.  **`rewrite_query_node`**: 接收到原始问题，进行“初始重写”，可能将其优化为更适合搜索引擎的查询，例如：“999感冒灵颗粒 药品说明书 成分 功效 用法用量 价格”。
    - **状态更新**: `updated_query = "..."`

5.  **检索执行 (条件边)**: 再次检测 `state["route"]`，值依然是 `"web_search"`，因此流程被导向 `web_search_node`。

6.  **`web_search_node`**: 使用重写后的查询 `updated_query` 调用 Tavily API，执行网络搜索，并将搜索结果（如网页摘要）填充到 `documents` 列表中。
    - **状态更新**: `documents = ["网络搜索结果1...", "网络搜索结果2..."]`

7.  **`generate_response_node`**: 接收到来自网络的 `documents` 和 `updated_query`，调用 LLM 生成一个包含商品详情的详细答案。
    - **状态更新**: `response = "根据网络信息，999感冒灵..."`

8.  **`grade_relevance_node`**: 将生成的答案与原始问题进行比较，LLM 评估认为答案是相关的。
    - **状态更新**: `is_relevant = True`

9.  **相关性评估与熔断 (条件边)**: `decide_after_grading` 函数检测到 `is_relevant` 为 `True`，返回 `"continue"`。

10. **`consolidate_memory_node`**: 流程走向终点前的最后一站。它分析整个成功的问答过程，可能会提炼出一条新的记忆，例如：“查询药品‘999感冒灵’的详细信息需要使用网络搜索”，并将其存入长期记忆库。

11. **`END`**: 流程成功结束，最终答案被返回给用户。
