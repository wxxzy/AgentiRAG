# Agentic RAG 评估模块说明

本目录包含了对 Agentic RAG 系统进行量化评估所需的脚本和数据。

## 目录结构

```
evaluation/
├── README.md                 # 本说明文件
├── golden_dataset.csv        # 用于评估的“黄金标准”测试数据集
├── evaluation.py             # 执行评估的主脚本
├── router_confusion_matrix.png # (输出) 路由环节性能混淆矩阵图
└── generator_ragas_report.csv  # (输出) 生成环节Ragas评估报告
```

---

## 如何使用

#### 步骤 1: 安装依赖

首先，请确保您已安装所有必要的Python库。在项目虚拟环境中运行：

```bash
pip install pandas scikit-learn matplotlib ragas datasets tqdm
```

#### 步骤 2: 准备/扩充测试数据

评估的质量取决于测试数据的质量。请打开 `golden_dataset.csv` 文件，根据您自己的业务场景，添加更多有代表性的问题。

**各列说明:**

- `question_id`: 问题的唯一标识符。
- `question`: 用户提出的原始问题。
- `ideal_route`: 您期望系统为该问题选择的理想路径 (`vectorstore`, `web_search`, `direct`)。
- `ideal_context_keywords`: (可选) 用于评估检索质量，列出理想上下文中应包含的关键词。
- `ideal_answer_summary`: 理想答案的摘要，用于Ragas评估答案的正确性。

#### 步骤 3: 运行评估脚本

打开命令行，然后执行 `evaluation.py` 脚本。

```bash
python .\evaluation\evaluation.py
```

脚本会自动执行所有评估流程。

#### 步骤 4: 分析评估结果

脚本运行完毕后，会在当前目录生成两个核心产出文件：

1.  **`router_confusion_matrix.png`**: 
    - **用途**: 直观地展示路由节点的分类性能。
    - **如何解读**: 一个理想的混淆矩阵应该只有对角线上有数字。如果非对角线有数字，则表示发生了错误分类。例如，如果“vectorstore”行、“web_search”列的数字是“2”，代表有2个本应走向量检索的问题被错误地路由到了网络搜索。

2.  **`generator_ragas_report.csv`**: 
    - **用途**: 量化评估RAG的检索和生成质量。
    - **如何解读**: 打开此CSV文件，关注以下核心指标 (分数越高越好, 范围0-1):
        - `faithfulness`: 答案是否忠实于检索到的上下文，分数低表示可能存在“幻觉”。
        - `answer_relevancy`: 答案是否与原始问题高度相关，分数低表示“答非所问”。
        - `context_recall`: 检索到的上下文是否包含了生成理想答案所需的全部信息。

根据这些量化结果，您可以有针对性地去优化对应环节的Prompt或模型。

---

## 评估脚本 (`evaluation.py`) 功能说明

- **`evaluate_router()`**: 评估路由节点的分类准确率。它会加载 `golden_dataset.csv`，逐一测试问题，并与 `ideal_route` 对比，最终生成分类报告和混淆矩阵图。

- **`evaluate_generator_and_retriever()`**: 评估检索和生成两个环节的综合表现。它使用 `Ragas` 框架，针对 `ideal_route` 为 `vectorstore` 的问题，进行端到端的测试，并计算多个核心RAG指标。

- **`evaluate_grader()`**: 这是一个占位函数，用于提示如何评估“相关性评估”节点。您需要仿照 `evaluate_router` 的逻辑，创建一个专门的数据集来测试这个二分类节点的性能。
