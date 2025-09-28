# Agentic RAG 智能问答系统

## 1. 项目简介

本项目是一个基于 **LangGraph** 和大语言模型（LLM）实现的 **Agentic RAG (检索增强生成)** 系统。它融合了动态查询分析和自我纠错机制，能够根据用户问题的复杂度智能地选择最优的策略（直接回答、向量库检索或网络搜索），并对生成的答案进行相关性评估，从而实现更高质量的问答效果。

## 2. 核心特性

- **智能路由**：根据问题类型自动选择直接回答、向量检索或网络搜索。
- **自我纠错**：当生成的答案与问题不相关时，系统会自动改写问题并重新尝试，形成一个修正循环。
- **动态知识源**：集成了本地向量数据库（ChromaDB）和实时网络搜索（Tavily）。
- **高度可配置**：支持切换不同的语言模型和嵌入模型，兼容OpenAI API格式的各种模型（包括本地部署的Ollama、VLLM等）和本地Sentence-Transformers模型。
- **模块化设计**：代码结构清晰，分为状态、节点、链、图等模块，便于维护和二次开发。
- **智能混合检索 (Intelligent Hybrid Retrieval)**：Agent可根据问题类型（概念理解 vs. 精确查找）自动选择最优检索策略（分层检索或直接检索），解决了特定实体查不准、查不全的问题。

## 3. 架构升级：智能混合检索

为了根治“大海捞针”和“特定条目查不全”的矛盾，本项目架构已升级为**智能混合检索（Intelligent Hybrid Retrieval）**模式。

Agent的核心不再是单一的检索链路，而是一个更聪明的**智能调度器**。它首先仅根据用户问题本身进行意图分析，然后决定调用哪一种最合适的“武器”去获取知识：

1.  **直接区块检索 (Direct Chunk Search)**：当用户查询一个**具体实体**（如药品名、产品型号）时，Agent会选择此策略。它将直接在最细粒度的“区块层”进行高精度搜索，确保最高的查全率。

2.  **分层检索 (Hierarchical Search)**：当用户提出一个**开放性、概念性**问题时，Agent则采用此策略。它会先通过“摘要层”定位相关文档，再到文档内部寻找答案，保证了上下文的完整性。

这种设计让Agent能“因材施教”，针对不同问题采用不同策略，兼顾了查全率与查准率，是构建强大RAG系统的关键。

## 4. 系统流程

系统的工作流程已升级，其核心步骤如下：

1.  **智能路由 (Route Query)**：接收用户问题，LLM**仅根据问题本身**分析其意图，决定最合适的处理路径（`direct_chunk_search`, `hierarchical_search`, `web_search`, `direct`）。

2.  **知识检索 (Retrieve)**：**仅当**路由决策需要从本地知识库检索时，此步骤被激活。Agent会根据第1步的决策，调用对应的检索函数（直接区块检索或分层检索）来获取上下文。

3.  **重写查询 (Rewrite Query)**：如果需要从知识库或网络检索，首先使用LLM优化和改写用户问题，使其更适合作为后续检索的输入。

4.  **生成答案 (Generate Response)**：LLM结合检索到的上下文信息和优化后的问题来生成答案。

5.  **评估相关性 (Grade Relevance)**：使用LLM评估生成的答案是否与原始问题相关。
    -   **如果相关**：流程结束，将答案返回给用户。
    -   **如果不相关**：流程自动跳转回第3步（重写查询），开始新一轮的尝试。

这个循环修正的机制确保了最终答案的质量和相关性。

## 4. 技术栈

- **语言**: Python 3.11+
- **流程编排**: LangGraph
- **LLM框架**: LangChain
- **核心语言模型**: Qwen, GPT, 或任何兼容OpenAI API的自定义模型
- **向量数据库**: ChromaDB
- **嵌入模型**: 支持OpenAI兼容模型和本地Sentence-Transformers模型
- **网络搜索**: Tavily Search

## 5. 安装与设置

### 步骤 1: 创建虚拟环境 (推荐)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 步骤 2: 安装依赖

```bash
pip install -r requirements.txt
```

### 步骤 3: 设置环境变量

1.  将 `.env.example` 文件复制并重命名为 `.env`。
2.  打开 `.env` 文件，填入您的API密钥和自定义API地址。

    ```dotenv
    # OpenAI 或 兼容OpenAI的模型的API Key
    OPENAI_API_KEY="your_openai_api_key"

    # 用于网络搜索的 Tavily API Key
    TAVILY_API_KEY="your_tavily_api_key"

    # --- Custom Endpoints ---
    # 用于聊天模型的主URL (例如: "http://localhost:11434/v1")
    OPENAI_API_BASE=

    # 如果您的嵌入模型使用了与聊天模型不同的URL，请在此处设置。
    # 如果留空，将默认使用上面的 OPENAI_API_BASE。
    EMBEDDING_API_BASE=
    ```

## 6. 配置说明

您可以在 `config.py` 文件中调整系统的核心行为：

- `LLM_MODEL_NAME`: 设置聊天模型的名称 (例如: `"gpt-4o"`, `"qwen-turbo"`)。
- `EMBEDDING_PROVIDER`: 选择嵌入模型的提供商。
  - `"openai"`: 使用兼容OpenAI API的模型。
  - `"local"`: 使用本地Sentence-Transformers模型。
- `EMBEDDING_MODEL_NAME`: 当提供商为`openai`时，指定嵌入模型的名称。
- `LOCAL_EMBEDDING_MODEL_PATH`: 当提供商为`local`时，指定本地模型的路径或HuggingFace Hub名称。

## 7. 如何运行

### 步骤 1: 初始化本地知识库 (首次运行必需)

1.  将您自己的知识库文档 (如 .txt, .md, .pdf, .xlsx 文件) 放入 `data` 文件夹中。
2.  运行以下命令来处理这些文档并构建向量数据库：

    ```bash
    python ingest.py
    ```
    该脚本会读取`data`目录下的所有文件，将它们处理并存储到 `chroma_db` 目录中。如果您的文档很多，或者您选择使用本地嵌入模型，此过程可能需要一些时间。

### 步骤 2: 运行主程序

知识库初始化完成后，运行主程序：

```bash
python main.py
```
程序启动后，您可以直接在命令行中输入问题进行交互。

### 步骤 3: 管理长期记忆

您可以通过特定的指令来与Agent的长期记忆进行交互：

- **查看记忆**: 输入 `!show_memories` 来查看Agent最近记住的10条关键信息。

- **删除记忆**: 输入 `!forget [您想忘记的主题]` 来让Agent删除相关的记忆。例如：
    ```
    !forget 我的项目ID
    ```
    系统会找出与该主题相关的记忆，并请求您最终确认是否删除。

### 运行模型连通性测试

如果您不确定自定义模型的配置是否正确，可以运行测试脚本：

```bash
python test_custom_model.py
```
该脚本会验证聊天模型的API地址和模型名称是否可用。
