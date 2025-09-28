# Agentic RAG 智能问答系统

## 1. 项目简介

本项目是一个基于 **LangGraph** 和大语言模型（LLM）实现的 **Agentic RAG (检索增强生成)** 系统。它融合了动态查询分析和自我纠错机制，能够根据用户问题的复杂度智能地选择最优的策略（直接回答、向量库检索或网络搜索），并对生成的答案进行相关性评估，从而实现更高质量的问答效果。

## 2. 核心特性

- **智能路由**：根据问题类型自动选择直接回答、向量检索或网络搜索。
- **自我纠错**：当生成的答案与问题不相关时，系统会自动改写问题并重新尝试，形成一个修正循环。
- **动态知识源**：集成了本地向量数据库（ChromaDB）和实时网络搜索（Tavily）。
- **高度可配置**：支持切换不同的语言模型和嵌入模型，兼容OpenAI API格式的各种模型（包括本地部署的Ollama、VLLM等）和本地Sentence-Transformers模型。
- **模块化设计**：代码结构清晰，分为状态、节点、链、图等模块，便于维护和二次开发。
- **分层检索 (Hierarchical Retrieval)**：采用“摘要-区块”两级索引，先通过摘要快速定位相关文档，再从相关文档中提取精准的上下文区块，显著提升复杂知识库下的检索精度和效率。

## 3. 架构升级：分层检索

为了解决传统RAG在大型知识库中检索精度不足和上下文碎片化的问题，本项目已升级为**分层检索（Hierarchical Retrieval）**架构。

其核心思想是将知识索引分为两个层级：

1.  **摘要层 (Summary Level)**：这一层存储了每份完整文档的AI生成摘要。它像一本书的目录，帮助系统快速理解每份文档的核心内容。
2.  **区块层 (Chunk Level)**：这一层存储了从文档中切分出的、细粒度的文本块，用于提供最终答案所需的精确引文。

检索过程分为两步：

- **第一步：文档定位**：用户的查询首先与“摘要层”进行匹配，快速找到最相关的几份**文档**。
- **第二步：内容提取**：系统仅在这几份已锁定的相关文档中，对其包含的“区块”进行二次检索，从而找到最精准的**上下文**。

这种“先找书，再翻页”的模式，极大地提高了检索的信噪比，为生成高质量答案提供了坚实的基础。

## 4. 系统流程

系统的工作流程遵循一个有向图（Graph），其核心步骤如下：

1.  **路由查询 (Route Query)**：接收用户问题，使用LLM分析其意图，决定最合适的处理路径（`direct`, `vectorstore`, `web_search`）。
2.  **直接回答 (Direct Response)**：如果问题简单，LLM直接生成答案。
3.  **重写查询 (Rewrite Query)**：如果需要检索，首先使用LLM优化和改写用户问题，使其更适合作为搜索引擎或向量数据库的输入。
4.  **知识检索 (Retrieve/Search)**：根据第1步的路由决策，从本地向量数据库或互联网上检索相关信息。
5.  **生成答案 (Generate Response)**：LLM结合检索到的上下文信息和优化后的问题来生成答案。
6.  **评估相关性 (Grade Relevance)**：使用LLM评估生成的答案是否与原始问题相关。
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

### 运行模型连通性测试

如果您不确定自定义模型的配置是否正确，可以运行测试脚本：

```bash
python test_custom_model.py
```
该脚本会验证聊天模型的API地址和模型名称是否可用。
