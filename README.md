# Agent 开发学习助手

这是一个用于学习 Agent 开发的命令行项目。当前主程序是 `rag_agent.py`：它让 DeepSeek 决定何时调用知识检索工具，使用本地 BGE 向量模型查找资料，再由模型判断候选资料是否足以回答问题。会话状态通过 SQLite 保存。项目目前不是 Web 服务，也没有部署。

正常问候可以直接回答；Agent 开发相关问题会先查本地知识库。程序启动时会自动读取 `knowledge` 目录下的所有 Markdown 文件。目前知识库包含 `agent_basics.md` 和 `frameworks.md`。新增或修改知识文件后需要重启程序以重新读取资料；切块或向量配置变化时也需要重启。启动时程序会检查向量缓存，内容和配置未变则复用缓存，变化后重新生成全部文档向量。没有资料支持的问题应明确说明资料不足，不应当作通用搜索引擎。

## 核心流程

```text
用户问题 → DeepSeek 决定是否调用工具 → search_agent_knowledge
                                  → BGE 检索候选文本块
                                  → DeepSeek 判断候选资料是否能回答
          ← 工具结果 ←───────────────┘
          → DeepSeek 生成最终回答
```

`thread_id` 用来区分会话；同一编号的历史状态保存在 `data/rag_agent_memory.sqlite`，重启后可恢复。运行时会打印 `HumanMessage`、`AIMessage`、`ToolMessage` 等消息轨迹，方便观察模型决策和工具执行。数据库可能包含私人对话，不要将其上传到公开仓库。

## 知识库切块与向量缓存

`rag_demo.py` 先按 Markdown 的一级、二级标题把资料分成小节，并保留标题；超过 500 个字符的小节再继续切分，目标重叠量为 20 个字符。重叠只发生在同一小节内部。每块的 `metadata` 保留来源文件名和标题。当前的 `start_index` 表示该块在所属小节文本中的起始位置，不是原始 Markdown 文件中的偏移。

文档向量缓存在 `data/document_vectors.json`。启动时，程序根据切块后的正文、metadata、BGE 模型名称和切块参数计算指纹。指纹一致时复用已有文档向量；缓存不存在、损坏或指纹变化时，重新生成并保存全部文档向量。查询时仍需为用户问题生成向量。当前尚不支持逐文件增量更新。

## 环境与安装

已在 Windows PowerShell、Python 3.12.7 下运行。请在本项目目录打开终端，确认 `python --version` 显示 Python 3.12。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

命令直接使用虚拟环境中的 Python，不需要运行 `Activate.ps1`。首次加载 `BAAI/bge-small-zh-v1.5` 时，如果本机尚无模型缓存，FastEmbed 可能需要联网下载模型。`requirements.txt` 记录了当前环境的主要依赖版本，尚未在另一台全新电脑上验证。

## 设置密钥并运行

先准备自己的 DeepSeek API Key。不要把密钥写入代码、README 或提交到仓库。以下命令只在当前 PowerShell 进程中设置环境变量；输入密钥时不会显示明文：

```powershell
$secureKey = Read-Host "DeepSeek API Key" -AsSecureString
$credential = [System.Management.Automation.PSCredential]::new("unused", $secureKey)
$env:DEEPSEEK_API_KEY = $credential.GetNetworkCredential().Password
.\.venv\Scripts\python.exe .\rag_agent.py
Remove-Item Env:\DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
```

程序会先询问会话编号。再次输入相同编号可恢复该会话；直接回车使用 `default-session`。输入 `退出`、`exit` 或 `quit` 结束。真实模型调用会产生 API 用量和费用，包括检索候选资料时的模型判断。运行结束后请清除环境变量，或直接关闭这个终端窗口。

`start_rag_agent.ps1` 是当前电脑使用的便捷脚本，依赖用户目录下一个特定名称的加密密钥文件；它不是新电脑的通用启动方式。

## 测试

以下命令覆盖规则版 Agent、LangChain 工具、会话记忆、RAG 和评估逻辑，同时**不运行真实 DeepSeek API 测试**：

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_agent test_rag_demo test_rag_agent test_langchain_memory test_langchain_agent.LangChainToolTest test_langchain_agent.LangChainAgentTest
```

2026-10-05 在当前环境运行结果为 `Ran 50 tests`、`OK`。测试中可能打印预设的模拟错误，用于验证失败处理；只要最后是 `OK`，这些模拟错误不代表真实 API 调用失败。首次运行时，测试导入检索模块也可能触发本地 BGE 模型下载。不要把无参数的 `python -m unittest -v` 当作无费用测试命令：如果已设置 `DEEPSEEK_API_KEY`，它可能运行真实 API 测试。

真实两阶段评测可运行：

```powershell
.\start_rag_eval_llm.ps1
```

该命令会调用 DeepSeek 并产生 API 用量。2026-10-04，用户在调整切块后报告当前 14 条评测数据为 `14/14`。这个结果只代表当前小型评测集全部通过，不代表系统已经覆盖所有真实问题。

## 文件索引

- `rag_agent.py`：当前命令行 Agent、工具注册和 SQLite 会话配置。
- `rag_eval.py`：运行本地 BGE 基础检索评测，并从独立数据文件读取评测问题。
- `rag_eval_llm.py`：对前三个候选逐个进行 DeepSeek 精判，并统计最终评测结果。
- `rag_demo.py`：自动读取 `knowledge/*.md`，按标题和长度切块、记录来源、缓存 BGE 文档向量并执行相似度检索。
- `data/rag_eval_cases.json`：独立的 RAG 评测数据集，保存测试问题、预期是否允许回答以及预期正文。
- `knowledge/agent_basics.md`：ToolCall、execute、会话状态、记忆和 RAG 等基础知识。
- `knowledge/frameworks.md`：LangChain 与 LangGraph 的定位和区别。
- `main.py`、`tools.py`：早期规则版 MiniAgent，供理解 `plan → execute → result`。
- [README.first-day.md](README.first-day.md)：第一天的原始练习说明。

当前范围是可解释的本地学习项目：没有 Web UI、HTTP API、用户认证或生产部署；知识库规模也很小。它适合展示 Agent 工具调用、RAG、记忆和测试机制，不应被描述成生产级系统。
