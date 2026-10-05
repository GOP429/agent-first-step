# Agent 第一天：先把核心循环跑起来

> 历史学习记录：以下内容保留了最初的规则版 MiniAgent 练习，不是当前 RAG Agent 的安装或运行说明。
> 其中的无参数测试命令也只适用于当时；当前项目请使用根 README 的无真实 API 测试命令。

这不是一个求职项目，而是你的第一块积木。它不需要 API Key、第三方库或 Agent 框架，目的是让你亲眼看到：

```text
用户输入 → plan 选择工具 → execute 执行工具 → 返回结果
```

## 1. 直接运行

打开 PowerShell：

```powershell
cd "C:\Users\lch\Documents\Codex\2026-08-29\agent\outputs\agent-first-step"
python main.py
```

如果系统使用 Windows Python Launcher，也可以：

```powershell
py main.py
```

依次输入：

```text
添加待办 周五交实验报告
查看待办
计算 (12 + 8) * 3
计算 （12+8）\*3
退出
```

你会看到两段轨迹：

- `[plan]`：程序选择了哪个工具，以及准备传入什么参数。
- `[tool]`：工具实际执行后返回了什么。

## 2. 运行测试

```powershell
python -m unittest -v
```

看到四个 `ok`，说明第一关通过。

## 3. 今天只读这三处代码

按顺序阅读 `main.py`：

1. `plan()`：决定使用哪个工具。
2. `execute()`：真正调用工具。
3. `run()`：把规划和执行串成一个循环。

然后阅读 `tools.py`，确认“工具只是普通 Python 函数或类方法”。

## 4. 你的唯一改造任务

自己增加第四个工具 `count_todos`，让下面的对话成立：

```text
你：待办有几个
[plan] 选择工具：count_todos，参数：{}
[tool] 工具返回：目前共有 1 个待办。
Agent：目前共有 1 个待办。
```

需要修改的位置只有三个：

1. 在 `tools.py` 的 `TodoStore` 中增加计数方法。
2. 在 `main.py` 的 `plan()` 中识别“待办有几个”。
3. 在 `execute()` 中注册 `count_todos`。

完成标准：重新运行测试，并手动验证以上对话。

## 5. 今天不要做的事

今天不学习 RAG、LangChain、多 Agent、向量数据库或模型微调。等你能解释 `plan → execute → result` 后，第二关才会把 `plan()` 的规则判断替换成真实大模型的 Tool Calling。
