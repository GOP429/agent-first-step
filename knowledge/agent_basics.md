# Agent 基础知识

## ToolCall
ToolCall 是模型生成的结构化工具调用单，通常包含工具名称、参数和调用编号。在本项目的教学比喻中，ToolCall 也可以称为“派工单”，表示交给 Agent 的任务说明。ToolCall 本身不会执行工具，必须由 execute 处理。

## execute

execute 负责读取 ToolCall，找到对应的本地工具并传入参数。真正的待办读写或数学计算由具体工具函数完成。

## thread_id

thread_id 是一整段会话的编号。相同的 thread_id 可以恢复原来的会话状态，不同的 thread_id 用于隔离不同会话。

## checkpointer

checkpointer 负责保存和恢复 Agent 状态。InMemorySaver 把状态放在当前进程内存中，SqliteSaver 可以把状态持久化到硬盘。

## RAG

RAG 会先从外部知识库中检索与用户问题相关的内容，再把检索结果交给大模型生成回答，从而减少只依赖模型内部知识造成的错误。

