# Agent 开发框架


## LangChain 和 LangGraph

LangChain 提供模型、工具和 Agent 循环等高层组件，适合快速构建常见的 Agent 应用。LangChain 的 Agent 实现使用 LangGraph 的底层能力。

LangGraph 用于编排有状态的工作流程，支持状态管理、分支、循环、持久化和更细粒度的流程控制。多智能体协作只是 LangGraph 的一种使用场景，并不是它的全部用途。

本节只介绍两者的定位和区别，不包含添加工作流节点的具体代码。