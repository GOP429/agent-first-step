"""Day 1: a zero-dependency mini-agent that exposes the agent loop clearly."""
from __future__ import annotations
from deepseek_client import DeepSeekAPIClient

import os
import re
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools import TodoStore, safe_calculate
from tool_schemas import TOOL_SCHEMAS


@dataclass(frozen=True)
class ToolCall:
    """The plan produced by the router."""

    name: str
    arguments: dict[str, Any]
    tool_call_id: str | None = None
@dataclass(frozen=True)
class ModelTurn:
    message: dict[str, Any]
    tool_calls: list[ToolCall]

class DeepSeekPlanner:
    def __init__(self, client) -> None:
        self.client = client
    def decide(self, messages: list[dict[str, Any]]) -> ModelTurn:
        response = self.client.create_chat_completion(
            messages=messages,
            tools=TOOL_SCHEMAS,
        )

        message = response["choices"][0]["message"]
        tool_calls = []

        for raw_tool_call in message.get("tool_calls", []):
            function = raw_tool_call["function"]
            arguments = json.loads(function["arguments"])

            tool_calls.append(
                ToolCall(
                    name=function["name"],
                    arguments=arguments,
                    tool_call_id=raw_tool_call.get("id"),
                )
            )

        return ModelTurn(
            message=message,
            tool_calls=tool_calls,
        )
    def plan(self, user_input: str):
        messages = [
            {
                "role": "user",
                "content": user_input,
            }
        ]

        turn = self.decide(messages)

        if turn.tool_calls:
            return turn.tool_calls[0]

        return turn.message.get("content")

class MiniAgent:
    """Plan a tool call, execute it, then turn the result into an answer."""

    def __init__(self, todo_store: TodoStore, planner= None) -> None:
        self.todo_store = todo_store
        self.planner = planner
        self.pending_action: str | None = None  

    def plan(self, user_input: str) -> ToolCall | None:
        """Choose one tool. Tomorrow, an LLM can replace only this method."""

        text = user_input.strip()

        add_match = re.match(
            r"^(?:添加待办|新增待办|记一下|帮我记下)[：:\s]*(.*)$",
            text,
        )
        if add_match:
            return ToolCall("add_todo", {"content": add_match.group(1).strip()})

        if any(keyword in text for keyword in ("查看待办", "列出待办", "查询待办", "待办有哪些")):
            return ToolCall("list_todos", {})
        if text == "待办有几个":
            return ToolCall("count_todos", {})

        calculate_match = re.match(
            r"^(?:计算|算一下|帮我算)[：:\s]*(.+?)(?:等于多少)?[？?]?$",
            text,
        )
        if calculate_match:
            return ToolCall("calculate", {"expression": calculate_match.group(1).strip()})

        return None

    def execute(self, call: ToolCall) -> str:
        """Execute only explicitly registered tools."""

        if call.name == "add_todo":
            content = call.arguments["content"]
            if not content:
                self.pending_action = "add_todo"
                return "请告诉我要添加什么"
            return self.todo_store.add(content)
        if call.name == "list_todos":
            return self.todo_store.list_all()
        if call.name == "calculate":
            return safe_calculate(call.arguments["expression"])
        if call.name == "count_todos":
            return self.todo_store.count()
        raise ValueError(f"未知工具：{call.name}")
    
    def _run_model_loop(self, user_input: str, show_trace: bool) -> str:
        messages = [
            {
                "role": "user",
                "content": user_input,
            }
        ]

        for step in range(5):
            turn = self.planner.decide(messages)
            messages.append(turn.message)

            if not turn.tool_calls:
                return turn.message.get("content") or "模型没有返回文字。"

            for call in turn.tool_calls:
                if call.tool_call_id is None:
                    return "模型返回的工具调用缺少编号。"

                if show_trace:
                    print(
                        f"[plan] 选择工具：{call.name}，"
                        f"参数：{call.arguments}"
                    )

                try:
                    result = self.execute(call)
                except (ValueError, ZeroDivisionError, SyntaxError) as error:
                    result = f"工具执行失败：{error}"

                if show_trace:
                    print(f"[tool] 工具返回：{result}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.tool_call_id,
                        "content": result,
                    }
                )

        return "模型连续调用工具次数过多，已停止。"
    
    def run(self, user_input: str, *, show_trace: bool = True) -> str:
        text = user_input.strip()
        if self.pending_action is not None and text in ("取消", "取消添加", "取消待办","算了"):
            self.pending_action = None
            return "已取消"
        if self.pending_action == "add_todo":
            call = ToolCall("add_todo", {"content": text})
            self.pending_action = None
        elif self.planner is not None and hasattr(self.planner, "decide"):
            return self._run_model_loop(text, show_trace)
        elif self.planner is not None:
            decision = self.planner.plan(text)

            if isinstance(decision, str):
                return decision

            call = decision
        else:
            call = self.plan(user_input)
        if call is None:
            return "我暂时只会：添加待办、查看待办、计算算式。"

        if show_trace:
            print(f"[plan] 选择工具：{call.name}，参数：{call.arguments}")

        try:
            result = self.execute(call)
            
        except (ValueError, ZeroDivisionError, SyntaxError) as error:
            result = f"工具执行失败：{error}"

        if show_trace:
            print(f"[tool] 工具返回：{result}")
        return result



def main() -> None:
    data_path = Path(__file__).parent / "data" / "todos.json"
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if api_key:
        client = DeepSeekAPIClient(api_key)
        planner = DeepSeekPlanner(client)
        agent = MiniAgent(TodoStore(data_path), planner=planner)
        mode = "DeepSeek"
    else:
        agent = MiniAgent(TodoStore(data_path))
        mode = "手写正则"
 
    print(f"Mini Agent 已启动（{mode}模式）。")
    print("试试：添加待办 周五交实验报告 / 查看待办 / 计算 (12 + 8) * 3")
    print("输入 exit 退出。\n")

    while True:
        user_input = input("你：").strip()
        if user_input.lower() in {"exit", "quit", "退出"}:
            print("Agent：再见。")
            break
        print(f"Agent：{agent.run(user_input)}\n")


if __name__ == "__main__":
    main()

