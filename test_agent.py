from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from main import MiniAgent, ToolCall
from tools import TodoStore, safe_calculate
from tool_schemas import TOOL_SCHEMAS

class FakePlanner:
    def plan(self, user_input: str) -> ToolCall:
        return ToolCall("count_todos", {})
class FakeDeepSeekClient:
    def create_chat_completion(self, messages, tools):
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "count_todos",
                                    "arguments": "{}",
                                }
                            }
                        ]
                    }
                }
            ]
        }
class FakeTextOnlyDeepSeekClient:
    def create_chat_completion(self, messages, tools):
        return {
            "choices": [
                {
                    "message": {
                        "content": "你好",
                        "tool_calls": [],
                    }
                }
            ]
        }



class FakeTwoStepDeepSeekClient:
    def __init__(self) -> None:
        self.call_count = 0
        self.tool_message = None

    def create_chat_completion(self, messages, tools):
        self.call_count += 1

        if self.call_count == 1:
            return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "count_todos",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    }
                }
            ]
        }

        self.tool_message = messages[-1]
        return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "你目前有 1 条待办。",
                }
            }
        ]
    }
class FakeThreeStepDeepSeekClient:
    def __init__(self) -> None:
        self.call_count = 0
        self.tool_messages = []
    def create_chat_completion(self, messages, tools):
        self.call_count += 1

        if self.call_count == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "count_todos",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        if self.call_count == 2:
            self.tool_messages.append(messages[-1])

            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {
                                        "name": "calculate",
                                        "arguments": '{"expression": "1 + 1"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        self.tool_messages.append(messages[-1])

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "你目前有 1 条待办，数量加 1 后是 2。",
                    }
                }
            ]
        }        
class FakeMultiToolDeepSeekClient:
    def __init__(self) -> None:
        self.call_count = 0
        self.tool_messages = []

    def create_chat_completion(self, messages, tools):
        self.call_count += 1

        if self.call_count == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "count_todos",
                                        "arguments": "{}",
                                    },
                                },
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {
                                        "name": "calculate",
                                        "arguments": '{"expression": "1 + 1"}',
                                    },
                                },
                            ],
                        }
                    }
                ]
            }

        self.tool_messages.extend(messages[-2:])

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "共有 1 条待办，1 加 1 等于 2。",
                    }
                }
            ]
        }
class FakeEndlessToolClient:
    def __init__(self) -> None:
        self.call_count = 0

    def create_chat_completion(self, messages, tools):
        self.call_count += 1

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_{self.call_count}",
                                "type": "function",
                                "function": {
                                    "name": "count_todos",
                                    "arguments": "{}",
                                },
                            }
                        ],
                    }
                }
            ]
        }
class FakeMissingToolCallIdClient:
    def __init__(self) -> None:
        self.call_count = 0

    def create_chat_completion(self, messages, tools):
        self.call_count += 1

        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "add_todo",
                                    "arguments": '{"content": "不应添加的待办"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
class MiniAgentTest(unittest.TestCase):

    def test_tool_schemas_cover_all_registered_tools(self) -> None:
        tool_names = []

        for schema in TOOL_SCHEMAS:
            tool_names.append(schema["function"]["name"])

        self.assertCountEqual(
            tool_names,
            ["add_todo", "list_todos", "calculate", "count_todos"],
        )
    def test_add_and_list_todo(self) -> None:
        with TemporaryDirectory() as directory:
            agent = MiniAgent(TodoStore(Path(directory) / "todos.json"))

            added = agent.run("添加待办 周五交实验报告", show_trace=False)
            listed = agent.run("查看待办", show_trace=False)

            self.assertIn("周五交实验报告", added)
            self.assertIn("1. 周五交实验报告", listed)

    def test_calculation(self) -> None:
        with TemporaryDirectory() as directory:
            agent = MiniAgent(TodoStore(Path(directory) / "todos.json"))
            answer = agent.run("计算 (12 + 8) * 3", show_trace=False)
            self.assertEqual(answer, "计算结果：60")

    def test_calculation_accepts_full_width_punctuation(self) -> None:
        with TemporaryDirectory() as directory:
            agent = MiniAgent(TodoStore(Path(directory) / "todos.json"))
            answer = agent.run(r"计算 （12+8）\*3", show_trace=False)
            self.assertEqual(answer, "计算结果：60")

    def test_calculator_rejects_python_code(self) -> None:
        with self.assertRaises(ValueError):
            safe_calculate("__import__('os').getcwd()")

    def test_unknown_request_returns_help(self) -> None:
        with TemporaryDirectory() as directory:
            agent = MiniAgent(TodoStore(Path(directory) / "todos.json"))
            answer = agent.run("帮我写论文", show_trace=False)
            self.assertIn("我暂时只会", answer)

    def test_count_todos(self) -> None:
        with TemporaryDirectory() as directory:
            agent = MiniAgent(TodoStore(Path(directory) / "todos.json"))
            agent.run("添加待办 复习Agent", show_trace=False)
            agent.run("添加待办 整理学习笔记", show_trace=False)
            answer = agent.run("待办有几个", show_trace=False)
            self.assertEqual(answer, "当前待办数量为 2")

    def test_add_todo_without_content_asks_for_content(self) -> None:
        with TemporaryDirectory() as directory:
            agent = MiniAgent(TodoStore(Path(directory) / "todos.json"))
            answer = agent.run("添加待办", show_trace=False)
            self.assertIn("请告诉我要添加什么", answer)
    def test_add_todo_can_continue_on_next_message(self) -> None:
        with TemporaryDirectory() as directory:
            agent = MiniAgent(TodoStore(Path(directory) / "todos.json"))

            first_answer = agent.run("添加待办", show_trace=False)
            second_answer = agent.run("周五交实验报告", show_trace=False)
            listed = agent.run("查看待办", show_trace=False)

            self.assertIn("请告诉我要添加什么", first_answer)
            self.assertIn("周五交实验报告", second_answer)
            self.assertIn("1. 周五交实验报告", listed)    
    def test_pending_add_todo_can_be_cancelled(self) -> None:
        with TemporaryDirectory() as directory:
            agent = MiniAgent(TodoStore(Path(directory) / "todos.json"))

            agent.run("添加待办", show_trace=False)
            answer = agent.run("取消", show_trace=False)
            listed = agent.run("查看待办", show_trace=False)

            self.assertIn("已取消", answer)
            self.assertEqual(listed, "目前没有待办。")
    def test_external_planner_can_choose_tool(self) -> None:
        with TemporaryDirectory() as directory:
            store = TodoStore(Path(directory) / "todos.json")
            store.add("复习 Agent")
            agent = MiniAgent(store, planner=FakePlanner())

            answer = agent.run("我的任务一共有多少个？", show_trace=False)

            self.assertEqual(answer, "当前待办数量为 1")   

    def test_deepseek_planner_converts_model_tool_call(self) -> None:
        from main import DeepSeekPlanner

        planner = DeepSeekPlanner(FakeDeepSeekClient())
        call = planner.plan("我的任务一共有多少个？")

        self.assertEqual(
            call,
            ToolCall("count_todos", {}),
        )
    def test_add_todo_schema_requires_content(self) -> None:
        add_schema = TOOL_SCHEMAS[1]["function"]

        self.assertEqual(add_schema["name"], "add_todo")
        self.assertEqual(
            add_schema["parameters"]["required"],
            ["content"],
        )
    def test_agent_returns_model_text_without_tool_call(self) -> None:
        from main import DeepSeekPlanner

        with TemporaryDirectory() as directory:
            planner = DeepSeekPlanner(FakeTextOnlyDeepSeekClient())
            agent = MiniAgent(
                TodoStore(Path(directory) / "todos.json"),
                planner=planner,
            )

            answer = agent.run("你好", show_trace=False)

        self.assertEqual(answer, "你好")
    def test_deepseek_api_client_returns_json_without_real_network(self) -> None:
        from unittest.mock import Mock, patch
        from deepseek_client import DeepSeekAPIClient

        fake_response = Mock()
        fake_response.json.return_value = {"choices": []}

        with patch(
            "deepseek_client.requests.post",
            return_value=fake_response,
        ):
            client = DeepSeekAPIClient("test-key")
            response = client.create_chat_completion(
                messages=[{"role": "user", "content": "你好"}],
                tools=[],
            )

        self.assertEqual(response, {"choices": []})



    def test_agent_sends_tool_result_back_to_model_for_final_answer(self) -> None:
        from main import DeepSeekPlanner

        with TemporaryDirectory() as directory:
            store = TodoStore(Path(directory) / "todos.json")
            store.add("复习 Agent")

            client = FakeTwoStepDeepSeekClient()
            planner = DeepSeekPlanner(client)
            agent = MiniAgent(store, planner=planner)

            answer = agent.run(
                "我现在有多少条待办？",
                show_trace=False,
            )

        self.assertEqual(answer, "你目前有 1 条待办。")
        self.assertEqual(client.call_count, 2)
        self.assertEqual(
            client.tool_message,
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "当前待办数量为 1",
            },
        )
    def test_agent_can_continue_for_three_model_rounds(self) -> None:
        from main import DeepSeekPlanner

        with TemporaryDirectory() as directory:
            store = TodoStore(Path(directory) / "todos.json")
            store.add("复习 Agent")

            client = FakeThreeStepDeepSeekClient()
            planner = DeepSeekPlanner(client)
            agent = MiniAgent(store, planner=planner)

            answer = agent.run(
                "统计待办数量，再把数量加 1。",
                show_trace=False,
            )

        self.assertEqual(
            answer,
            "你目前有 1 条待办，数量加 1 后是 2。",
        )
        self.assertEqual(client.call_count, 3)
        self.assertEqual(
            client.tool_messages,
            [
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "当前待办数量为 1",
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_2",
                    "content": "计算结果：2",
                },
            ],
        )
    def test_agent_executes_multiple_tools_in_one_model_turn(self) -> None:
        from main import DeepSeekPlanner

        with TemporaryDirectory() as directory:
            store = TodoStore(Path(directory) / "todos.json")
            store.add("复习 Agent")

            client = FakeMultiToolDeepSeekClient()
            planner = DeepSeekPlanner(client)
            agent = MiniAgent(store, planner=planner)

            answer = agent.run(
                "统计待办数量，并计算 1 + 1。",
                show_trace=False,
            )

        self.assertEqual(
            answer,
            "共有 1 条待办，1 加 1 等于 2。",
        )
        self.assertEqual(client.call_count, 2)
        self.assertEqual(
            client.tool_messages,
            [
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "当前待办数量为 1",
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_2",
                    "content": "计算结果：2",
                },
            ],
        )
    def test_agent_rejects_tool_call_without_id(self) -> None:
        from main import DeepSeekPlanner

        with TemporaryDirectory() as directory:
            store = TodoStore(Path(directory) / "todos.json")
            client = FakeMissingToolCallIdClient()
            planner = DeepSeekPlanner(client)
            agent = MiniAgent(store, planner=planner)

            answer = agent.run(
                "添加一条待办。",
                show_trace=False,
            )

            todos_after_run = store.list_all()

        self.assertEqual(
            answer,
            "模型返回的工具调用缺少编号。",
        )
        self.assertEqual(client.call_count, 1)
        self.assertEqual(todos_after_run, "目前没有待办。")
    def test_deepseek_planner_keeps_tool_call_id(self) -> None:
        from main import DeepSeekPlanner

        client = FakeTwoStepDeepSeekClient()
        planner = DeepSeekPlanner(client)

        call = planner.plan("我现在有多少条待办？")

        self.assertEqual(call.tool_call_id, "call_1")
    def test_deepseek_planner_builds_complete_model_turn(self) -> None:
        from main import DeepSeekPlanner

        client = FakeTwoStepDeepSeekClient()
        planner = DeepSeekPlanner(client)
        messages = [
            {
                "role": "user",
                "content": "我现在有多少条待办？",
            }
        ]

        turn = planner.decide(messages)

        self.assertEqual(
            turn.message["tool_calls"][0]["id"],
            "call_1",
        )
        self.assertEqual(
            turn.tool_calls,
            [
                ToolCall(
                    name="count_todos",
                    arguments={},
                    tool_call_id="call_1",
                )
            ],
        )
    def test_agent_stops_after_too_many_model_rounds(self) -> None:
        from main import DeepSeekPlanner

        with TemporaryDirectory() as directory:
            store = TodoStore(Path(directory) / "todos.json")
            client = FakeEndlessToolClient()
            planner = DeepSeekPlanner(client)
            agent = MiniAgent(store, planner=planner)

            answer = agent.run(
                "一直统计待办。",
                show_trace=False,
            )

        self.assertEqual(
            answer,
            "模型连续调用工具次数过多，已停止。",
        )
        self.assertEqual(client.call_count, 5)


if __name__ == "__main__":
    unittest.main()
