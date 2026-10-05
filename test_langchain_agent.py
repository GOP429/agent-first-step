import os
import unittest

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, ToolMessage

from langchain_agent import build_agent, calculate

class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

class LangChainToolTest(unittest.TestCase):
    def test_calculate_tool(self):
        result = calculate.invoke(
            {"expression": "(12+8)*3"}
        )

        self.assertEqual(
            result,
            "计算结果：60",
        )
class LangChainAgentTest(unittest.TestCase):
    def test_agent_executes_tool_call(self):
        model = ToolCallingFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "calculate",
                            "args": {
                                "expression": "(12+8)*3",
                            },
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="计算已经完成。",
                ),
            ]
        )

        agent = create_agent(
            model=model,
            tools=[calculate],
        )

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "请计算（12+8）乘以3",
                    }
                ]
            }
        )

        tool_messages = [
            message
            for message in result["messages"]
            if message.type == "tool"
        ]

        self.assertEqual(
            len(tool_messages),
            1,
        )
        self.assertEqual(
            tool_messages[0].content,
            "计算结果：60",
        )
        self.assertEqual(
            result["messages"][-1].content,
            "计算已经完成。",
        )
@unittest.skipUnless(
    os.getenv("DEEPSEEK_API_KEY"),
    "需要 DEEPSEEK_API_KEY 才能运行真实 API 测试",
)
class LangChainRealApiTest(unittest.TestCase):
    def test_real_model_can_call_count_todos(self):
        agent = build_agent()

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "请务必使用 count_todos 工具"
                            "统计当前待办数量。"
                        ),
                    }
                ]
            }
        )

        tool_messages = [
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage)
        ]

        self.assertTrue(tool_messages)
        self.assertEqual(
            tool_messages[0].name,
            "count_todos",
        )