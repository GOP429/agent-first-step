import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver


class MemoryFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

class LangChainMemoryTest(unittest.TestCase):
    def test_different_threads_are_isolated(self):
        with TemporaryDirectory() as directory:
            database_path = (
                Path(directory)
                / "memory.sqlite"
            )

            model = MemoryFakeModel(
                responses=[
                    AIMessage(content="名字已保存。"),
                    AIMessage(content="当前会话不知道名字。"),
                ]
            )

            with SqliteSaver.from_conn_string(
                str(database_path)
            ) as checkpointer:
                agent = create_agent(
                    model=model,
                    tools=[],
                    checkpointer=checkpointer,
                )

                agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "我叫小李。",
                            }
                        ]
                    },
                    {
                        "configurable": {
                            "thread_id": "thread-a",
                        }
                    },
                )

                result_b = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "我叫什么？",
                            }
                        ]
                    },
                    {
                        "configurable": {
                            "thread_id": "thread-b",
                        }
                    },
                )

                thread_b_contents = [
                    message.content
                    for message in result_b["messages"]
                ]

                self.assertNotIn(
                    "我叫小李。",
                    thread_b_contents,
                )
    def test_same_thread_recovers_after_restart(self):
        with TemporaryDirectory() as directory:
            database_path = (
                Path(directory)
                / "memory.sqlite"
            )

            thread_config = {
                "configurable": {
                    "thread_id": "thread-a",
                }
            }

            first_model = MemoryFakeModel(
                responses=[
                    AIMessage(content="名字已保存。"),
                ]
            )

            with SqliteSaver.from_conn_string(
                str(database_path)
            ) as checkpointer:
                first_agent = create_agent(
                    model=first_model,
                    tools=[],
                    checkpointer=checkpointer,
                )

                first_agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "我叫小李。",
                            }
                        ]
                    },
                    thread_config,
                )

            second_model = MemoryFakeModel(
                responses=[
                    AIMessage(content="已恢复原会话。"),
                ]
            )

            with SqliteSaver.from_conn_string(
                str(database_path)
            ) as checkpointer:
                restarted_agent = create_agent(
                    model=second_model,
                    tools=[],
                    checkpointer=checkpointer,
                )

                result = restarted_agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "我叫什么？",
                            }
                        ]
                    },
                    thread_config,
                )

            restored_contents = [
                message.content
                for message in result["messages"]
            ]

            self.assertIn(
                "我叫小李。",
                restored_contents,
            )