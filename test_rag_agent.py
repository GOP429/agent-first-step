import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from openai import APIConnectionError, APIStatusError, AuthenticationError
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import (
    AIMessage,
    ToolMessage,
)

from rag_agent import (
    build_rag_agent,
    search_agent_knowledge,
)
from rag_eval_llm import (
    evaluate_case_with_llm,
    is_retryable_model_error,
    judge_answerability,
    parse_judge_decision,
)
class FakeJudgeModel:
    def __init__(self):
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        return SimpleNamespace(
            content=(
                '{"supported": true, '
                '"reason": "资料能够回答问题"}'
            ),
        )


class ToolCallingFakeModel(
    FakeMessagesListChatModel
):
    def bind_tools(
        self,
        tools,
        **kwargs,
    ):
        return self


class RagAgentTest(unittest.TestCase):



    @patch(
        "rag_agent.review_candidate",
    )
    @patch(
        "rag_agent.search_top_documents",
    )
    def test_search_tool_rejects_unsupported_candidate(
        self,
        search_top_documents,
        review_candidate,
    ):
        search_top_documents.return_value = [
            (
                SimpleNamespace(
                    metadata={
                        "source": "agent_basics.md",
                    },
                    page_content=(
                        "SqliteSaver 可以把状态持久化到硬盘。"
                    ),
                ),
                0.55,
            ),
        ]

        review_candidate.return_value = False

        result = search_agent_knowledge.invoke(
            {
                "query": (
                    "Agent 连接 MySQL 数据库"
                ),
                "original_question": (
                    "Agent 怎样连接 MySQL 数据库？"
                ),
            }
        )

        self.assertIn(
            "没有足够相关的资料",
            result,
        )
        review_candidate.assert_called_once()

    @patch("rag_agent.review_candidate", return_value=None)
    @patch("rag_agent.search_top_documents")
    def test_search_tool_reports_judge_failure(
        self,
        search_top_documents,
        review_candidate,
    ):
        search_top_documents.return_value = [
            (
                SimpleNamespace(
                    metadata={"source": "agent_basics.md"},
                    page_content="SqliteSaver 保存到硬盘。",
                ),
                0.55,
            ),
        ]

        result = search_agent_knowledge.invoke(
            {
                "query": "SqliteSaver 保存位置",
                "original_question": "SqliteSaver 保存在哪里？",
            }
        )

        self.assertIn("审核服务暂时不可用", result)
        review_candidate.assert_called_once()

    @patch(
        "rag_agent.review_candidate",
        return_value=True,
    )        
    def test_agent_executes_knowledge_search_tool(
        self,
        review_candidate,
    ):
        model = ToolCallingFakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": (
                                "search_agent_knowledge"
                            ),
                                                "args": {
                            "query": (
                                "SqliteSaver 状态 硬盘 持久化"
                            ),
                            "original_question": (
                                "SqliteSaver把状态保存在哪里？"
                            ),
                        },   
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content=(
                        "根据资料，SqliteSaver"
                        "把状态保存到硬盘。"
                    ),
                ),
            ]
        )

        agent = build_rag_agent(
            model=model,
        )

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "SqliteSaver把状态"
                            "保存在哪里？"
                        ),
                    }
                ]
            }
        )

        tool_messages = [
            message
            for message in result["messages"]
            if isinstance(
                message,
                ToolMessage,
            )
        ]

        self.assertEqual(
            len(tool_messages),
            1,
        )
        self.assertIn(
            "SqliteSaver",
            tool_messages[0].content,
        )
        self.assertIn(
            "硬盘",
            tool_messages[0].content,
        )
        self.assertEqual(
            result["messages"][-1].content,
            (
                "根据资料，SqliteSaver"
                "把状态保存到硬盘。"
            ),
        )
        review_candidate.assert_called_once()
        
    @patch(
        "rag_agent.review_candidate",
    )
    @patch(
        "rag_agent.search_top_documents",
    )
    def test_search_tool_uses_second_candidate_when_first_is_rejected(
        self,
        search_top_documents,
        review_candidate,
    ):
        wrong_document = SimpleNamespace(
            metadata={
                "source": "wrong.md",
            },
            page_content=(
                "这是一段不能回答问题的资料。"
            ),
        )

        correct_document = SimpleNamespace(
            metadata={
                "source": "agent_basics.md",
            },
            page_content=(
                "ToolCall 本身不会执行工具，"
                "必须由 execute 处理。"
            ),
        )

        search_top_documents.return_value = [
            (
                wrong_document,
                0.60,
            ),
            (
                correct_document,
                0.55,
            ),
        ]

        review_candidate.side_effect = [
            False,
            True,
        ]

        result = search_agent_knowledge.invoke(
            {
                "query": (
                    "ToolCall execute 工具执行"
                ),
                "original_question": (
                    "派工单自己能干活吗？"
                ),
            }
        )

        self.assertIn(
            "ToolCall",
            result,
        )
        self.assertIn(
            "execute",
            result,
        )
        self.assertEqual(
            review_candidate.call_count,
            2,
        )
        self.assertEqual(
            search_top_documents.call_args.args[0],
            "ToolCall execute 工具执行",
        )
        self.assertEqual(
            review_candidate.call_args_list[0].args[0],
            "派工单自己能干活吗？",
        )

    @patch("rag_agent.review_candidate")
    @patch("rag_agent.search_top_documents")
    def test_search_tool_uses_second_candidate_after_judge_failure(
        self,
        search_top_documents,
        review_candidate,
    ):
        failed_document = SimpleNamespace(
            metadata={"source": "failed.md"},
            page_content="第一段候选资料。",
        )
        supported_document = SimpleNamespace(
            metadata={"source": "agent_basics.md"},
            page_content=(
                "ToolCall 本身不会执行工具，"
                "必须由 execute 处理。"
            ),
        )

        search_top_documents.return_value = [
            (failed_document, 0.60),
            (supported_document, 0.55),
        ]
        review_candidate.side_effect = [
            None,
            True,
        ]

        result = search_agent_knowledge.invoke(
            {
                "query": "ToolCall execute 工具执行",
                "original_question": "派工单自己能干活吗？",
            }
        )

        self.assertIn("agent_basics.md", result)
        self.assertIn("ToolCall", result)
        self.assertEqual(review_candidate.call_count, 2)

    def test_agent_can_answer_without_search_tool(
        self,
    ):
        model = ToolCallingFakeModel(
            responses=[
                AIMessage(
                    content="你好！",
                ),
            ]
        )

        agent = build_rag_agent(
            model=model,
        )

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "你好",
                    }
                ]
            }
        )

        tool_messages = [
            message
            for message in result["messages"]
            if isinstance(
                message,
                ToolMessage,
            )
        ]

        self.assertEqual(
            len(tool_messages),
            0,
        )
        self.assertEqual(
            result["messages"][-1].content,
            "你好！",
        )


class RagEvalLlmTest(unittest.TestCase):

    def test_judge_rejects_invalid_model_output(
        self,
    ):
        model = Mock()
        model.invoke.return_value = SimpleNamespace(
            content="SUPPORTED",
        )

        document = SimpleNamespace(
            page_content="测试资料",
        )

        result = judge_answerability(
            "测试问题",
            document,
            model,
        )

        self.assertIsNone(result)
        self.assertEqual(
            model.invoke.call_count,
            1,
        )

    @patch("rag_eval_llm.time.sleep")
    def test_judge_rejects_when_model_call_fails(
        self,
        sleep_mock,
    ):
        model = Mock()
        model.invoke.side_effect = APIConnectionError(request=Mock())

        document = SimpleNamespace(
            page_content="测试资料",
        )

        result = judge_answerability(
            "测试问题",
            document,
            model,
        )

        self.assertIsNone(result)
        self.assertEqual(
            model.invoke.call_count,
            2,
        )
        sleep_mock.assert_called_once_with(1)
    @patch("rag_eval_llm.time.sleep")
    def test_judge_retries_after_temporary_failure(
        self,
        sleep_mock,
    ):
        model = Mock()
        model.invoke.side_effect = [
            APIConnectionError(request=Mock()),
            SimpleNamespace(
                content=(
                    '{"supported": true, '
                    '"reason": "第二次调用成功"}'
                ),
            ),
        ]

        document = SimpleNamespace(
            page_content="测试资料",
        )

        result = judge_answerability(
            "测试问题",
            document,
            model,
        )

        self.assertTrue(result)
        self.assertEqual(
            model.invoke.call_count,
            2,
        )
        sleep_mock.assert_called_once_with(1)

    @patch("rag_eval_llm.time.sleep")
    def test_judge_does_not_retry_authentication_error(self, sleep_mock):
        model = Mock()
        response = Mock(status_code=401, headers={}, request=Mock())
        model.invoke.side_effect = AuthenticationError(
            "API Key 无效",
            response=response,
            body=None,
        )
        document = SimpleNamespace(page_content="测试资料")

        result = judge_answerability("测试问题", document, model)

        self.assertIsNone(result)
        model.invoke.assert_called_once()
        sleep_mock.assert_not_called()

    def test_http_400_is_not_retryable(self):
        response = Mock(status_code=400, headers={}, request=Mock())
        error = APIStatusError("模拟参数错误", response=response, body=None)

        self.assertFalse(is_retryable_model_error(error))

    def test_http_503_is_retryable(self):
        response = Mock(status_code=503, headers={}, request=Mock())
        error = APIStatusError("模拟服务器繁忙", response=response, body=None)

        self.assertTrue(is_retryable_model_error(error))

    def test_http_429_is_retryable(self):
        response = Mock(status_code=429, headers={}, request=Mock())
        error = APIStatusError("模拟请求过快", response=response, body=None)
        self.assertTrue(is_retryable_model_error(error))

    def test_http_402_is_not_retryable(self):
        response = Mock(status_code=402, headers={}, request=Mock())
        error = APIStatusError("模拟余额不足", response=response, body=None)
        self.assertFalse(is_retryable_model_error(error))

    def test_parse_judge_decision_accepts_valid_json(
        self,
    ):
        decision = parse_judge_decision(
            (
                '{"supported": true, '
                '"reason": "资料能够回答问题"}'
            )
        )

        self.assertTrue(
            decision.supported,
        )
        self.assertEqual(
            decision.reason,
            "资料能够回答问题",
        )

    def test_parse_judge_decision_rejects_invalid_json(
        self,
    ):
        with self.assertRaises(
            ValueError,
        ):
            parse_judge_decision(
                "SUPPORTED",
            )

    @patch("rag_eval_llm.evaluate_case")
    def test_evaluation_records_judge_failure(self, evaluate_case):
        evaluate_case.return_value = {
            "query": "Agent怎样连接MySQL数据库？",
            "score": 0.55,
            "predicted_retrieve": True,
            "content_hit": False,
            "document": SimpleNamespace(
                page_content="SqliteSaver把状态保存到硬盘。"
            ),
        }
        model = Mock()
        model.invoke.return_value = SimpleNamespace(content="INVALID")

        result = evaluate_case_with_llm(
            {
                "query": "Agent怎样连接MySQL数据库？",
                "should_retrieve": False,
            },
            model,
        )

        self.assertIs(result["final_retrieve"], False)
        self.assertTrue(result["judge_failed"])
        self.assertFalse(result["final_passed"])

    @patch("rag_eval_llm.evaluate_case")
    def test_low_score_candidate_enters_llm_review(
        self,
        evaluate_case,
    ):
        evaluate_case.return_value = {
            "query": (
                "机器人关闭以后还记得我吗？"
            ),
            "score": 0.3867,
            "predicted_retrieve": False,
            "content_hit": True,
            "document": SimpleNamespace(
                page_content=(
                    "SqliteSaver 可以把状态持久化到硬盘。"
                ),
            ),
        }

        case = {
            "query": (
                "机器人关闭以后还记得我吗？"
            ),
            "should_retrieve": True,
            "expected_text": "SqliteSaver",
        }
        model = FakeJudgeModel()

        result = evaluate_case_with_llm(
            case,
            model,
        )

        self.assertTrue(result["llm_checked"])
        self.assertEqual(len(model.calls), 1)
        self.assertTrue(result["llm_approved"])
        self.assertTrue(result["final_retrieve"])
        self.assertTrue(result["final_passed"])

    @patch("rag_eval_llm.evaluate_case")
    def test_evaluation_uses_second_candidate_when_first_is_rejected(
        self,
        evaluate_case,
    ):
        wrong_document = SimpleNamespace(
            page_content=(
                "本节只介绍两者的定位和区别，"
                "不包含添加工作流节点的具体代码。"
            ),
        )
        correct_document = SimpleNamespace(
            page_content=(
                "LangChain 提供模型、工具和 "
                "Agent 循环等高层组件。"
            ),
        )

        evaluate_case.return_value = {
            "query": (
                "LangChain和LangGraph的定位有什么不同？"
            ),
            "score": 0.5657,
            "predicted_retrieve": True,
            "content_hit": False,
            "document": wrong_document,
            "candidates": [
                (
                    wrong_document,
                    0.5657,
                ),
                (
                    correct_document,
                    0.5646,
                ),
            ],
        }

        model = Mock()
        model.invoke.side_effect = [
            SimpleNamespace(
                content=(
                    '{"supported": false, '
                    '"reason": "第一块不能回答"}'
                ),
            ),
            SimpleNamespace(
                content=(
                    '{"supported": true, '
                    '"reason": "第二块能够回答"}'
                ),
            ),
        ]

        case = {
            "query": (
                "LangChain和LangGraph的定位有什么不同？"
            ),
            "should_retrieve": True,
            "expected_text": "LangChain",
        }

        result = evaluate_case_with_llm(
            case,
            model,
        )

        self.assertEqual(
            model.invoke.call_count,
            2,
        )
        self.assertIs(
            result["document"],
            correct_document,
        )
        self.assertTrue(
            result["content_hit"],
        )
        self.assertTrue(
            result["final_retrieve"],
        )
        self.assertTrue(
            result["final_passed"],
        )
