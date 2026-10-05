from pathlib import Path
from rag_eval_llm import (
    LLM_REVIEW_THRESHOLD,
    judge_answerability,
)
from langgraph.checkpoint.sqlite import SqliteSaver


from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek

from langchain.tools import tool

from rag_demo import (
    document_vectors,
    documents,
    embedding_model,
    search_top_documents,
)
memory_path = (
    Path(__file__).parent
    / "data"
    / "rag_agent_memory.sqlite"
)

def review_candidate(
    query: str,
    document,
) -> bool | None:
    judge_model = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
    )

    return judge_answerability(
        query,
        document,
        judge_model,
    )

@tool
def search_agent_knowledge(
    query: str,
    original_question: str,
) -> str:
    """
    检索Agent开发知识库中的相关资料。

    query 用于BGE向量检索。
    original_question 是用户的原始问题，
    用于审核候选资料能否真正回答用户。
    """

    candidates = search_top_documents(
        query,
        documents,
        document_vectors,
        embedding_model,
        top_k=3,
    )

    review_failed = False

    for matched_document, score in candidates:
        if score < LLM_REVIEW_THRESHOLD:
            break

        review_result = review_candidate(
            original_question,
            matched_document,
        )
        if review_result is None:
            review_failed = True
            continue
        if not review_result:
            continue

        source = matched_document.metadata[
            "source"
        ]

        return (
            f"资料来源：{source}"
            f"\n匹配分数：{score:.4f}"
            f"\n\n{matched_document.page_content}"
        )
    if review_failed:
        return "知识库审核服务暂时不可用，请稍后重试。"

    return "知识库中没有足够相关的资料。"


def build_rag_agent(
    model=None,
    checkpointer=None,
):
    if model is None:
        model = ChatDeepSeek(
            model="deepseek-chat",
            temperature=0,
        )
    return create_agent(
        model=model,
        tools=[
            search_agent_knowledge,
        ],
        system_prompt=(
            "你是一个Agent开发学习助手。"
            "遇到Agent开发、LangChain、LangGraph、"
            "工具调用、记忆或RAG相关的知识问题时，"
            "必须先使用search_agent_knowledge工具检索资料。"
            "调用search_agent_knowledge时，"
            "original_question必须原样保留用户当前问题；"
            "query可以改写成更适合检索的关键词，"
            "但不能改变问题原意。"
            "普通问候和闲聊不需要使用工具。"
            "使用工具后，只能根据工具返回的资料回答；"
            "如果资料不足，要明确告诉用户资料不足。"
        ),
        checkpointer=checkpointer,
    )

def main():
    memory_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SqliteSaver.from_conn_string(
        str(memory_path)
    ) as checkpointer:
        agent = build_rag_agent(
            checkpointer=checkpointer,
        )

        thread_id = input(
            "请输入会话编号："
        ).strip()

        if not thread_id:
            thread_id = "default-session"

        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        while True:
            user_input = input("你：").strip()

            if user_input in {
                "退出",
                "exit",
                "quit",
            }:
                break

            if not user_input:
                continue

            result = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input,
                        }
                    ]
                },
                config,
            )

            print("\n--- 消息轨迹 ---")

            for index, message in enumerate(
                result["messages"],
            ):
                print(
                    index,
                    type(message).__name__,
                    "tool_calls=",
                    getattr(
                        message,
                        "tool_calls",
                        None,
                    ),
                    "content=",
                    message.content,
                )

            print(
                f"\nAgent："
                f"{result['messages'][-1].content}"
            )
if __name__ == "__main__":
    main()