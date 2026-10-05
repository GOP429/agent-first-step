import json
import time
from dataclasses import dataclass
from langchain_deepseek import ChatDeepSeek
from openai import APIConnectionError, APIStatusError
from rag_eval import (
    MINIMUM_SIMILARITY,
    evaluate_case,
    evaluation_cases,
)

LLM_REVIEW_THRESHOLD = 0.3
MAX_JUDGE_ATTEMPTS = 2
RETRY_DELAY_SECONDS = 1

def is_retryable_model_error(error: Exception) -> bool:
    if isinstance(error, APIConnectionError):
        return True

    if isinstance(error, APIStatusError):
        return (
            error.status_code in (408, 409, 429)
            or error.status_code >= 500
        )

    return False

@dataclass(frozen=True)
class JudgeDecision:
    supported: bool
    reason: str

def parse_judge_decision(
    content: str,
) -> JudgeDecision:
    if not isinstance(content, str):
        raise TypeError(
            "审核结果必须是字符串。"
        )

    data = json.loads(content)

    if not isinstance(data, dict):
        raise ValueError(
            "审核结果必须是JSON对象。"
        )

    supported = data.get("supported")
    reason = data.get("reason")

    if not isinstance(supported, bool):
        raise ValueError(
            "supported必须是布尔值。"
        )

    if (
        not isinstance(reason, str)
        or not reason.strip()
    ):
        raise ValueError(
            "reason必须是非空字符串。"
        )

    return JudgeDecision(
        supported=supported,
        reason=reason.strip(),
    )
    
def judge_answerability(
    query: str,
    document,
    model,
) -> bool | None:
    prompt = (
        "你是知识库证据裁判。"
        "你的任务不是回答用户问题，"
        "而是判断候选资料是否提供了回答依据。"
        "只根据候选资料判断，不使用自己的知识。"
        "用户问题可能使用口语、简称或近义表达，"
        "不要求与资料使用完全相同的词。"
        "只要通过合理的同义转换就能直接得到答案，"
        "也应该判定为支持。"
        "如果资料明确给出了肯定答案或否定答案，"
        "都属于有回答依据，应判定为支持。"
        "如果问题需要按照不同条件分别回答，"
        "而资料明确给出了这些条件和对应结果，"
        "也属于有回答依据，应判定为支持。"
        "例如：问题是“关闭程序后还保留记忆吗？”，"
        "资料说“InMemorySaver保存在内存中，"
        "SqliteSaver持久化到硬盘”，"
        "这足以回答“取决于使用哪种存档方式”，"
        "应该判定为支持。"
        "只有资料完全缺少问题所需的信息时，"
        "才判定为不支持。"
        "例如：问题是“设备会自动关机吗？”，"
        "资料说“设备不会自动关机”，"
        "仍然应该判定为支持。"
        "最终只输出一个合法JSON对象，"
        "不要使用Markdown代码块或额外文字。"
        '格式为：{"supported": true, '
        '"reason": "简短审核理由"}。'
        "supported必须是JSON布尔值true或false。"
        f"\n\n用户问题：{query}"
        f"\n\n候选资料：\n{document.page_content}"
    )
    response = None

    for attempt in range(
        1,
        MAX_JUDGE_ATTEMPTS + 1,
    ):
        try:
            response = model.invoke(prompt)
            break
        except Exception as error:
            print(
                f"[judge-error] "
                f"第{attempt}次模型调用失败："
                f"{error}"
            )

            if not is_retryable_model_error(error):
                return None

            if attempt < MAX_JUDGE_ATTEMPTS:
                time.sleep(RETRY_DELAY_SECONDS)

    if response is None:
        return None

    try:
        decision = parse_judge_decision(
            response.content,
        )
    except (TypeError, ValueError) as error:
        print(
            f"[judge-error] 审核结果无效：{error}"
        )
        return None

    print(
        f"[judge] {query} -> "
        f"supported={decision.supported}, "
        f"reason={decision.reason}"
    )

    return decision.supported
def evaluate_case_with_llm(
    case: dict,
    model,
) -> dict:
    result = evaluate_case(case)

    candidates = result.get("candidates")

    if candidates is None:
        candidates = []

        if result["document"] is not None:
            candidates.append(
                (
                    result["document"],
                    result["score"],
                )
            )

    llm_checked = False
    review_failed = False
    selected_document = None
    selected_score = -1.0

    for document, score in candidates:
        if score < LLM_REVIEW_THRESHOLD:
            break

        llm_checked = True

        review_result = judge_answerability(
            result["query"],
            document,
            model,
        )

        if review_result is None:
            review_failed = True
            continue

        if not review_result:
            continue

        selected_document = document
        selected_score = score
        break

    judge_failed = (
        selected_document is None
        and review_failed
    )

    if selected_document is not None:
        llm_approved = True
    elif judge_failed:
        llm_approved = None
    else:
        llm_approved = False

    final_retrieve = (
        selected_document is not None
    )

    expected_text = case.get(
        "expected_text"
    )

    content_hit = (
        expected_text is not None
        and selected_document is not None
        and expected_text
        in selected_document.page_content
    )

    final_passed = (
        not judge_failed
        and final_retrieve
        == case["should_retrieve"]
        and (
            not case["should_retrieve"]
            or content_hit
        )
    )

    if selected_document is not None:
        final_document = selected_document
        final_score = selected_score
    else:
        final_document = result["document"]
        final_score = result["score"]

    return {
        **result,
        "score": final_score,
        "content_hit": content_hit,
        "document": final_document,
        "llm_checked": llm_checked,
        "llm_approved": llm_approved,
        "judge_failed": judge_failed,
        "final_retrieve": final_retrieve,
        "final_passed": final_passed,
    }

def main():
    model = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        max_retries=0,
    )

    passed_count = 0

    for index, case in enumerate(
        evaluation_cases,
        start=1,
    ):
        print(f"\n--- case {index} ---")
        result = evaluate_case_with_llm(
            case,
            model,
        )

        if not result["llm_checked"]:
            llm_status = "未调用"
        elif result["judge_failed"]:
            llm_status = "出错"
        elif result["llm_approved"]:
            llm_status = "允许"
        else:
            llm_status = "拒绝"

        final_status = (
            "PASS"
            if result["final_passed"]
            else "FAIL"
        )

        if result["final_passed"]:
            passed_count += 1

        
        print(f"问题：{result['query']}")
        print(f"BGE分数：{result['score']:.4f}")
        print(
            "BGE结果："
            + (
                "放行"
                if result["predicted_retrieve"]
                else "拦截"
            )
        )
        print(
            "进入精判："
            + (
                "是"
                if result["llm_checked"]
                else "否"
            )
        )
        print(f"DeepSeek精判：{llm_status}")
        print(f"最终判定：{final_status}")

    print("\n--- 总结 ---")
    print(
        f"通过：{passed_count}/"
        f"{len(evaluation_cases)}"
    )


if __name__ == "__main__":
    main()