import json
from pathlib import Path

from rag_demo import (
    MINIMUM_SIMILARITY,
    document_vectors,
    documents,
    embedding_model,
    search_top_documents,
)

EVALUATION_CASES_PATH = (
    Path(__file__).parent
    / "data"
    / "rag_eval_cases.json"
)


def load_evaluation_cases(
    path: Path,
) -> list[dict]:
    evaluation_text = path.read_text(
        encoding="utf-8",
    )

    return json.loads(evaluation_text)


evaluation_cases = load_evaluation_cases(
    EVALUATION_CASES_PATH,
)


def evaluate_case(case: dict) -> dict:
    candidates = search_top_documents(
        case["query"],
        documents,
        document_vectors,
        embedding_model,
        top_k=3,
    )

    if candidates:
        document, score = candidates[0]
    else:
        document = None
        score = -1.0

    predicted_retrieve = (
        document is not None
        and score >= MINIMUM_SIMILARITY
    )

    content_hit = (
        case["expected_text"] is not None
        and document is not None
        and case["expected_text"] in document.page_content
    )

    decision_correct = (
        predicted_retrieve
        == case["should_retrieve"]
    )

    passed = (
        decision_correct
        and (
            not case["should_retrieve"]
            or content_hit
        )
    )

    return {
        "query": case["query"],
        "score": score,
        "predicted_retrieve": predicted_retrieve,
        "content_hit": content_hit,
        "passed": passed,
        "document": document,
        "candidates": candidates,
    }
def main():
    passed_count = 0
    false_positive_count = 0
    false_negative_count = 0
    wrong_chunk_count = 0
    print(
        f"当前检索阈值：{MINIMUM_SIMILARITY}"
    )

    for index, case in enumerate(
        evaluation_cases,
        start=1,
    ):
        result = evaluate_case(case)

        expected = (
            "应该检索"
            if case["should_retrieve"]
            else "应该拦截"
        )
        predicted = (
            "实际检索"
            if result["predicted_retrieve"]
            else "实际拦截"
        )
        status = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        if result["passed"]:
            passed_count += 1
        if (
            not case["should_retrieve"]
            and result["predicted_retrieve"]
        ):
            false_positive_count += 1

        elif (
            case["should_retrieve"]
            and not result["predicted_retrieve"]
        ):
            false_negative_count += 1

        elif (
            case["should_retrieve"]
            and not result["content_hit"]
        ):
            wrong_chunk_count += 1

        print(f"\n--- case {index} ---")
        print(f"问题：{result['query']}")
        print(f"相似度：{result['score']:.4f}")
        print(f"预期：{expected}")
        print(f"结果：{predicted}")
        print(f"正文命中：{result['content_hit']}")
        print(f"判定：{status}")
        if not result["passed"]:
            print("实际命中的文本块：")

            if result["document"] is None:
                print("没有找到文本块")
            else:
                print(
                    result["document"].page_content
                )

    total_count = len(evaluation_cases)

    print("\n--- 总结 ---")
    print(
        f"通过：{passed_count}/{total_count}"
    )
    print(f"误放：{false_positive_count}")
    print(f"误拦：{false_negative_count}")
    print(f"错块：{wrong_chunk_count}")


if __name__ == "__main__":
    main()