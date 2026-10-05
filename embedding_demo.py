import numpy as np

from fastembed import TextEmbedding


model = TextEmbedding(
    model_name="BAAI/bge-small-zh-v1.5",
)

query = "状态怎样长期保存到电脑中？"

passages = [
    "SqliteSaver 可以把状态持久化到硬盘。",
    "ToolCall 本身不会执行工具。",
]

query_vector = list(
    model.query_embed(query)
)[0]

passage_vectors = list(
    model.passage_embed(passages)
)


def cosine_similarity(
    vector_a,
    vector_b,
) -> float:
    return float(
        np.dot(vector_a, vector_b)
        / (
            np.linalg.norm(vector_a)
            * np.linalg.norm(vector_b)
        )
    )


print(f"问题向量维度：{len(query_vector)}")
print(f"向量前8个数字：{query_vector[:8]}")

for passage, passage_vector in zip(
    passages,
    passage_vectors,
):
    similarity = cosine_similarity(
        query_vector,
        passage_vector,
    )

    print("\n资料：", passage)
    print("相似度：", similarity)