import json
import hashlib

from langchain_deepseek import ChatDeepSeek
from pathlib import Path
import numpy as np
from fastembed import TextEmbedding
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 20

knowledge_directory = (
    Path(__file__).parent
    / "knowledge"
)

VECTOR_CACHE_PATH = (
    Path(__file__).parent
    / "data"
    / "document_vectors.json"
)

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "title"),
        ("##", "section"),
    ],
    strip_headers=False,
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True,
)

def calculate_text_hash(text: str) -> str:
    text_bytes = text.encode("utf-8")
    return hashlib.sha256(text_bytes).hexdigest()

def build_cache_fingerprint(document_list: list) -> str:
    cache_information = {
        "model_name": EMBEDDING_MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "documents": [
            {
                "content": document.page_content,
                "metadata": document.metadata,
            }
            for document in document_list
        ],
    }

    information_text = json.dumps(
        cache_information,
        ensure_ascii=False,
        sort_keys=True,
    )

    return calculate_text_hash(information_text)

def save_vector_cache(
    path: Path,
    fingerprint: str,
    vectors: list,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_data = {
        "fingerprint": fingerprint,
        "vectors": [
            vector.tolist()
            for vector in vectors
        ],
    }

    path.write_text(
        json.dumps(
            cache_data,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

def load_vector_cache(
    path: Path,
    fingerprint: str,
) -> list | None:
    if not path.exists():
        return None

    try:
        cache_data = json.loads(
            path.read_text(encoding="utf-8")
        )

        if cache_data["fingerprint"] != fingerprint:
            return None

        return [
            np.array(vector, dtype=float)
            for vector in cache_data["vectors"]
        ]
    except (OSError, ValueError, KeyError, TypeError):
        return None

def load_knowledge_documents(
    directory: Path,
    splitter: RecursiveCharacterTextSplitter,
) -> list:
    knowledge_paths = sorted(
        directory.glob("*.md")
    )

    if not knowledge_paths:
        raise FileNotFoundError(
            "knowledge 目录中没有 Markdown 文件。"
        )

    section_documents = []

    for path in knowledge_paths:
        text = path.read_text(encoding="utf-8")

        sections = header_splitter.split_text(text)

        for section in sections:
            section.metadata["source"] = path.name

        section_documents.extend(sections)

    return splitter.split_documents(section_documents)


documents = load_knowledge_documents(
    knowledge_directory,
    text_splitter,
)
embedding_model = TextEmbedding(
    model_name=EMBEDDING_MODEL_NAME,

)

document_texts = [
    document.page_content
    for document in documents
]

cache_fingerprint = build_cache_fingerprint(
    documents,
)

document_vectors = load_vector_cache(
    VECTOR_CACHE_PATH,
    cache_fingerprint,
)

if document_vectors is None:
    document_vectors = list(
        embedding_model.passage_embed(
            document_texts
        )
    )

    save_vector_cache(
        VECTOR_CACHE_PATH,
        cache_fingerprint,
        document_vectors,
    )

    print("[cache] 已重新生成并保存文档向量。")
else:
    print("[cache] 已复用硬盘中的文档向量。")

MINIMUM_SIMILARITY = 0.4


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


def search_documents(
    query: str,
    document_list: list,
    vector_list: list,
    model: TextEmbedding,
):
    query_vector = list(
        model.query_embed(query)
    )[0]

    best_document = None
    best_score = -1.0

    for document, document_vector in zip(
        document_list,
        vector_list,
    ):
        score = cosine_similarity(
            query_vector,
            document_vector,
        )

        if score > best_score:
            best_score = score
            best_document = document

    return best_document, best_score

def search_top_documents(
    query: str,
    document_list: list,
    vector_list: list,
    model: TextEmbedding,
    top_k: int = 3,
):
    if top_k <= 0:
        raise ValueError(
            "top_k 必须大于 0。"
        )

    query_vector = list(
        model.query_embed(query)
    )[0]

    scored_documents = []

    for document, document_vector in zip(
        document_list,
        vector_list,
    ):
        score = cosine_similarity(
            query_vector,
            document_vector,
        )

        scored_documents.append(
            (
                document,
                score,
            )
        )

    scored_documents.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return scored_documents[:top_k]
def generate_answer(
    query: str,
    document,
) -> str:
    model = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
    )

    source = document.metadata["source"]

    prompt = (
        "你是一个基于资料回答问题的助手。"
        "只能根据下面提供的资料回答；"
        "如果资料不足，请明确回答“资料不足”。"
        f"\n\n资料来源：{source}"
        f"\n资料内容：\n{document.page_content}"
        f"\n\n用户问题：{query}"
        "\n\n请给出简洁回答，并在结尾标注资料来源。"
    )

    response = model.invoke(prompt)
    return response.content

def main():
    print(
    f"已为 {len(document_vectors)} "
    f"个文本块生成向量。"
)

    for index, document in enumerate(
        documents,
        start=1,
    ):
        print(f"\n--- chunk {index} ---")
        print(
            f"source: "
            f"{document.metadata['source']}"
        )
        print(
            f"start_index: "
            f"{document.metadata['start_index']}"
        )
        print(document.page_content)

    query = input(
        "\n请输入要检索的问题："
    ).strip()

    matched_document, score = search_documents(
        query,
        documents,
        document_vectors,
        embedding_model,
    )

    if (
        matched_document is None
        or score < MINIMUM_SIMILARITY
    ):
        print(
            "\n没有找到足够相关的资料。"
            f"\n最高匹配分数：{score:.4f}"
        )
    else:
        print("\n--- 检索结果 ---")
        print(f"匹配分数：{score}")
        print(
            f"来源："
            f"{matched_document.metadata['source']}"
        )
        print(matched_document.page_content)
        answer = generate_answer(
            query,
            matched_document,
        )

        print("\n--- DeepSeek 回答 ---")
        print(answer)

if __name__ == "__main__":
    main()  