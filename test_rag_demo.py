import numpy as np
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from rag_demo import (
    load_knowledge_documents,
    search_top_documents,
    load_vector_cache,
    save_vector_cache,
    build_cache_fingerprint,
)


class FakeEmbeddingModel:
    def query_embed(self, query):
        return [
            [1.0, 0.0],
        ]


class RagDemoTest(unittest.TestCase):
    def test_search_top_documents_returns_highest_scores_first(
        self,
    ):
        document_list = [
            SimpleNamespace(
                page_content="低分资料",
            ),
            SimpleNamespace(
                page_content="中分资料",
            ),
            SimpleNamespace(
                page_content="高分资料",
            ),
        ]

        vector_list = [
            [0.0, 1.0],
            [0.8, 0.2],
            [1.0, 0.0],
        ]

        results = search_top_documents(
            query="测试问题",
            document_list=document_list,
            vector_list=vector_list,
            model=FakeEmbeddingModel(),
            top_k=2,
        )

        contents = [
            document.page_content
            for document, score in results
        ]

        self.assertEqual(
            len(results),
            2,
        )
        self.assertEqual(
            contents,
            [
                "高分资料",
                "中分资料",
            ],
        )
        self.assertGreaterEqual(
            results[0][1],
            results[1][1],
        )

    def test_vector_cache_requires_matching_fingerprint(self):
        with TemporaryDirectory() as directory:
            cache_path = Path(directory) / "vectors.json"

            save_vector_cache(
                cache_path,
                "旧指纹",
                [np.array([0.1, 0.2])],
            )

            cached_vectors = load_vector_cache(
                cache_path,
                "旧指纹",
            )

            self.assertIsNotNone(cached_vectors)
            np.testing.assert_allclose(
                cached_vectors,
                [[0.1, 0.2]],
            )

            outdated_vectors = load_vector_cache(
                cache_path,
                "新指纹",
            )

            self.assertIsNone(outdated_vectors)

    def test_cache_fingerprint_changes_with_content(self):
        document = SimpleNamespace(
            page_content="原来的知识",
            metadata={"source": "example.md"},
        )

        original_fingerprint = build_cache_fingerprint(
            [document],
        )

        same_fingerprint = build_cache_fingerprint(
            [document],
        )

        self.assertEqual(
            original_fingerprint,
            same_fingerprint,
        )

        document.page_content = "修改后的知识"

        changed_fingerprint = build_cache_fingerprint(
            [document],
        )

        self.assertNotEqual(
            original_fingerprint,
            changed_fingerprint,
        )

    def test_load_knowledge_documents_reads_all_markdown_files(
        self,
    ):
        with TemporaryDirectory() as directory:
            knowledge_directory = Path(directory)

            (
                knowledge_directory / "a.md"
            ).write_text(
                "第一份知识：ToolCall 是工具调用单。",
                encoding="utf-8",
            )

            (
                knowledge_directory / "b.md"
            ).write_text(
                "第二份知识：LangGraph 负责编排工作流。",
                encoding="utf-8",
            )

            (
                knowledge_directory / "ignored.txt"
            ).write_text(
                "这份内容不应该被加载。",
                encoding="utf-8",
            )

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=0,
                add_start_index=True,
            )

            documents = load_knowledge_documents(
                knowledge_directory,
                splitter,
            )

            documents_by_source = {
                document.metadata["source"]:
                document.page_content
                for document in documents
            }

            self.assertEqual(
                set(documents_by_source),
                {
                    "a.md",
                    "b.md",
                },
            )
            self.assertIn(
                "ToolCall",
                documents_by_source["a.md"],
            )
            self.assertIn(
                "LangGraph",
                documents_by_source["b.md"],
            )
            self.assertNotIn(
                "ignored.txt",
                documents_by_source,
            )

    def test_markdown_sections_stay_together(self):
        with TemporaryDirectory() as directory:
            knowledge_directory = Path(directory)

            (knowledge_directory / "example.md").write_text(
                "# 测试知识\n\n"
                "## 记忆\n\n"
                "InMemorySaver 保存到内存。\n\n"
                "SqliteSaver 保存到硬盘。\n\n"
                "## 工具\n\n"
                "ToolCall 本身不会执行工具。\n",
                encoding="utf-8",
            )

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=20,
                add_start_index=True,
            )

            documents = load_knowledge_documents(
                knowledge_directory,
                splitter,
            )

            self.assertEqual(len(documents), 2)

            memory_document = documents[0]
            self.assertIn("## 记忆", memory_document.page_content)
            self.assertIn("InMemorySaver", memory_document.page_content)
            self.assertIn("SqliteSaver", memory_document.page_content)
            self.assertNotIn("ToolCall", memory_document.page_content)

            self.assertEqual(
                memory_document.metadata["source"],
                "example.md",
            )
            self.assertEqual(
                memory_document.metadata["section"],
                "记忆",
            )
            self.assertIn("ToolCall", documents[1].page_content)

if __name__ == "__main__":
    unittest.main()