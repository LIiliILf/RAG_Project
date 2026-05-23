import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    # 保持 tests 脱离 src 后仍可导入核心模块。
    sys.path.append(str(SRC_DIR))

from bm25_index import BM25IndexManager


class BM25IndexManagerTests(unittest.TestCase):
    """验证 BM25 索引构建、检索与输入校验。"""

    def setUp(self):
        self.documents = [
            "显卡型号 RTX4090 配备 24GB 显存",
            "错误码 ERR_CONN_RESET 与网络中断有关",
            "RAG 系统先检索再生成",
        ]
        self.doc_ids = ["doc_0001", "doc_0002", "doc_0003"]

    def test_build_index_stores_mappings(self):
        """build_index 后应记录文档数量与映射关系。"""
        manager = BM25IndexManager()
        manager.build_index(self.documents, self.doc_ids)

        self.assertTrue(manager.is_ready())
        self.assertEqual(manager.total_docs, 3)
        self.assertEqual(manager.doc_mapping[0], "doc_0001")
        self.assertEqual(len(manager.tokenized_corpus), 3)

    def test_search_returns_expected_top_document(self):
        """术语类 query 应命中对应文档。"""
        manager = BM25IndexManager()
        manager.build_index(self.documents, self.doc_ids)

        results = manager.search("ERR_CONN_RESET 怎么解决", top_k=2)

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].doc_id, "doc_0002")
        self.assertIn("ERR_CONN_RESET", results[0].content)
        self.assertGreater(results[0].score, 0)

    def test_build_index_rejects_mismatched_lengths(self):
        """documents 与 doc_ids 数量不一致时应抛错。"""
        manager = BM25IndexManager()
        with self.assertRaises(ValueError):
            manager.build_index(self.documents, ["doc_0001"])

    def test_search_rejects_empty_store(self):
        """索引未构建时不允许检索。"""
        manager = BM25IndexManager()
        with self.assertRaises(RuntimeError):
            manager.search("RTX4090", top_k=1)

    def test_search_rejects_invalid_top_k(self):
        """top_k 必须大于 0。"""
        manager = BM25IndexManager()
        manager.build_index(self.documents, self.doc_ids)
        with self.assertRaises(ValueError):
            manager.search("RTX4090", top_k=0)

    def test_search_returns_empty_when_no_terms_match(self):
        """无匹配词项时应返回空结果。"""
        manager = BM25IndexManager()
        manager.build_index(self.documents, self.doc_ids)

        results = manager.search("UNSEEN_TOKEN_12345", top_k=3)
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
