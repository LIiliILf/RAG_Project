import unittest
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    # 保持 tests 脱离 src 后仍可导入核心模块。
    sys.path.append(str(SRC_DIR))

from reranker import rerank_results


class RerankerTests(unittest.TestCase):
    """验证重排序模块的核心行为。"""

    def setUp(self):
        self.docs = [
            "今天适合散步。",
            "显卡型号 RTX4090 配备 24GB 显存。",
            "RAG 系统先检索再回答。",
        ]
        self.doc_ids = ["doc_0001", "doc_0002", "doc_0003"]
        self.metadatas = [
            {"source": "sample", "idx": 0},
            {"source": "sample", "idx": 1},
            {"source": "sample", "idx": 2},
        ]

    def test_baseline_mode_keeps_original_order(self):
        """method=none 时应保持原始候选顺序。"""
        results = rerank_results(
            query="RTX4090 显存",
            docs=self.docs,
            doc_ids=self.doc_ids,
            metadata_list=self.metadatas,
            method="none",
            top_k=2,
        )
        self.assertEqual([item.doc_id for item in results], ["doc_0001", "doc_0002"])
        self.assertTrue(all(item.method == "none" for item in results))

    def test_overlap_mode_promotes_relevant_document(self):
        """overlap 模式应把关键词更匹配的候选排在前面。"""
        results = rerank_results(
            query="RTX4090 显存",
            docs=self.docs,
            doc_ids=self.doc_ids,
            metadata_list=self.metadatas,
            method="overlap",
            top_k=3,
        )
        self.assertEqual(results[0].doc_id, "doc_0002")
        self.assertEqual(results[0].method, "overlap")

    def test_length_mismatch_raises_error(self):
        """docs/doc_ids/metadata 数量不一致时应抛错。"""
        with self.assertRaises(ValueError):
            rerank_results(
                query="test",
                docs=["a", "b"],
                doc_ids=["doc_1"],
                metadata_list=[{"i": 0}, {"i": 1}],
                method="overlap",
                top_k=1,
            )

    def test_invalid_top_k_raises_error(self):
        """top_k 非法时应抛错。"""
        with self.assertRaises(ValueError):
            rerank_results(
                query="test",
                docs=self.docs,
                doc_ids=self.doc_ids,
                metadata_list=self.metadatas,
                method="overlap",
                top_k=0,
            )

    def test_cross_encoder_failure_falls_back(self):
        """cross_encoder 失败时应自动回退 baseline。"""
        with patch("reranker.rerank_with_cross_encoder", side_effect=RuntimeError("mock failed")):
            results = rerank_results(
                query="RTX4090 显存",
                docs=self.docs,
                doc_ids=self.doc_ids,
                metadata_list=self.metadatas,
                method="cross_encoder",
                top_k=2,
            )

        self.assertEqual([item.doc_id for item in results], ["doc_0001", "doc_0002"])
        self.assertTrue(all(item.method == "cross_encoder_fallback" for item in results))


if __name__ == "__main__":
    unittest.main()
