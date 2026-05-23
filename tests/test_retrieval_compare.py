import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.append(str(EXPERIMENTS_DIR))

from retrieval_compare import hit_at_1, top1_agreement_rate, top1_id


class RetrievalCompareUtilityTests(unittest.TestCase):
    """验证 BM25/FAISS 对比脚本中的统计函数。"""

    class _Item:
        """用于构造最小检索结果对象。"""

        def __init__(self, value):
            self.doc_id = value
            self.chunk_id = value

    def test_top1_id_returns_none_on_empty_results(self):
        """空结果列表时，top1_id 应返回 None。"""
        self.assertIsNone(top1_id([], "doc_id"))

    def test_top1_id_returns_first_result_id(self):
        """有结果时，top1_id 应返回第一条的 id。"""
        results = [self._Item("doc_0002"), self._Item("doc_0001")]
        self.assertEqual(top1_id(results, "doc_id"), "doc_0002")

    def test_hit_at_1_computes_ratio(self):
        """hit_at_1 应返回命中比例。"""
        predictions = {"q1": "doc_0001", "q2": "doc_0003", "q3": "doc_0003"}
        expected = {"q1": "doc_0001", "q2": "doc_0002", "q3": "doc_0003"}
        self.assertAlmostEqual(hit_at_1(predictions, expected), 2 / 3, places=6)

    def test_top1_agreement_rate_computes_ratio(self):
        """top1_agreement_rate 应返回两路 Top1 一致比例。"""
        left = {"q1": "a", "q2": "b", "q3": "c"}
        right = {"q1": "a", "q2": "x", "q3": "c"}
        self.assertAlmostEqual(top1_agreement_rate(left, right, ["q1", "q2", "q3"]), 2 / 3, places=6)


if __name__ == "__main__":
    unittest.main()
