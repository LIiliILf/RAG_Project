import unittest
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    # 保持 tests 脱离 src 后仍可导入核心模块。
    sys.path.append(str(SRC_DIR))

from embeddings import cosine_similarity


class EmbeddingUtilityTests(unittest.TestCase):
    """验证 embedding 工具函数的基础数学行为。"""

    def test_cosine_similarity_returns_expected_shape(self):
        """query 与多文档比较后应返回一维分数数组。"""
        query = np.array([[1.0, 0.0]], dtype="float32")
        documents = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ],
            dtype="float32",
        )

        scores = cosine_similarity(query, documents)

        self.assertEqual(scores.shape, (3,))

    def test_cosine_similarity_ranks_same_direction_highest(self):
        """方向一致的向量应有最高相似度。"""
        query = np.array([[1.0, 0.0]], dtype="float32")
        documents = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [-1.0, 0.0],
            ],
            dtype="float32",
        )

        scores = cosine_similarity(query, documents)
        best_index = int(np.argmax(scores))

        self.assertEqual(best_index, 1)
        self.assertAlmostEqual(float(scores[1]), 1.0, places=5)

    def test_cosine_similarity_rejects_empty_documents(self):
        """空文档矩阵应触发参数错误。"""
        query = np.array([[1.0, 0.0]], dtype="float32")
        documents = np.empty((0, 2), dtype="float32")

        with self.assertRaises(ValueError):
            cosine_similarity(query, documents)


if __name__ == "__main__":
    unittest.main()
