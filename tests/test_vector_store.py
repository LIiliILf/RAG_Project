import unittest
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    # 保持 tests 脱离 src 后仍可导入核心模块。
    sys.path.append(str(SRC_DIR))

from vector_store import FaissVectorStore


class FaissVectorStoreTests(unittest.TestCase):
    """验证向量索引构建、检索和输入校验逻辑。"""

    def test_build_index_stores_all_chunks(self):
        """build_index 后应记录完整状态。"""
        chunks = ["文档解析", "文本分块", "向量检索"]
        chunk_ids = ["chunk_0001", "chunk_0002", "chunk_0003"]
        metadatas = [
            {"source": "sample.txt", "chunk_index": 0},
            {"source": "sample.txt", "chunk_index": 1},
            {"source": "sample.txt", "chunk_index": 2},
        ]
        embeddings = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype="float32",
        )

        store = FaissVectorStore()
        store.build_index(chunks, chunk_ids, metadatas, embeddings)

        self.assertTrue(store.is_ready())
        self.assertEqual(store.total_chunks, 3)
        self.assertEqual(store.dimension, 2)
        self.assertEqual(store.id_order, chunk_ids)

    def test_search_returns_nearest_chunks_with_metadata(self):
        """search 应返回按距离排序且可回溯 metadata 的结果。"""
        chunks = ["文档解析负责读取文件", "天气很好适合散步", "FAISS 负责向量检索"]
        chunk_ids = ["chunk_0001", "chunk_0002", "chunk_0003"]
        metadatas = [
            {"source": "sample.txt", "chunk_index": 0},
            {"source": "sample.txt", "chunk_index": 1},
            {"source": "sample.txt", "chunk_index": 2},
        ]
        embeddings = np.array(
            [
                [0.0, 0.0],
                [10.0, 10.0],
                [1.0, 0.0],
            ],
            dtype="float32",
        )

        store = FaissVectorStore()
        store.build_index(chunks, chunk_ids, metadatas, embeddings)

        results = store.search(np.array([[0.2, 0.0]], dtype="float32"), top_k=2)

        self.assertEqual([result.chunk_id for result in results], ["chunk_0001", "chunk_0003"])
        self.assertEqual(results[0].content, "文档解析负责读取文件")
        self.assertEqual(results[0].metadata["source"], "sample.txt")
        self.assertEqual(results[0].rank, 1)
        self.assertLess(results[0].distance, results[1].distance)

    def test_build_index_rejects_mismatched_lengths(self):
        """输入列表数量不一致时应抛出错误。"""
        store = FaissVectorStore()

        with self.assertRaises(ValueError):
            store.build_index(
                chunks=["文档解析", "文本分块"],
                chunk_ids=["chunk_0001"],
                metadatas=[{"source": "sample.txt"}, {"source": "sample.txt"}],
                embeddings=np.array([[0.0, 0.0], [1.0, 0.0]], dtype="float32"),
            )

    def test_search_rejects_empty_store(self):
        """索引未构建时不允许检索。"""
        store = FaissVectorStore()

        with self.assertRaises(RuntimeError):
            store.search(np.array([[0.0, 0.0]], dtype="float32"), top_k=1)

    def test_search_rejects_wrong_dimension(self):
        """query 向量维度不匹配时应抛出错误。"""
        store = FaissVectorStore()
        store.build_index(
            chunks=["文档解析"],
            chunk_ids=["chunk_0001"],
            metadatas=[{"source": "sample.txt"}],
            embeddings=np.array([[0.0, 0.0]], dtype="float32"),
        )

        with self.assertRaises(ValueError):
            store.search(np.array([[0.0, 0.0, 0.0]], dtype="float32"), top_k=1)


if __name__ == "__main__":
    unittest.main()
