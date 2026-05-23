import unittest
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    # 保持 tests 脱离 src 后仍可导入核心模块。
    sys.path.append(str(SRC_DIR))

from hybrid_retriever import (
    HybridRetriever,
    fuse_scores,
    normalize_bm25_scores,
    normalize_semantic_scores,
)


class _BM25Item:
    """模拟 BM25 检索结果对象。"""

    def __init__(self, doc_id, score, content=""):
        self.doc_id = doc_id
        self.score = score
        self.content = content


class _FaissItem:
    """模拟 FAISS 检索结果对象。"""

    def __init__(self, chunk_id, distance, content="", metadata=None):
        self.chunk_id = chunk_id
        self.distance = distance
        self.content = content
        self.metadata = metadata or {}


class _FakeBM25Index:
    """用于单测的 BM25 假索引，避免依赖真实分词与排序细节。"""

    def __init__(self, result_map):
        self._result_map = result_map
        self._ready = False

    def build_index(self, documents, doc_ids):
        self._ready = bool(documents) and bool(doc_ids)

    def is_ready(self):
        return self._ready

    def clear(self):
        self._ready = False

    def search(self, query, top_k=5):
        return list(self._result_map.get(query, []))[:top_k]


class _FakeVectorStore:
    """用于单测的 FAISS 假向量库，按 query 向量精确映射返回结果。"""

    def __init__(self, result_map):
        self._result_map = result_map
        self._ready = False

    def build_index(self, chunks, chunk_ids, metadatas, embeddings):
        self._ready = bool(chunks) and bool(chunk_ids) and embeddings is not None

    def is_ready(self):
        return self._ready

    def clear(self):
        self._ready = False

    def search(self, query_embedding, top_k=5):
        key = tuple(float(v) for v in np.asarray(query_embedding).reshape(-1))
        return list(self._result_map.get(key, []))[:top_k]


class HybridRetrieverUtilityTests(unittest.TestCase):
    """验证混合检索工具函数（归一化与融合公式）。"""

    def test_normalize_bm25_scores_scales_by_max(self):
        """BM25 分数应按最大值缩放到 [0,1]。"""
        normalized = normalize_bm25_scores(
            [
                _BM25Item("doc_1", 2.0),
                _BM25Item("doc_2", 1.0),
            ]
        )
        self.assertAlmostEqual(normalized["doc_1"], 1.0, places=6)
        self.assertAlmostEqual(normalized["doc_2"], 0.5, places=6)

    def test_normalize_semantic_scores_uses_distance_inverse(self):
        """语义分数应由距离反推相似度并归一化。"""
        normalized = normalize_semantic_scores(
            [
                _FaissItem("doc_1", 0.0),
                _FaissItem("doc_2", 1.0),
            ]
        )
        self.assertAlmostEqual(normalized["doc_1"], 1.0, places=6)
        self.assertAlmostEqual(normalized["doc_2"], 0.5, places=6)

    def test_fuse_scores_applies_alpha(self):
        """融合分数应满足 alpha 加权公式。"""
        merged = fuse_scores(
            bm25_scores={"doc_1": 1.0},
            semantic_scores={"doc_1": 0.5},
            alpha=0.7,
        )
        # 按融合公式可得：0.7 * 0.5 + 0.3 * 1.0 = 0.65
        self.assertAlmostEqual(merged["doc_1"]["hybrid_score"], 0.65, places=6)

    def test_fuse_scores_rejects_invalid_alpha(self):
        """alpha 越界时应抛出参数错误。"""
        with self.assertRaises(ValueError):
            fuse_scores({"doc_1": 1.0}, {"doc_1": 1.0}, alpha=1.2)


class HybridRetrieverFlowTests(unittest.TestCase):
    """验证 HybridRetriever 的索引构建与检索流程。"""

    def setUp(self):
        self.documents = ["doc one", "doc two", "doc three"]
        self.doc_ids = ["doc_1", "doc_2", "doc_3"]
        self.metadatas = [{"i": 0}, {"i": 1}, {"i": 2}]
        self.embeddings = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype="float32",
        )

    def test_search_merges_and_ranks_union_candidates(self):
        """两路候选合并后应按融合分数正确排序。"""
        bm25_map = {
            "q": [
                _BM25Item("doc_1", 2.0, "doc one"),
                _BM25Item("doc_2", 1.0, "doc two"),
            ]
        }
        query_key = (1.0, 0.0)
        vector_map = {
            query_key: [
                _FaissItem("doc_2", 0.1, "doc two", {"i": 1}),
                _FaissItem("doc_3", 0.2, "doc three", {"i": 2}),
            ]
        }

        retriever = HybridRetriever(
            alpha=0.7,
            bm25_index=_FakeBM25Index(bm25_map),
            vector_store=_FakeVectorStore(vector_map),
        )
        retriever.build_index(self.documents, self.doc_ids, self.metadatas, self.embeddings)

        results = retriever.search("q", np.array([[1.0, 0.0]], dtype="float32"), top_k=3)

        self.assertEqual([item.doc_id for item in results], ["doc_2", "doc_3", "doc_1"])
        self.assertGreater(results[0].hybrid_score, results[1].hybrid_score)
        self.assertGreater(results[1].hybrid_score, results[2].hybrid_score)

    def test_search_rejects_invalid_top_k(self):
        """top_k 非法时应抛出参数错误。"""
        retriever = HybridRetriever(
            alpha=0.7,
            bm25_index=_FakeBM25Index({"q": []}),
            vector_store=_FakeVectorStore({(1.0,): []}),
        )
        retriever.build_index(
            ["doc"],
            ["doc_1"],
            [{"i": 0}],
            np.array([[1.0]], dtype="float32"),
        )
        with self.assertRaises(ValueError):
            retriever.search("q", np.array([[1.0]], dtype="float32"), top_k=0)

    def test_build_index_rejects_length_mismatch(self):
        """构建索引时输入长度不一致应被拒绝。"""
        retriever = HybridRetriever(
            alpha=0.7,
            bm25_index=_FakeBM25Index({}),
            vector_store=_FakeVectorStore({}),
        )
        with self.assertRaises(ValueError):
            retriever.build_index(
                documents=["a", "b"],
                doc_ids=["doc_1"],
                metadatas=[{"i": 0}, {"i": 1}],
                embeddings=np.array([[1.0], [2.0]], dtype="float32"),
            )


if __name__ == "__main__":
    unittest.main()
