"""
混合检索模块。

职责：
1. 在同一语料上并行执行 BM25 与向量检索。
2. 将异构分数归一化到可比较的 [0, 1] 区间。
3. 按权重融合两路结果并返回排序后的候选集。
"""

from dataclasses import dataclass

from bm25_index import BM25IndexManager
from vector_store import FaissVectorStore


DEFAULT_HYBRID_ALPHA = 0.7
DEFAULT_CANDIDATE_MULTIPLIER = 2


@dataclass(frozen=True)
class HybridSearchResult:
    """混合检索的标准化输出结构。"""

    rank: int
    doc_id: str
    content: str
    metadata: dict
    hybrid_score: float
    semantic_score: float
    bm25_score: float


def normalize_bm25_scores(bm25_results):
    """
    使用最大值缩放将 BM25 分数归一化到 [0, 1]。
    """
    if not bm25_results:
        return {}

    raw = {}
    for item in bm25_results:
        score = max(float(item.score), 0.0)
        previous = raw.get(item.doc_id, 0.0)
        if score > previous:
            raw[item.doc_id] = score

    if not raw:
        return {}

    max_score = max(raw.values())
    if max_score <= 0:
        return {doc_id: 0.0 for doc_id in raw}
    return {doc_id: score / max_score for doc_id, score in raw.items()}


def normalize_semantic_scores(faiss_results):
    """
    将 FAISS L2 距离转为相似度，再归一化到 [0, 1]。

    similarity = 1 / (1 + distance)
    """
    if not faiss_results:
        return {}

    raw = {}
    for item in faiss_results:
        similarity = 1.0 / (1.0 + max(float(item.distance), 0.0))
        previous = raw.get(item.chunk_id, 0.0)
        if similarity > previous:
            raw[item.chunk_id] = similarity

    if not raw:
        return {}

    max_score = max(raw.values())
    if max_score <= 0:
        return {doc_id: 0.0 for doc_id in raw}
    return {doc_id: score / max_score for doc_id, score in raw.items()}


def fuse_scores(bm25_scores, semantic_scores, alpha):
    """
    加权融合：
    hybrid = alpha * semantic + (1 - alpha) * bm25
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be in [0, 1]")

    merged = {}
    all_doc_ids = set(bm25_scores) | set(semantic_scores)
    for doc_id in all_doc_ids:
        bm25_score = float(bm25_scores.get(doc_id, 0.0))
        semantic_score = float(semantic_scores.get(doc_id, 0.0))
        hybrid_score = alpha * semantic_score + (1 - alpha) * bm25_score
        merged[doc_id] = {
            "bm25_score": bm25_score,
            "semantic_score": semantic_score,
            "hybrid_score": hybrid_score,
        }
    return merged


class HybridRetriever:
    """
    融合 BM25 与 FAISS 输出的混合检索器。
    """

    def __init__(
        self,
        alpha=DEFAULT_HYBRID_ALPHA,
        bm25_index=None,
        vector_store=None,
    ):
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = float(alpha)
        self.bm25_index = bm25_index or BM25IndexManager()
        self.vector_store = vector_store or FaissVectorStore()
        self.doc_ids = []
        self.content_map = {}
        self.metadata_map = {}

    @property
    def total_docs(self):
        return len(self.doc_ids)

    def is_ready(self):
        return self.bm25_index.is_ready() and self.vector_store.is_ready() and self.total_docs > 0

    def clear(self):
        self.bm25_index.clear()
        self.vector_store.clear()
        self.doc_ids = []
        self.content_map = {}
        self.metadata_map = {}

    def build_index(self, documents, doc_ids, metadatas, embeddings):
        """
        在同一语料上同时构建 BM25 与 FAISS 索引。
        """
        documents = list(documents)
        doc_ids = list(doc_ids)
        metadatas = [dict(metadata) for metadata in metadatas]

        if not documents:
            raise ValueError("documents must not be empty")
        if len(documents) != len(doc_ids):
            raise ValueError("documents and doc_ids must have the same length")
        if len(documents) != len(metadatas):
            raise ValueError("documents and metadatas must have the same length")
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("doc_ids must be unique")

        self.bm25_index.build_index(documents, doc_ids)
        self.vector_store.build_index(documents, doc_ids, metadatas, embeddings)

        self.doc_ids = doc_ids
        self.content_map = dict(zip(doc_ids, documents))
        self.metadata_map = dict(zip(doc_ids, metadatas))

    def search(self, query, query_embedding, top_k=5, candidate_multiplier=DEFAULT_CANDIDATE_MULTIPLIER):
        """
        执行 BM25 + FAISS 检索并返回融合排序结果。
        """
        if not self.is_ready():
            raise RuntimeError("hybrid retriever is not ready, call build_index first")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be greater than 0")

        candidate_k = min(max(top_k * int(candidate_multiplier), top_k), self.total_docs)

        bm25_results = self.bm25_index.search(query, top_k=candidate_k)
        semantic_results = self.vector_store.search(query_embedding, top_k=candidate_k)

        bm25_scores = normalize_bm25_scores(bm25_results)
        semantic_scores = normalize_semantic_scores(semantic_results)
        fused = fuse_scores(bm25_scores, semantic_scores, self.alpha)

        ranked_doc_ids = sorted(
            fused.keys(),
            key=lambda doc_id: (
                fused[doc_id]["hybrid_score"],
                fused[doc_id]["semantic_score"],
                fused[doc_id]["bm25_score"],
            ),
            reverse=True,
        )

        results = []
        for rank, doc_id in enumerate(ranked_doc_ids[:top_k], start=1):
            score_pack = fused[doc_id]
            results.append(
                HybridSearchResult(
                    rank=rank,
                    doc_id=doc_id,
                    content=self.content_map[doc_id],
                    metadata=dict(self.metadata_map[doc_id]),
                    hybrid_score=float(score_pack["hybrid_score"]),
                    semantic_score=float(score_pack["semantic_score"]),
                    bm25_score=float(score_pack["bm25_score"]),
                )
            )
        return results
