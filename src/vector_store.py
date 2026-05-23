from dataclasses import dataclass

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None


@dataclass(frozen=True)
class SearchResult:
    """标准化检索输出结构，便于实验和后续集成。"""
    rank: int
    chunk_id: str
    content: str
    metadata: dict
    distance: float


class FaissVectorStore:
    """
    本地 FAISS 向量索引。

    只负责保存向量和返回检索结果，不负责读取文件、文本分块或生成 embedding。
    """

    def __init__(self):
        # id_order 记录“FAISS 向量位置 -> chunk_id”的映射顺序。
        self.index = None
        self.dimension = None
        self.id_order = []
        self.contents_map = {}
        self.metadatas_map = {}

    @property
    def total_chunks(self):
        """当前索引中的 chunk 数量。"""
        return len(self.id_order)

    def is_ready(self):
        """索引是否可检索。"""
        return self.index is not None and self.dimension is not None and self.total_chunks > 0

    def clear(self):
        """清空索引与映射缓存。"""
        self.index = None
        self.dimension = None
        self.id_order = []
        self.contents_map = {}
        self.metadatas_map = {}

    def build_index(self, chunks, chunk_ids, metadatas, embeddings):
        """
        构建 FAISS 索引。

        Args:
            chunks: chunk 文本列表。
            chunk_ids: 与 chunks 一一对应的 id 列表。
            metadatas: 与 chunks 一一对应的 metadata 列表。
            embeddings: 形状为 (n_chunks, dim) 的 float32 向量矩阵。
        """
        _require_faiss()

        chunks = list(chunks)
        chunk_ids = list(chunk_ids)
        metadatas = [dict(metadata) for metadata in metadatas]
        embeddings = _as_2d_float32("embeddings", embeddings)

        if not chunks:
            raise ValueError("chunks 不能为空")
        if len(chunks) != len(chunk_ids):
            raise ValueError("chunks 和 chunk_ids 数量必须一致")
        if len(chunks) != len(metadatas):
            raise ValueError("chunks 和 metadatas 数量必须一致")
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks 数量必须等于 embeddings 行数")
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("chunk_ids 不能重复")

        dimension = int(embeddings.shape[1])
        if dimension <= 0:
            raise ValueError("embedding 维度必须大于 0")

        # 学习阶段使用精确检索 IndexFlatL2，行为最直观。
        index = faiss.IndexFlatL2(dimension)
        index.add(np.ascontiguousarray(embeddings, dtype="float32"))

        self.index = index
        self.dimension = dimension
        self.id_order = chunk_ids
        self.contents_map = dict(zip(chunk_ids, chunks))
        self.metadatas_map = dict(zip(chunk_ids, metadatas))

    def search(self, query_embedding, top_k=3):
        """
        使用 query embedding 检索最相似的 chunks。

        使用 IndexFlatL2 时，distance 越小表示越相似。
        """
        if not self.is_ready():
            raise RuntimeError("FAISS 索引尚未构建，请先调用 build_index()")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        query_embedding = _as_2d_float32("query_embedding", query_embedding)
        if query_embedding.shape[0] != 1:
            raise ValueError("query_embedding 形状必须是 (1, dim)")
        if query_embedding.shape[1] != self.dimension:
            raise ValueError("query_embedding 和 FAISS 索引维度必须一致")

        search_k = min(top_k, self.total_chunks)
        distances, indices = self.index.search(
            np.ascontiguousarray(query_embedding, dtype="float32"),
            search_k,
        )

        results = []
        for rank, (distance, index_position) in enumerate(zip(distances[0], indices[0]), start=1):
            if index_position < 0:
                continue
            # FAISS 返回的是位置索引，需要映射回业务字段。
            chunk_id = self.id_order[int(index_position)]
            results.append(
                SearchResult(
                    rank=rank,
                    chunk_id=chunk_id,
                    content=self.contents_map[chunk_id],
                    metadata=dict(self.metadatas_map[chunk_id]),
                    distance=float(distance),
                )
            )

        return results


def _require_faiss():
    """确保运行环境已安装 faiss-cpu。"""
    if faiss is None:
        raise RuntimeError("缺少依赖 faiss-cpu，请先运行：python -m pip install faiss-cpu")


def _as_2d_float32(name, value):
    """统一输入类型和形状检查。"""
    array = np.asarray(value, dtype="float32")
    if array.ndim != 2:
        raise ValueError(f"{name} 必须是二维矩阵")
    if array.shape[0] == 0:
        raise ValueError(f"{name} 不能为空")
    return array
