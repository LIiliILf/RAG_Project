"""
Embedding 工具模块。

职责：
1. 决定使用哪个本地/在线 embedding 模型。
2. 提供文档与查询向量化函数。
3. 提供基础余弦相似度计算。
"""

import logging
import os
import numpy as np
from pathlib import Path

from functools import lru_cache

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_CACHE_DIR = PROJECT_ROOT / "models" / "huggingface"
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(MODEL_CACHE_DIR / "sentence-transformers"))

LOCAL_BGE_ZH_MODEL_PATH = PROJECT_ROOT / "models" / "bge-small-zh-v1.5"
LOCAL_MINILM_MODEL_PATH = PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"

# 模型选择说明：
# - models/bge-small-zh-v1.5: 中文优化模型，优先使用
# - models/all-MiniLM-L6-v2: 英文优化模型，作为备用
# - all-MiniLM-L6-v2: 英文优化，384维，轻量快速
# - shibing624/text2vec-base-chinese: 中文优化
# - BAAI/bge-small-zh-v1.5: 中文优化，性能更好
def resolve_embed_model_name():
    """按优先级选择可用模型路径或模型名。"""
    if LOCAL_BGE_ZH_MODEL_PATH.exists():
        return str(LOCAL_BGE_ZH_MODEL_PATH)
    if LOCAL_MINILM_MODEL_PATH.exists():
        return str(LOCAL_MINILM_MODEL_PATH)
    return "all-MiniLM-L6-v2"


EMBED_MODEL_NAME = resolve_embed_model_name()

# 第一次调用时加载模型，后续直接复用同一个模型对象
@lru_cache(maxsize=1)
def get_embed_model():
    """
    获取向量化模型（单例 + 缓存）

    首次调用时加载模型，后续调用直接返回缓存的实例。
    """
    from sentence_transformers import SentenceTransformer
    logging.info(f"加载向量化模型: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    # 新旧 sentence-transformers 版本的维度获取接口不同。
    if hasattr(model, "get_embedding_dimension"):
        dimension = model.get_embedding_dimension()
    else:
        dimension = model.get_sentence_embedding_dimension()
    logging.info(f"向量化模型加载完成，输出维度: {dimension}")
    return model


def encode_texts(texts, show_progress=False):
    """
    将文本列表编码为向量

    Args:
        texts: 文本列表
        show_progress: 是否显示进度条

    Returns:
        numpy 数组，形状为 (n_texts, embedding_dim)
    """
    model = get_embed_model()
    embeddings = model.encode(texts, show_progress_bar=show_progress)
    # FAISS 默认要求 float32，统一在这里转换。
    return np.array(embeddings).astype('float32')


def encode_query(query):
    """
    将单个查询文本编码为向量

    Returns:
        numpy 数组，形状为 (1, embedding_dim)
    """
    model = get_embed_model()
    # query 保持二维形状 (1, dim)，方便后续直接检索。
    embedding = model.encode([query])
    return np.array(embedding).astype('float32')


def cosine_similarity(query_embedding, document_embeddings):
    """
    计算一个 query 向量和多个 document 向量的余弦相似度。

    Args:
        query_embedding: 形状为 (1, dim) 的查询向量。
        document_embeddings: 形状为 (n_docs, dim) 的文档向量矩阵。

    Returns:
        形状为 (n_docs,) 的相似度分数。
    """
    query_embedding = np.asarray(query_embedding, dtype="float32")
    document_embeddings = np.asarray(document_embeddings, dtype="float32")

    if query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
        raise ValueError("query_embedding 形状必须是 (1, dim)")
    if document_embeddings.ndim != 2 or document_embeddings.shape[0] == 0:
        raise ValueError("document_embeddings 形状必须是 (n_docs, dim)，且不能为空")
    if query_embedding.shape[1] != document_embeddings.shape[1]:
        raise ValueError("query_embedding 和 document_embeddings 的维度必须一致")

    query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
    document_norms = np.linalg.norm(document_embeddings, axis=1, keepdims=True)

    if np.any(query_norm == 0) or np.any(document_norms == 0):
        raise ValueError("向量不能是零向量")

    normalized_query = query_embedding / query_norm
    normalized_documents = document_embeddings / document_norms
    # (n_docs, dim) @ (dim,) -> (n_docs,)
    return normalized_documents @ normalized_query[0]
