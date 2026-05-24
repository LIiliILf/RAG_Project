"""
重排序模块（Reranker）。

职责：
1. 对召回候选进行二次精排，而不是重新做全量检索。
2. 提供统一入口，支持 baseline / overlap / cross_encoder 等方法。
3. 在模型不可用时提供可回退路径，保证流程可运行。
"""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import logging
import os
import re


DEFAULT_RERANK_METHOD = "cross_encoder"
# cross_encoder 模型切换说明：
# 1) 默认值：sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
# 2) 可通过环境变量覆盖：RERANK_CROSS_ENCODER_MODEL
# 3) 可通过 reranker_experiment.py 的 --cross-encoder-model 参数再次覆盖（优先级更高）
DEFAULT_CROSS_ENCODER_MODEL = os.getenv(
    "RERANK_CROSS_ENCODER_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)


@dataclass(frozen=True)
class RerankResult:
    """重排序后的标准化结果。"""

    rank: int
    doc_id: str
    content: str
    metadata: dict
    score: float
    method: str


def rerank_results(
    query,
    docs,
    doc_ids,
    metadata_list,
    method=DEFAULT_RERANK_METHOD,
    top_k=5,
    cross_encoder_model=None,
):
    """
    重排序统一入口。

    Args:
        query: 用户问题
        docs: 候选文本列表
        doc_ids: 与 docs 一一对应的文档 id
        metadata_list: 与 docs 一一对应的 metadata
        method: 重排序方法（none / overlap / cross_encoder / llm）
        top_k: 返回数量
        cross_encoder_model: cross_encoder 模型名或本地路径，仅 method=cross_encoder 时生效
    """
    docs, doc_ids, metadata_list = _validate_and_copy(docs, doc_ids, metadata_list, top_k)
    if not docs:
        return []

    normalized_method = str(method or "").strip().lower()
    if normalized_method in {"", "none", "baseline"}:
        return _fallback_results(doc_ids, docs, metadata_list, top_k, method="none")

    if normalized_method in {"overlap", "keyword", "lexical"}:
        return rerank_with_overlap(query, docs, doc_ids, metadata_list, top_k=top_k)

    if normalized_method == "cross_encoder":
        model_name = cross_encoder_model or DEFAULT_CROSS_ENCODER_MODEL
        try:
            return rerank_with_cross_encoder(
                query,
                docs,
                doc_ids,
                metadata_list,
                top_k=top_k,
                model_name=model_name,
            )
        except Exception as exc:
            logging.warning("cross_encoder 不可用（model=%s），回退 baseline：%s", model_name, exc)
            return _fallback_results(doc_ids, docs, metadata_list, top_k, method="cross_encoder_fallback")

    if normalized_method == "llm":
        # 学习阶段默认不引入远程调用，先回退到 overlap。
        logging.warning("llm 重排序当前未启用，回退 overlap。")
        return rerank_with_overlap(query, docs, doc_ids, metadata_list, top_k=top_k)

    raise ValueError(f"unsupported rerank method: {method}")


def rerank_with_overlap(query, docs, doc_ids, metadata_list, top_k=5):
    """基于关键词重叠的轻量重排序（本地可运行，便于教学验证）。"""
    query_tokens = _tokenize(query)
    query_counter = Counter(query_tokens)

    scored = []
    for doc_id, doc, metadata in zip(doc_ids, docs, metadata_list):
        score = _overlap_score(query_counter, _tokenize(doc))
        scored.append((doc_id, doc, metadata, score))

    scored.sort(key=lambda item: item[3], reverse=True)
    return _to_results(scored[:top_k], method="overlap")


def rerank_with_cross_encoder(query, docs, doc_ids, metadata_list, top_k=5, model_name=DEFAULT_CROSS_ENCODER_MODEL):
    """
    使用 cross-encoder 对候选进行精排。
    若模型加载失败，应由上层统一入口做回退。
    """
    encoder = get_cross_encoder(model_name)
    pairs = [[query, doc] for doc in docs]
    scores = encoder.predict(pairs)

    scored = []
    for doc_id, doc, metadata, score in zip(doc_ids, docs, metadata_list, scores):
        scored.append((doc_id, doc, metadata, float(score)))

    scored.sort(key=lambda item: item[3], reverse=True)
    return _to_results(scored[:top_k], method="cross_encoder")


@lru_cache(maxsize=2)
def get_cross_encoder(model_name=DEFAULT_CROSS_ENCODER_MODEL):
    """懒加载 cross-encoder 模型。"""
    from sentence_transformers import CrossEncoder

    logging.info("加载 cross_encoder 模型: %s", model_name)
    return CrossEncoder(model_name)


def _validate_and_copy(docs, doc_ids, metadata_list, top_k):
    """统一输入校验，避免重排序阶段静默错误。"""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    docs = list(docs)
    doc_ids = list(doc_ids)
    metadata_list = [dict(metadata) for metadata in metadata_list]

    if len(docs) != len(doc_ids):
        raise ValueError("docs and doc_ids must have the same length")
    if len(docs) != len(metadata_list):
        raise ValueError("docs and metadata_list must have the same length")
    if len(set(doc_ids)) != len(doc_ids):
        raise ValueError("doc_ids must be unique")

    return docs, doc_ids, metadata_list


def _fallback_results(doc_ids, docs, metadata_list, top_k, method):
    """回退策略：保持原始顺序并赋予递减分数。"""
    total = max(len(docs), 1)
    scored = []
    for idx, (doc_id, doc, metadata) in enumerate(zip(doc_ids, docs, metadata_list)):
        score = 1.0 - (idx / total)
        scored.append((doc_id, doc, metadata, float(score)))
    return _to_results(scored[:top_k], method=method)


def _to_results(scored_items, method):
    """把内部元组结构转换为标准化 dataclass 输出。"""
    results = []
    for rank, (doc_id, doc, metadata, score) in enumerate(scored_items, start=1):
        results.append(
            RerankResult(
                rank=rank,
                doc_id=doc_id,
                content=doc,
                metadata=dict(metadata),
                score=float(score),
                method=method,
            )
        )
    return results


def _tokenize(text):
    """
    简化分词：
    - 英文/数字/下划线按词切分
    - 中文按单字切分（教学场景保底）
    """
    if text is None:
        return []
    raw = str(text).strip().lower()
    if not raw:
        return []
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", raw)


def _overlap_score(query_counter, doc_tokens):
    """基于覆盖率与密度计算重叠分数，范围约在 [0, 1]。"""
    if not query_counter or not doc_tokens:
        return 0.0

    doc_counter = Counter(doc_tokens)
    matched = 0
    for token, q_count in query_counter.items():
        matched += min(q_count, doc_counter.get(token, 0))

    query_len = sum(query_counter.values())
    doc_len = len(doc_tokens)
    if query_len == 0 or doc_len == 0:
        return 0.0

    coverage = matched / query_len
    density = matched / doc_len
    return float(0.8 * coverage + 0.2 * density)
