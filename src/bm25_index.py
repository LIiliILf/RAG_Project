"""
BM25 稀疏检索模块。

职责：
1. 构建关键词索引（documents -> tokenized_corpus）。
2. 执行 top_k 检索（query -> score 排序）。
3. 维护 doc_id 与原文映射。

说明：
- 若安装了 rank_bm25 + jieba，优先使用它们。
- 若缺少依赖，自动回退到本地简化实现。
"""

from collections import Counter
from dataclasses import dataclass
from math import log
import re

import numpy as np

try:
    import jieba
except ImportError:
    jieba = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None


@dataclass(frozen=True)
class BM25SearchResult:
    """标准化 BM25 检索结果。"""
    rank: int
    doc_id: str
    content: str
    score: float


class BM25IndexManager:
    """
    BM25 检索索引管理器。

    外部只需要关注 3 个方法：
    - build_index(documents, doc_ids)
    - search(query, top_k=5)
    - clear()
    """

    def __init__(self):
        self.bm25_index = None
        self.doc_mapping = {}
        self.tokenized_corpus = []
        self.raw_corpus = []
        self._backend = None

    @property
    def total_docs(self):
        """当前索引中的文档数量。"""
        return len(self.raw_corpus)

    def is_ready(self):
        """索引是否可检索。"""
        return self.bm25_index is not None and self.total_docs > 0

    def build_index(self, documents, doc_ids):
        """
        构建 BM25 索引。

        Args:
            documents: 文本列表。
            doc_ids: 与 documents 一一对应的唯一 id 列表。
        """
        documents = list(documents)
        doc_ids = list(doc_ids)

        if not documents:
            raise ValueError("documents 不能为空")
        if len(documents) != len(doc_ids):
            raise ValueError("documents 和 doc_ids 数量必须一致")
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("doc_ids 不能重复")

        tokenized_corpus = [self._tokenize(document) for document in documents]
        if any(len(tokens) == 0 for tokens in tokenized_corpus):
            raise ValueError("documents 中存在无法分词的空文本")

        self.raw_corpus = documents
        self.doc_mapping = {index: doc_id for index, doc_id in enumerate(doc_ids)}
        self.tokenized_corpus = tokenized_corpus

        if BM25Okapi is not None:
            self.bm25_index = BM25Okapi(self.tokenized_corpus)
            self._backend = "rank_bm25"
        else:
            self.bm25_index = _SimpleBM25(self.tokenized_corpus)
            self._backend = "simple"

    def search(self, query, top_k=5):
        """
        使用 BM25 检索相关文档。

        返回按 score 从高到低排序的结果；score 越大表示越相关。
        """
        if not self.is_ready():
            raise RuntimeError("BM25 索引尚未构建，请先调用 build_index()")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = np.asarray(self.bm25_index.get_scores(tokenized_query), dtype="float32")
        search_k = min(top_k, self.total_docs)
        top_indices = np.argsort(scores)[::-1][:search_k]

        results = []
        rank = 1
        for index in top_indices:
            score = float(scores[index])
            if score <= 0:
                continue

            doc_id = self.doc_mapping[int(index)]
            results.append(
                BM25SearchResult(
                    rank=rank,
                    doc_id=doc_id,
                    content=self.raw_corpus[int(index)],
                    score=score,
                )
            )
            rank += 1

        return results

    def clear(self):
        """清空索引与缓存。"""
        self.bm25_index = None
        self.doc_mapping = {}
        self.tokenized_corpus = []
        self.raw_corpus = []
        self._backend = None

    @staticmethod
    def _tokenize(text):
        """
        文本分词。

        优先使用 jieba；若未安装则回退到正则分词：
        - 英文/数字/下划线按词切分
        - 中文按单字切分（仅用于学习场景的保底）
        """
        if text is None:
            return []

        text = str(text).strip()
        if not text:
            return []

        if jieba is not None:
            return [token for token in jieba.cut(text) if token.strip()]

        return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text)


class _SimpleBM25:
    """
    无第三方依赖时的简化 BM25 实现。

    仅用于学习和测试环境兜底，避免因为依赖缺失阻塞课程进度。
    """

    def __init__(self, tokenized_corpus, k1=1.5, b=0.75):
        self.corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.N = len(tokenized_corpus)
        self.doc_lengths = np.array([len(doc) for doc in tokenized_corpus], dtype="float32")
        self.avgdl = float(np.mean(self.doc_lengths)) if self.N else 0.0
        self.term_freqs = [Counter(doc) for doc in tokenized_corpus]

        doc_freq = Counter()
        for doc in tokenized_corpus:
            for term in set(doc):
                doc_freq[term] += 1
        self.doc_freq = doc_freq

    def _idf(self, term):
        df = self.doc_freq.get(term, 0)
        return log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def get_scores(self, query_tokens):
        scores = np.zeros(self.N, dtype="float32")
        if self.N == 0 or self.avgdl == 0:
            return scores

        for term in query_tokens:
            idf = self._idf(term)
            for doc_index, tf_dict in enumerate(self.term_freqs):
                tf = tf_dict.get(term, 0)
                if tf == 0:
                    continue
                doc_len = self.doc_lengths[doc_index]
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                scores[doc_index] += idf * (tf * (self.k1 + 1)) / denom

        return scores
