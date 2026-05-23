"""
BM25 vs FAISS 对比实验（同语料、同 query）。

目标：
1. 在同一批 documents 上分别执行 BM25 与 FAISS 检索。
2. 并排输出两种检索器的 TopK 结果。
3. 统计 Top1 命中率和两者 Top1 一致率。
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.append(str(EXPERIMENTS_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# 在导入 numpy 相关模块前限制线程，降低低内存环境的初始化失败概率。
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from bm25_index import BM25IndexManager
from embeddings import EMBED_MODEL_NAME, encode_query, encode_texts
from vector_store import FaissVectorStore


TOP_K = 3

DOCUMENTS = [
    "显卡型号 RTX4090 配备 24GB 显存，适合高分辨率推理任务。",
    "常见报错 ERR_CONN_RESET 通常与网络中断或代理配置有关。",
    "配置项 HYBRID_ALPHA 用于控制混合检索中语义分数的权重。",
    "RAG 系统会先检索资料，再结合上下文生成回答。",
    "BM25 属于关键词检索，FAISS 属于向量语义检索。",
]

QUERIES = [
    "RTX4090 显存是多少",
    "ERR_CONN_RESET 怎么排查",
    "HYBRID_ALPHA 是什么",
    "RAG 为什么先检索再回答",
    "纯关键词检索用什么",
]

EXPECTED_TOP1 = {
    "RTX4090 显存是多少": "doc_0001",
    "ERR_CONN_RESET 怎么排查": "doc_0002",
    "HYBRID_ALPHA 是什么": "doc_0003",
    "RAG 为什么先检索再回答": "doc_0004",
    "纯关键词检索用什么": "doc_0005",
}


@dataclass(frozen=True)
class CompareSummary:
    bm25_hit_at_1: float
    faiss_hit_at_1: float
    top1_agreement_rate: float
    faiss_enabled: bool


def top1_id(results, id_attr):
    """提取检索结果 Top1 的 id；无结果则返回 None。"""
    if not results:
        return None
    return getattr(results[0], id_attr, None)


def hit_at_1(predictions, expected):
    """计算 Hit@1（预测 Top1 与预期 id 一致的比例）。"""
    if not expected:
        return 0.0
    matched = 0
    for query, expected_id in expected.items():
        if predictions.get(query) == expected_id:
            matched += 1
    return matched / len(expected)


def top1_agreement_rate(left_predictions, right_predictions, queries):
    """计算两种检索器 Top1 一致率。"""
    if not queries:
        return 0.0
    agreed = 0
    for query in queries:
        if left_predictions.get(query) == right_predictions.get(query):
            agreed += 1
    return agreed / len(queries)


def run_compare(documents, queries, top_k, bm25_only=False):
    """执行 BM25 与 FAISS 并排检索。"""
    doc_ids = [f"doc_{index:04d}" for index in range(1, len(documents) + 1)]

    bm25 = BM25IndexManager()
    bm25.build_index(documents, doc_ids)

    vector_store = None
    faiss_enabled = not bm25_only
    if faiss_enabled:
        embeddings = encode_texts(documents)
        vector_store = FaissVectorStore()
        vector_store.build_index(
            documents,
            doc_ids,
            [{"source": "retrieval_compare", "chunk_index": i} for i in range(len(documents))],
            embeddings,
        )

    bm25_results = {}
    faiss_results = {}
    bm25_top1 = {}
    faiss_top1 = {}

    for query in queries:
        b_results = bm25.search(query, top_k=top_k)
        if faiss_enabled:
            f_results = vector_store.search(encode_query(query), top_k=top_k)
        else:
            f_results = []
        bm25_results[query] = b_results
        faiss_results[query] = f_results
        bm25_top1[query] = top1_id(b_results, "doc_id")
        faiss_top1[query] = top1_id(f_results, "chunk_id")

    faiss_hit = hit_at_1(faiss_top1, EXPECTED_TOP1) if faiss_enabled else 0.0
    agreement = top1_agreement_rate(bm25_top1, faiss_top1, queries) if faiss_enabled else 0.0
    summary = CompareSummary(
        bm25_hit_at_1=hit_at_1(bm25_top1, EXPECTED_TOP1),
        faiss_hit_at_1=faiss_hit,
        top1_agreement_rate=agreement,
        faiss_enabled=faiss_enabled,
    )

    return bm25, bm25_results, faiss_results, summary


def print_results(bm25, bm25_results, faiss_results, summary, top_k):
    """格式化输出对比结果。"""
    print("documents 数量:", len(DOCUMENTS))
    print("queries 数量:", len(QUERIES))
    print("top_k:", top_k)
    print("BM25 后端:", "rank_bm25" if bm25._backend == "rank_bm25" else "simple-fallback")
    if summary.faiss_enabled:
        print("embedding 模型:", EMBED_MODEL_NAME)
    else:
        print("embedding 模型: 已跳过（bm25-only）")
    print()

    for query in QUERIES:
        print("=" * 96)
        print("query:", query)
        print("BM25 TopK:")
        b_results = bm25_results[query]
        if not b_results:
            print("  无命中结果")
        else:
            for result in b_results:
                print(f"  {result.rank}. {result.doc_id} | score={result.score:.4f} | {result.content}")

        print("FAISS TopK:")
        f_results = faiss_results[query]
        if not summary.faiss_enabled:
            print("  已跳过（bm25-only）")
        elif not f_results:
            print("  无命中结果")
        else:
            for result in f_results:
                print(f"  {result.rank}. {result.chunk_id} | distance={result.distance:.4f} | {result.content}")

        b_top1 = top1_id(b_results, "doc_id")
        f_top1 = top1_id(f_results, "chunk_id")
        if summary.faiss_enabled:
            print("Top1 是否一致:", "是" if b_top1 == f_top1 else "否")
        else:
            print("Top1 是否一致: 已跳过（bm25-only）")
        print("预期 Top1:", EXPECTED_TOP1.get(query))
        print("BM25 Top1:", b_top1)
        print("FAISS Top1:", f_top1)

    print()
    print("-" * 96)
    print("BM25 Hit@1:", f"{summary.bm25_hit_at_1:.2%}")
    if summary.faiss_enabled:
        print("FAISS Hit@1:", f"{summary.faiss_hit_at_1:.2%}")
        print("Top1 一致率:", f"{summary.top1_agreement_rate:.2%}")
    else:
        print("FAISS Hit@1: 已跳过（bm25-only）")
        print("Top1 一致率: 已跳过（bm25-only）")


def parse_args():
    parser = argparse.ArgumentParser(description="Run BM25 vs FAISS retrieval comparison.")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Top K results to show for each retriever.")
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help="Skip FAISS branch and run BM25 only (for low-memory / torch load issues).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("top-k 必须大于 0")

    try:
        bm25, bm25_results, faiss_results, summary = run_compare(
            DOCUMENTS,
            QUERIES,
            args.top_k,
            bm25_only=args.bm25_only,
        )
    except OSError as exc:
        if "WinError 1455" in str(exc) or "WinError 1114" in str(exc):
            print("FAISS 分支依赖的模型加载失败：Windows 页面文件/内存不足。")
            print("建议：先用 --bm25-only 跑通，或调大页面文件后重试。")
            print("示例：python experiments\\retrieval_compare.py --bm25-only")
            return
        raise
    print_results(bm25, bm25_results, faiss_results, summary, args.top_k)


if __name__ == "__main__":
    main()


# 运行结果示例（2026-05-23，Windows PowerShell）
# 完整模式：
# documents 数量: 5
# queries 数量: 5
# top_k: 3
# BM25 后端: simple-fallback
# embedding 模型: D:\Projects\Codex\Projects\RAG_Project\models\bge-small-zh-v1.5
# BM25 Hit@1: 100.00%
# FAISS Hit@1: 100.00%
# Top1 一致率: 100.00%
#
# bm25-only 模式：
# python experiments\retrieval_compare.py --bm25-only --top-k 3
# BM25 Hit@1: 100.00%
# FAISS Hit@1: 已跳过（bm25-only）
# Top1 一致率: 已跳过（bm25-only）
