"""
混合检索实验脚本。

目标：
1. 在同一语料上构建 BM25 + FAISS 双索引。
2. 在相同 query 集上比较不同 alpha 配置。
3. 观察 TopK 排序和 Hit@1 变化。
"""

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

# 在低内存 Windows 环境下，降低 BLAS 初始化压力。
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from embeddings import EMBED_MODEL_NAME, encode_query, encode_texts
from hybrid_retriever import HybridRetriever
from retrieval_compare import DOCUMENTS, EXPECTED_TOP1, QUERIES, hit_at_1


TOP_K = 3
DEFAULT_ALPHAS = "0.3,0.7,1.0"


def parse_alphas(raw):
    """
    解析 alpha 列表。

    支持输入形式：
    1) "0.3,0.7,1.0"
    2) ["0.3", "0.7", "1.0"]
    3) PowerShell 中未加引号的逗号参数（会被拆成多个 token）
    """
    raw_items = [raw] if isinstance(raw, str) else list(raw)
    values = []
    for raw_item in raw_items:
        for item in str(raw_item).split(","):
            item = item.strip()
            if not item:
                continue
            value = float(item)
            if value < 0 or value > 1:
                raise ValueError("alpha must be in [0, 1]")
            values.append(value)
    if not values:
        raise ValueError("at least one alpha is required")
    # 保持原有顺序，同时去重。
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def run_hybrid(alpha, documents, queries, doc_ids, metadatas, document_embeddings, query_embeddings, top_k):
    """运行单个 alpha 配置，并返回检索结果与 Hit@1 指标。"""
    retriever = HybridRetriever(alpha=alpha)
    retriever.build_index(documents, doc_ids, metadatas, document_embeddings)

    query_results = {}
    predictions = {}
    for query in queries:
        results = retriever.search(query, query_embeddings[query], top_k=top_k)
        query_results[query] = results
        predictions[query] = results[0].doc_id if results else None

    summary = {
        "alpha": alpha,
        "hit_at_1": hit_at_1(predictions, EXPECTED_TOP1),
        "predictions": predictions,
    }
    return query_results, summary


def print_alpha_results(alpha, query_results, summary, top_k):
    """格式化打印单个 alpha 实验结果。"""
    print()
    print("#" * 96)
    print(f"alpha = {alpha:.2f}")
    print("Hit@1:", f"{summary['hit_at_1']:.2%}")

    for query in QUERIES:
        print("-" * 96)
        print("query:", query)
        print(f"Top {top_k}:")
        results = query_results[query]
        if not results:
            print("  no results")
            continue
        for item in results:
            print(
                f"  {item.rank}. {item.doc_id} | hybrid={item.hybrid_score:.4f} "
                f"| semantic={item.semantic_score:.4f} | bm25={item.bm25_score:.4f}"
            )
            print("     content:", item.content)


def parse_args():
    parser = argparse.ArgumentParser(description="运行混合检索实验。")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Top K results for each query.")
    parser.add_argument(
        "--alphas",
        type=str,
        nargs="+",
        default=[DEFAULT_ALPHAS],
        help='alpha 列表，支持 "--alphas 0.3,0.7,1.0" 或 "--alphas 0.3 0.7 1.0"',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("top-k must be greater than 0")

    alphas = parse_alphas(args.alphas)
    doc_ids = [f"doc_{index:04d}" for index in range(1, len(DOCUMENTS) + 1)]
    metadatas = [
        {"source": "hybrid_experiment", "chunk_index": index}
        for index in range(len(DOCUMENTS))
    ]

    print("documents count:", len(DOCUMENTS))
    print("queries count:", len(QUERIES))
    print("top_k:", args.top_k)
    print("alphas:", ", ".join(f"{value:.2f}" for value in alphas))
    print("embedding model:", EMBED_MODEL_NAME)

    try:
        document_embeddings = encode_texts(DOCUMENTS)
        query_embeddings = {query: encode_query(query) for query in QUERIES}
    except OSError as exc:
        if "WinError 1455" in str(exc) or "WinError 1114" in str(exc):
            print("embedding model load failed due to low memory/pagefile on Windows.")
            print("suggestion: increase pagefile or use BM25-only scripts first.")
            return
        raise

    for alpha in alphas:
        query_results, summary = run_hybrid(
            alpha,
            DOCUMENTS,
            QUERIES,
            doc_ids,
            metadatas,
            document_embeddings,
            query_embeddings,
            args.top_k,
        )
        print_alpha_results(alpha, query_results, summary, args.top_k)


if __name__ == "__main__":
    main()


# 运行结果示例（2026-05-23，Windows PowerShell）：
# python experiments\hybrid_experiment.py --top-k 3 --alphas 0.3,0.7,1.0
# documents count: 5
# queries count: 5
# top_k: 3
# alphas: 0.30, 0.70, 1.00
# embedding model: D:\Projects\Codex\Projects\RAG_Project\models\bge-small-zh-v1.5
# alpha = 0.30 -> Hit@1: 100.00%
# alpha = 0.70 -> Hit@1: 100.00%
# alpha = 1.00 -> Hit@1: 100.00%
