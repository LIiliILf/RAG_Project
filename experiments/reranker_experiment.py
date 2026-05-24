"""
Reranker 重排序实验脚本。

目标：
1. 基于混合检索产出候选集。
2. 对比是否启用重排序时的 TopK 与 Hit@1。
3. 记录重排序带来的额外耗时。
"""

from pathlib import Path
import argparse
import os
import sys
import time

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
from reranker import rerank_results
from retrieval_compare import DOCUMENTS, EXPECTED_TOP1, QUERIES, hit_at_1


HYBRID_ALPHA = 0.7
RETRIEVAL_TOP_K = 5
RERANK_TOP_K = 3
DEFAULT_METHODS = "none,cross_encoder"

# cross_encoder 模型预设：
# - 如果你已经把模型下载到本地，优先用 local 路径（更稳定）。
# - 如果要切换模型，优先改下面的 key 或在命令行用 --cross-encoder-model 传路径。
CROSS_ENCODER_MODEL_PRESETS = {
    "bge_base_local": str(PROJECT_ROOT / "models" / "bge-reranker-base"),
    "bge_v2_m3_local": str(PROJECT_ROOT / "models" / "bge-reranker-v2-m3"),
    "minilm_online": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}
DEFAULT_CROSS_ENCODER_MODEL_KEY = "bge_base_local"

TOY_CASES = [
    {
        "query": "RTX4090 24GB 显存是多少",
        "expected_top1": "doc_b",
        "candidates": [
            ("doc_a", "RTX4090 配备 16GB 显存，适合轻量任务。"),
            ("doc_b", "RTX4090 配备 24GB 显存，适合高分辨率推理任务。"),
            ("doc_c", "RAG 系统会先检索资料，再结合上下文生成回答。"),
        ],
    },
    {
        "query": "证书过期报错怎么排查",
        "expected_top1": "doc_e",
        "candidates": [
            ("doc_d", "ERR_CONN_RESET 通常与网络中断或代理配置有关。"),
            ("doc_e", "ERR_CERT_DATE_INVALID 通常与证书过期或系统时间错误有关。"),
            ("doc_f", "BM25 属于关键词检索，FAISS 属于向量语义检索。"),
        ],
    },
    {
        "query": "RAG 为什么先检索再回答",
        "expected_top1": "doc_h",
        "candidates": [
            ("doc_g", "RAG 用于把外部知识引入大模型。"),
            ("doc_h", "RAG 系统会先检索资料，再结合上下文生成回答。"),
            ("doc_i", "HYBRID_ALPHA 用于控制混合检索中语义分数的权重。"),
        ],
    },
]


def parse_methods(raw):
    """解析方法列表，兼容逗号写法和空格写法。"""
    raw_items = [raw] if isinstance(raw, str) else list(raw)
    methods = []
    for raw_item in raw_items:
        for item in str(raw_item).split(","):
            method = item.strip().lower()
            if not method:
                continue
            methods.append(method)
    if not methods:
        raise ValueError("at least one rerank method is required")

    seen = set()
    unique = []
    for method in methods:
        if method in seen:
            continue
        seen.add(method)
        unique.append(method)
    return unique


def resolve_cross_encoder_model(model_key, explicit_model):
    """
    解析最终使用的 cross_encoder 模型。

    优先级：
    1) --cross-encoder-model（显式路径/模型名）
    2) --cross-encoder-model-key（预设 key）
    3) 空字符串（交由 reranker.py 默认值处理）
    """
    explicit_model = str(explicit_model or "").strip()
    if explicit_model:
        return explicit_model

    key = str(model_key or "").strip()
    if not key:
        return ""

    if key not in CROSS_ENCODER_MODEL_PRESETS:
        raise ValueError(f"unknown cross-encoder-model-key: {key}")
    return CROSS_ENCODER_MODEL_PRESETS[key]


def build_hybrid_retriever(documents):
    """构建混合检索器，作为本节候选召回入口。"""
    doc_ids = [f"doc_{index:04d}" for index in range(1, len(documents) + 1)]
    metadatas = [
        {"source": "reranker_experiment", "chunk_index": index}
        for index in range(len(documents))
    ]
    embeddings = encode_texts(documents)
    retriever = HybridRetriever(alpha=HYBRID_ALPHA)
    retriever.build_index(documents, doc_ids, metadatas, embeddings)
    return retriever


def run_method(method, retriever, queries, retrieval_top_k, rerank_top_k, cross_encoder_model=""):
    """运行单个重排序方法并返回统计结果。"""
    predictions = {}
    details = {}
    retrieval_ms = 0.0
    rerank_ms = 0.0

    for query in queries:
        start = time.perf_counter()
        candidates = retriever.search(query, encode_query(query), top_k=retrieval_top_k)
        retrieval_ms += (time.perf_counter() - start) * 1000

        docs = [item.content for item in candidates]
        doc_ids = [item.doc_id for item in candidates]
        metadata_list = [item.metadata for item in candidates]

        start = time.perf_counter()
        reranked = rerank_results(
            query=query,
            docs=docs,
            doc_ids=doc_ids,
            metadata_list=metadata_list,
            method=method,
            top_k=rerank_top_k,
            cross_encoder_model=(cross_encoder_model or None),
        )
        rerank_ms += (time.perf_counter() - start) * 1000

        predictions[query] = reranked[0].doc_id if reranked else None
        details[query] = {
            "before": candidates[:rerank_top_k],
            "after": reranked,
        }

    summary = {
        "method": method,
        "hit_at_1": hit_at_1(predictions, EXPECTED_TOP1),
        "before_hit_at_1": hit_at_1(
            {query: details[query]["before"][0].doc_id if details[query]["before"] else None for query in queries},
            EXPECTED_TOP1,
        ),
        "changed_top1_count": sum(
            1
            for query in queries
            if details[query]["before"]
            and details[query]["after"]
            and details[query]["before"][0].doc_id != details[query]["after"][0].doc_id
        ),
        "predictions": predictions,
        "retrieval_avg_ms": retrieval_ms / len(queries) if queries else 0.0,
        "rerank_avg_ms": rerank_ms / len(queries) if queries else 0.0,
        "total_avg_ms": (retrieval_ms + rerank_ms) / len(queries) if queries else 0.0,
    }
    return details, summary


def run_toy_method(method, rerank_top_k, cross_encoder_model=""):
    """运行教学用 toy case：用于直观看到重排效果。"""
    details = {}
    before_predictions = {}
    after_predictions = {}
    rerank_ms = 0.0

    for case in TOY_CASES:
        query = case["query"]
        expected = case["expected_top1"]
        candidates = case["candidates"]

        doc_ids = [item[0] for item in candidates]
        docs = [item[1] for item in candidates]
        metadata_list = [{"source": "toy_case"} for _ in candidates]

        before = doc_ids[:rerank_top_k]
        before_predictions[query] = before[0] if before else None

        start = time.perf_counter()
        reranked = rerank_results(
            query=query,
            docs=docs,
            doc_ids=doc_ids,
            metadata_list=metadata_list,
            method=method,
            top_k=rerank_top_k,
            cross_encoder_model=(cross_encoder_model or None),
        )
        rerank_ms += (time.perf_counter() - start) * 1000

        after = [item.doc_id for item in reranked]
        after_predictions[query] = after[0] if after else None

        details[query] = {
            "before_ids": before,
            "after_ids": after,
            "expected_top1": expected,
        }

    expected_map = {case["query"]: case["expected_top1"] for case in TOY_CASES}
    summary = {
        "method": method,
        "before_hit_at_1": hit_at_1(before_predictions, expected_map),
        "hit_at_1": hit_at_1(after_predictions, expected_map),
        "changed_top1_count": sum(
            1 for query in expected_map if before_predictions.get(query) != after_predictions.get(query)
        ),
        "retrieval_avg_ms": 0.0,
        "rerank_avg_ms": rerank_ms / len(TOY_CASES),
        "total_avg_ms": rerank_ms / len(TOY_CASES),
    }
    return details, summary


def print_method_result(details, summary, rerank_top_k, mode):
    """格式化输出单个方法结果。"""
    print()
    print("#" * 96)
    print("method:", summary["method"])
    print("mode:", mode)
    print("重排前 Hit@1:", f"{summary['before_hit_at_1']:.2%}")
    print("重排后 Hit@1:", f"{summary['hit_at_1']:.2%}")
    print("Top1 变化数:", summary["changed_top1_count"])
    print("平均召回耗时(ms):", f"{summary['retrieval_avg_ms']:.2f}")
    print("平均重排耗时(ms):", f"{summary['rerank_avg_ms']:.2f}")
    print("平均总耗时(ms):", f"{summary['total_avg_ms']:.2f}")

    if mode == "pipeline":
        query_list = QUERIES
        expected_getter = lambda q: EXPECTED_TOP1.get(q)
    else:
        query_list = [case["query"] for case in TOY_CASES]
        toy_expected = {case["query"]: case["expected_top1"] for case in TOY_CASES}
        expected_getter = lambda q: toy_expected.get(q)

    for query in query_list:
        print("-" * 96)
        print("query:", query)
        expected = expected_getter(query)

        if mode == "pipeline":
            before_ids = [item.doc_id for item in details[query]["before"]]
            after_ids = [item.doc_id for item in details[query]["after"]]
        else:
            before_ids = details[query]["before_ids"]
            after_ids = details[query]["after_ids"]

        top1_after = after_ids[0] if after_ids else None

        print(f"召回前 Top{rerank_top_k}:", " > ".join(before_ids) if before_ids else "无")
        print(f"重排后 Top{rerank_top_k}:", " > ".join(after_ids) if after_ids else "无")
        print("预期 Top1:", expected)
        print("重排后 Top1:", top1_after)
        print("是否命中:", "是" if top1_after == expected else "否")


def parse_args():
    parser = argparse.ArgumentParser(description="运行 reranker 重排序实验。")
    parser.add_argument(
        "--mode",
        type=str,
        default="pipeline",
        choices=["pipeline", "toy"],
        help="pipeline=端到端召回+重排，toy=教学用固定候选重排",
    )
    parser.add_argument("--retrieval-top-k", type=int, default=RETRIEVAL_TOP_K, help="召回候选数量。")
    parser.add_argument("--rerank-top-k", type=int, default=RERANK_TOP_K, help="重排序输出数量。")
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=[DEFAULT_METHODS],
        help='重排方法，支持 "--methods none,overlap" 或 "--methods none overlap"',
    )
    parser.add_argument(
        "--cross-encoder-model",
        type=str,
        default="",
        help="cross_encoder 模型名或本地路径（仅在 methods 包含 cross_encoder 时生效）",
    )
    parser.add_argument(
        "--cross-encoder-model-key",
        type=str,
        default=DEFAULT_CROSS_ENCODER_MODEL_KEY,
        choices=sorted(CROSS_ENCODER_MODEL_PRESETS.keys()),
        help="cross_encoder 模型预设 key（仅在 methods 包含 cross_encoder 时生效）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.retrieval_top_k <= 0:
        raise ValueError("retrieval-top-k must be greater than 0")
    if args.rerank_top_k <= 0:
        raise ValueError("rerank-top-k must be greater than 0")
    if args.rerank_top_k > args.retrieval_top_k:
        raise ValueError("rerank-top-k must be <= retrieval-top-k")

    methods = parse_methods(args.methods)
    selected_cross_encoder_model = resolve_cross_encoder_model(
        model_key=args.cross_encoder_model_key,
        explicit_model=args.cross_encoder_model,
    )

    print("documents count:", len(DOCUMENTS))
    print("queries count:", len(QUERIES))
    print("hybrid alpha:", HYBRID_ALPHA)
    print("mode:", args.mode)
    print("retrieval_top_k:", args.retrieval_top_k)
    print("rerank_top_k:", args.rerank_top_k)
    print("methods:", ", ".join(methods))
    print("embedding model:", EMBED_MODEL_NAME)
    if "cross_encoder" in methods:
        print("cross_encoder_model_key:", args.cross_encoder_model_key)
        print("cross_encoder_model:", selected_cross_encoder_model or "(default from reranker.py)")

    retriever = None
    if args.mode == "pipeline":
        try:
            retriever = build_hybrid_retriever(DOCUMENTS)
        except OSError as exc:
            if "WinError 1455" in str(exc) or "WinError 1114" in str(exc):
                print("embedding 模型加载失败：Windows 页面文件/内存不足。")
                print("建议先扩容页面文件，或先改用 toy 模式观察重排效果。")
                print("示例：python experiments\\reranker_experiment.py --mode toy --methods none,overlap")
                return
            raise

    for method in methods:
        if args.mode == "pipeline":
            details, summary = run_method(
                method=method,
                retriever=retriever,
                queries=QUERIES,
                retrieval_top_k=args.retrieval_top_k,
                rerank_top_k=args.rerank_top_k,
                cross_encoder_model=selected_cross_encoder_model,
            )
        else:
            details, summary = run_toy_method(
                method=method,
                rerank_top_k=args.rerank_top_k,
                cross_encoder_model=selected_cross_encoder_model,
            )
        print_method_result(details, summary, args.rerank_top_k, args.mode)


if __name__ == "__main__":
    main()


# 运行结果示例（2026-05-24，Windows PowerShell）：
# python experiments\reranker_experiment.py --retrieval-top-k 5 --rerank-top-k 3 --methods none,cross_encoder --cross-encoder-model-key bge_base_local
# method: none
# method: cross_encoder
#
# cross_encoder 切换示例：
# 1) 用预设 key（推荐，本地模型）：
# python experiments\reranker_experiment.py --mode toy --methods cross_encoder --cross-encoder-model-key bge_base_local
#
# 2) 直接指定路径/模型名：
# python experiments\reranker_experiment.py --mode toy --methods cross_encoder --cross-encoder-model D:\Projects\Codex\Projects\RAG_Project\models\bge-reranker-base
