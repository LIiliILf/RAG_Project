"""
第12节：多文档问答实验（检索 -> 重排 -> Prompt -> 大模型回答）。

说明：
1. 默认使用 fixed 候选，方便先验证“调用链路”。
2. 如需端到端检索，可切换 mode=pipeline。
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
from generator import DEFAULT_CHAT_MODEL, generate_answer_with_rag, resolve_base_url
from hybrid_retriever import HybridRetriever
from prompt_builder import build_prompt
from reranker import rerank_results
from retrieval_compare import DOCUMENTS


DEFAULT_MODE = "fixed"
DEFAULT_QUERY = "RTX4090 显存是多少？"
DEFAULT_RETRIEVAL_TOP_K = 5
DEFAULT_RERANK_TOP_K = 3
DEFAULT_RERANK_METHOD = "overlap"
DEFAULT_ANSWER_RULES = (
    "若证据冲突，先回答“根据当前资料无法确定”，并明确指出冲突字段及对应 doc_id；"
    "若证据不足，说明缺失了哪类信息。"
)

FIXED_CANDIDATES = [
    {
        "doc_id": "doc_0001",
        "content": "RTX4090 配备 16GB 显存，适合轻量任务。",
        "metadata": {"source": "gpu_a.md"},
    },
    {
        "doc_id": "doc_0002",
        "content": "RTX4090 配备 24GB 显存，适合高分辨率推理任务。",
        "metadata": {"source": "gpu_b.md"},
    },
    {
        "doc_id": "doc_0003",
        "content": "RAG 系统通常先检索，再把资料交给模型生成答案。",
        "metadata": {"source": "rag_intro.md"},
    },
    {
        "doc_id": "doc_0004",
        "content": "ERR_CERT_DATE_INVALID 通常与证书过期或系统时间错误有关。",
        "metadata": {"source": "network.md"},
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="运行第12节多文档问答实验。")
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE, choices=["fixed", "pipeline"], help="候选来源模式。")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="用户问题。")
    parser.add_argument("--retrieval-top-k", type=int, default=DEFAULT_RETRIEVAL_TOP_K, help="召回候选数量（pipeline 模式生效）。")
    parser.add_argument("--rerank-top-k", type=int, default=DEFAULT_RERANK_TOP_K, help="重排输出数量。")
    parser.add_argument(
        "--rerank-method",
        type=str,
        default=DEFAULT_RERANK_METHOD,
        choices=["none", "overlap", "cross_encoder"],
        help="重排方法。",
    )
    parser.add_argument("--cross-encoder-model", type=str, default="", help="cross_encoder 模型路径或模型名。")
    parser.add_argument("--llm-model", type=str, default=DEFAULT_CHAT_MODEL, help="大模型名称。")
    parser.add_argument("--base-url", type=str, default="", help="OpenAI 兼容接口 base_url。")
    parser.add_argument("--print-prompt-only", action="store_true", help="仅打印 Prompt，不调用大模型。")
    return parser.parse_args()


def build_fixed_candidates():
    """返回固定候选，便于验证后半段链路。"""
    return list(FIXED_CANDIDATES)


def build_pipeline_candidates(query, retrieval_top_k):
    """从现有语料做一次混合检索，产出候选。"""
    doc_ids = [f"doc_{index:04d}" for index in range(1, len(DOCUMENTS) + 1)]
    metadatas = [
        {"source": "retrieval_compare", "chunk_index": index}
        for index in range(len(DOCUMENTS))
    ]
    embeddings = encode_texts(DOCUMENTS)
    retriever = HybridRetriever(alpha=0.7)
    retriever.build_index(DOCUMENTS, doc_ids, metadatas, embeddings)

    candidates = retriever.search(query, encode_query(query), top_k=retrieval_top_k)
    return [
        {
            "doc_id": item.doc_id,
            "content": item.content,
            "metadata": item.metadata,
            "score": item.hybrid_score,
        }
        for item in candidates
    ]


def do_rerank(query, candidates, method, top_k, cross_encoder_model):
    """对候选做重排，得到可直接进入 Prompt 的上下文列表。"""
    docs = [item["content"] for item in candidates]
    doc_ids = [item["doc_id"] for item in candidates]
    metadatas = [item["metadata"] for item in candidates]

    reranked = rerank_results(
        query=query,
        docs=docs,
        doc_ids=doc_ids,
        metadata_list=metadatas,
        method=method,
        top_k=top_k,
        cross_encoder_model=(cross_encoder_model or None),
    )
    return [
        {
            "doc_id": item.doc_id,
            "content": item.content,
            "metadata": item.metadata,
            "score": item.score,
        }
        for item in reranked
    ]


def print_candidates(title, candidates):
    """格式化打印候选列表。"""
    print(title)
    if not candidates:
        print("  (无)")
        return

    for index, item in enumerate(candidates, start=1):
        score = item.get("score")
        score_text = f"{float(score):.4f}" if score is not None else "-"
        print(f"  {index}. {item['doc_id']} | score={score_text} | {item['content']}")


def main():
    args = parse_args()
    if args.retrieval_top_k <= 0:
        raise ValueError("retrieval-top-k must be greater than 0")
    if args.rerank_top_k <= 0:
        raise ValueError("rerank-top-k must be greater than 0")

    print("mode:", args.mode)
    print("query:", args.query)
    print("rerank_method:", args.rerank_method)
    print("llm_model:", args.llm_model)
    print("base_url:", resolve_base_url(args.base_url))
    if args.mode == "pipeline":
        print("embedding_model:", EMBED_MODEL_NAME)

    if args.mode == "fixed":
        candidates = build_fixed_candidates()
    else:
        try:
            candidates = build_pipeline_candidates(args.query, args.retrieval_top_k)
        except OSError as exc:
            if "WinError 1455" in str(exc) or "WinError 1114" in str(exc):
                print("embedding 模型加载失败：Windows 页面文件/内存不足。")
                print("建议先使用 fixed 模式跑通第12节链路。")
                print("示例：python experiments\\qa_pipeline_experiment.py --mode fixed --print-prompt-only")
                return
            raise

    reranked = do_rerank(
        query=args.query,
        candidates=candidates,
        method=args.rerank_method,
        top_k=args.rerank_top_k,
        cross_encoder_model=args.cross_encoder_model,
    )

    print()
    print_candidates("召回候选：", candidates)
    print()
    print_candidates("重排后候选：", reranked)

    if args.print_prompt_only:
        prompt = build_prompt(
            question=args.query,
            reranked_results=reranked,
            answer_rules=DEFAULT_ANSWER_RULES,
            strict_mode=True,
            require_citation=True,
        )
        print()
        print("system_prompt:")
        print(prompt.system_prompt)
        print()
        print("user_prompt:")
        print(prompt.user_prompt)
        print()
        print("context_block:")
        print(prompt.context_block)
        return

    try:
        result = generate_answer_with_rag(
            question=args.query,
            reranked_results=reranked,
            answer_rules=DEFAULT_ANSWER_RULES,
            model=args.llm_model,
            base_url=args.base_url,
            temperature=0.2,
            max_tokens=256,
            strict_mode=True,
            require_citation=True,
        )
    except RuntimeError as exc:
        print("调用失败:", exc)
        print("建议：")
        print("1) 检查 base_url 和 API Key 是否同一服务商；")
        print("2) 设置 API Key 环境变量（DEEPSEEK_API_KEY / SILICONFLOW_API_KEY / OPENAI_API_KEY）；")
        print("3) 再运行同一命令；若只想先看 Prompt，追加参数 --print-prompt-only。")
        return
    except Exception as exc:
        print("调用失败:", exc)
        print("建议先用 --print-prompt-only 验证前半链路，再检查模型名/base_url/API Key。")
        return

    print()
    print("=" * 96)
    print("模型回答：")
    print(result.answer)
    print("-" * 96)
    print("model:", result.model)
    print("finish_reason:", result.finish_reason)
    print("usage:", result.usage)
    print("unknown_hint_triggered:", result.unknown_hint_triggered)


if __name__ == "__main__":
    main()


# 运行示例：
# 1) 只看 Prompt（不调用大模型）：
# python experiments\qa_pipeline_experiment.py --mode fixed --query "RTX4090 显存是多少？" --print-prompt-only
#
# 2) 调用大模型（需先配 API Key）：
# python experiments\qa_pipeline_experiment.py --mode fixed --query "RTX4090 显存是多少？" --llm-model deepseek-chat
