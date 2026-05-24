"""
第12节：多文档问答与来源追踪实验。

目标：
1. 读取多个本地文档，统一构建索引。
2. 检索 + 重排后保留 source/doc_id/chunk_id。
3. 生成可追踪来源的回答。
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

from generator import DEFAULT_CHAT_MODEL, generate_answer_with_rag, resolve_base_url
from multi_doc_pipeline import (
    build_chunk_records,
    build_hybrid_retriever_from_chunks,
    discover_source_files,
    load_documents_from_files,
    retrieve_and_rerank,
)
from prompt_builder import build_prompt


DEFAULT_INPUT_PATHS = [
    str(PROJECT_ROOT / "data"),
    str(PROJECT_ROOT / "test_files"),
]
DEFAULT_QUERY = "RAG 的流程可以分为哪些步骤？"
DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 40
DEFAULT_HYBRID_ALPHA = 0.7
DEFAULT_RETRIEVAL_TOP_K = 8
DEFAULT_RERANK_TOP_K = 4
DEFAULT_RERANK_METHOD = "overlap"
DEFAULT_ANSWER_RULES = (
    "优先引用最相关证据；若证据不足，回答“根据当前资料无法确定”，并说明缺失信息。"
)


def parse_args():
    parser = argparse.ArgumentParser(description="运行第12节多文档问答实验。")
    parser.add_argument(
        "--input-paths",
        type=str,
        nargs="+",
        default=DEFAULT_INPUT_PATHS,
        help="输入文件或目录列表（可混合）。",
    )
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="用户问题。")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="分块大小。")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="分块重叠。")
    parser.add_argument("--hybrid-alpha", type=float, default=DEFAULT_HYBRID_ALPHA, help="混合检索权重。")
    parser.add_argument("--retrieval-top-k", type=int, default=DEFAULT_RETRIEVAL_TOP_K, help="召回数量。")
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


def print_document_summary(documents, chunks, skipped):
    """打印多文档构建概览。"""
    print("文档数量:", len(documents))
    print("chunk 数量:", len(chunks))
    print("跳过文件数:", len(skipped))
    if skipped:
        for item in skipped:
            print("  skipped:", item["filepath"], "| reason:", item["reason"])

    counts = {}
    for item in chunks:
        counts[item.doc_id] = counts.get(item.doc_id, 0) + 1

    print("每个文档的 chunk 数：")
    for document in documents:
        print(f"  {document.doc_id} | {document.source} | chunks={counts.get(document.doc_id, 0)}")


def print_retrieval_results(title, items, include_score=True):
    """打印检索或重排结果。"""
    print(title)
    if not items:
        print("  (无)")
        return

    for index, item in enumerate(items, start=1):
        metadata = item.metadata if hasattr(item, "metadata") else item.get("metadata", {})
        source = metadata.get("source", "-")
        doc_id = metadata.get("doc_id", "-")
        chunk_id = metadata.get("chunk_id", "-")

        if include_score:
            score = getattr(item, "hybrid_score", None)
            if score is None:
                score = getattr(item, "score", None)
            score_text = f"{float(score):.4f}" if score is not None else "-"
        else:
            score_text = "-"

        content = item.content if hasattr(item, "content") else item.get("content", "")
        print(
            f"  {index}. doc={doc_id} chunk={chunk_id} source={source} score={score_text}\n"
            f"     {content}"
        )


def main():
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be greater than 0")
    if args.chunk_overlap < 0:
        raise ValueError("chunk-overlap must be >= 0")
    if args.chunk_overlap >= args.chunk_size:
        raise ValueError("chunk-overlap must be < chunk-size")
    if args.retrieval_top_k <= 0:
        raise ValueError("retrieval-top-k must be greater than 0")
    if args.rerank_top_k <= 0:
        raise ValueError("rerank-top-k must be greater than 0")
    if args.hybrid_alpha < 0 or args.hybrid_alpha > 1:
        raise ValueError("hybrid-alpha must be in [0, 1]")

    print("query:", args.query)
    print("llm_model:", args.llm_model)
    print("base_url:", resolve_base_url(args.base_url))
    print("input_paths:", ", ".join(args.input_paths))

    source_files = discover_source_files(args.input_paths)
    documents, skipped = load_documents_from_files(source_files)
    chunks = build_chunk_records(
        documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print_document_summary(documents, chunks, skipped)

    try:
        retriever = build_hybrid_retriever_from_chunks(chunks, alpha=args.hybrid_alpha)
        retrieved, reranked = retrieve_and_rerank(
            query=args.query,
            retriever=retriever,
            retrieval_top_k=args.retrieval_top_k,
            rerank_top_k=args.rerank_top_k,
            rerank_method=args.rerank_method,
            cross_encoder_model=args.cross_encoder_model,
        )
    except OSError as exc:
        if "WinError 1455" in str(exc) or "WinError 1114" in str(exc):
            print("embedding 模型加载失败：Windows 页面文件/内存不足。")
            print("建议先减少语料/分块，或先扩容页面文件后再试。")
            return
        raise

    print()
    print_retrieval_results("召回结果：", retrieved, include_score=True)
    print()
    print_retrieval_results("重排结果：", reranked, include_score=True)

    reranked_for_prompt = [
        {
            "doc_id": item.doc_id,
            "content": item.content,
            "metadata": item.metadata,
            "score": item.score,
        }
        for item in reranked
    ]

    if args.print_prompt_only:
        prompt = build_prompt(
            question=args.query,
            reranked_results=reranked_for_prompt,
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
        return

    try:
        result = generate_answer_with_rag(
            question=args.query,
            reranked_results=reranked_for_prompt,
            answer_rules=DEFAULT_ANSWER_RULES,
            strict_mode=True,
            require_citation=True,
            model=args.llm_model,
            base_url=args.base_url,
            temperature=0.2,
            max_tokens=256,
        )
    except RuntimeError as exc:
        print("调用失败:", exc)
        print("建议：")
        print("1) 检查 base_url 与 API Key 是否同一服务商；")
        print("2) 设置本地环境变量 API Key 后重试。")
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
# 1) 仅看 Prompt（推荐先跑）
# python experiments\multi_doc_qa_experiment.py --query "RAG 的流程可以分为哪些步骤？" --print-prompt-only
#
# 2) 在线调用大模型
# python experiments\multi_doc_qa_experiment.py --query "RAG 的流程可以分为哪些步骤？" --llm-model deepseek-chat
