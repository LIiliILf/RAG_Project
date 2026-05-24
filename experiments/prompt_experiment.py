"""
Prompt A/B 实验脚本。

目标：
1. 使用同一批候选上下文，构建两版 Prompt（A: 弱约束，B: 强约束）。
2. 直观看到 system/user prompt 在“边界控制”上的差异。
3. 验证 context 裁剪、引用格式、未知提示是否按预期生效。
"""

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from prompt_builder import build_prompt


DEFAULT_MAX_DOCS = 3
DEFAULT_MAX_CHARS_PER_DOC = 180
DEFAULT_MAX_TOTAL_CHARS = 520

CASES = [
    {
        "name": "已知问题-显存",
        "question": "RTX4090 24GB 显存是多少？",
        "answer_rules": "优先引用含数字的证据；不要编造额外参数。",
        "contexts": [
            {
                "doc_id": "doc_a",
                "content": "RTX4090 配备 16GB 显存，适合轻量任务。",
                "metadata": {"source": "gpu_a.md"},
                "score": 0.72,
            },
            {
                "doc_id": "doc_b",
                "content": "RTX4090 配备 24GB 显存，适合高分辨率推理任务。",
                "metadata": {"source": "gpu_b.md"},
                "score": 0.96,
            },
            {
                "doc_id": "doc_c",
                "content": "RAG 系统通常先检索，再把资料交给模型生成答案。",
                "metadata": {"source": "rag_intro.md"},
                "score": 0.41,
            },
        ],
    },
    {
        "name": "未知问题-证据不足",
        "question": "RTX4090 的整卡功耗上限是多少？",
        "answer_rules": "如果资料不足，请明确说无法确定。",
        "contexts": [
            {
                "doc_id": "doc_x",
                "content": "RAG 流程分为：解析、分块、检索、生成。",
                "metadata": {"source": "rag_flow.md"},
                "score": 0.67,
            },
            {
                "doc_id": "doc_y",
                "content": "BM25 是关键词检索方法，FAISS 是向量检索方法。",
                "metadata": {"source": "retrieval.md"},
                "score": 0.61,
            },
        ],
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="运行 Prompt A/B 对比实验。")
    parser.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS, help="上下文最多保留文档数。")
    parser.add_argument(
        "--max-chars-per-doc",
        type=int,
        default=DEFAULT_MAX_CHARS_PER_DOC,
        help="每条文档最多保留字符数。",
    )
    parser.add_argument(
        "--max-total-chars",
        type=int,
        default=DEFAULT_MAX_TOTAL_CHARS,
        help="上下文总字符预算。",
    )
    return parser.parse_args()


def print_prompt_package(label, package):
    """格式化输出 Prompt 包。"""
    print("-" * 110)
    print(label)
    print("unknown_hint_triggered:", package.unknown_hint_triggered)
    print("truncated:", package.truncated)
    print("omitted_count:", package.omitted_count)
    print("context_count:", len(package.contexts))
    print()
    print("[system prompt]")
    print(package.system_prompt)
    print()
    print("[user prompt]")
    print(package.user_prompt)


def run_case(case, args):
    """单个案例的 A/B Prompt 对比。"""
    question = case["question"]
    contexts = case["contexts"]
    answer_rules = case["answer_rules"]

    prompt_a = build_prompt(
        question=question,
        reranked_results=contexts,
        answer_rules=answer_rules,
        strict_mode=False,
        require_citation=False,
        unknown_fallback="",
        max_docs=args.max_docs,
        max_chars_per_doc=args.max_chars_per_doc,
        max_total_chars=args.max_total_chars,
    )
    prompt_b = build_prompt(
        question=question,
        reranked_results=contexts,
        answer_rules=answer_rules,
        strict_mode=True,
        require_citation=True,
        max_docs=args.max_docs,
        max_chars_per_doc=args.max_chars_per_doc,
        max_total_chars=args.max_total_chars,
    )

    print("=" * 110)
    print("case:", case["name"])
    print("question:", question)
    print_prompt_package("Prompt A（弱约束）", prompt_a)
    print_prompt_package("Prompt B（强约束）", prompt_b)


def main():
    args = parse_args()
    if args.max_docs <= 0:
        raise ValueError("max-docs must be greater than 0")
    if args.max_chars_per_doc <= 0:
        raise ValueError("max-chars-per-doc must be greater than 0")
    if args.max_total_chars <= 0:
        raise ValueError("max-total-chars must be greater than 0")

    print("Prompt A/B 实验开始")
    print("cases:", len(CASES))
    print("max_docs:", args.max_docs)
    print("max_chars_per_doc:", args.max_chars_per_doc)
    print("max_total_chars:", args.max_total_chars)
    print()

    for case in CASES:
        run_case(case, args)


if __name__ == "__main__":
    main()


# 运行示例（2026-05-24）：
# python experiments\prompt_experiment.py --max-docs 3 --max-chars-per-doc 180 --max-total-chars 520
