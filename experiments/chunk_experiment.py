"""
分块实验脚本。

用途：
1. 对比不同 `chunk_size` / `chunk_overlap` 参数的分块效果。
2. 输出可直接粘贴到学习记录的统计表。
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    # 保持 experiments 脱离 src 后仍可导入核心模块。
    sys.path.append(str(SRC_DIR))

from document_loader import extract_text
from text_splitter import split_text


# (chunk_size, chunk_overlap) 参数组合。
PARAMETER_SETS = [
    (200, 0),
    (200, 40),
    (400, 40),
    (800, 40),
    (800, 100),
]


def summarize_chunks(chunks):
    """统计 chunk 数量和长度分布。"""
    if not chunks:
        return {
            "count": 0,
            "min_len": 0,
            "max_len": 0,
            "avg_len": 0,
        }

    lengths = [len(chunk) for chunk in chunks]
    return {
        "count": len(chunks),
        "min_len": min(lengths),
        "max_len": max(lengths),
        "avg_len": round(sum(lengths) / len(lengths), 1),
    }


def print_chunk_preview(chunks, limit=3):
    """打印前几个 chunk 的预览内容，便于人工检查可读性。"""
    for index, chunk in enumerate(chunks[:limit], start=1):
        preview = " ".join(chunk.split())
        if len(preview) > 160:
            preview = preview[:160] + "..."
        print(f"chunk {index}: {preview}")


def main():
    """执行分块参数实验并打印结果表。"""
    sample_file = PROJECT_ROOT / "test_files" / "sample_rag.txt"

    text = extract_text(str(sample_file))
    if not text.strip():
        raise RuntimeError(f"测试文本为空: {sample_file}")

    # 测试文件较短，重复几次便于观察 chunk_size 和 overlap 的变化。
    experiment_text = "\n\n".join([text] * 4)

    print("测试文件:", sample_file)
    print("原始字符数:", len(text))
    print("实验字符数:", len(experiment_text))
    print()
    print("| chunk_size | overlap | chunk 数量 | 最短 | 最长 | 平均 |")
    print("| --- | --- | --- | --- | --- | --- |")

    all_results = []
    for chunk_size, overlap in PARAMETER_SETS:
        chunks = split_text(
            experiment_text,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
        summary = summarize_chunks(chunks)
        all_results.append((chunk_size, overlap, chunks, summary))
        print(
            f"| {chunk_size} | {overlap} | {summary['count']} | "
            f"{summary['min_len']} | {summary['max_len']} | {summary['avg_len']} |"
        )

    print()
    print("默认参数预览：chunk_size=400, overlap=40")
    # 只打印推荐参数的预览，避免输出过长。
    for chunk_size, overlap, chunks, _summary in all_results:
        if chunk_size == 400 and overlap == 40:
            print_chunk_preview(chunks)
            break


if __name__ == "__main__":
    main()


# 运行结果示例（2026-05-23，Windows PowerShell）
# 测试文件: D:\Projects\Codex\Projects\RAG_Project\test_files\sample_rag.txt
# 原始字符数: 345
# 实验字符数: 1386
#
# | chunk_size | overlap | chunk 数量 | 最短 | 最长 | 平均 |
# | --- | --- | --- | --- | --- | --- |
# | 200 | 0 | 8 | 146 | 197 | 171.4 |
# | 200 | 40 | 12 | 101 | 186 | 150.2 |
# | 400 | 40 | 4 | 340 | 388 | 374.0 |
# | 800 | 40 | 2 | 687 | 735 | 711.0 |
# | 800 | 100 | 2 | 735 | 748 | 741.5 |
# 默认参数预览：chunk_size=400, overlap=40（chunk 1~3 已打印）
