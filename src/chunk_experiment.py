from pathlib import Path

from document_loader import extract_text
from text_splitter import split_text


PARAMETER_SETS = [
    (200, 0),
    (200, 40),
    (400, 40),
    (800, 40),
    (800, 100),
]


def summarize_chunks(chunks):
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
    for index, chunk in enumerate(chunks[:limit], start=1):
        preview = " ".join(chunk.split())
        if len(preview) > 160:
            preview = preview[:160] + "..."
        print(f"chunk {index}: {preview}")


def main():
    project_root = Path(__file__).resolve().parents[1]
    sample_file = project_root / "test_files" / "sample_rag.txt"

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
    for chunk_size, overlap, chunks, _summary in all_results:
        if chunk_size == 400 and overlap == 40:
            print_chunk_preview(chunks)
            break


if __name__ == "__main__":
    main()
