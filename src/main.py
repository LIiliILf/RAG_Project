"""
文档解析入口脚本。

用途：
1. 逐个读取测试文件。
2. 输出解析状态、字符数、关键字命中和内容预览。
"""

from pathlib import Path

from document_loader import extract_text


TEST_KEYWORD = "RAG_TEST_KEYWORD_2026"

EMPTY_TEXT_HINTS = {
    ".docx": "如果这是 Word 文件，可能缺少 python-docx，或文档没有可读取段落。",
    ".pdf": "如果这是 PDF 文件，可能缺少 pdfminer.six，或 PDF 是扫描版图片。",
    ".txt": "这个 TXT 文件可能本身就是空文件。",
    ".md": "这个 Markdown 文件可能本身就是空文件。",
}


def preview(text, limit=120):
    """压缩空白并截断长文本，用于终端预览。"""
    compact_text = " ".join(text.split())
    if len(compact_text) <= limit:
        return compact_text
    return compact_text[:limit] + "..."


def inspect_file(filepath):
    """打印单个文件的解析结果。"""
    print("=" * 80)
    print(f"文件: {filepath.name}")

    try:
        text = extract_text(str(filepath))
    except ModuleNotFoundError as exc:
        # 单独捕获缺依赖场景，给出可执行提示。
        print("状态: 解析失败")
        print(f"原因: 缺少依赖 {exc.name}")
        print("建议: 安装对应依赖后再试")
        return
    except Exception as exc:
        print("状态: 解析失败")
        print(f"原因: {exc}")
        return

    is_empty = not text.strip()
    has_keyword = TEST_KEYWORD in text

    print(f"状态: {'空文本或不支持格式' if is_empty else '解析成功'}")
    print(f"字符数: {len(text)}")
    print(f"包含测试关键词: {'是' if has_keyword else '否'}")

    if not is_empty:
        print("内容预览:")
        print(preview(text))
    else:
        print("提示:")
        print(EMPTY_TEXT_HINTS.get(filepath.suffix.lower(), "该格式不在当前解析器支持范围内。"))


def main():
    """运行全部测试文件的解析检查。"""
    project_root = Path(__file__).resolve().parents[1]
    test_files_dir = project_root / "test_files"

    if not test_files_dir.exists():
        raise FileNotFoundError(f"测试文件目录不存在: {test_files_dir}")

    files = [
        test_files_dir / "sample_rag.txt",
        test_files_dir / "sample_rag.md",
        test_files_dir / "sample_rag.docx",
        test_files_dir / "sample_rag.pdf",
        test_files_dir / "empty.txt",
        test_files_dir / "unsupported_sample.xyz",
    ]

    for filepath in files:
        inspect_file(filepath)


if __name__ == "__main__":
    main()

