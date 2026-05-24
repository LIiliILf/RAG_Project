import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    # 保持 tests 脱离 src 后仍可导入核心模块。
    sys.path.append(str(SRC_DIR))

from multi_doc_pipeline import (
    DocumentRecord,
    build_chunk_records,
    discover_source_files,
    load_documents_from_files,
)


class MultiDocPipelineTests(unittest.TestCase):
    """验证第12节多文档处理链路的基础行为。"""

    def test_discover_source_files_filters_extensions(self):
        """目录扫描应只返回支持格式。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "a.md").write_text("markdown text", encoding="utf-8")
            (root / "b.txt").write_text("plain text", encoding="utf-8")
            (root / "c.xyz").write_text("unsupported", encoding="utf-8")

            files = discover_source_files([str(root)])
            names = [item.name for item in files]

            self.assertEqual(names, ["a.md", "b.txt"])

    def test_load_documents_skips_empty_files(self):
        """空文本应被跳过，非空文本应分配连续 doc_id。"""
        filepaths = [
            Path("a.md"),
            Path("b.md"),
            Path("c.md"),
        ]

        def fake_extractor(filepath):
            mapping = {
                "a.md": "doc A content",
                "b.md": "",
                "c.md": "doc C content",
            }
            return mapping[Path(filepath).name]

        documents, skipped = load_documents_from_files(filepaths, text_extractor=fake_extractor)

        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0].doc_id, "doc_0001")
        self.assertEqual(documents[1].doc_id, "doc_0002")
        self.assertEqual([item["filepath"] for item in skipped], ["b.md"])

    def test_build_chunk_records_contains_source_fields(self):
        """chunk metadata 应包含 source/doc_id/chunk_id。"""
        documents = [
            DocumentRecord(
                doc_id="doc_0001",
                source="sample.md",
                filepath="D:/sample.md",
                text="第一句。第二句。第三句。",
            )
        ]

        chunks = build_chunk_records(
            documents=documents,
            chunk_size=6,
            chunk_overlap=1,
        )

        self.assertGreaterEqual(len(chunks), 2)
        first = chunks[0]
        self.assertEqual(first.doc_id, "doc_0001")
        self.assertTrue(first.chunk_id.startswith("doc_0001_chunk_"))
        self.assertEqual(first.metadata["source"], "sample.md")
        self.assertEqual(first.metadata["doc_id"], "doc_0001")
        self.assertEqual(first.metadata["chunk_id"], first.chunk_id)

    def test_load_documents_raises_when_all_empty(self):
        """所有文档为空时应报错。"""
        filepaths = [Path("a.md"), Path("b.md")]

        def always_empty(_):
            return ""

        with self.assertRaises(ValueError):
            load_documents_from_files(filepaths, text_extractor=always_empty)


if __name__ == "__main__":
    unittest.main()
