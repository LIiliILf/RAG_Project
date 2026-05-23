import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    # 保持 tests 脱离 src 后仍可导入核心模块。
    sys.path.append(str(SRC_DIR))

from text_splitter import split_text


class SplitTextTests(unittest.TestCase):
    """验证文本分块函数在常见边界条件下的行为。"""

    def test_short_text_returns_single_chunk(self):
        """短文本不应被额外切分。"""
        text = "短文本不需要切分。"

        chunks = split_text(text, chunk_size=100, chunk_overlap=10)

        self.assertEqual(chunks, [text])

    def test_long_text_is_split_into_bounded_chunks(self):
        """长文本应被切成多个且不超过上限的 chunk。"""
        text = "第一段说明文档解析。" * 20

        chunks = split_text(text, chunk_size=40, chunk_overlap=5)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 40 for chunk in chunks))

    def test_overlap_keeps_boundary_context(self):
        """overlap 应在相邻 chunk 边界保留重复上下文。"""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        chunks = split_text(text, chunk_size=10, chunk_overlap=3)

        self.assertEqual(chunks[0][-3:], chunks[1][:3])

    def test_blank_text_returns_empty_list(self):
        """空白输入应返回空列表。"""
        chunks = split_text("   \n\t", chunk_size=100, chunk_overlap=10)

        self.assertEqual(chunks, [])

    def test_invalid_overlap_raises_error(self):
        """chunk_overlap >= chunk_size 应被拒绝。"""
        with self.assertRaises(ValueError):
            split_text("abcdef", chunk_size=10, chunk_overlap=10)


if __name__ == "__main__":
    unittest.main()
