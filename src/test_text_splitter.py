import unittest

from text_splitter import split_text


class SplitTextTests(unittest.TestCase):
    def test_short_text_returns_single_chunk(self):
        text = "短文本不需要切分。"

        chunks = split_text(text, chunk_size=100, chunk_overlap=10)

        self.assertEqual(chunks, [text])

    def test_long_text_is_split_into_bounded_chunks(self):
        text = "第一段说明文档解析。" * 20

        chunks = split_text(text, chunk_size=40, chunk_overlap=5)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 40 for chunk in chunks))

    def test_overlap_keeps_boundary_context(self):
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        chunks = split_text(text, chunk_size=10, chunk_overlap=3)

        self.assertEqual(chunks[0][-3:], chunks[1][:3])

    def test_blank_text_returns_empty_list(self):
        chunks = split_text("   \n\t", chunk_size=100, chunk_overlap=10)

        self.assertEqual(chunks, [])

    def test_invalid_overlap_raises_error(self):
        with self.assertRaises(ValueError):
            split_text("abcdef", chunk_size=10, chunk_overlap=10)


if __name__ == "__main__":
    unittest.main()
