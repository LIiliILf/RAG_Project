import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    # 保持 tests 脱离 src 后仍可导入核心模块。
    sys.path.append(str(SRC_DIR))

from prompt_builder import (
    PromptContext,
    build_context_block,
    build_prompt,
    build_system_prompt,
    build_user_prompt,
    normalize_contexts,
    should_answer_unknown,
)


class _FakeCandidate:
    """模拟重排候选对象（非 dict 输入）。"""

    def __init__(self, doc_id, content, metadata=None, score=None):
        self.doc_id = doc_id
        self.content = content
        self.metadata = metadata or {}
        self.score = score


class PromptBuilderTests(unittest.TestCase):
    """验证 Prompt 构建模块的核心行为。"""

    def setUp(self):
        self.contexts = [
            {
                "doc_id": "doc_0001",
                "content": "RTX4090 配备 24GB 显存，适合高分辨率推理任务。",
                "metadata": {"source": "gpu.md"},
                "score": 0.95,
            },
            {
                "doc_id": "doc_0002",
                "content": "RAG 系统会先检索资料，再生成回答。",
                "metadata": {"source": "rag.md"},
                "score": 0.61,
            },
            {
                "doc_id": "doc_0003",
                "content": "BM25 属于关键词检索方法。",
                "metadata": {"source": "retrieval.md"},
                "score": 0.52,
            },
        ]

    def test_normalize_contexts_supports_dict_and_object(self):
        """normalize_contexts 应兼容 dict 与对象输入。"""
        mixed = [
            self.contexts[0],
            _FakeCandidate("doc_1000", "示例文本", {"source": "obj.md"}, 0.5),
        ]
        normalized = normalize_contexts(mixed)
        self.assertEqual(len(normalized), 2)
        self.assertIsInstance(normalized[0], PromptContext)
        self.assertEqual(normalized[1].doc_id, "doc_1000")

    def test_build_context_block_formats_doc_id_and_source(self):
        """context block 应包含统一的 doc_id/source 标记。"""
        result = build_context_block(
            reranked_results=self.contexts,
            max_docs=2,
            max_chars_per_doc=100,
            max_total_chars=500,
        )
        self.assertEqual(len(result.contexts), 2)
        self.assertIn("[doc_id=doc_0001 source=gpu.md]", result.context_block)
        self.assertIn("[doc_id=doc_0002 source=rag.md]", result.context_block)

    def test_build_context_block_respects_total_char_budget(self):
        """总字符预算过小时应触发裁剪。"""
        result = build_context_block(
            reranked_results=self.contexts,
            max_docs=3,
            max_chars_per_doc=200,
            max_total_chars=80,
        )
        self.assertLessEqual(len(result.context_block), 80)
        self.assertTrue(result.truncated)

    def test_build_system_prompt_contains_strict_rules(self):
        """strict_mode 应写入“仅基于上下文”约束。"""
        system_prompt = build_system_prompt(
            answer_rules="不要编造参数。",
            strict_mode=True,
            require_citation=True,
        )
        self.assertIn("只能基于提供的参考资料回答", system_prompt)
        self.assertIn("回答结尾必须标注引用 doc_id", system_prompt)
        self.assertIn("不要编造参数", system_prompt)

    def test_build_user_prompt_rejects_empty_question(self):
        """问题为空时应抛出参数错误。"""
        with self.assertRaises(ValueError):
            build_user_prompt(question="", context_block="x")

    def test_should_answer_unknown_returns_true_when_no_overlap(self):
        """无词元重叠时应触发“资料不足”提示。"""
        question = "RTX4090 功耗上限是多少"
        contexts = [
            {"doc_id": "doc_x", "content": "RAG 系统先检索再回答。", "metadata": {"source": "a.md"}},
            {"doc_id": "doc_y", "content": "BM25 是关键词检索。", "metadata": {"source": "b.md"}},
        ]
        self.assertTrue(should_answer_unknown(question, contexts, min_overlap_tokens=1))

    def test_build_prompt_appends_unknown_hint_when_triggered(self):
        """build_prompt 在资料不足时应附加未知回答提示。"""
        package = build_prompt(
            question="RTX4090 功耗上限是多少",
            reranked_results=[
                {"doc_id": "doc_x", "content": "RAG 系统先检索再回答。", "metadata": {"source": "a.md"}},
            ],
            strict_mode=True,
            require_citation=True,
            max_docs=1,
            max_chars_per_doc=120,
            max_total_chars=200,
            min_overlap_tokens=1,
        )
        self.assertTrue(package.unknown_hint_triggered)
        self.assertIn("资料不足", package.user_prompt)
        self.assertIn("回答结尾必须标注引用 doc_id", package.system_prompt)


if __name__ == "__main__":
    unittest.main()
