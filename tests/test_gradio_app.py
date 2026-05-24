import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    # 保持 tests 脱离 src 后仍可导入核心模块。
    sys.path.append(str(SRC_DIR))

from gradio_app import (
    PipelineSessionState,
    clear_chat_history,
    normalize_uploaded_files,
    process_uploaded_files,
    run_qa_turn,
    validate_build_options,
)


class _FakeFile:
    """模拟 Gradio 上传文件对象（只保留 name 字段）。"""

    def __init__(self, name):
        self.name = name


class GradioAppTests(unittest.TestCase):
    """验证第13节界面编排层的核心行为。"""

    def test_normalize_uploaded_files_supports_str_and_file_object(self):
        """上传输入兼容路径字符串和带 name 的对象。"""
        paths = normalize_uploaded_files(["a.md", _FakeFile("b.txt")])
        self.assertEqual(paths, ["a.md", "b.txt"])

    def test_validate_build_options_rejects_invalid_overlap(self):
        """chunk_overlap >= chunk_size 时应拒绝。"""
        with self.assertRaises(ValueError):
            validate_build_options(chunk_size=100, chunk_overlap=100, hybrid_alpha=0.7)

    @patch("gradio_app.build_hybrid_retriever_from_chunks")
    @patch("gradio_app.build_chunk_records")
    @patch("gradio_app.load_documents_from_files")
    @patch("gradio_app.discover_source_files")
    def test_process_uploaded_files_builds_session(
        self,
        mock_discover_source_files,
        mock_load_documents,
        mock_build_chunk_records,
        mock_build_retriever,
    ):
        """上传处理成功后应返回会话状态与摘要。"""
        fake_doc = SimpleNamespace(
            doc_id="doc_0001",
            source="sample.md",
            filepath="D:/sample.md",
            text="RAG 测试文本",
        )
        fake_chunk = SimpleNamespace(
            doc_id="doc_0001",
            chunk_id="doc_0001_chunk_0001",
            source="sample.md",
            content="RAG 测试文本分块",
        )

        mock_discover_source_files.return_value = [Path("sample.md")]
        mock_load_documents.return_value = ([fake_doc], [])
        mock_build_chunk_records.return_value = [fake_chunk]
        mock_build_retriever.return_value = object()

        session, status, chunk_rows, summary = process_uploaded_files(
            uploaded_files=["sample.md"],
            chunk_size=200,
            chunk_overlap=20,
            hybrid_alpha=0.6,
        )

        self.assertIsInstance(session, PipelineSessionState)
        self.assertIn("处理成功", status)
        self.assertEqual(len(chunk_rows), 1)
        self.assertIn("文档数: 1", summary)
        self.assertIn("chunk 数: 1", summary)

    def test_run_qa_turn_requires_ready_session(self):
        """未建索引时提问应直接提示。"""
        history, status, candidates = run_qa_turn(
            session=None,
            question="RAG 是什么？",
            chat_history=[],
        )
        self.assertEqual(history, [])
        self.assertIn("请先上传并处理文档", status)
        self.assertIn("索引未就绪", candidates)

    @patch("gradio_app.generate_answer_with_rag")
    @patch("gradio_app.retrieve_and_rerank")
    def test_run_qa_turn_success(self, mock_retrieve_and_rerank, mock_generate_answer):
        """问答成功后应写入聊天记录并输出候选证据。"""
        retrieved = [
            SimpleNamespace(
                doc_id="doc_0001_chunk_0001",
                content="RTX4090 配备 24GB 显存。",
                metadata={"doc_id": "doc_0001", "chunk_id": "doc_0001_chunk_0001", "source": "gpu.md"},
                hybrid_score=0.91,
            )
        ]
        reranked = [
            SimpleNamespace(
                doc_id="doc_0001_chunk_0001",
                content="RTX4090 配备 24GB 显存。",
                metadata={"doc_id": "doc_0001", "chunk_id": "doc_0001_chunk_0001", "source": "gpu.md"},
                score=0.87,
            )
        ]
        mock_retrieve_and_rerank.return_value = (retrieved, reranked)
        mock_generate_answer.return_value = SimpleNamespace(
            answer="RTX4090 显存是 24GB。[doc_id=doc_0001]",
            model="deepseek-chat",
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
            unknown_hint_triggered=False,
        )

        session = PipelineSessionState(
            retriever=object(),
            documents=[SimpleNamespace(doc_id="doc_0001")],
            chunks=[SimpleNamespace(chunk_id="doc_0001_chunk_0001")],
            skipped=[],
            source_files=["sample.md"],
            chunk_size=400,
            chunk_overlap=40,
            hybrid_alpha=0.7,
        )

        history, status, candidates = run_qa_turn(
            session=session,
            question="RTX4090 显存是多少？",
            chat_history=[],
        )

        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["role"], "assistant")
        self.assertIn("RTX4090 显存是 24GB", history[1]["content"])
        self.assertIn("问答完成", status)
        self.assertIn("召回候选", candidates)
        self.assertIn("重排候选", candidates)

    def test_clear_chat_history(self):
        """清空动作应回到初始状态。"""
        history, status, candidates = clear_chat_history()
        self.assertEqual(history, [])
        self.assertIn("清空", status)
        self.assertIn("已清空", candidates)


if __name__ == "__main__":
    unittest.main()
