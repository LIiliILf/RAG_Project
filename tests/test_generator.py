import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    # 保持 tests 脱离 src 后仍可导入核心模块。
    sys.path.append(str(SRC_DIR))

from generator import (
    build_messages,
    generate_answer_with_rag,
    generate_chat_completion,
    resolve_api_key,
    resolve_base_url,
)


class _FakeCompletionMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = _FakeCompletionMessage(content)
        self.finish_reason = finish_reason


class _FakeUsage:
    def __init__(self, prompt_tokens=10, completion_tokens=20, total_tokens=30):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class _FakeResponse:
    def __init__(self, content="ok", model="fake-model"):
        self.id = "req_fake_123"
        self.model = model
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


class _FakeChatCompletions:
    def create(self, **kwargs):
        _ = kwargs
        return _FakeResponse(content="这是一个测试回答。", model="fake-chat-model")


class _FakeChat:
    def __init__(self):
        self.completions = _FakeChatCompletions()


class _FakeClient:
    def __init__(self):
        self.chat = _FakeChat()


class GeneratorTests(unittest.TestCase):
    """验证 generator 模块的输入校验与调用流程。"""

    def test_build_messages_rejects_empty_inputs(self):
        """system/user 任一为空应抛出参数错误。"""
        with self.assertRaises(ValueError):
            build_messages("", "hello")
        with self.assertRaises(ValueError):
            build_messages("hi", "")

    def test_resolve_api_key_prefers_explicit_value(self):
        """显式传入的 key 优先级最高。"""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "env_key"}, clear=False):
            self.assertEqual(resolve_api_key("explicit_key"), "explicit_key")

    def test_resolve_api_key_reads_env_candidates(self):
        """未显式传入时应按候选环境变量读取。"""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "deepseek_key"}, clear=True):
            self.assertEqual(resolve_api_key(None), "deepseek_key")

    def test_resolve_api_key_prefers_provider_matched_env(self):
        """根据 base_url 应优先选择匹配服务商的 key。"""
        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "deepseek_key", "OPENAI_API_KEY": "openai_key"},
            clear=True,
        ):
            self.assertEqual(resolve_api_key(None, base_url="https://api.openai.com/v1"), "openai_key")

    def test_resolve_base_url_uses_env_when_no_explicit(self):
        """base_url 未显式传入时应读取环境变量。"""
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://api.deepseek.com"}, clear=True):
            self.assertEqual(resolve_base_url(""), "https://api.deepseek.com")

    def test_generate_chat_completion_raises_when_no_api_key(self):
        """缺少 API Key 时应给出可执行提示。"""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                generate_chat_completion(
                    system_prompt="system",
                    user_prompt="user",
                    model="test-model",
                    client=_FakeClient(),
                )

    def test_generate_chat_completion_with_fake_client(self):
        """使用假客户端时应返回标准化结果。"""
        result = generate_chat_completion(
            system_prompt="你是助手",
            user_prompt="你好",
            model="test-model",
            api_key="fake_key",
            client=_FakeClient(),
        )
        self.assertEqual(result.answer, "这是一个测试回答。")
        self.assertEqual(result.model, "fake-chat-model")
        self.assertEqual(result.finish_reason, "stop")
        self.assertEqual(result.usage["total_tokens"], 30)

    def test_generate_answer_with_rag_connects_prompt_and_generation(self):
        """应完成 prompt 构建并调用生成，输出统一结构。"""
        reranked_results = [
            {
                "doc_id": "doc_0001",
                "content": "RTX4090 配备 24GB 显存。",
                "metadata": {"source": "gpu.md"},
                "score": 0.9,
            },
            {
                "doc_id": "doc_0002",
                "content": "RAG 先检索再回答。",
                "metadata": {"source": "rag.md"},
                "score": 0.6,
            },
        ]
        result = generate_answer_with_rag(
            question="RTX4090 显存是多少",
            reranked_results=reranked_results,
            model="test-model",
            api_key="fake_key",
            client=_FakeClient(),
            max_docs=2,
            max_chars_per_doc=120,
            max_total_chars=300,
        )
        self.assertIn("doc_0001", result.context_block)
        self.assertEqual(result.answer, "这是一个测试回答。")
        self.assertEqual(result.model, "fake-chat-model")


if __name__ == "__main__":
    unittest.main()
