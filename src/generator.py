"""
生成模块（Generator）。

职责：
1. 把 prompt_builder 产出的 system/user prompt 组装成消息。
2. 调用 OpenAI 兼容接口获取模型回答。
3. 返回标准化结果，便于实验脚本记录与评估。
"""

from dataclasses import dataclass
import os

from prompt_builder import build_prompt


DEFAULT_CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "deepseek-chat")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 512
API_KEY_ENV_CANDIDATES = (
    "DEEPSEEK_API_KEY",
    "SILICONFLOW_API_KEY",
    "OPENAI_API_KEY",
)


@dataclass(frozen=True)
class ChatGenerationResult:
    """模型生成结果。"""

    answer: str
    model: str
    finish_reason: str | None
    usage: dict
    request_id: str | None


@dataclass(frozen=True)
class RAGGenerationResult:
    """完整 RAG 生成结果（含 prompt）。"""

    question: str
    system_prompt: str
    user_prompt: str
    context_block: str
    answer: str
    model: str
    usage: dict
    finish_reason: str | None
    unknown_hint_triggered: bool


def resolve_base_url(explicit_base_url=None):
    """
    解析最终使用的 base_url。

    优先级：
    1) 显式参数
    2) 环境变量 OPENAI_BASE_URL
    3) 默认值 https://api.deepseek.com
    """
    explicit = str(explicit_base_url or "").strip()
    if explicit:
        return explicit
    env_value = str(os.getenv("OPENAI_BASE_URL", "")).strip()
    if env_value:
        return env_value
    return DEFAULT_BASE_URL


def _api_key_candidates_for_base_url(base_url):
    """根据 base_url 推断优先尝试的 API Key 环境变量顺序。"""
    normalized = str(base_url or "").strip().lower()
    if "deepseek" in normalized:
        return ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "SILICONFLOW_API_KEY")
    if "siliconflow" in normalized:
        return ("SILICONFLOW_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY")
    if "openai.com" in normalized:
        return ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "SILICONFLOW_API_KEY")
    return API_KEY_ENV_CANDIDATES


def resolve_api_key(explicit_api_key=None, base_url=None):
    """按优先级读取可用 API Key。"""
    explicit = str(explicit_api_key or "").strip()
    if explicit:
        return explicit

    for env_name in _api_key_candidates_for_base_url(base_url):
        value = str(os.getenv(env_name, "")).strip()
        if value:
            return value
    return ""


def build_messages(system_prompt, user_prompt):
    """构建 Chat Completions 所需消息。"""
    system_text = str(system_prompt or "").strip()
    user_text = str(user_prompt or "").strip()

    if not system_text:
        raise ValueError("system_prompt must not be empty")
    if not user_text:
        raise ValueError("user_prompt must not be empty")

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


def create_client(api_key, base_url=DEFAULT_BASE_URL, timeout=DEFAULT_TIMEOUT_SECONDS):
    """创建 OpenAI 兼容客户端（懒加载）。"""
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("缺少依赖 openai，请先运行：python -m pip install openai") from exc

    normalized_base_url = resolve_base_url(base_url)
    kwargs = {
        "api_key": str(api_key).strip(),
        "base_url": normalized_base_url,
        "timeout": float(timeout),
    }
    return OpenAI(**kwargs)


def generate_chat_completion(
    system_prompt,
    user_prompt,
    model=DEFAULT_CHAT_MODEL,
    temperature=DEFAULT_TEMPERATURE,
    max_tokens=DEFAULT_MAX_TOKENS,
    api_key=None,
    base_url=DEFAULT_BASE_URL,
    timeout=DEFAULT_TIMEOUT_SECONDS,
    client=None,
):
    """
    调用大模型并返回标准化结果。
    """
    if str(model or "").strip() == "":
        raise ValueError("model must not be empty")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")

    resolved_base_url = resolve_base_url(base_url)
    resolved_api_key = resolve_api_key(api_key, base_url=resolved_base_url)
    if not resolved_api_key:
        raise RuntimeError(
            f"未找到可用 API Key（base_url={resolved_base_url}）。"
            "请设置环境变量 DEEPSEEK_API_KEY / SILICONFLOW_API_KEY / OPENAI_API_KEY，"
            "或在函数参数传入 api_key。"
        )

    messages = build_messages(system_prompt, user_prompt)

    active_client = client or create_client(
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        timeout=timeout,
    )
    try:
        response = active_client.chat.completions.create(
            model=str(model).strip(),
            messages=messages,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
    except Exception as exc:
        if exc.__class__.__name__ == "AuthenticationError":
            raise RuntimeError(
                "鉴权失败（401）。请检查 API Key 与 base_url 是否匹配："
                f"当前 base_url={resolved_base_url}。"
            ) from exc
        raise

    if not getattr(response, "choices", None):
        raise RuntimeError("模型响应异常：choices 为空")

    message = response.choices[0].message
    answer = str(getattr(message, "content", "") or "").strip()
    usage_obj = getattr(response, "usage", None)
    usage = {}
    if usage_obj is not None:
        usage = {
            "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
            "completion_tokens": getattr(usage_obj, "completion_tokens", None),
            "total_tokens": getattr(usage_obj, "total_tokens", None),
        }

    return ChatGenerationResult(
        answer=answer,
        model=str(getattr(response, "model", model)),
        finish_reason=getattr(response.choices[0], "finish_reason", None),
        usage=usage,
        request_id=getattr(response, "id", None),
    )


def generate_answer_with_rag(
    question,
    reranked_results,
    answer_rules="",
    strict_mode=True,
    require_citation=True,
    max_docs=3,
    max_chars_per_doc=280,
    max_total_chars=1200,
    min_overlap_tokens=1,
    unknown_fallback="根据当前资料无法确定。",
    model=DEFAULT_CHAT_MODEL,
    temperature=DEFAULT_TEMPERATURE,
    max_tokens=DEFAULT_MAX_TOKENS,
    api_key=None,
    base_url=DEFAULT_BASE_URL,
    timeout=DEFAULT_TIMEOUT_SECONDS,
    client=None,
):
    """
    一步完成：构建 Prompt -> 调用模型 -> 返回结果。
    """
    prompt_package = build_prompt(
        question=question,
        reranked_results=reranked_results,
        answer_rules=answer_rules,
        strict_mode=strict_mode,
        require_citation=require_citation,
        max_docs=max_docs,
        max_chars_per_doc=max_chars_per_doc,
        max_total_chars=max_total_chars,
        min_overlap_tokens=min_overlap_tokens,
        unknown_fallback=unknown_fallback,
    )

    generation = generate_chat_completion(
        system_prompt=prompt_package.system_prompt,
        user_prompt=prompt_package.user_prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        client=client,
    )

    return RAGGenerationResult(
        question=prompt_package.question,
        system_prompt=prompt_package.system_prompt,
        user_prompt=prompt_package.user_prompt,
        context_block=prompt_package.context_block,
        answer=generation.answer,
        model=generation.model,
        usage=generation.usage,
        finish_reason=generation.finish_reason,
        unknown_hint_triggered=prompt_package.unknown_hint_triggered,
    )
