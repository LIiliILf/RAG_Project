"""
Prompt 构建模块。

职责：
1. 把重排后的候选组织成统一的上下文块（context block）。
2. 构建 system / user prompt，明确“仅基于上下文回答”的约束。
3. 提供“资料不足”提示，降低回答越界风险。
"""

from dataclasses import dataclass
import re


DEFAULT_MAX_DOCS = 3
DEFAULT_MAX_CHARS_PER_DOC = 280
DEFAULT_MAX_TOTAL_CHARS = 1200
DEFAULT_UNKNOWN_FALLBACK = "根据当前资料无法确定。"
STOP_TOKENS = {
    # 中文高频虚词 / 疑问词
    "的",
    "是",
    "了",
    "在",
    "和",
    "或",
    "与",
    "及",
    "吗",
    "呢",
    "啊",
    "把",
    "被",
    "对",
    "为",
    "上",
    "下",
    "中",
    "这",
    "那",
    "多少",
    "什么",
    "怎么",
    "如何",
    "是否",
    "请",
    "问",
    # 英文高频虚词
    "the",
    "a",
    "an",
    "is",
    "are",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "or",
    "with",
    "what",
    "how",
    "why",
    "when",
}


@dataclass(frozen=True)
class PromptContext:
    """规范化后的单条上下文。"""

    doc_id: str
    content: str
    metadata: dict
    score: float | None = None


@dataclass(frozen=True)
class ContextBuildResult:
    """上下文拼装结果。"""

    context_block: str
    contexts: list[PromptContext]
    truncated: bool
    omitted_count: int


@dataclass(frozen=True)
class PromptPackage:
    """最终可直接喂给模型的 Prompt 包。"""

    question: str
    system_prompt: str
    user_prompt: str
    context_block: str
    contexts: list[PromptContext]
    unknown_hint_triggered: bool
    truncated: bool
    omitted_count: int


def normalize_contexts(reranked_results):
    """
    兼容 dict / dataclass / object 输入，统一为 PromptContext 列表。
    空内容会被自动过滤。
    """
    if reranked_results is None:
        return []

    contexts = []
    for index, item in enumerate(list(reranked_results), start=1):
        if isinstance(item, dict):
            doc_id = item.get("doc_id") or item.get("chunk_id") or f"doc_{index:04d}"
            content = item.get("content") or item.get("text") or ""
            metadata = item.get("metadata") or {}
            score = item.get("score")
        else:
            doc_id = getattr(item, "doc_id", None) or getattr(item, "chunk_id", None) or f"doc_{index:04d}"
            content = getattr(item, "content", "")
            metadata = getattr(item, "metadata", {}) or {}
            score = getattr(item, "score", None)

        content_text = _compact_whitespace(str(content or ""))
        if not content_text:
            continue

        if not isinstance(metadata, dict):
            metadata = {"raw_metadata": str(metadata)}

        parsed_score = None
        if score is not None:
            try:
                parsed_score = float(score)
            except (TypeError, ValueError):
                parsed_score = None

        contexts.append(
            PromptContext(
                doc_id=str(doc_id),
                content=content_text,
                metadata=dict(metadata),
                score=parsed_score,
            )
        )
    return contexts


def build_context_block(
    reranked_results,
    max_docs=DEFAULT_MAX_DOCS,
    max_chars_per_doc=DEFAULT_MAX_CHARS_PER_DOC,
    max_total_chars=DEFAULT_MAX_TOTAL_CHARS,
    include_source=True,
):
    """
    将候选上下文拼装为统一文本块。

    裁剪策略：
    1) 仅保留前 max_docs 条；
    2) 每条最多 max_chars_per_doc；
    3) 全部拼接后最多 max_total_chars。
    """
    if max_docs <= 0:
        raise ValueError("max_docs must be greater than 0")
    if max_chars_per_doc <= 0:
        raise ValueError("max_chars_per_doc must be greater than 0")
    if max_total_chars <= 0:
        raise ValueError("max_total_chars must be greater than 0")

    contexts = normalize_contexts(reranked_results)
    selected_contexts = contexts[:max_docs]
    omitted_count = max(len(contexts) - len(selected_contexts), 0)

    lines = []
    used_contexts = []
    truncated = False

    for index, context in enumerate(selected_contexts):
        header = _build_context_header(context.doc_id, context.metadata, include_source)
        clipped_content, clipped_by_doc = _truncate_with_ellipsis(context.content, max_chars_per_doc)
        line = f"{header} {clipped_content}".strip()

        candidate_block = "\n".join(lines + [line]) if lines else line
        if len(candidate_block) <= max_total_chars:
            lines.append(line)
            used_contexts.append(
                PromptContext(
                    doc_id=context.doc_id,
                    content=clipped_content,
                    metadata=dict(context.metadata),
                    score=context.score,
                )
            )
            truncated = truncated or clipped_by_doc
            continue

        current_text = "\n".join(lines)
        remaining = max_total_chars - len(current_text)
        if lines:
            remaining -= 1  # 预留换行符

        if remaining > 0:
            clipped_line, clipped_by_total = _truncate_with_ellipsis(line, remaining)
            clipped_line = clipped_line.strip()
            if clipped_line:
                lines.append(clipped_line)
                used_contexts.append(
                    PromptContext(
                        doc_id=context.doc_id,
                        content=clipped_content,
                        metadata=dict(context.metadata),
                        score=context.score,
                    )
                )
                truncated = True or clipped_by_doc or clipped_by_total
                omitted_count += len(selected_contexts) - index - 1
                break

        truncated = True
        omitted_count += len(selected_contexts) - index
        break

    return ContextBuildResult(
        context_block="\n".join(lines),
        contexts=used_contexts,
        truncated=truncated,
        omitted_count=omitted_count,
    )


def build_system_prompt(
    answer_rules="",
    strict_mode=True,
    require_citation=True,
    unknown_fallback=DEFAULT_UNKNOWN_FALLBACK,
):
    """构建 system prompt。"""
    lines = [
        "你是一个检索增强问答助手。",
        "回答需要简洁、准确、可追踪。",
    ]

    if strict_mode:
        lines.append("只能基于提供的参考资料回答，不得补充资料外结论。")
    if unknown_fallback:
        lines.append(
            f"若证据不足或证据冲突，先给出结论“{unknown_fallback}”，"
            "再用一句话说明不足或冲突点，并标注对应 doc_id。"
        )
    if require_citation:
        lines.append("回答结尾必须标注引用 doc_id，例如：[doc_id=doc_0001]。")

    extra_rules = [line.strip() for line in str(answer_rules or "").splitlines() if line.strip()]
    if extra_rules:
        lines.append("附加规则：")
        lines.extend([f"- {rule}" for rule in extra_rules])

    return "\n".join(lines)


def build_user_prompt(question, context_block, require_citation=True):
    """构建 user prompt。"""
    question_text = str(question or "").strip()
    if not question_text:
        raise ValueError("question must not be empty")

    context_text = str(context_block or "").strip() or "(无可用参考资料)"
    lines = [
        f"问题：{question_text}",
        "",
        "参考资料：",
        context_text,
        "",
    ]

    if require_citation:
        lines.append("请基于参考资料作答，并在结尾标注引用的 doc_id。")
    else:
        lines.append("请基于参考资料作答。")

    return "\n".join(lines)


def should_answer_unknown(question, contexts, min_overlap_tokens=1):
    """
    判断是否应触发“资料不足”提示。

    规则：
    - question 与所有 context 均缺少最小词元重叠时，返回 True。
    """
    if min_overlap_tokens <= 0:
        raise ValueError("min_overlap_tokens must be greater than 0")

    question_tokens = _signal_tokens(_tokenize(question))
    if not question_tokens:
        return True

    normalized_contexts = contexts
    if normalized_contexts and not isinstance(normalized_contexts[0], PromptContext):
        normalized_contexts = normalize_contexts(normalized_contexts)

    if not normalized_contexts:
        return True

    for context in normalized_contexts:
        context_tokens = _signal_tokens(_tokenize(context.content))
        if len(question_tokens & context_tokens) >= min_overlap_tokens:
            return False
    return True


def build_prompt(
    question,
    reranked_results,
    answer_rules="",
    strict_mode=True,
    require_citation=True,
    max_docs=DEFAULT_MAX_DOCS,
    max_chars_per_doc=DEFAULT_MAX_CHARS_PER_DOC,
    max_total_chars=DEFAULT_MAX_TOTAL_CHARS,
    min_overlap_tokens=1,
    unknown_fallback=DEFAULT_UNKNOWN_FALLBACK,
):
    """构建完整 Prompt 包（system + user + context）。"""
    context_result = build_context_block(
        reranked_results=reranked_results,
        max_docs=max_docs,
        max_chars_per_doc=max_chars_per_doc,
        max_total_chars=max_total_chars,
    )

    unknown_hint = should_answer_unknown(
        question=question,
        contexts=context_result.contexts,
        min_overlap_tokens=min_overlap_tokens,
    )

    system_prompt = build_system_prompt(
        answer_rules=answer_rules,
        strict_mode=strict_mode,
        require_citation=require_citation,
        unknown_fallback=unknown_fallback,
    )
    user_prompt = build_user_prompt(
        question=question,
        context_block=context_result.context_block,
        require_citation=require_citation,
    )

    if unknown_hint and unknown_fallback:
        user_prompt = (
            f"{user_prompt}\n\n"
            f"补充要求：若资料不足，请直接回答“{unknown_fallback}”，不要猜测。"
        )

    return PromptPackage(
        question=str(question).strip(),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        context_block=context_result.context_block,
        contexts=context_result.contexts,
        unknown_hint_triggered=unknown_hint,
        truncated=context_result.truncated,
        omitted_count=context_result.omitted_count,
    )


def _build_context_header(doc_id, metadata, include_source):
    """构建 context 行头，保留 doc_id 与 source。"""
    source = ""
    if include_source:
        source = str((metadata or {}).get("source", "")).strip()

    if source:
        return f"[doc_id={doc_id} source={source}]"
    return f"[doc_id={doc_id}]"


def _truncate_with_ellipsis(text, limit):
    """截断文本并在必要时补省略号。"""
    text = str(text or "")
    if limit <= 0:
        return "", bool(text)
    if len(text) <= limit:
        return text, False
    if limit <= 3:
        return text[:limit], True
    return text[: limit - 3].rstrip() + "...", True


def _compact_whitespace(text):
    """压缩多余空白，避免 prompt 噪声。"""
    return " ".join(str(text or "").split())


def _tokenize(text):
    """
    简化分词：
    - 英文/数字/下划线按词切分
    - 中文按单字切分（教学场景保底）
    """
    raw = str(text or "").strip().lower()
    if not raw:
        return []
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", raw)


def _signal_tokens(tokens):
    """
    过滤掉高频虚词，保留更有判别力的词元。
    若过滤后为空，则回退到原始词元集合。
    """
    raw = [str(token) for token in list(tokens or []) if str(token).strip()]
    filtered = [token for token in raw if token not in STOP_TOKENS]
    return set(filtered or raw)
