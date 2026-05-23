"""
2.文本分块模块。

策略：
1. 以 `chunk_size` 控制每段最大长度。
2. 以 `chunk_overlap` 保留相邻段上下文。
3. 优先按分隔符切分，最后再回退到硬切。
"""

DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 40
DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "，", "；", "：", " ", ""]


def split_text(
    text,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    separators=None,
):
    """
    将长文本切成多个 chunk。

    Args:
        text: 待切分的纯文本。
        chunk_size: 每个 chunk 的最大字符数。
        chunk_overlap: 相邻 chunk 的重叠字符数。
        separators: 优先使用的分隔符列表。

    Returns:
        chunks 列表。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap 不能小于 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    if not text or not text.strip():
        return []

    separators = DEFAULT_SEPARATORS if separators is None else separators
    text = text.strip()
    chunks = []
    start = 0

    while start < len(text):
        max_end = min(start + chunk_size, len(text))
        # 优先在窗口内找最近分隔符，避免切断句子。
        end = _find_split_position(text, start, max_end, separators, chunk_size)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        # 通过 overlap 保留边界上下文。
        next_start = end - chunk_overlap
        if next_start <= start:
            # 防止异常参数导致死循环。
            next_start = end
        start = next_start

    return chunks


def _find_split_position(text, start, max_end, separators, chunk_size):
    """在 [start, max_end) 范围内选择最合适的切分点。"""
    if max_end >= len(text):
        return len(text)

    window = text[start:max_end]
    # 至少要保留窗口 35% 的有效内容，避免切得过碎。
    min_break = max(1, int(chunk_size * 0.35))

    for separator in separators:
        if separator == "":
            continue

        position = window.rfind(separator)
        if position >= min_break:
            return start + position + len(separator)

    # 找不到分隔符时退化为硬切。
    return max_end
