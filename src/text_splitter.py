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
        end = _find_split_position(text, start, max_end, separators, chunk_size)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def _find_split_position(text, start, max_end, separators, chunk_size):
    if max_end >= len(text):
        return len(text)

    window = text[start:max_end]
    min_break = max(1, int(chunk_size * 0.35))

    for separator in separators:
        if separator == "":
            continue

        position = window.rfind(separator)
        if position >= min_break:
            return start + position + len(separator)

    return max_end
