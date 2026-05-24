"""
多文档问答链路模块。

职责：
1. 发现并加载多个本地文档。
2. 将文档切分为带来源信息的 chunks。
3. 构建混合检索索引并执行检索 + 重排。
"""

from dataclasses import dataclass
from pathlib import Path

from document_loader import extract_text
from embeddings import encode_query, encode_texts
from hybrid_retriever import HybridRetriever
from reranker import rerank_results
from text_splitter import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, split_text


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".docx",
    ".xlsx",
    ".xls",
    ".pptx",
}


@dataclass(frozen=True)
class DocumentRecord:
    """文档记录。"""

    doc_id: str
    source: str
    filepath: str
    text: str


@dataclass(frozen=True)
class ChunkRecord:
    """分块记录。"""

    chunk_id: str
    doc_id: str
    source: str
    filepath: str
    chunk_index: int
    content: str

    @property
    def metadata(self):
        """给检索/重排/Prompt 使用的来源元信息。"""
        return {
            "source": self.source,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "filepath": self.filepath,
        }


def discover_source_files(input_paths, allowed_extensions=None):
    """
    从文件/目录输入中发现可处理文件（去重 + 稳定排序）。
    """
    if not input_paths:
        raise ValueError("input_paths must not be empty")

    allowed = {
        ext.lower() for ext in (allowed_extensions or SUPPORTED_EXTENSIONS)
    }
    discovered = {}

    for raw_path in input_paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"输入路径不存在: {path}")

        if path.is_file():
            _add_file_if_supported(path, allowed, discovered)
            continue

        for child in sorted(path.rglob("*"), key=lambda p: p.as_posix().lower()):
            if child.is_file():
                _add_file_if_supported(child, allowed, discovered)

    files = [Path(raw) for raw in sorted(discovered)]
    if not files:
        raise ValueError("未找到可处理文件，请检查输入目录和文件后缀")
    return files


def load_documents_from_files(filepaths, text_extractor=extract_text):
    """
    将文件列表加载为 DocumentRecord。

    Returns:
        (documents, skipped)
        documents: list[DocumentRecord]
        skipped: list[dict]，记录空文本或解析失败文件
    """
    documents = []
    skipped = []

    for filepath in [Path(path) for path in filepaths]:
        raw_text = text_extractor(str(filepath))
        text = str(raw_text or "").strip()
        if not text:
            skipped.append({"filepath": str(filepath), "reason": "empty_or_unsupported"})
            continue

        doc_id = f"doc_{len(documents) + 1:04d}"
        documents.append(
            DocumentRecord(
                doc_id=doc_id,
                source=filepath.name,
                filepath=str(filepath.resolve(strict=False)),
                text=text,
            )
        )

    if not documents:
        raise ValueError("所有输入文档都为空或无法解析")
    return documents, skipped


def build_chunk_records(
    documents,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
):
    """将文档切分为 chunks 并补齐来源信息。"""
    chunk_records = []
    for document in list(documents):
        chunks = split_text(
            text=document.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for index, chunk in enumerate(chunks, start=1):
            chunk_id = f"{document.doc_id}_chunk_{index:04d}"
            chunk_records.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    source=document.source,
                    filepath=document.filepath,
                    chunk_index=index - 1,
                    content=chunk,
                )
            )

    if not chunk_records:
        raise ValueError("未生成有效 chunk，请检查文档内容和分块参数")
    return chunk_records


def build_hybrid_retriever_from_chunks(chunk_records, alpha=0.7):
    """基于 chunks 构建混合检索器。"""
    chunk_records = list(chunk_records)
    documents = [item.content for item in chunk_records]
    doc_ids = [item.chunk_id for item in chunk_records]
    metadatas = [item.metadata for item in chunk_records]
    embeddings = encode_texts(documents)

    retriever = HybridRetriever(alpha=alpha)
    retriever.build_index(documents, doc_ids, metadatas, embeddings)
    return retriever


def retrieve_and_rerank(
    query,
    retriever,
    retrieval_top_k=8,
    rerank_top_k=4,
    rerank_method="overlap",
    cross_encoder_model="",
):
    """
    执行检索 + 重排，返回两段中间结果。
    """
    if retrieval_top_k <= 0:
        raise ValueError("retrieval_top_k must be greater than 0")
    if rerank_top_k <= 0:
        raise ValueError("rerank_top_k must be greater than 0")

    retrieved = retriever.search(
        query=query,
        query_embedding=encode_query(query),
        top_k=retrieval_top_k,
    )

    docs = [item.content for item in retrieved]
    doc_ids = [item.doc_id for item in retrieved]
    metadata_list = [item.metadata for item in retrieved]

    reranked = rerank_results(
        query=query,
        docs=docs,
        doc_ids=doc_ids,
        metadata_list=metadata_list,
        method=rerank_method,
        top_k=rerank_top_k,
        cross_encoder_model=(cross_encoder_model or None),
    )
    return retrieved, reranked


def _add_file_if_supported(path, allowed_extensions, target):
    """将符合后缀的文件加入目标集合。"""
    suffix = path.suffix.lower()
    if suffix not in allowed_extensions:
        return
    resolved = str(path.resolve(strict=False))
    target[resolved] = True
