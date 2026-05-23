"""
embedding 实验脚本。

读取测试 Markdown 中的 `documents` / `queries`，
输出 query 与 document 的相似度排序结果。
"""

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    # 保持 experiments 脱离 src 后仍可导入核心模块。
    sys.path.append(str(SRC_DIR))

from embeddings import cosine_similarity, encode_query, encode_texts


def read_numbered_section(markdown_text, heading):
    """读取形如 `1. xxx` 的小节条目列表。"""
    lines = markdown_text.splitlines()
    in_section = False
    items = []

    for line in lines:
        stripped = line.strip()
        if stripped == f"## {heading}":
            in_section = True
            continue
        if in_section and stripped.startswith("## "):
            break
        if in_section and ". " in stripped:
            prefix, value = stripped.split(". ", 1)
            if prefix.isdigit() and value:
                items.append(value)

    return items


def main():
    """执行 embedding 相似度实验。"""
    data_file = PROJECT_ROOT / "test_files" / "embedding_sentences.md"

    markdown_text = data_file.read_text(encoding="utf-8")
    documents = read_numbered_section(markdown_text, "documents")
    queries = read_numbered_section(markdown_text, "queries")

    if not documents:
        raise RuntimeError("没有读取到 documents")
    if not queries:
        raise RuntimeError("没有读取到 queries")

    print("测试文件:", data_file)
    print("documents 数量:", len(documents))
    print("queries 数量:", len(queries))
    print()

    document_embeddings = encode_texts(documents)
    print("document_embeddings shape:", document_embeddings.shape)
    print("document_embeddings dtype:", document_embeddings.dtype)
    print()

    for query in queries:
        query_embedding = encode_query(query)
        scores = cosine_similarity(query_embedding, document_embeddings)
        best_index = int(np.argmax(scores))

        print("=" * 80)
        print("query:", query)
        print("query_embedding shape:", query_embedding.shape)
        print("最相似 document:", documents[best_index])
        print("相似度:", round(float(scores[best_index]), 4))

        # 从高到低取前 3 个分数。
        ranked_indices = np.argsort(scores)[::-1][:3]
        print("Top 3:")
        for rank, index in enumerate(ranked_indices, start=1):
            print(f"{rank}. {round(float(scores[index]), 4)} | {documents[index]}")


if __name__ == "__main__":
    main()


# 运行结果示例（2026-05-23，Windows PowerShell）
# 测试文件: D:\Projects\Codex\Projects\RAG_Project\test_files\embedding_sentences.md
# documents 数量: 8
# queries 数量: 6
# document_embeddings shape: (8, 512)
# document_embeddings dtype: float32
#
# query: 如何把文件内容变成可以处理的文本？
# 最相似 document: 文档解析负责把 PDF、Word、Markdown 和 TXT 文件转换成纯文本。
# 相似度: 0.6503
#
# query: 怎么把一篇很长的文档拆成小段？
# 最相似 document: 文本分块会把长文本切成适合检索的小片段。
# 相似度: 0.6649
#
# query: 怎样把句子变成向量？
# 最相似 document: Embedding 可以把文本转换成向量，让机器计算语义相似度。
# 相似度: 0.5709
#
# query: 哪个组件负责保存向量并做相似度搜索？
# 最相似 document: FAISS 可以把文档向量存起来，并支持本地相似度检索。
# 相似度: 0.6835
#
# query: RAG 为什么要先查资料再回答？
# 最相似 document: RAG 系统会先检索资料，再让大模型基于上下文回答问题。
# 相似度: 0.7276
#
# query: 今天适合出门走走吗？
# 最相似 document: 今天天气很好，适合去公园散步。
# 相似度: 0.6910
