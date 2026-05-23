"""
FAISS 检索实验脚本。

流程：
documents -> embeddings -> build FAISS index -> query embedding -> top_k 检索。
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(EXPERIMENTS_DIR) not in sys.path:
    # 允许该脚本在被外部加载时仍能导入同目录的 embedding_experiment。
    sys.path.append(str(EXPERIMENTS_DIR))
if str(SRC_DIR) not in sys.path:
    # 保持 experiments 脱离 src 后仍可导入核心模块。
    sys.path.append(str(SRC_DIR))

from embeddings import EMBED_MODEL_NAME, encode_query, encode_texts
from embedding_experiment import read_numbered_section
from vector_store import FaissVectorStore


TOP_K = 3


def main():
    """执行 FAISS 本地语义检索实验。"""
    data_file = PROJECT_ROOT / "test_files" / "embedding_sentences.md"

    markdown_text = data_file.read_text(encoding="utf-8")
    documents = read_numbered_section(markdown_text, "documents")
    queries = read_numbered_section(markdown_text, "queries")

    if not documents:
        raise RuntimeError("没有读取到 documents")
    if not queries:
        raise RuntimeError("没有读取到 queries")

    # 向量位置通过顺序映射回 chunk_id，再映射到文本和 metadata。
    chunk_ids = [f"chunk_{index:04d}" for index in range(1, len(documents) + 1)]
    metadatas = [
        {
            "source": data_file.name,
            "chunk_index": index,
        }
        for index in range(len(documents))
    ]

    print("测试文件:", data_file)
    print("embedding 模型:", EMBED_MODEL_NAME)
    print("documents 数量:", len(documents))
    print("top_k:", TOP_K)
    print()

    document_embeddings = encode_texts(documents)
    print("document_embeddings shape:", document_embeddings.shape)
    print("document_embeddings dtype:", document_embeddings.dtype)

    vector_store = FaissVectorStore()
    vector_store.build_index(documents, chunk_ids, metadatas, document_embeddings)
    print("FAISS index 类型: IndexFlatL2")
    print("FAISS index chunk 数量:", vector_store.total_chunks)
    print("FAISS index 维度:", vector_store.dimension)
    print("distance 说明: 使用 L2 距离，越小越相似")
    print()

    for query in queries:
        query_embedding = encode_query(query)
        results = vector_store.search(query_embedding, top_k=TOP_K)

        print("=" * 80)
        print("query:", query)
        print("query_embedding shape:", query_embedding.shape)
        print("Top 3:")
        for result in results:
            print(f"{result.rank}. distance={result.distance:.4f}")
            print("   chunk_id:", result.chunk_id)
            print("   content:", result.content)
            print("   metadata:", result.metadata)


if __name__ == "__main__":
    main()


# 运行结果示例（2026-05-23，Windows PowerShell）
# 测试文件: D:\Projects\Codex\Projects\RAG_Project\test_files\embedding_sentences.md
# embedding 模型: D:\Projects\Codex\Projects\RAG_Project\models\bge-small-zh-v1.5
# documents 数量: 8
# top_k: 3
# document_embeddings shape: (8, 512)
# document_embeddings dtype: float32
# FAISS index 类型: IndexFlatL2
# FAISS index chunk 数量: 8
# FAISS index 维度: 512
# distance 说明: 使用 L2 距离，越小越相似
#
# query: 如何把文件内容变成可以处理的文本？
# Top 1: chunk_0001 | distance=0.6995
#
# query: 怎么把一篇很长的文档拆成小段？
# Top 1: chunk_0002 | distance=0.6703
#
# query: 怎样把句子变成向量？
# Top 1: chunk_0003 | distance=0.8583
#
# query: 哪个组件负责保存向量并做相似度搜索？
# Top 1: chunk_0004 | distance=0.6330
#
# query: RAG 为什么要先查资料再回答？
# Top 1: chunk_0006 | distance=0.5449
#
# query: 今天适合出门走走吗？
# Top 1: chunk_0007 | distance=0.6179

# 测试文件: RAG_Project\test_files\embedding_sentences.md
# embedding 模型: D:\Projects\Codex\Projects\RAG_Project\models\bge-small-zh-v1.5
# documents 数量: 8
# top_k: 3
#
# Loading weights: 100%|██████████| 71/71 [00:00<00:00, 5461.03it/s]
# document_embeddings shape: (8, 512)
# document_embeddings dtype: float32
# FAISS index 类型: IndexFlatL2
# FAISS index chunk 数量: 8
# FAISS index 维度: 512
# distance 说明: 使用 L2 距离，越小越相似
#
# ================================================================================
# query: 如何把文件内容变成可以处理的文本？
# query_embedding shape: (1, 512)
# Top 3:
# 1. distance=0.6995
#    chunk_id: chunk_0001
#    content: 文档解析负责把 PDF、Word、Markdown 和 TXT 文件转换成纯文本。
#    metadata: {'source': 'embedding_sentences.md', 'chunk_index': 0}
# 2. distance=0.7952
#    chunk_id: chunk_0002
#    content: 文本分块会把长文本切成适合检索的小片段。
#    metadata: {'source': 'embedding_sentences.md', 'chunk_index': 1}
# 3. distance=0.9409
#    chunk_id: chunk_0003
#    content: Embedding 可以把文本转换成向量，让机器计算语义相似度。
#    metadata: {'source': 'embedding_sentences.md', 'chunk_index': 2}
# ================================================================================
# query: 怎么把一篇很长的文档拆成小段？
# query_embedding shape: (1, 512)
# Top 3:
# 1. distance=0.6703
#    chunk_id: chunk_0002
#    content: 文本分块会把长文本切成适合检索的小片段。
#    metadata: {'source': 'embedding_sentences.md', 'chunk_index': 1}
# 2. distance=0.9157
#    chunk_id: chunk_0001
#    content: 文档解析负责把 PDF、Word、Markdown 和 TXT 文件转换成纯文本。
#    metadata: {'source': 'embedding_sentences.md', 'chunk_index': 0}
# 3. distance=1.1023
#    chunk_id: chunk_0004
#    content: FAISS 可以把文档向量存起来，并支持本地相似度检索。
#    metadata: {'source': 'embedding_sentences.md', 'chunk_index': 3}
# ================================================================================
# query: 怎样把句子变成向量？
# query_embedding shape: (1, 512)
# Top 3:
# 1. distance=0.8583
#    chunk_id: chunk_0003
#    content: Embedding 可以把文本转换成向量，让机器计算语义相似度。
#    metadata: {'source': 'embedding_sentences.md', 'chunk_index': 2}
# 2. distance=1.0278
#    chunk_id: chunk_0004
#    content: FAISS 可以把文档向量存起来，并支持本地相似度检索。
#    metadata: {'source': 'embedding_sentences.md', 'chunk_index': 3}
# 3. distance=1.1782
#    chunk_id: chunk_0006
#    content: RAG 系统会先检索资料，再让大模型基于上下文回答问题。
#    metadata: {'source': 'embedding_sentences.md', 'chunk_index': 5}
# ================================================================================
# query: 哪个组件负责保存向量并做相似度搜索？
# query_embedding shape: (1, 512)
# Top 3:
# 1. distance=0.6330
#    chunk_id: chunk_0004
#    content: FAISS 可以把文档向量存起来，并支持本地相似度检索。
#    metadata: {'source': 'embedding_sentences.md', 'chunk_index': 3}
# 2. distance=0.8399
#    chunk_id: chunk_0003
#    content: Embedding 可以把文本转换成向量，让机器计算语义相似度。
#    metadata: {'source': 'embedding_sentences.md', 'chunk_index': 2}
# 3. distance=0.9977
#    chunk_id: chunk_0005
#    content: Prompt 构建会把用户问题和检索到的上下文组合起来。
#    metadata: {'source': 'embedding_sentences.md', 'chunk_index': 4}
# ================================================================================
# query: RAG 为什么要先查资料再回答？
# query_embedding shape: (1, 512)
# Top 3:
# 1. distance=0.5449
#    chunk_id: chunk_0006
#    content: RAG 系统会先检索资料，再让大模型基于上下文回答问题。
#    metadata: {'source': 'embedding_sentences.md', 'chunk_index': 5}
# 2. distance=1.0914
#    chunk_id: chunk_0004
#    content: FAISS 可以把文档向量存起来，并支持本地相似度检索。
#    metadata: {'source': 'embedding_sentences.md', 'chunk_index': 3}
# 3. distance=1.1141
#    chunk_id: chunk_0005
#    content: Prompt 构建会把用户问题和检索到的上下文组合起来。
#    metadata: {'source': 'embedding_sentences.md', 'chunk_index': 4}
# ================================================================================
# query: 今天适合出门走走吗？
# query_embedding shape: (1, 512)
# Top 3:
# 1. distance=0.6179
#    chunk_id: chunk_0007
#    content: 今天天气很好，适合去公园散步。
#    metadata: {'source': 'embedding_sentences.md', 'chunk_index': 6}
# 2. distance=1.2792
#    chunk_id: chunk_0008
#    content: 晚餐可以选择米饭、青菜和鸡蛋。
#    metadata: {'source': 'embedding_sentences.md', 'chunk_index': 7}
# 3. distance=1.4223
#    chunk_id: chunk_0002
#    content: 文本分块会把长文本切成适合检索的小片段。
#    metadata: {'source': 'embedding_sentences.md', 'chunk_index': 1}