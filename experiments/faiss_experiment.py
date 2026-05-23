"""
第 7 节 FAISS 检索实验脚本。

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
