from pathlib import Path

import numpy as np

from embeddings import cosine_similarity, encode_query, encode_texts


def read_numbered_section(markdown_text, heading):
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
    project_root = Path(__file__).resolve().parents[1]
    data_file = project_root / "test_files" / "embedding_sentences.md"

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

        ranked_indices = np.argsort(scores)[::-1][:3]
        print("Top 3:")
        for rank, index in enumerate(ranked_indices, start=1):
            print(f"{rank}. {round(float(scores[index]), 4)} | {documents[index]}")


if __name__ == "__main__":
    main()
