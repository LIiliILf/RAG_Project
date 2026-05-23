"""
BM25 实验脚本。

目标：
1. 构建 BM25 索引。
2. 用术语/编号类 query 演示关键词检索效果。
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    # 保持 experiments 脱离 src 后仍可导入核心模块。
    sys.path.append(str(SRC_DIR))

from bm25_index import BM25IndexManager


TOP_K = 3

DOCUMENTS = [
    "显卡型号 RTX4090 配备 24GB 显存，适合高分辨率推理任务。",
    "常见报错 ERR_CONN_RESET 通常与网络中断或代理配置有关。",
    "配置项 HYBRID_ALPHA 用于控制混合检索中语义分数的权重。",
    "RAG 系统会先检索资料，再结合上下文生成回答。",
    "BM25 属于关键词检索，FAISS 属于向量语义检索。",
]

QUERIES = [
    "RTX4090 显存是多少",
    "ERR_CONN_RESET 怎么排查",
    "HYBRID_ALPHA 是什么",
    "RAG 为什么先检索再回答",
    "纯关键词检索用什么",
]


def main():
    """执行 BM25 检索实验。"""
    doc_ids = [f"doc_{index:04d}" for index in range(1, len(DOCUMENTS) + 1)]

    manager = BM25IndexManager()
    manager.build_index(DOCUMENTS, doc_ids)

    print("documents 数量:", len(DOCUMENTS))
    print("top_k:", TOP_K)
    print("后端实现:", "rank_bm25" if manager._backend == "rank_bm25" else "simple-fallback")
    print()

    for query in QUERIES:
        results = manager.search(query, top_k=TOP_K)
        print("=" * 80)
        print("query:", query)
        print("Top 3:")

        if not results:
            print("无命中结果")
            continue

        for result in results:
            print(f"{result.rank}. score={result.score:.4f}")
            print("   doc_id:", result.doc_id)
            print("   content:", result.content)


if __name__ == "__main__":
    main()


# 运行结果示例（2026-05-23，Windows PowerShell）
# documents 数量: 5
# top_k: 3
# 后端实现: simple-fallback
#
# query: RTX4090 显存是多少
# Top 1: doc_0001 | score=4.6559
#
# query: ERR_CONN_RESET 怎么排查
# Top 1: doc_0002 | score=1.3863
#
# query: HYBRID_ALPHA 是什么
# Top 1: doc_0003 | score=1.3542
#
# query: RAG 为什么先检索再回答
# Top 1: doc_0004 | score=8.0095
#
# query: 纯关键词检索用什么
# Top 1: doc_0005 | score=5.4234
