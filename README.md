# RAG_Project

目标是从本地文档解析开始，逐步搭出可控、可测试的 RAG 链路。

## 当前进度（2026-05-24）

已完成到（完整 RAG 链路），当前已落地模块：

```text
文档解析 -> 文本分块 -> 向量化 -> FAISS -> BM25 -> 混合检索 -> 重排序 -> Prompt 构建 -> 大模型调用
```

当前状态说明：

- `src/main.py` 仍是“文档解析入口”脚本。
- 检索、重排、Prompt 主要通过 `experiments/` 脚本验证。
- `cross-encoder` 手动下载脚本已归档到 `experiments/archive/`。

## 目录结构

```text
RAG_Project/
├── README.md
├── src/
│   ├── document_loader.py
│   ├── text_splitter.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── bm25_index.py
│   ├── hybrid_retriever.py
│   ├── reranker.py
│   ├── prompt_builder.py
│   ├── generator.py
│   ├── multi_doc_pipeline.py
│   └── main.py
├── experiments/
│   ├── chunk_experiment.py
│   ├── embedding_experiment.py
│   ├── faiss_experiment.py
│   ├── bm25_experiment.py
│   ├── hybrid_experiment.py
│   ├── retrieval_compare.py
│   ├── reranker_experiment.py
│   ├── prompt_experiment.py
│   ├── qa_pipeline_experiment.py
│   ├── multi_doc_qa_experiment.py
│   ├── chunk实验.md
│   ├── hybrid检索实验.md
│   ├── rerank实验.md
│   └── archive/
│       └── download_cross_encoder.py
├── tests/
│   ├── test_text_splitter.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_bm25_index.py
│   ├── test_hybrid_retriever.py
│   ├── test_reranker.py
│   ├── test_prompt_builder.py
│   ├── test_generator.py
│   ├── test_multi_doc_pipeline.py
│   └── test_retrieval_compare.py
└── test_files/
    ├── sample_rag.txt
    ├── sample_rag.md
    ├── sample_rag.docx
    ├── sample_rag.pdf
    ├── embedding_sentences.md
    ├── empty.txt
    └── unsupported_sample.xyz
```

## 安装依赖

```powershell
python -m pip install numpy faiss-cpu rank-bm25 sentence-transformers python-docx pdfminer.six
```

国内网络不稳定可切镜像（按需）：

```powershell
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn numpy faiss-cpu rank-bm25 sentence-transformers python-docx pdfminer.six
```

## 常用运行命令

1) 文档解析检查：

```powershell
python src\main.py
```

2) BM25 vs FAISS 对比（低内存可先 bm25-only）：

```powershell
python experiments\retrieval_compare.py --top-k 3
python experiments\retrieval_compare.py --bm25-only --top-k 3
```

3) Reranker 实验（推荐先 toy）：

```powershell
python experiments\reranker_experiment.py --mode toy --methods none,cross_encoder --cross-encoder-model-key bge_base_local --rerank-top-k 3
```

4) Prompt A/B 实验：

```powershell
python experiments\prompt_experiment.py --max-docs 3 --max-chars-per-doc 180 --max-total-chars 520
```

5) 运行Prompt构建：

```powershell
python -m unittest tests\test_prompt_builder.py
```

6) RAG完整链路（先离线看 Prompt）：

```powershell
python experiments\qa_pipeline_experiment.py --mode fixed --query "RTX4090 显存是多少？" --print-prompt-only
```

调用 LLM（需先在本机环境变量中配置 API Key）：

```powershell
python experiments\qa_pipeline_experiment.py --mode fixed --query "RTX4090 显存是多少？" --llm-model deepseek-chat
```

7) 多文档来源追踪（先离线看 Prompt）：

```powershell
python experiments\multi_doc_qa_experiment.py --query "RAG 的流程可以分为哪些步骤？" --print-prompt-only
```

调用 LLM（需先在本机环境变量中配置 API Key）：

```powershell
python experiments\multi_doc_qa_experiment.py --query "RAG 的流程可以分为哪些步骤？" --llm-model deepseek-chat
```

8) 启动 Gradio 界面（代理环境推荐）：

```powershell
$env:NO_PROXY="localhost,127.0.0.1,::1"
$env:no_proxy="localhost,127.0.0.1,::1"
python experiments\gradio_app_experiment.py
```

## 模块说明（src）

- `document_loader.py`：读取 txt/md/docx/pdf 等文件并抽取文本。
- `text_splitter.py`：把长文本切成可检索的 chunk。
- `embeddings.py`：加载 embedding 模型并编码文档/查询。
- `vector_store.py`：FAISS 向量索引构建、检索、持久化接口。
- `bm25_index.py`：关键词检索索引构建与查询。
- `hybrid_retriever.py`：融合 BM25 与向量检索分数。
- `reranker.py`：对候选做 `none / overlap / cross_encoder` 重排。
- `prompt_builder.py`：构建 context block、system prompt、user prompt。
- `generator.py`：封装 OpenAI 兼容调用，并串联 Prompt 到模型回答。
- `multi_doc_pipeline.py`：多文档发现、分块、统一索引、来源信息透传。
- `main.py`：当前文档解析阶段的演示入口。

## TODO（下一阶段）

- [ ] 多文档问答与来源追踪（source/doc_id/chunk_id，进行中）。
- [ ] 本地界面（Gradio）与交互流程。
- [ ] 质量评估（命中率、延迟、可追踪性）。
- [ ] 项目复盘与发布整理。
