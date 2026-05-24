"""
Gradio 界面编排模块。

职责：
1. 提供“上传文档 -> 建索引 -> 提问”的本地 UI 编排。
2. 复用已有多文档链路（multi_doc_pipeline）与生成链路（generator）。
3. 输出可追踪证据（召回/重排）与会话状态，不重写核心算法。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from generator import DEFAULT_CHAT_MODEL, generate_answer_with_rag, resolve_base_url
from multi_doc_pipeline import (
    build_chunk_records,
    build_hybrid_retriever_from_chunks,
    discover_source_files,
    load_documents_from_files,
    retrieve_and_rerank,
)


DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 40
DEFAULT_HYBRID_ALPHA = 0.7
DEFAULT_RETRIEVAL_TOP_K = 8
DEFAULT_RERANK_TOP_K = 4
DEFAULT_UI_RERANK_METHOD = "overlap"
DEFAULT_ANSWER_RULES = "优先引用最相关证据；若证据不足，回答“根据当前资料无法确定”。"
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.2


@dataclass
class PipelineSessionState:
    """UI 会话状态：保存已构建的索引和统计信息。"""

    retriever: Any
    documents: list
    chunks: list
    skipped: list
    source_files: list[str]
    chunk_size: int
    chunk_overlap: int
    hybrid_alpha: float

    @property
    def doc_count(self):
        return len(self.documents)

    @property
    def chunk_count(self):
        return len(self.chunks)


def normalize_uploaded_files(uploaded_files):
    """
    兼容 Gradio File 组件返回值：
    - str 路径
    - pathlib.Path
    - 含 .name 的文件对象
    """
    if not uploaded_files:
        return []

    normalized = []
    items = uploaded_files if isinstance(uploaded_files, (list, tuple)) else [uploaded_files]
    for item in items:
        if isinstance(item, Path):
            normalized.append(str(item))
            continue
        if isinstance(item, str):
            normalized.append(item)
            continue
        name = getattr(item, "name", None)
        if name:
            normalized.append(str(name))
            continue
        raise ValueError(f"无法识别上传文件对象: {type(item)}")
    return normalized


def validate_build_options(chunk_size, chunk_overlap, hybrid_alpha):
    """校验“上传并建索引”参数。"""
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if int(chunk_overlap) < 0:
        raise ValueError("chunk_overlap 不能小于 0")
    if int(chunk_overlap) >= int(chunk_size):
        raise ValueError("chunk_overlap 必须小于 chunk_size")
    alpha = float(hybrid_alpha)
    if alpha < 0 or alpha > 1:
        raise ValueError("hybrid_alpha 必须在 [0, 1] 区间内")


def validate_query_options(retrieval_top_k, rerank_top_k):
    """校验问答阶段参数。"""
    if int(retrieval_top_k) <= 0:
        raise ValueError("retrieval_top_k 必须大于 0")
    if int(rerank_top_k) <= 0:
        raise ValueError("rerank_top_k 必须大于 0")


def summarize_session(session):
    """把会话状态转换为可读摘要。"""
    return (
        f"文档数: {session.doc_count} | "
        f"chunk 数: {session.chunk_count} | "
        f"跳过文件: {len(session.skipped)} | "
        f"chunk_size: {session.chunk_size} | "
        f"chunk_overlap: {session.chunk_overlap} | "
        f"hybrid_alpha: {session.hybrid_alpha}"
    )


def build_chunk_table_rows(chunks, preview_chars=120):
    """构建 chunk 可视化表格行。"""
    rows = []
    for item in list(chunks):
        content = str(getattr(item, "content", "") or "")
        preview = content if len(content) <= preview_chars else content[:preview_chars] + "..."
        rows.append(
            [
                str(getattr(item, "doc_id", "-")),
                str(getattr(item, "chunk_id", "-")),
                str(getattr(item, "source", "-")),
                len(content),
                preview,
            ]
        )
    return rows


def format_candidates_markdown(title, items, score_field):
    """格式化候选列表，展示来源和分数。"""
    if not items:
        return f"### {title}\n(无)"

    lines = [f"### {title}"]
    for index, item in enumerate(list(items), start=1):
        metadata = dict(getattr(item, "metadata", {}) or {})
        source = metadata.get("source", "-")
        doc_id = metadata.get("doc_id", "-")
        chunk_id = metadata.get("chunk_id", "-")
        score = getattr(item, score_field, None)
        if score is None and score_field != "score":
            score = getattr(item, "score", None)
        score_text = "-" if score is None else f"{float(score):.4f}"
        content = str(getattr(item, "content", "") or "")
        lines.append(
            f"{index}. doc={doc_id} chunk={chunk_id} source={source} score={score_text}\n"
            f"   {content}"
        )
    return "\n".join(lines)


def process_uploaded_files(
    uploaded_files,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    hybrid_alpha=DEFAULT_HYBRID_ALPHA,
):
    """
    处理上传文件并构建检索索引。

    Returns:
        (session_state, status_text, chunk_rows, session_summary)
    """
    try:
        validate_build_options(chunk_size, chunk_overlap, hybrid_alpha)
        filepaths = normalize_uploaded_files(uploaded_files)
        if not filepaths:
            raise ValueError("请先上传至少一个文件")

        source_files = discover_source_files(filepaths)
        documents, skipped = load_documents_from_files(source_files)
        chunks = build_chunk_records(
            documents=documents,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
        )
        retriever = build_hybrid_retriever_from_chunks(chunks, alpha=float(hybrid_alpha))
        session = PipelineSessionState(
            retriever=retriever,
            documents=list(documents),
            chunks=list(chunks),
            skipped=list(skipped),
            source_files=[str(path) for path in source_files],
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            hybrid_alpha=float(hybrid_alpha),
        )

        lines = [
            "处理成功",
            f"文档数: {session.doc_count}",
            f"chunk 数: {session.chunk_count}",
            f"跳过文件: {len(session.skipped)}",
        ]
        if session.skipped:
            for item in session.skipped:
                lines.append(f"- skipped: {item['filepath']} | {item['reason']}")

        return session, "\n".join(lines), build_chunk_table_rows(session.chunks), summarize_session(session)
    except Exception as exc:
        return None, f"处理失败: {exc}", [], "状态: 未就绪"


def run_qa_turn(
    session,
    question,
    chat_history,
    retrieval_top_k=DEFAULT_RETRIEVAL_TOP_K,
    rerank_top_k=DEFAULT_RERANK_TOP_K,
    rerank_method=DEFAULT_UI_RERANK_METHOD,
    cross_encoder_model="",
    llm_model=DEFAULT_CHAT_MODEL,
    base_url="",
    answer_rules=DEFAULT_ANSWER_RULES,
    strict_mode=True,
    require_citation=True,
    temperature=DEFAULT_TEMPERATURE,
    max_tokens=DEFAULT_MAX_TOKENS,
):
    """
    执行单轮问答，返回更新后的聊天记录与候选展示。

    Returns:
        (chat_history, status_text, candidates_markdown)
    """
    history = list(chat_history or [])
    query = str(question or "").strip()

    if not query:
        return history, "请输入问题后再提问", "### 候选证据\n(尚未提问)"
    if session is None:
        return history, "请先上传并处理文档，再开始提问", "### 候选证据\n(索引未就绪)"

    try:
        validate_query_options(retrieval_top_k, rerank_top_k)
        retrieved, reranked = retrieve_and_rerank(
            query=query,
            retriever=session.retriever,
            retrieval_top_k=int(retrieval_top_k),
            rerank_top_k=int(rerank_top_k),
            rerank_method=rerank_method,
            cross_encoder_model=str(cross_encoder_model or ""),
        )

        reranked_for_prompt = [
            {
                "doc_id": item.doc_id,
                "content": item.content,
                "metadata": dict(item.metadata),
                "score": item.score,
            }
            for item in reranked
        ]

        generation = generate_answer_with_rag(
            question=query,
            reranked_results=reranked_for_prompt,
            answer_rules=answer_rules,
            strict_mode=bool(strict_mode),
            require_citation=bool(require_citation),
            model=str(llm_model).strip() or DEFAULT_CHAT_MODEL,
            base_url=resolve_base_url(base_url),
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )

        usage_text = "-"
        if generation.usage:
            usage_text = (
                f"prompt={generation.usage.get('prompt_tokens')}, "
                f"completion={generation.usage.get('completion_tokens')}, "
                f"total={generation.usage.get('total_tokens')}"
            )
        answer_text = (
            f"{generation.answer}\n\n"
            f"[model={generation.model}] [finish_reason={generation.finish_reason}] [usage={usage_text}]"
        )
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer_text})

        candidates_markdown = "\n\n".join(
            [
                format_candidates_markdown("召回候选", retrieved, score_field="hybrid_score"),
                format_candidates_markdown("重排候选", reranked, score_field="score"),
            ]
        )
        status = (
            f"问答完成：召回 {len(retrieved)} 条，重排 {len(reranked)} 条，"
            f"unknown_hint_triggered={generation.unknown_hint_triggered}"
        )
        return history, status, candidates_markdown
    except Exception as exc:
        return history, f"问答失败: {exc}", "### 候选证据\n(本轮执行失败)"


def clear_chat_history():
    """清空聊天记录。"""
    return [], "对话已清空", "### 候选证据\n(已清空)"


def build_gradio_app():
    """构建 Gradio UI。"""
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:
        raise RuntimeError("缺少依赖 gradio，请先运行：python -m pip install gradio") from exc

    with gr.Blocks(title="RAG Project - 第13节 Gradio 界面") as app:
        gr.Markdown("# 第13节：本地 RAG 问答界面")
        gr.Markdown("上传文档后建立索引，再提问并查看召回/重排候选。")

        session_state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=5):
                files = gr.File(
                    label="上传文档（支持 PDF/TXT/MD/DOCX/XLSX/XLS/PPTX）",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".md", ".docx", ".xlsx", ".xls", ".pptx"],
                )
                with gr.Row():
                    chunk_size = gr.Number(label="chunk_size", value=DEFAULT_CHUNK_SIZE, precision=0)
                    chunk_overlap = gr.Number(label="chunk_overlap", value=DEFAULT_CHUNK_OVERLAP, precision=0)
                    hybrid_alpha = gr.Slider(label="hybrid_alpha", value=DEFAULT_HYBRID_ALPHA, minimum=0.0, maximum=1.0, step=0.05)
                process_button = gr.Button("处理文档并建立索引", variant="primary")
                process_status = gr.Textbox(label="处理状态", lines=6, interactive=False)
                session_summary = gr.Textbox(label="系统状态", lines=2, interactive=False, value="状态: 未就绪")

            with gr.Column(scale=7):
                chunk_table = gr.Dataframe(
                    headers=["doc_id", "chunk_id", "source", "chars", "preview"],
                    label="分块可视化（预览）",
                    interactive=False,
                    wrap=True,
                )

        gr.Markdown("## 提问区")
        with gr.Row():
            question = gr.Textbox(label="问题", placeholder="请输入问题...", lines=2, scale=6)
            ask_button = gr.Button("开始提问", variant="primary", scale=1)
            clear_button = gr.Button("清空对话", scale=1)

        with gr.Row():
            retrieval_top_k = gr.Number(label="retrieval_top_k", value=DEFAULT_RETRIEVAL_TOP_K, precision=0)
            rerank_top_k = gr.Number(label="rerank_top_k", value=DEFAULT_RERANK_TOP_K, precision=0)
            rerank_method = gr.Dropdown(
                label="rerank_method",
                choices=["none", "overlap", "cross_encoder"],
                value=DEFAULT_UI_RERANK_METHOD,
            )
            cross_encoder_model = gr.Textbox(label="cross_encoder_model（可选）", placeholder="本地路径或模型名")

        with gr.Row():
            llm_model = gr.Textbox(label="llm_model", value=DEFAULT_CHAT_MODEL)
            base_url = gr.Textbox(label="base_url（可选）", placeholder="留空则用环境变量或默认值")
            temperature = gr.Slider(label="temperature", minimum=0.0, maximum=1.0, step=0.05, value=DEFAULT_TEMPERATURE)
            max_tokens = gr.Number(label="max_tokens", value=DEFAULT_MAX_TOKENS, precision=0)

        answer_rules = gr.Textbox(label="answer_rules", lines=2, value=DEFAULT_ANSWER_RULES)
        with gr.Row():
            strict_mode = gr.Checkbox(label="strict_mode", value=True)
            require_citation = gr.Checkbox(label="require_citation", value=True)

        chat_history = gr.Chatbot(label="问答记录", height=420)
        qa_status = gr.Textbox(label="问答状态", lines=2, interactive=False)
        candidates_markdown = gr.Markdown("### 候选证据\n(尚未提问)")

        process_button.click(
            fn=process_uploaded_files,
            inputs=[files, chunk_size, chunk_overlap, hybrid_alpha],
            outputs=[session_state, process_status, chunk_table, session_summary],
        )
        ask_button.click(
            fn=run_qa_turn,
            inputs=[
                session_state,
                question,
                chat_history,
                retrieval_top_k,
                rerank_top_k,
                rerank_method,
                cross_encoder_model,
                llm_model,
                base_url,
                answer_rules,
                strict_mode,
                require_citation,
                temperature,
                max_tokens,
            ],
            outputs=[chat_history, qa_status, candidates_markdown],
        )
        question.submit(
            fn=run_qa_turn,
            inputs=[
                session_state,
                question,
                chat_history,
                retrieval_top_k,
                rerank_top_k,
                rerank_method,
                cross_encoder_model,
                llm_model,
                base_url,
                answer_rules,
                strict_mode,
                require_citation,
                temperature,
                max_tokens,
            ],
            outputs=[chat_history, qa_status, candidates_markdown],
        )
        clear_button.click(
            fn=clear_chat_history,
            outputs=[chat_history, qa_status, candidates_markdown],
        )

    return app


def launch_gradio_app(server_name="127.0.0.1", server_port=7860, share=False):
    """启动 Gradio 应用。"""
    app = build_gradio_app()
    app.launch(server_name=server_name, server_port=int(server_port), share=bool(share))
