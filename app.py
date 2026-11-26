"""Streamlit frontend for AI-Powered GitHub Repository Assistant."""

from __future__ import annotations

import inspect
from typing import Any

import pandas as pd
import streamlit as st

import config
from src.embeddings import EmbeddingError
from src.gemini_llm import (
    GeminiAPIError,
    GeminiCapacityError,
    GeminiConfigError,
    GeminiModelNotFoundError,
    GeminiQuotaError,
)
from src.github_loader import GitHubLoaderError, parse_github_url
from src.local_project_loader import LocalProjectLoaderError
from src.rag_pipeline import IndexedProject, RAGAnswer, RAGPipeline
from src.repository_processor import RepositoryProcessingError
from src.reranker import RerankerError
from src.retriever import RetrievalError
from src.vector_store import VectorStoreError
from utils.helpers import ensure_windows_dll_paths, format_file_size

ensure_windows_dll_paths()

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

INPUT_SOURCE_OPTIONS: tuple[str, ...] = (
    "GitHub Repository URL",
    "Upload ZIP",
    "Upload Project Folder",
)


def _inject_dark_theme_css() -> None:
    """Apply cohesive dark SaaS styling across sidebar, inputs, cards, and chat."""
    st.markdown(
        """
        <style>
            /* Global dark canvas */
            .stApp {
                background-color: #0b1120;
                color: #e5e7eb;
            }
            [data-testid="stHeader"] {
                background-color: rgba(11, 17, 32, 0.85);
                backdrop-filter: blur(8px);
            }
            [data-testid="stSidebar"] {
                background-color: #0f172a;
                border-right: 1px solid #1e293b;
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 3rem;
                max-width: 1120px;
            }

            /* Typography & Header */
            .saas-header {
                padding: 1.25rem 1.5rem;
                background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
                border: 1px solid #1e293b;
                border-radius: 12px;
                margin-bottom: 1.25rem;
            }
            .saas-title {
                margin: 0;
                font-size: 1.85rem;
                font-weight: 700;
                letter-spacing: -0.02em;
                color: #f8fafc;
            }
            .saas-subtitle {
                margin: 0.45rem 0 0 0;
                font-size: 0.96rem;
                line-height: 1.5;
                color: #94a3b8;
            }

            /* Sidebar brand & status pills */
            .sidebar-brand {
                padding: 0.9rem 1rem;
                background-color: #111827;
                border: 1px solid #1e293b;
                border-radius: 10px;
                margin-bottom: 1rem;
            }
            .sidebar-brand-title {
                font-size: 1rem;
                font-weight: 700;
                color: #f8fafc;
                margin: 0;
            }
            .sidebar-brand-sub {
                font-size: 0.8rem;
                color: #94a3b8;
                margin-top: 0.25rem;
            }
            .status-pill {
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                padding: 0.28rem 0.7rem;
                border-radius: 999px;
                font-size: 0.78rem;
                font-weight: 600;
                letter-spacing: 0.01em;
                margin-bottom: 0.5rem;
            }
            .pill-ok {
                background-color: rgba(16, 185, 129, 0.14);
                color: #34d399;
                border: 1px solid rgba(52, 211, 153, 0.35);
            }
            .pill-blue {
                background-color: rgba(59, 130, 246, 0.15);
                color: #60a5fa;
                border: 1px solid rgba(59, 130, 246, 0.35);
            }
            .pill-warn {
                background-color: rgba(245, 158, 11, 0.14);
                color: #fbbf24;
                border: 1px solid rgba(251, 191, 36, 0.35);
            }
            .pill-idle {
                background-color: rgba(148, 163, 184, 0.12);
                color: #94a3b8;
                border: 1px solid rgba(148, 163, 184, 0.28);
            }

            /* Bordered containers & cards */
            [data-testid="stVerticalBlockBorderWrapper"] {
                background-color: #111827;
                border: 1px solid #1e293b !important;
                border-radius: 12px !important;
                padding: 0.4rem 0.6rem;
            }
            .section-heading {
                font-size: 1.2rem;
                font-weight: 650;
                color: #f1f5f9;
                margin-bottom: 0.25rem;
            }
            .section-caption {
                font-size: 0.88rem;
                color: #94a3b8;
                margin-bottom: 0.85rem;
            }
            .info-strip {
                background-color: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 8px;
                padding: 0.65rem 0.9rem;
                font-size: 0.88rem;
                color: #cbd5e1;
                margin: 0.5rem 0 0.75rem 0;
            }

            /* Dark inputs & file uploader */
            [data-testid="stTextInput"] input,
            [data-testid="stTextArea"] textarea {
                background-color: #0f172a !important;
                color: #f1f5f9 !important;
                border: 1px solid #334155 !important;
                border-radius: 8px !important;
            }
            [data-testid="stTextInput"] input:focus,
            [data-testid="stTextArea"] textarea:focus {
                border-color: #3b82f6 !important;
                box-shadow: 0 0 0 1px #3b82f6 !important;
            }
            [data-testid="stFileUploaderDropzone"] {
                background-color: #0f172a !important;
                border: 1px dashed #334155 !important;
                border-radius: 10px !important;
                color: #cbd5e1 !important;
            }
            [data-testid="stFileUploaderDropzone"]:hover {
                border-color: #3b82f6 !important;
            }

            /* Segmented control dark polish */
            [data-testid="stSegmentedControl"] {
                background-color: #0f172a;
                padding: 0.25rem;
                border-radius: 10px;
                border: 1px solid #1e293b;
            }

            /* Button hierarchy */
            button[kind="primary"],
            [data-testid="stFormSubmitButton"] button[kind="primary"] {
                background-color: #2563eb !important;
                color: #ffffff !important;
                border: 1px solid #3b82f6 !important;
                border-radius: 8px !important;
                font-weight: 600 !important;
            }
            button[kind="primary"]:hover,
            [data-testid="stFormSubmitButton"] button[kind="primary"]:hover {
                background-color: #1d4ed8 !important;
                border-color: #60a5fa !important;
            }
            button[kind="secondary"],
            [data-testid="stFormSubmitButton"] button[kind="secondary"] {
                background-color: #1e293b !important;
                color: #e2e8f0 !important;
                border: 1px solid #334155 !important;
                border-radius: 8px !important;
            }
            button[kind="secondary"]:hover,
            [data-testid="stFormSubmitButton"] button[kind="secondary"]:hover {
                background-color: #283548 !important;
                border-color: #475569 !important;
            }

            /* Metrics, Expanders, Chat */
            [data-testid="stMetric"] {
                background-color: #0f172a;
                border: 1px solid #1e293b;
                border-radius: 10px;
                padding: 0.75rem 0.9rem;
            }
            [data-testid="stMetricLabel"] {
                color: #94a3b8 !important;
            }
            [data-testid="stMetricValue"] {
                color: #f8fafc !important;
            }
            [data-testid="stExpander"] {
                background-color: #0f172a !important;
                border: 1px solid #1e293b !important;
                border-radius: 10px !important;
            }
            [data-testid="stChatMessage"] {
                background-color: #111827;
                border: 1px solid #1e293b;
                border-radius: 10px;
                padding: 0.85rem 1rem;
                margin-bottom: 0.75rem;
            }
            .source-meta {
                font-size: 0.83rem;
                color: #94a3b8;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _supports_directory_uploader() -> bool:
    """Detect whether the installed Streamlit file_uploader supports directory mode."""
    try:
        sig = inspect.signature(st.file_uploader)
        param = sig.parameters.get("accept_multiple_files")
        if param is None:
            return False
        annotation_str = str(param.annotation)
        return "directory" in annotation_str or tuple(
            int(x) for x in st.__version__.split(".")[:2]
        ) >= (1, 50)
    except Exception:
        return False


def _init_session_state() -> None:
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline()
    if "indexed_project" not in st.session_state:
        st.session_state.indexed_project = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "selected_input_source" not in st.session_state:
        st.session_state.selected_input_source = INPUT_SOURCE_OPTIONS[0]


def _get_pipeline() -> RAGPipeline:
    return st.session_state.rag_pipeline


def _reset_project() -> None:
    pipeline = _get_pipeline()
    indexed: IndexedProject | None = st.session_state.indexed_project
    if indexed is not None:
        pipeline.reset_project(indexed.repository.project_id)
    st.session_state.indexed_project = None
    st.session_state.chat_history = []
    st.session_state.pending_question = None


def _clear_chat() -> None:
    st.session_state.chat_history = []
    st.session_state.pending_question = None


def _format_input_source_label(source_key: str) -> str:
    return {
        "github_url": "GitHub Repository URL",
        "zip_upload": "Local Project ZIP Upload",
        "folder_upload": "Local Project Folder Upload",
    }.get(source_key, source_key)


def _run_analysis(action_fn: Any) -> None:
    """Execute a project ingestion action with progress feedback and error handling."""
    progress_bar = st.progress(0.0, text="Starting project analysis...")
    status_placeholder = st.empty()

    def _on_progress(message: str, pct: float) -> None:
        progress_bar.progress(pct, text=message)
        status_placeholder.caption(message)

    try:
        prev: IndexedProject | None = st.session_state.indexed_project
        indexed = action_fn(_on_progress)
        if prev is not None and prev.repository.project_id != indexed.repository.project_id:
            _get_pipeline().reset_project(prev.repository.project_id)
            st.session_state.chat_history = []

        st.session_state.indexed_project = indexed
        progress_bar.progress(1.0, text="Project ready!")
        status_placeholder.success(
            f"Successfully analyzed **{indexed.repository.project_name}** "
            f"({indexed.repository.supported_count} supported files processed, "
            f"{indexed.chunk_count} searchable chunks)."
        )
    except (
        GitHubLoaderError,
        LocalProjectLoaderError,
        RepositoryProcessingError,
        EmbeddingError,
        VectorStoreError,
        ValueError,
    ) as exc:
        progress_bar.empty()
        status_placeholder.error(f"Analysis failed: {exc}")
    except Exception as exc:
        progress_bar.empty()
        status_placeholder.error(f"Unexpected error during analysis: {exc}")


def _render_sidebar() -> None:
    """Render clean dark sidebar with title, API status, and project status."""
    pipeline = _get_pipeline()
    indexed: IndexedProject | None = st.session_state.indexed_project

    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-brand">
                <div class="sidebar-brand-title">⚡ {config.APP_TITLE}</div>
                <div class="sidebar-brand-sub">Codebase Intelligence &amp; RAG Search</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Gemini API Status")
        if pipeline.gemini_client.is_configured():
            st.markdown(
                '<span class="status-pill pill-ok">● Connected</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"Active model: `{pipeline.gemini_client.model_name}`")
        else:
            st.markdown(
                '<span class="status-pill pill-warn">● Key Not Configured</span>',
                unsafe_allow_html=True,
            )
            st.warning(
                "Add `GEMINI_API_KEY` from https://aistudio.google.com/apikey "
                "to `.env` in the project root and restart."
            )

        st.divider()
        st.subheader("Project Status")
        if indexed is None:
            st.markdown(
                '<span class="status-pill pill-idle">● Waiting for Project</span>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Choose an input source in **Analyze Your Project** to index a codebase."
            )
        else:
            repo = indexed.repository
            st.markdown(
                '<span class="status-pill pill-blue">● Ready for Questions</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Project:** `{repo.project_name}`")
            st.markdown(
                f"**Source:** {_format_input_source_label(repo.input_source)}"
            )
            st.markdown(f"**Status:** {indexed.indexing_status}")
            st.caption(
                f"{repo.supported_count} of {repo.discovered_count} files processed"
            )

            if repo.input_source == "github_url" and repo.metadata:
                branch = repo.metadata.get("branch", "main")
                stars = repo.metadata.get("stars", 0)
                st.caption(f"Branch: `{branch}` · Stars: {stars}")

            if st.button(
                "Reset Active Project",
                type="secondary",
                use_container_width=True,
                key="sidebar_reset_btn",
            ):
                _reset_project()
                st.rerun()


def _render_github_input(pipeline: RAGPipeline) -> None:
    """Render GitHub Repository URL controls inside the unified section."""
    repo_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/username/repository-name",
        key="github_url_input",
    )

    url_valid = False
    if repo_url.strip():
        try:
            parsed = parse_github_url(repo_url)
            url_valid = True
            branch_note = f" · branch `{parsed.branch}`" if parsed.branch else ""
            st.markdown(
                f'<div class="info-strip">Valid public repository: '
                f'<code>{parsed.owner}/{parsed.repo}</code>{branch_note}</div>',
                unsafe_allow_html=True,
            )
        except GitHubLoaderError as exc:
            st.warning(str(exc))

    if st.button(
        "Analyze Repository",
        key="analyze_github_btn",
        type="primary",
    ):
        if not repo_url.strip():
            st.error("Please enter a public GitHub repository URL.")
        elif not url_valid:
            try:
                parse_github_url(repo_url)
            except GitHubLoaderError as exc:
                st.error(str(exc))
        else:
            _run_analysis(
                lambda cb: pipeline.ingest_github_url(
                    repo_url,
                    chunk_size=config.DEFAULT_CHUNK_SIZE,
                    chunk_overlap=config.DEFAULT_CHUNK_OVERLAP,
                    progress_callback=cb,
                )
            )


def _render_zip_input(pipeline: RAGPipeline) -> None:
    """Render ZIP archive uploader and controls inside the unified section."""
    st.caption(
        f"Supports large `.zip` project archives (up to {config.MAX_UPLOAD_SIZE_MB} MB) "
        "with incremental extraction and relative path preservation."
    )
    zip_file = st.file_uploader(
        "Select a project ZIP archive",
        type=["zip"],
        accept_multiple_files=False,
        key="zip_uploader",
    )

    if zip_file is not None:
        size_ok = zip_file.size <= config.MAX_UPLOAD_SIZE_BYTES
        st.markdown(
            f'<div class="info-strip">Selected archive: <code>{zip_file.name}</code> '
            f'· Size: <strong>{format_file_size(zip_file.size)}</strong></div>',
            unsafe_allow_html=True,
        )
        if not size_ok:
            st.error(
                f"Archive exceeds the maximum upload limit of {config.MAX_UPLOAD_SIZE_MB} MB."
            )

        if st.button(
            "Analyze ZIP",
            key="analyze_zip_btn",
            type="primary",
            disabled=not size_ok,
        ):
            _run_analysis(
                lambda cb: pipeline.ingest_zip(
                    zip_file,
                    filename=zip_file.name,
                    chunk_size=config.DEFAULT_CHUNK_SIZE,
                    chunk_overlap=config.DEFAULT_CHUNK_OVERLAP,
                    progress_callback=cb,
                )
            )


def _render_folder_input(pipeline: RAGPipeline) -> None:
    """Render folder / multiple-file uploader inside the unified section."""
    dir_upload_supported = _supports_directory_uploader()
    mode_options = (
        ["Browser Folder Picker", "Multiple Files"]
        if dir_upload_supported
        else ["Multiple Files"]
    )
    selected_mode = st.radio(
        "Selection mode",
        options=mode_options,
        horizontal=True,
        key="folder_upload_mode",
    )

    if selected_mode == "Browser Folder Picker" and dir_upload_supported:
        st.caption(
            "Preserves relative folder structure in Chromium, Edge, and Firefox browsers. "
            "If your browser does not support folder selection, switch to Multiple Files or Upload ZIP."
        )
        uploaded_items = st.file_uploader(
            "Select a local project folder",
            accept_multiple_files="directory",
            key="folder_dir_uploader",
        )
    else:
        if not dir_upload_supported:
            st.info(
                "Direct folder picking is unavailable in this runtime. "
                "Select multiple project files below or choose Upload ZIP."
            )
        uploaded_items = st.file_uploader(
            "Select multiple project files",
            accept_multiple_files=True,
            key="folder_multi_uploader",
        )

    custom_name = st.text_input(
        "Project Name (optional fallback if browser omits folder root)",
        value="local-project",
        key="folder_project_name",
    )

    if uploaded_items:
        total_size = sum(getattr(f, "size", 0) for f in uploaded_items)
        first_name = getattr(uploaded_items[0], "name", "")
        inferred_root = (
            first_name.replace("\\", "/").split("/", 1)[0]
            if "/" in first_name.replace("\\", "/")
            else (custom_name or "local-project")
        )
        st.markdown(
            f'<div class="info-strip">Project root: <code>{inferred_root}</code> '
            f'· Selected files: <strong>{len(uploaded_items)}</strong> '
            f'({format_file_size(total_size)})</div>',
            unsafe_allow_html=True,
        )

        if st.button(
            "Analyze Folder",
            key="analyze_folder_btn",
            type="primary",
        ):
            _run_analysis(
                lambda cb: pipeline.ingest_uploaded_folder(
                    uploaded_items,
                    fallback_project_name=custom_name or "local-project",
                    chunk_size=config.DEFAULT_CHUNK_SIZE,
                    chunk_overlap=config.DEFAULT_CHUNK_OVERLAP,
                    progress_callback=cb,
                )
            )


def _render_unified_project_input() -> None:
    """Render the single unified 'Analyze Your Project' section."""
    pipeline = _get_pipeline()

    with st.container(border=True):
        st.markdown(
            '<div class="section-heading">Analyze Your Project</div>'
            '<div class="section-caption">'
            "Select one project source below to extract, index, and explore your codebase."
            "</div>",
            unsafe_allow_html=True,
        )

        selected = st.segmented_control(
            "Project Source",
            options=list(INPUT_SOURCE_OPTIONS),
            default=st.session_state.selected_input_source,
            selection_mode="single",
            label_visibility="collapsed",
            key="project_source_selector",
        )
        if selected:
            st.session_state.selected_input_source = selected
        active_source = st.session_state.selected_input_source

        st.markdown("")
        if active_source == "GitHub Repository URL":
            _render_github_input(pipeline)
        elif active_source == "Upload ZIP":
            _render_zip_input(pipeline)
        else:
            _render_folder_input(pipeline)


def _render_project_results(indexed: IndexedProject) -> None:
    """Render project summary, file counts, and indexed/skipped file reports."""
    repo = indexed.repository
    languages = repo.processing_info.get("languages", {})
    top_langs = ", ".join(f"{lang} ({cnt})" for lang, cnt in list(languages.items())[:5])

    with st.container(border=True):
        header_col, status_col = st.columns([3, 1])
        with header_col:
            st.markdown(
                f'<div class="section-heading">Project Summary: <code>{repo.project_name}</code></div>'
                f'<div class="section-caption">'
                f"Source: {_format_input_source_label(repo.input_source)} · "
                f"Languages: {top_langs or 'Text'}"
                f"</div>",
                unsafe_allow_html=True,
            )
        with status_col:
            st.markdown(
                f'<div style="text-align:right;"><span class="status-pill pill-ok">'
                f"● {indexed.indexing_status}</span></div>",
                unsafe_allow_html=True,
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Discovered", repo.discovered_count)
        m2.metric("Indexed Files", repo.supported_count)
        m3.metric("Skipped Files", len(repo.skipped_files))
        m4.metric("Total Lines", repo.processing_info.get("total_lines", 0))

        col_files, col_langs = st.columns([3, 2])
        with col_files:
            with st.expander(f"Indexed Files ({repo.supported_count})", expanded=False):
                for path in repo.file_paths[:200]:
                    st.text(path)
                if len(repo.file_paths) > 200:
                    st.caption(f"...and {len(repo.file_paths) - 200} more files")

            if repo.skipped_files:
                with st.expander(
                    f"Skipped Files ({len(repo.skipped_files)})", expanded=False
                ):
                    summary = repo.processing_info.get("skip_summary", {})
                    for reason, count in summary.items():
                        st.caption(f"• `{reason}`: {count}")
                    for item in repo.skipped_files[:50]:
                        st.text(f"{item['path']} ({item['reason']})")
                    if len(repo.skipped_files) > 50:
                        st.caption(
                            f"...and {len(repo.skipped_files) - 50} more skipped entries"
                        )

        with col_langs:
            if languages:
                with st.expander("Language Breakdown", expanded=False):
                    lang_df = pd.DataFrame(
                        [
                            {"Language": lang, "Files": count}
                            for lang, count in languages.items()
                        ]
                    )
                    st.dataframe(lang_df, use_container_width=True, hide_index=True)


def _render_sources(answer: RAGAnswer) -> None:
    """Render verified source file references and syntax-highlighted code snippets."""
    if not answer.sources:
        return

    with st.expander(f"Source References ({len(answer.sources)})", expanded=True):
        unique_paths = list(dict.fromkeys(src.file_path for src in answer.sources))
        st.markdown("**Referenced Files:** " + ", ".join(f"`{p}`" for p in unique_paths))
        for idx, src in enumerate(answer.sources, start=1):
            st.markdown(
                f"**{idx}. `{src.file_path}`** "
                f"<span class='source-meta'>(Lines {src.start_line}-{src.end_line})</span>",
                unsafe_allow_html=True,
            )
            st.code(src.snippet, language=src.language)


def _ask_and_append(question: str) -> None:
    indexed: IndexedProject | None = st.session_state.indexed_project
    if indexed is None:
        st.warning("Please analyze a repository, ZIP, or folder first.")
        return

    cleaned = question.strip()
    if not cleaned:
        st.warning("Please enter a valid question.")
        return

    pipeline = _get_pipeline()
    history_payload = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.chat_history
    ]

    with st.spinner("Searching codebase and generating grounded answer..."):
        try:
            rag_answer = pipeline.answer_question(
                indexed_project=indexed,
                question=cleaned,
                top_k_retrieval=config.DEFAULT_TOP_K_RETRIEVAL,
                top_k_rerank=config.DEFAULT_TOP_K_RERANK,
                chat_history=history_payload,
            )
        except (
            GeminiConfigError,
            GeminiQuotaError,
            GeminiCapacityError,
            GeminiModelNotFoundError,
            GeminiAPIError,
            RetrievalError,
            RerankerError,
            EmbeddingError,
            VectorStoreError,
        ) as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"An unexpected error occurred while answering: {exc}")
            return

    st.session_state.chat_history.append({"role": "user", "content": cleaned})
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": rag_answer.answer,
            "rag_answer": rag_answer,
        }
    )


def _render_chat_interface() -> None:
    """Render the dark-themed AI chat Q&A interface."""
    indexed: IndexedProject | None = st.session_state.indexed_project
    if indexed is None:
        return

    with st.container(border=True):
        st.markdown(
            '<div class="section-heading">Ask Questions About Your Codebase</div>'
            '<div class="section-caption">'
            "Ask about architecture, modules, dependencies, or specific functions."
            "</div>",
            unsafe_allow_html=True,
        )

        sample_questions = [
            "What is the purpose of this project?",
            "Explain the overall project architecture.",
            "Which libraries and frameworks are used?",
            "Explain the execution flow and key modules.",
        ]
        btn_cols = st.columns(len(sample_questions))
        for col, sample_q in zip(btn_cols, sample_questions):
            if col.button(sample_q, use_container_width=True):
                st.session_state.pending_question = sample_q

        if st.session_state.pending_question:
            q_to_run = st.session_state.pending_question
            st.session_state.pending_question = None
            _ask_and_append(q_to_run)
            st.rerun()

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                rag_ans = message.get("rag_answer")
                if isinstance(rag_ans, RAGAnswer):
                    _render_sources(rag_ans)

        with st.form("qa_form", clear_on_submit=True):
            question_input = st.text_input(
                "Ask a question",
                placeholder="e.g., How does authentication work? What does app.py do?",
                label_visibility="collapsed",
            )
            col_send, col_clear, _ = st.columns([1, 1, 4])
            send_clicked = col_send.form_submit_button(
                "Send", type="primary", use_container_width=True
            )
            clear_clicked = col_clear.form_submit_button(
                "Clear Chat", type="secondary", use_container_width=True
            )

        if clear_clicked:
            _clear_chat()
            st.rerun()

        if send_clicked:
            if not question_input.strip():
                st.warning("Please type a question before clicking Send.")
            else:
                _ask_and_append(question_input)
                st.rerun()


def main() -> None:
    _inject_dark_theme_css()
    _init_session_state()

    st.markdown(
        f"""
        <div class="saas-header">
            <h1 class="saas-title">{config.APP_TITLE}</h1>
            <p class="saas-subtitle">{config.APP_DESCRIPTION}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_sidebar()
    _render_unified_project_input()

    indexed: IndexedProject | None = st.session_state.indexed_project
    if indexed is not None:
        st.markdown("")
        _render_project_results(indexed)
        st.markdown("")
        _render_chat_interface()


if __name__ == "__main__":
    main()
