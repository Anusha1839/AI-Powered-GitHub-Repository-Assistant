"""UI regression tests for the redesigned dark-theme Streamlit interface."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.chunking import CodeChunk
from src.rag_pipeline import IndexedProject
from src.repository_processor import ProcessedFile, ProcessedRepository


APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def _make_dummy_indexed_project() -> IndexedProject:
    pf = ProcessedFile(
        relative_path="app.py",
        file_name="app.py",
        extension=".py",
        language="python",
        content="print('hello')\n",
        size_bytes=15,
        line_count=1,
        sha256="deadbeef",
    )
    repo = ProcessedRepository(
        project_id="proj123",
        project_name="demo-repo",
        input_source="zip_upload",
        source_detail="demo-repo.zip",
        files=[pf],
        file_paths=["app.py"],
        discovered_count=2,
        supported_count=1,
        skipped_files=[{"path": "logo.png", "reason": "binary_extension"}],
        processing_info={
            "total_lines": 1,
            "languages": {"python": 1},
            "skip_summary": {"binary_extension": 1},
        },
    )
    chunk = CodeChunk(
        chunk_id="app.py::chunk_0",
        source_file_path="app.py",
        file_name="app.py",
        language="python",
        text="print('hello')",
        chunk_index=0,
        start_line=1,
        end_line=1,
    )
    return IndexedProject(
        repository=repo,
        chunks=[chunk],
        chunk_count=1,
        embedding_model="all-MiniLM-L6-v2",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        vector_db_status="Ready (FAISS IndexFlatIP)",
        indexing_status="Indexed",
    )


class TestStreamlitUI:
    def test_no_rag_settings_sliders_and_unified_input_switches_dynamically(self) -> None:
        at = AppTest.from_file(APP_PATH, default_timeout=90).run(timeout=90)
        assert not at.exception

        # 1. No RAG Pipeline Settings or sliders exposed
        assert len(at.slider) == 0
        all_markdown = "\n".join(m.value for m in at.markdown)
        assert "RAG Pipeline Settings" not in all_markdown
        assert "Analyze Your Project" in all_markdown

        # 2. No separate numbered tabs
        assert len(at.tabs) == 0

        # 3. Default source: GitHub Repository URL
        assert [b.label for b in at.button] == ["Analyze Repository"]
        assert any("GitHub Repository URL" == ti.label for ti in at.text_input)

        # 4. Switch dynamically to Upload ZIP
        at.segmented_control(key="project_source_selector").set_value("Upload ZIP").run(
            timeout=90
        )
        assert not at.exception
        assert [b.label for b in at.button] == []  # Analyze ZIP renders once a file is selected
        assert not any("GitHub Repository URL" == ti.label for ti in at.text_input)

        # 5. Switch dynamically to Upload Project Folder
        at.segmented_control(key="project_source_selector").set_value(
            "Upload Project Folder"
        ).run(timeout=90)
        assert not at.exception
        assert any("Project Name" in ti.label for ti in at.text_input)

    def test_results_and_chat_preserve_state_when_switching_source(self) -> None:
        at = AppTest.from_file(APP_PATH, default_timeout=90).run(timeout=90)
        at.session_state["indexed_project"] = _make_dummy_indexed_project()
        at.session_state["chat_history"] = [
            {"role": "user", "content": "What does app.py do?"},
            {"role": "assistant", "content": "It prints hello."},
        ]
        at.run(timeout=90)
        assert not at.exception

        all_markdown = "\n".join(m.value for m in at.markdown)
        assert "demo-repo" in all_markdown
        assert "Ask Questions About Your Codebase" in all_markdown

        # Switch input source while project is loaded; session state remains intact
        at.segmented_control(key="project_source_selector").set_value("Upload ZIP").run(
            timeout=90
        )
        assert not at.exception
        assert at.session_state["indexed_project"] is not None
        assert len(at.session_state["chat_history"]) == 2
