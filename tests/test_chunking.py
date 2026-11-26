"""Tests for line-aware CodeChunker."""

from __future__ import annotations

import pytest

from src.chunking import CodeChunker
from src.repository_processor import ProcessedFile, ProcessedRepository


def _make_processed_file(path: str, content: str) -> ProcessedFile:
    lines = content.count("\n") + 1
    return ProcessedFile(
        relative_path=path,
        file_name=path.split("/")[-1],
        extension=".py",
        language="python",
        content=content,
        size_bytes=len(content.encode("utf-8")),
        line_count=lines,
        sha256="abc",
    )


class TestCodeChunker:
    def test_chunk_preserves_line_numbers_and_metadata(self) -> None:
        content = "\n".join(f"def func_{i}():\n    return {i}" for i in range(40))
        pf = _make_processed_file("src/module.py", content)
        repo = ProcessedRepository(
            project_id="p1",
            project_name="demo",
            input_source="zip_upload",
            source_detail="demo.zip",
            files=[pf],
            file_paths=[pf.relative_path],
            discovered_count=1,
            supported_count=1,
            skipped_files=[],
        )

        chunker = CodeChunker(chunk_size=300, chunk_overlap=60)
        chunks = chunker.chunk_repository(repo)

        assert len(chunks) > 1
        assert chunks[0].start_line == 1
        assert chunks[0].source_file_path == "src/module.py"
        assert chunks[0].file_name == "module.py"
        assert chunks[0].chunk_index == 0
        # Ensure monotonic progression across overlapping chunks
        for prev, curr in zip(chunks, chunks[1:]):
            assert curr.start_line >= prev.start_line
            assert curr.end_line >= curr.start_line
            assert curr.chunk_index == prev.chunk_index + 1

    def test_single_long_line_is_split(self) -> None:
        long_line = "x" * 750
        pf = _make_processed_file("data/long.txt", long_line)
        chunker = CodeChunker(chunk_size=300, chunk_overlap=50)
        chunks = chunker.chunk_file(pf)
        assert len(chunks) >= 3
        assert all(c.start_line == 1 and c.end_line == 1 for c in chunks)

    def test_invalid_chunk_params_raise(self) -> None:
        with pytest.raises(ValueError):
            CodeChunker(chunk_size=50, chunk_overlap=10)
        with pytest.raises(ValueError):
            CodeChunker(chunk_size=400, chunk_overlap=400)
