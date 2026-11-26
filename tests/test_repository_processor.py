"""Tests for the unified RepositoryProcessor filtering and validation."""

from __future__ import annotations

import pytest

from src.repository_processor import (
    ProjectDocumentBundle,
    RawProjectFile,
    RepositoryProcessingError,
    RepositoryProcessor,
)


class TestRepositoryProcessor:
    def test_filters_excluded_dirs_binaries_empty_and_large_files(self) -> None:
        bundle = ProjectDocumentBundle(
            project_name="Demo App",
            input_source="zip_upload",
            source_detail="demo.zip",
            raw_files=[
                RawProjectFile("app.py", b"def run():\n    return True\n", 26),
                RawProjectFile("README.md", b"# Demo\nDocs here.", 17),
                RawProjectFile("config.properties", b"app.port=8080\n", 14),
                RawProjectFile(".git/config", b"[core]\n", 7),
                RawProjectFile("node_modules/pkg/index.js", b"module.exports = {};", 20),
                RawProjectFile("__pycache__/app.cpython-311.pyc", b"\x00\x01\x02", 3),
                RawProjectFile("assets/logo.png", b"\x89PNG\x00\x00", 6),
                RawProjectFile("empty.py", b"   \n", 4),
                RawProjectFile("huge.py", b"x = 1\n" * 500, 3000),
                RawProjectFile("app.py", b"duplicate", 9),
                RawProjectFile("weird.unknownext", b"hello", 5),
            ],
        )

        processor = RepositoryProcessor(max_file_size_bytes=500)
        processed = processor.process(bundle)

        assert processed.project_name == "Demo-App"
        assert processed.discovered_count == 11
        assert processed.supported_count == 3
        assert processed.file_paths == ["README.md", "app.py", "config.properties"]
        reasons = {item["path"]: item["reason"] for item in processed.skipped_files}
        assert reasons[".git/config"].startswith("excluded_directory")
        assert reasons["node_modules/pkg/index.js"].startswith("excluded_directory")
        assert reasons["assets/logo.png"] == "binary_extension"
        assert reasons["empty.py"] == "empty_file"
        assert reasons["huge.py"] == "exceeds_max_file_size"
        assert reasons["weird.unknownext"] == "unsupported_extension"

    def test_all_files_unsupported_raises(self) -> None:
        bundle = ProjectDocumentBundle(
            project_name="Empty",
            input_source="folder_upload",
            source_detail="empty",
            raw_files=[RawProjectFile("pic.jpg", b"\xff\xd8\xff\x00", 4)],
        )
        processor = RepositoryProcessor()
        with pytest.raises(RepositoryProcessingError):
            processor.process(bundle)

    def test_deterministic_project_id_for_identical_content(self) -> None:
        files = [RawProjectFile("src/a.py", b"x = 1\n", 6)]
        b1 = ProjectDocumentBundle("proj", "zip_upload", "a.zip", files)
        b2 = ProjectDocumentBundle("proj", "zip_upload", "b.zip", files)
        processor = RepositoryProcessor()
        assert processor.process(b1).project_id == processor.process(b2).project_id
