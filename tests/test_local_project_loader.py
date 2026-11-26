"""Tests for local ZIP, uploaded folder/multi-file, and directory loaders."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from src.local_project_loader import (
    LocalProjectLoaderError,
    load_from_local_directory,
    load_from_uploaded_files,
    load_from_zip,
)


def _build_zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


class _FakeUploadedFile:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._buf = io.BytesIO(data)
        self.size = len(data)

    def seek(self, pos: int) -> None:
        self._buf.seek(pos)

    def read(self) -> bytes:
        return self._buf.read()


class TestZipLoader:
    def test_valid_nested_zip_strips_single_root(self) -> None:
        zip_bytes = _build_zip(
            {
                "my-project/app.py": b"print('hello')\n",
                "my-project/README.md": b"# My Project\n",
                "my-project/src/main.py": b"def run():\n    pass\n",
                "my-project/utils/helpers.py": b"X = 1\n",
            }
        )
        bundle = load_from_zip(io.BytesIO(zip_bytes), filename="my-project.zip")
        assert bundle.project_name == "my-project"
        assert bundle.input_source == "zip_upload"
        paths = sorted(f.relative_path for f in bundle.raw_files)
        assert paths == ["README.md", "app.py", "src/main.py", "utils/helpers.py"]

    def test_invalid_zip_raises(self) -> None:
        with pytest.raises(LocalProjectLoaderError, match="not a valid ZIP"):
            load_from_zip(b"not-a-zip-archive", filename="bad.zip")

    def test_empty_zip_raises(self) -> None:
        empty_zip = _build_zip({})
        with pytest.raises(LocalProjectLoaderError, match="no files"):
            load_from_zip(empty_zip, filename="empty.zip")

    @pytest.mark.parametrize(
        "unsafe_name",
        [
            "../outside.py",
            "project/../../escape.py",
            "/etc/passwd",
            "C:\\Windows\\system.ini",
        ],
    )
    def test_unsafe_zip_paths_rejected(self, unsafe_name: str) -> None:
        zip_bytes = _build_zip({unsafe_name: b"malicious"})
        with pytest.raises(LocalProjectLoaderError, match="Unsafe"):
            load_from_zip(zip_bytes, filename="unsafe.zip")

    def test_oversized_zip_rejected(self) -> None:
        zip_bytes = _build_zip({"proj/app.py": b"a" * 500})
        with pytest.raises(LocalProjectLoaderError, match="maximum upload limit"):
            load_from_zip(zip_bytes, filename="big.zip", max_upload_bytes=100)

    def test_large_zip_incremental_skips_noisy_and_oversized_files_with_summary(self) -> None:
        zip_bytes = _build_zip(
            {
                "big-proj/app.py": b"print('ok')\n",
                "big-proj/node_modules/pkg/index.js": b"module.exports = 1;\n",
                "big-proj/assets/logo.png": b"\x89PNG\x00\x01",
                "big-proj/huge.py": b"x = 1\n" * 200,
            }
        )
        bundle = load_from_zip(
            io.BytesIO(zip_bytes),
            filename="big-proj.zip",
            max_file_size_bytes=200,
        )
        assert [f.relative_path for f in bundle.raw_files] == ["app.py"]
        summary = bundle.metadata["extraction_summary"]
        assert summary["extracted_files"] == 1
        assert summary["skipped_files"] == 3
        skipped = {e["path"]: e["reason"] for e in bundle.metadata["pre_skipped_files"]}
        assert skipped["node_modules/pkg/index.js"].startswith("excluded_directory")
        assert skipped["assets/logo.png"] == "binary_extension"
        assert "exceeds_max_file_size" in skipped["huge.py"]


class TestFolderAndMultiFileLoader:
    def test_uploaded_folder_preserves_relative_paths(self) -> None:
        files = [
            _FakeUploadedFile("demo-app/app.py", b"print('hi')\n"),
            _FakeUploadedFile("demo-app/src/core.py", b"def f():\n    return 1\n"),
            _FakeUploadedFile("demo-app/image.png", b"\x89PNG\r\n\x1a\n"),
        ]
        bundle = load_from_uploaded_files(files)
        assert bundle.project_name == "demo-app"
        assert bundle.input_source == "folder_upload"
        paths = sorted(f.relative_path for f in bundle.raw_files)
        assert paths == ["app.py", "image.png", "src/core.py"]

    def test_uploaded_flat_multiple_files_uses_fallback_name(self) -> None:
        files = [
            _FakeUploadedFile("app.py", b"print(1)\n"),
            _FakeUploadedFile("utils.py", b"print(2)\n"),
        ]
        bundle = load_from_uploaded_files(files, fallback_project_name="custom-proj")
        assert bundle.project_name == "custom-proj"
        assert sorted(f.relative_path for f in bundle.raw_files) == ["app.py", "utils.py"]

    def test_empty_upload_raises(self) -> None:
        with pytest.raises(LocalProjectLoaderError):
            load_from_uploaded_files([])

    def test_local_directory_nested_and_empty(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty_proj"
        empty_dir.mkdir()
        with pytest.raises(LocalProjectLoaderError, match="empty"):
            load_from_local_directory(empty_dir)

        proj_dir = tmp_path / "sample_proj"
        (proj_dir / "src").mkdir(parents=True)
        (proj_dir / "__pycache__").mkdir()
        (proj_dir / "main.py").write_text("print('ok')\n", encoding="utf-8")
        (proj_dir / "src" / "helper.py").write_text("x = 1\n", encoding="utf-8")
        (proj_dir / "__pycache__" / "main.cpython-311.pyc").write_bytes(b"\x00\x01")

        bundle = load_from_local_directory(proj_dir)
        assert bundle.project_name == "sample_proj"
        paths = sorted(f.relative_path for f in bundle.raw_files)
        assert paths == ["main.py", "src/helper.py"]
