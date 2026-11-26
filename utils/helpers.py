"""Shared helper utilities for path normalization, encoding, and formatting."""

from __future__ import annotations

import os
import posixpath
import re
import sys
from pathlib import PurePosixPath
from typing import Optional


_EXTENSION_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "jsx",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    ".json": "json",
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "rst",
    ".txt": "text",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sql": "sql",
    ".properties": "properties",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "ini",
    ".toml": "toml",
    ".xml": "xml",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".bat": "batch",
    ".cmd": "batch",
    ".ps1": "powershell",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".swift": "swift",
    ".r": "r",
    ".scala": "scala",
    ".gradle": "groovy",
    ".vue": "vue",
    ".svelte": "svelte",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".proto": "protobuf",
}


def normalize_rel_path(path: str) -> str:
    """Normalize any relative path to clean POSIX forward-slash form."""
    cleaned = path.replace("\\", "/").strip()
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    cleaned = cleaned.lstrip("/")
    if not cleaned:
        return ""
    normalized = posixpath.normpath(cleaned)
    return "" if normalized == "." else normalized


def detect_language(file_path: str) -> str:
    """Infer a display/highlighting language from a filename or path."""
    name = PurePosixPath(normalize_rel_path(file_path)).name.lower()
    if name == "dockerfile" or name.endswith(".dockerfile"):
        return "dockerfile"
    if name == "makefile":
        return "makefile"
    if name.startswith(".env"):
        return "dotenv"
    ext = PurePosixPath(name).suffix.lower()
    return _EXTENSION_LANGUAGE_MAP.get(ext, "text")


def is_binary_bytes(data: bytes) -> bool:
    """Heuristic check to determine whether raw bytes represent binary data."""
    if not data:
        return False
    if b"\x00" in data[:8192]:
        return True
    sample = data[:4096]
    # Count non-text control bytes (excluding tab, LF, CR, form feed)
    control_bytes = sum(
        1 for b in sample if b < 32 and b not in (9, 10, 13, 12)
    )
    return (control_bytes / len(sample)) > 0.10


def decode_text_safely(data: bytes) -> Optional[str]:
    """Decode raw bytes to UTF-8 text safely, returning None for binary data."""
    if is_binary_bytes(data):
        return None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def format_file_size(num_bytes: int) -> str:
    """Format byte count into human-readable units."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def sanitize_project_name(name: str, fallback: str = "uploaded-project") -> str:
    """Sanitize a project name for display and safe identifier creation."""
    cleaned = re.sub(r"[^\w.\-]+", "-", name.strip()).strip("-.")
    return cleaned or fallback


def is_safe_subpath(base_dir: str, target_path: str) -> bool:
    """Verify target_path resolves strictly inside base_dir (prevents traversal)."""
    base_real = os.path.realpath(base_dir)
    target_real = os.path.realpath(target_path)
    try:
        return os.path.commonpath([base_real, target_real]) == base_real
    except ValueError:
        return False


def ensure_windows_dll_paths() -> None:
    """Register venv DLL directories on Windows so bundled MSVC runtime DLLs resolve."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        for candidate in (sys.prefix, os.path.join(sys.prefix, "Scripts")):
            if os.path.isdir(candidate):
                try:
                    os.add_dll_directory(candidate)
                except OSError:
                    pass
