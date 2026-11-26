"""Central configuration for the AI-Powered GitHub Repository Assistant."""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Load environment variables from .env in project root without overriding explicit env vars
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

APP_TITLE = "AI-Powered GitHub Repository Assistant"
APP_DESCRIPTION = (
    "Analyze GitHub repositories, ZIP archives, or local project folders using "
    "Sentence Transformers, FAISS vector search, Cross-Encoder reranking, and Google Gemini."
)

# Model configurations
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
FALLBACK_GEMINI_MODELS: tuple[str, ...] = ("gemini-3.8-flash", "gemini-3.6-flash")

# Chunking defaults
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 200
MAX_CHUNK_SIZE = 4000

# Retrieval & reranking defaults
DEFAULT_TOP_K_RETRIEVAL = 15
DEFAULT_TOP_K_RERANK = 5
MAX_RERANK_CANDIDATES = 25
MAX_RERANK_TEXT_CHARS = 2500
MAX_CONTEXT_CHARS = 28000

# File and upload limits
MAX_UPLOAD_SIZE_MB = 200
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
MAX_TOTAL_EXTRACTED_BYTES = 200 * 1024 * 1024
MAX_FILE_SIZE_BYTES = 1024 * 1024  # 1 MB per source file
MAX_FILES_COUNT = 5000
EMBEDDING_BATCH_SIZE = 32

# Network timeouts
GITHUB_REQUEST_TIMEOUT = 20

# Supported text-based source and configuration extensions
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".java",
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
        ".ts",
        ".tsx",
        ".html",
        ".htm",
        ".css",
        ".scss",
        ".sass",
        ".less",
        ".json",
        ".md",
        ".markdown",
        ".rst",
        ".txt",
        ".yaml",
        ".yml",
        ".sql",
        ".properties",
        ".ini",
        ".cfg",
        ".conf",
        ".toml",
        ".xml",
        ".sh",
        ".bash",
        ".zsh",
        ".bat",
        ".cmd",
        ".ps1",
        ".c",
        ".cpp",
        ".cc",
        ".h",
        ".hpp",
        ".cs",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".kt",
        ".kts",
        ".swift",
        ".r",
        ".scala",
        ".gradle",
        ".vue",
        ".svelte",
        ".graphql",
        ".gql",
        ".proto",
        ".dockerfile",
    }
)

# Extensionless or special filenames that should be included
SUPPORTED_SPECIAL_FILENAMES: frozenset[str] = frozenset(
    {
        "dockerfile",
        "makefile",
        "procfile",
        "rakefile",
        "gemfile",
        "jenkinsfile",
        "vagrantfile",
        "cmakelists.txt",
        ".gitignore",
        ".gitattributes",
        ".dockerignore",
        ".editorconfig",
        ".env.example",
        ".env.template",
        ".eslintrc",
        ".prettierrc",
        "readme",
        "license",
    }
)

# Directories excluded from repository analysis
EXCLUDED_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        "venv",
        ".venv",
        "env",
        "dist",
        "build",
        ".idea",
        ".vscode",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".eggs",
        "target",
        "bin",
        "obj",
        "out",
        "coverage",
        "htmlcov",
        ".next",
        ".nuxt",
        "vendor",
    }
)

# Lockfiles and generated files that add noise rather than architectural insight
EXCLUDED_FILENAMES: frozenset[str] = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "pipfile.lock",
        "composer.lock",
        "cargo.lock",
        ".ds_store",
        "thumbs.db",
    }
)

# Compiled and binary file extensions explicitly excluded
BINARY_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pyc",
        ".pyo",
        ".pyd",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".class",
        ".jar",
        ".war",
        ".ear",
        ".o",
        ".a",
        ".lib",
        ".obj",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".ico",
        ".svg",
        ".webp",
        ".tiff",
        ".pdf",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".wav",
        ".flac",
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
        ".eot",
        ".db",
        ".sqlite",
        ".sqlite3",
        ".pkl",
        ".pickle",
        ".npy",
        ".npz",
        ".pt",
        ".pth",
        ".onnx",
        ".safetensors",
        ".faiss",
        ".index",
        ".bin",
        ".dat",
        ".iso",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
    }
)


def get_gemini_api_key() -> str:
    """Return the configured Gemini API key from environment variables."""
    return os.getenv("GEMINI_API_KEY", "").strip()


def get_github_token() -> str:
    """Return an optional GitHub token from environment variables."""
    return os.getenv("GITHUB_TOKEN", "").strip()
