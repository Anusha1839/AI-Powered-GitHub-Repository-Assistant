"""Unified repository processor and common project data models."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Optional

import config
from utils.helpers import (
    decode_text_safely,
    detect_language,
    normalize_rel_path,
    sanitize_project_name,
)
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class RawProjectFile:
    """Raw file extracted by any input loader before filtering and decoding."""

    relative_path: str
    content: bytes | str
    size_bytes: int
    source_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProjectDocumentBundle:
    """Common representation produced by GitHub, ZIP, and Folder loaders."""

    project_name: str
    input_source: str  # "github_url" | "zip_upload" | "folder_upload"
    source_detail: str
    raw_files: list[RawProjectFile]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProcessedFile:
    """Validated, decoded text file ready for chunking."""

    relative_path: str
    file_name: str
    extension: str
    language: str
    content: str
    size_bytes: int
    line_count: int
    sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProcessedRepository:
    """Unified output consumed by the downstream RAG pipeline."""

    project_id: str
    project_name: str
    input_source: str
    source_detail: str
    files: list[ProcessedFile]
    file_paths: list[str]
    discovered_count: int
    supported_count: int
    skipped_files: list[dict[str, str]]
    metadata: dict[str, Any] = field(default_factory=dict)
    processing_info: dict[str, Any] = field(default_factory=dict)


class RepositoryProcessingError(Exception):
    """Raised when a project contains no processable source files."""


class RepositoryProcessor:
    """Filters, validates, decodes, and normalizes files from any loader."""

    def __init__(
        self,
        max_file_size_bytes: int = config.MAX_FILE_SIZE_BYTES,
        max_files_count: int = config.MAX_FILES_COUNT,
    ) -> None:
        self.max_file_size_bytes = max_file_size_bytes
        self.max_files_count = max_files_count

    @staticmethod
    def is_excluded_path(rel_path: str) -> Optional[str]:
        """Return exclusion reason if path hits an excluded directory or file."""
        pure = PurePosixPath(rel_path)
        parts = [part.lower() for part in pure.parts[:-1]]
        for part in parts:
            if part in config.EXCLUDED_DIRS or part.endswith(".egg-info"):
                return f"excluded_directory:{part}"

        filename_lower = pure.name.lower()
        if filename_lower in config.EXCLUDED_FILENAMES:
            return "excluded_filename"
        if filename_lower.endswith(".min.js") or filename_lower.endswith(".min.css"):
            return "minified_asset"
        return None

    @staticmethod
    def is_supported_file(rel_path: str) -> tuple[bool, str]:
        """Determine whether a file path is a supported text/code format."""
        pure = PurePosixPath(rel_path)
        name_lower = pure.name.lower()
        ext_lower = pure.suffix.lower()

        if ext_lower in config.BINARY_EXTENSIONS:
            return False, "binary_extension"
        if name_lower in config.SUPPORTED_SPECIAL_FILENAMES:
            return True, "supported_special_file"
        if ext_lower in config.SUPPORTED_EXTENSIONS:
            return True, "supported_extension"
        return False, "unsupported_extension"

    def process(self, bundle: ProjectDocumentBundle) -> ProcessedRepository:
        """Validate, filter, and decode a ProjectDocumentBundle into ProcessedRepository."""
        project_name = sanitize_project_name(bundle.project_name)
        seen_paths: set[str] = set()
        processed_files: list[ProcessedFile] = []
        skipped_files: list[dict[str, str]] = []
        language_counts: dict[str, int] = {}
        total_bytes = 0
        total_lines = 0

        pre_skipped = list(bundle.metadata.get("pre_skipped_files", []))
        discovered_count = len(bundle.raw_files) + len(pre_skipped)
        if discovered_count == 0:
            raise RepositoryProcessingError(
                "No files were found in the provided project source."
            )

        skipped_files.extend(pre_skipped)
        for entry in pre_skipped:
            if entry.get("path"):
                seen_paths.add(normalize_rel_path(entry["path"]))

        for raw_file in bundle.raw_files:
            rel_path = normalize_rel_path(raw_file.relative_path)
            if not rel_path or ".." in PurePosixPath(rel_path).parts:
                skipped_files.append(
                    {"path": raw_file.relative_path, "reason": "invalid_path"}
                )
                continue

            if rel_path in seen_paths:
                skipped_files.append({"path": rel_path, "reason": "duplicate_path"})
                continue
            seen_paths.add(rel_path)

            exclusion_reason = self.is_excluded_path(rel_path)
            if exclusion_reason:
                skipped_files.append({"path": rel_path, "reason": exclusion_reason})
                continue

            supported, support_reason = self.is_supported_file(rel_path)
            if not supported:
                skipped_files.append({"path": rel_path, "reason": support_reason})
                continue

            if raw_file.size_bytes == 0:
                skipped_files.append({"path": rel_path, "reason": "empty_file"})
                continue

            if raw_file.size_bytes > self.max_file_size_bytes:
                skipped_files.append(
                    {"path": rel_path, "reason": "exceeds_max_file_size"}
                )
                continue

            if isinstance(raw_file.content, bytes):
                text_content = decode_text_safely(raw_file.content)
                if text_content is None:
                    skipped_files.append(
                        {"path": rel_path, "reason": "binary_or_encoding_error"}
                    )
                    continue
                raw_bytes = raw_file.content
            else:
                text_content = raw_file.content
                raw_bytes = text_content.encode("utf-8", errors="replace")

            if not text_content.strip():
                skipped_files.append({"path": rel_path, "reason": "empty_file"})
                continue

            if len(processed_files) >= self.max_files_count:
                skipped_files.append(
                    {"path": rel_path, "reason": "exceeds_max_files_count"}
                )
                continue

            pure = PurePosixPath(rel_path)
            language = detect_language(rel_path)
            line_count = text_content.count("\n") + 1
            file_hash = hashlib.sha256(raw_bytes).hexdigest()

            processed = ProcessedFile(
                relative_path=rel_path,
                file_name=pure.name,
                extension=pure.suffix.lower(),
                language=language,
                content=text_content,
                size_bytes=len(raw_bytes),
                line_count=line_count,
                sha256=file_hash,
                metadata=dict(raw_file.source_metadata),
            )
            processed_files.append(processed)
            language_counts[language] = language_counts.get(language, 0) + 1
            total_bytes += processed.size_bytes
            total_lines += line_count

        if not processed_files:
            raise RepositoryProcessingError(
                "No supported non-empty text or source code files were found after filtering."
            )

        processed_files.sort(key=lambda item: item.relative_path)
        file_paths = [f.relative_path for f in processed_files]

        # Deterministic content fingerprint avoids re-indexing unchanged projects
        hasher = hashlib.sha256()
        hasher.update(project_name.encode("utf-8"))
        hasher.update(bundle.input_source.encode("utf-8"))
        for item in processed_files:
            hasher.update(item.relative_path.encode("utf-8"))
            hasher.update(item.sha256.encode("utf-8"))
        project_id = hasher.hexdigest()[:16]

        skip_summary: dict[str, int] = {}
        for entry in skipped_files:
            reason = entry["reason"].split(":")[0].split(" ")[0]
            skip_summary[reason] = skip_summary.get(reason, 0) + 1

        processing_info = {
            "total_size_bytes": total_bytes,
            "total_lines": total_lines,
            "languages": dict(
                sorted(language_counts.items(), key=lambda kv: kv[1], reverse=True)
            ),
            "skipped_count": len(skipped_files),
            "skip_summary": skip_summary,
            "extraction_summary": bundle.metadata.get("extraction_summary"),
        }

        logger.info(
            "Processed project '%s' (%s): %d/%d supported files, %d lines.",
            project_name,
            bundle.input_source,
            len(processed_files),
            discovered_count,
            total_lines,
        )

        return ProcessedRepository(
            project_id=project_id,
            project_name=project_name,
            input_source=bundle.input_source,
            source_detail=bundle.source_detail,
            files=processed_files,
            file_paths=file_paths,
            discovered_count=discovered_count,
            supported_count=len(processed_files),
            skipped_files=skipped_files,
            metadata=dict(bundle.metadata),
            processing_info=processing_info,
        )
