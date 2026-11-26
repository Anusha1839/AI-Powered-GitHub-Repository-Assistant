"""Local project loaders for ZIP archives, browser folder uploads, and local directories."""

from __future__ import annotations

import io
import os
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Optional, Sequence

import config
from src.repository_processor import (
    ProjectDocumentBundle,
    RawProjectFile,
    RepositoryProcessor,
)
from utils.helpers import (
    is_safe_subpath,
    normalize_rel_path,
    sanitize_project_name,
)
from utils.logger import get_logger

logger = get_logger(__name__)

_EXTRACTION_CHUNK_BYTES = 64 * 1024


class LocalProjectLoaderError(Exception):
    """Raised when a ZIP upload, folder upload, or multi-file upload is invalid."""


def _strip_single_root_folder(paths: list[str]) -> tuple[list[str], str]:
    """Detect and strip a single shared top-level directory across relative paths."""
    if not paths:
        return paths, ""

    first_parts: set[str] = set()
    all_nested = True
    for path in paths:
        parts = PurePosixPath(path).parts
        if len(parts) < 2:
            all_nested = False
            break
        first_parts.add(parts[0])

    if all_nested and len(first_parts) == 1:
        root_name = next(iter(first_parts))
        stripped = [
            "/".join(PurePosixPath(p).parts[1:])
            for p in paths
        ]
        return stripped, root_name

    return paths, ""


def _open_zip_handle(
    zip_source: bytes | BinaryIO | str | Path,
    filename: Optional[str],
    max_upload_bytes: int,
) -> tuple[zipfile.ZipFile, str, int]:
    """Validate source size/extension and return an open ZipFile handle without extra copies."""
    if isinstance(zip_source, (str, Path)):
        path = Path(zip_source)
        if not path.exists() or not path.is_file():
            raise LocalProjectLoaderError(f"ZIP file not found: {path}")
        archive_name = filename or path.name
        if not archive_name.lower().endswith(".zip"):
            raise LocalProjectLoaderError("Uploaded file must have a .zip extension.")
        archive_size = path.stat().st_size
        if archive_size == 0:
            raise LocalProjectLoaderError("The uploaded ZIP file is empty.")
        if archive_size > max_upload_bytes:
            raise LocalProjectLoaderError(
                f"ZIP archive exceeds the maximum upload limit of "
                f"{max_upload_bytes // (1024 * 1024)} MB."
            )
        try:
            return zipfile.ZipFile(path, "r"), archive_name, archive_size
        except zipfile.BadZipFile as exc:
            raise LocalProjectLoaderError(
                "The uploaded file is not a valid ZIP archive."
            ) from exc

    if isinstance(zip_source, (bytes, bytearray)):
        archive_name = filename or "uploaded_project.zip"
        if not archive_name.lower().endswith(".zip"):
            raise LocalProjectLoaderError("Only .zip archives are supported in ZIP upload mode.")
        archive_size = len(zip_source)
        if archive_size == 0:
            raise LocalProjectLoaderError("The uploaded ZIP file is empty.")
        if archive_size > max_upload_bytes:
            raise LocalProjectLoaderError(
                f"ZIP archive exceeds the maximum upload limit of "
                f"{max_upload_bytes // (1024 * 1024)} MB."
            )
        try:
            return zipfile.ZipFile(io.BytesIO(zip_source), "r"), archive_name, archive_size
        except zipfile.BadZipFile as exc:
            raise LocalProjectLoaderError(
                "The uploaded file is not a valid ZIP archive."
            ) from exc

    # File-like object (e.g., Streamlit UploadedFile or BytesIO)
    archive_name = (
        filename
        or getattr(zip_source, "name", "uploaded_project.zip")
        or "uploaded_project.zip"
    )
    if not archive_name.lower().endswith(".zip"):
        raise LocalProjectLoaderError("Only .zip archives are supported in ZIP upload mode.")

    if hasattr(zip_source, "seek"):
        zip_source.seek(0, os.SEEK_END)
        archive_size = zip_source.tell()
        zip_source.seek(0)
    else:
        data = zip_source.read()
        if not isinstance(data, (bytes, bytearray)):
            raise LocalProjectLoaderError("Could not read binary contents from the ZIP file.")
        zip_source = io.BytesIO(data)
        archive_size = len(data)

    if archive_size == 0:
        raise LocalProjectLoaderError("The uploaded ZIP file is empty.")
    if archive_size > max_upload_bytes:
        raise LocalProjectLoaderError(
            f"ZIP archive exceeds the maximum upload limit of "
            f"{max_upload_bytes // (1024 * 1024)} MB."
        )

    try:
        return zipfile.ZipFile(zip_source, "r"), archive_name, archive_size
    except zipfile.BadZipFile as exc:
        raise LocalProjectLoaderError(
            "The uploaded file is not a valid ZIP archive."
        ) from exc


def _extract_member_incremental(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    dest_path: str,
    max_file_bytes: int,
    max_retries: int = 1,
) -> bytes:
    """Extract one ZIP member in 64KB chunks with retry support on transient errors."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            written = 0
            with zf.open(info, "r") as src, open(dest_path, "wb") as dst:
                while True:
                    chunk = src.read(_EXTRACTION_CHUNK_BYTES)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > max_file_bytes:
                        raise ValueError("exceeds_max_file_size")
                    dst.write(chunk)
            with open(dest_path, "rb") as fh:
                return fh.read()
        except ValueError:
            raise
        except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(0.05)
                continue
            raise
    assert last_exc is not None
    raise last_exc


def load_from_zip(
    zip_source: bytes | BinaryIO | str | Path,
    filename: str | None = None,
    max_upload_bytes: int = config.MAX_UPLOAD_SIZE_BYTES,
    max_total_extracted_bytes: int = config.MAX_TOTAL_EXTRACTED_BYTES,
    max_file_size_bytes: int = config.MAX_FILE_SIZE_BYTES,
    max_files_count: int = config.MAX_FILES_COUNT,
) -> ProjectDocumentBundle:
    """Safely validate and incrementally extract supported files from a ZIP archive."""
    zf, archive_name, archive_size = _open_zip_handle(
        zip_source, filename=filename, max_upload_bytes=max_upload_bytes
    )

    raw_files: list[RawProjectFile] = []
    pre_skipped_files: list[dict[str, str]] = []
    retried_files: list[str] = []
    extracted_bytes_total = 0
    temp_dir_obj = tempfile.TemporaryDirectory(prefix="repo_assistant_zip_")

    try:
        temp_dir = temp_dir_obj.name
        with zf:
            file_infos = [info for info in zf.infolist() if not info.is_dir()]
            if not file_infos:
                raise LocalProjectLoaderError("The uploaded ZIP archive contains no files.")

            # Security validation pass across all entries before extracting anything
            normalized_members: list[tuple[zipfile.ZipInfo, str]] = []
            for info in file_infos:
                if info.flag_bits & 0x1:
                    raise LocalProjectLoaderError(
                        "Password-protected or encrypted ZIP archives are not supported."
                    )

                unix_mode = (info.external_attr >> 16) & 0o170000
                if unix_mode == 0o120000:
                    raise LocalProjectLoaderError(
                        f"Unsafe symlink entry detected in ZIP archive: {info.filename}"
                    )

                member_name = info.filename.replace("\\", "/")
                if (
                    member_name.startswith("/")
                    or (len(member_name) >= 2 and member_name[1] == ":")
                    or ".." in PurePosixPath(member_name).parts
                ):
                    raise LocalProjectLoaderError(
                        f"Unsafe file path detected in ZIP archive: {info.filename}"
                    )

                rel_member = normalize_rel_path(info.filename)
                if rel_member:
                    normalized_members.append((info, rel_member))

            if not normalized_members:
                raise LocalProjectLoaderError("The uploaded ZIP archive contains no readable files.")

            raw_paths = [entry[1] for entry in normalized_members]
            clean_paths, detected_root = _strip_single_root_folder(raw_paths)

            # Incremental filtering and chunked extraction so large archives with
            # node_modules/.git/media do not exhaust memory or abort supported files
            for (info, rel_member), clean_path in zip(normalized_members, clean_paths):
                exclusion_reason = RepositoryProcessor.is_excluded_path(clean_path)
                if exclusion_reason:
                    pre_skipped_files.append(
                        {"path": clean_path, "reason": exclusion_reason}
                    )
                    continue

                supported, support_reason = RepositoryProcessor.is_supported_file(clean_path)
                if not supported:
                    pre_skipped_files.append(
                        {"path": clean_path, "reason": support_reason}
                    )
                    continue

                if info.file_size == 0:
                    pre_skipped_files.append(
                        {"path": clean_path, "reason": "empty_file"}
                    )
                    continue

                if info.file_size > max_file_size_bytes:
                    pre_skipped_files.append(
                        {
                            "path": clean_path,
                            "reason": (
                                f"exceeds_max_file_size "
                                f"({info.file_size // 1024} KB > {max_file_size_bytes // 1024} KB)"
                            ),
                        }
                    )
                    continue

                if len(raw_files) >= max_files_count:
                    pre_skipped_files.append(
                        {"path": clean_path, "reason": "exceeds_max_files_count"}
                    )
                    continue

                if extracted_bytes_total + info.file_size > max_total_extracted_bytes:
                    pre_skipped_files.append(
                        {"path": clean_path, "reason": "exceeds_total_extracted_limit"}
                    )
                    continue

                dest_path = os.path.join(temp_dir, *PurePosixPath(rel_member).parts)
                if not is_safe_subpath(temp_dir, dest_path):
                    raise LocalProjectLoaderError(
                        f"Unsafe extraction path outside temporary directory: {info.filename}"
                    )

                try:
                    data = _extract_member_incremental(
                        zf=zf,
                        info=info,
                        dest_path=dest_path,
                        max_file_bytes=max_file_size_bytes,
                        max_retries=1,
                    )
                except ValueError:
                    pre_skipped_files.append(
                        {"path": clean_path, "reason": "exceeds_max_file_size"}
                    )
                    continue
                except Exception as exc:
                    pre_skipped_files.append(
                        {"path": clean_path, "reason": f"extraction_failed:{type(exc).__name__}"}
                    )
                    logger.warning("Failed extracting '%s' after retry: %s", clean_path, exc)
                    continue

                extracted_bytes_total += len(data)
                raw_files.append(
                    RawProjectFile(
                        relative_path=clean_path,
                        content=data,
                        size_bytes=len(data),
                    )
                )
    finally:
        temp_dir_obj.cleanup()

    if not raw_files and pre_skipped_files:
        raise LocalProjectLoaderError(
            "No supported text/source files could be extracted from the ZIP archive "
            f"({len(pre_skipped_files)} entries were skipped due to filters or size limits)."
        )

    default_name = Path(archive_name).stem
    project_name = sanitize_project_name(detected_root or default_name)

    skip_breakdown: dict[str, int] = {}
    for entry in pre_skipped_files:
        reason = entry["reason"].split(":")[0].split(" ")[0]
        skip_breakdown[reason] = skip_breakdown.get(reason, 0) + 1

    extraction_summary = {
        "total_archive_entries": len(raw_files) + len(pre_skipped_files),
        "extracted_files": len(raw_files),
        "skipped_files": len(pre_skipped_files),
        "retried_files": len(retried_files),
        "extracted_bytes": extracted_bytes_total,
        "skip_breakdown": skip_breakdown,
    }

    logger.info(
        "Extracted ZIP '%s' (%d supported files, %d skipped) as project '%s'.",
        archive_name,
        len(raw_files),
        len(pre_skipped_files),
        project_name,
    )

    return ProjectDocumentBundle(
        project_name=project_name,
        input_source="zip_upload",
        source_detail=archive_name,
        raw_files=raw_files,
        metadata={
            "archive_filename": archive_name,
            "archive_size_bytes": archive_size,
            "extracted_file_count": len(raw_files),
            "pre_skipped_files": pre_skipped_files,
            "extraction_summary": extraction_summary,
        },
    )


def load_from_uploaded_files(
    uploaded_files: Sequence[Any],
    fallback_project_name: str = "local-project",
    max_upload_bytes: int = config.MAX_UPLOAD_SIZE_BYTES,
    max_files_count: int = config.MAX_FILES_COUNT,
) -> ProjectDocumentBundle:
    """Load files from Streamlit folder upload or multiple-file upload."""
    if not uploaded_files:
        raise LocalProjectLoaderError(
            "No files were selected. Please choose a project folder or multiple source files."
        )

    if len(uploaded_files) > max_files_count * 3:
        raise LocalProjectLoaderError(
            "Too many files selected in folder upload."
        )

    entries: list[tuple[str, bytes]] = []
    total_bytes = 0

    for item in uploaded_files:
        raw_name = getattr(item, "name", "") or ""
        norm_name = raw_name.replace("\\", "/").strip()
        if (
            not norm_name
            or norm_name.startswith("/")
            or (len(norm_name) >= 2 and norm_name[1] == ":")
            or ".." in PurePosixPath(norm_name).parts
        ):
            raise LocalProjectLoaderError(
                f"Unsafe or invalid uploaded file path: {raw_name}"
            )

        rel_path = normalize_rel_path(norm_name)
        if not rel_path:
            continue

        if hasattr(item, "seek"):
            item.seek(0)
        data = item.read() if hasattr(item, "read") else bytes(item)
        if not isinstance(data, (bytes, bytearray)):
            raise LocalProjectLoaderError(
                f"Could not read binary data from uploaded file: {raw_name}"
            )

        file_bytes = bytes(data)
        total_bytes += len(file_bytes)
        if total_bytes > max_upload_bytes:
            raise LocalProjectLoaderError(
                f"Total uploaded folder size exceeds {max_upload_bytes // (1024 * 1024)} MB."
            )

        entries.append((rel_path, file_bytes))

    if not entries:
        raise LocalProjectLoaderError("No valid files were found in the selection.")

    raw_paths = [entry[0] for entry in entries]
    normalized_paths, detected_root = _strip_single_root_folder(raw_paths)

    raw_files = [
        RawProjectFile(
            relative_path=norm_path,
            content=file_bytes,
            size_bytes=len(file_bytes),
        )
        for norm_path, (_, file_bytes) in zip(normalized_paths, entries)
    ]

    project_name = sanitize_project_name(detected_root or fallback_project_name)
    logger.info(
        "Loaded %d uploaded files for local project '%s'.",
        len(raw_files),
        project_name,
    )

    return ProjectDocumentBundle(
        project_name=project_name,
        input_source="folder_upload",
        source_detail=detected_root or project_name,
        raw_files=raw_files,
        metadata={
            "uploaded_file_count": len(raw_files),
            "total_upload_bytes": total_bytes,
            "detected_root_folder": detected_root or None,
        },
    )


def load_from_local_directory(
    directory_path: str | Path,
    max_total_bytes: int = config.MAX_TOTAL_EXTRACTED_BYTES,
    max_files_count: int = config.MAX_FILES_COUNT,
) -> ProjectDocumentBundle:
    """Load files directly from a local folder path on disk while preserving relative paths."""
    root = Path(directory_path)
    if not root.exists() or not root.is_dir():
        raise LocalProjectLoaderError(
            f"Local folder does not exist or is not a directory: {directory_path}"
        )

    raw_files: list[RawProjectFile] = []
    total_bytes = 0

    for current_dir, dirnames, filenames in os.walk(root, followlinks=False):
        # Prune excluded directories early during traversal
        dirnames[:] = [
            d
            for d in dirnames
            if d.lower() not in config.EXCLUDED_DIRS and not d.endswith(".egg-info")
        ]

        for fname in filenames:
            full_path = Path(current_dir) / fname
            if full_path.is_symlink():
                continue
            rel_path = normalize_rel_path(str(full_path.relative_to(root)))
            if not rel_path:
                continue

            try:
                data = full_path.read_bytes()
            except OSError as exc:
                logger.warning("Could not read file '%s': %s", rel_path, exc)
                continue

            total_bytes += len(data)
            if total_bytes > max_total_bytes:
                raise LocalProjectLoaderError(
                    f"Folder size exceeds the {max_total_bytes // (1024 * 1024)} MB safety limit."
                )

            raw_files.append(
                RawProjectFile(
                    relative_path=rel_path,
                    content=data,
                    size_bytes=len(data),
                )
            )
            if len(raw_files) >= max_files_count * 3:
                raise LocalProjectLoaderError(
                    "Folder contains more files than the maximum allowed limit."
                )

    if not raw_files:
        raise LocalProjectLoaderError(
            f"The selected folder '{root.name}' is empty or contains no readable files."
        )

    project_name = sanitize_project_name(root.name)
    return ProjectDocumentBundle(
        project_name=project_name,
        input_source="folder_upload",
        source_detail=str(root),
        raw_files=raw_files,
        metadata={
            "folder_path": str(root),
            "discovered_file_count": len(raw_files),
            "total_bytes": total_bytes,
        },
    )
