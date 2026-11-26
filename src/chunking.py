"""Line-aware source code and documentation chunking module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import config
from src.repository_processor import ProcessedFile, ProcessedRepository
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class CodeChunk:
    """Single chunk of source code or documentation with full provenance metadata."""

    chunk_id: str
    source_file_path: str
    file_name: str
    language: str
    text: str
    chunk_index: int
    start_line: int
    end_line: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_embedding_text(self) -> str:
        """Format chunk with file path and line range context for embedding."""
        return (
            f"File: {self.source_file_path} "
            f"(Language: {self.language}, Lines {self.start_line}-{self.end_line})\n"
            f"{self.text}"
        )


class CodeChunker:
    """Splits processed repository files into overlapping, line-indexed chunks."""

    def __init__(
        self,
        chunk_size: int = config.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        if chunk_size < config.MIN_CHUNK_SIZE:
            raise ValueError(
                f"chunk_size must be at least {config.MIN_CHUNK_SIZE} characters."
            )
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be non-negative and smaller than chunk_size."
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_file(self, file: ProcessedFile) -> list[CodeChunk]:
        """Split a single ProcessedFile into overlapping CodeChunk objects."""
        lines = file.content.splitlines(keepends=True)
        if not lines:
            return []

        chunks: list[CodeChunk] = []
        total_lines = len(lines)
        idx = 0
        chunk_index = 0

        while idx < total_lines:
            current_line = lines[idx]

            # Handle an unusually long single line by slicing it safely
            if len(current_line) > self.chunk_size:
                step = self.chunk_size - self.chunk_overlap
                for offset in range(0, len(current_line), step):
                    slice_text = current_line[offset : offset + self.chunk_size].strip("\r\n")
                    if slice_text.strip():
                        chunks.append(
                            CodeChunk(
                                chunk_id=f"{file.relative_path}::chunk_{chunk_index}",
                                source_file_path=file.relative_path,
                                file_name=file.file_name,
                                language=file.language,
                                text=slice_text,
                                chunk_index=chunk_index,
                                start_line=idx + 1,
                                end_line=idx + 1,
                                metadata={
                                    "extension": file.extension,
                                    "file_size_bytes": file.size_bytes,
                                    "file_line_count": file.line_count,
                                },
                            )
                        )
                        chunk_index += 1
                    if offset + self.chunk_size >= len(current_line):
                        break
                idx += 1
                continue

            start_idx = idx
            char_count = 0
            end_idx = idx

            while end_idx < total_lines:
                line_len = len(lines[end_idx])
                if char_count > 0 and char_count + line_len > self.chunk_size:
                    break
                char_count += line_len
                end_idx += 1

            raw_chunk_text = "".join(lines[start_idx:end_idx]).rstrip("\r\n")
            if raw_chunk_text.strip():
                chunks.append(
                    CodeChunk(
                        chunk_id=f"{file.relative_path}::chunk_{chunk_index}",
                        source_file_path=file.relative_path,
                        file_name=file.file_name,
                        language=file.language,
                        text=raw_chunk_text,
                        chunk_index=chunk_index,
                        start_line=start_idx + 1,
                        end_line=end_idx,
                        metadata={
                            "extension": file.extension,
                            "file_size_bytes": file.size_bytes,
                            "file_line_count": file.line_count,
                        },
                    )
                )
                chunk_index += 1

            if end_idx >= total_lines:
                break

            # Compute overlap rewind while guaranteeing forward progress
            if self.chunk_overlap == 0:
                idx = end_idx
            else:
                overlap_chars = 0
                next_idx = end_idx
                while next_idx > start_idx + 1:
                    prev_len = len(lines[next_idx - 1])
                    if overlap_chars + prev_len > self.chunk_overlap:
                        break
                    overlap_chars += prev_len
                    next_idx -= 1
                idx = next_idx if next_idx > start_idx else end_idx

        return chunks

    def chunk_repository(self, repository: ProcessedRepository) -> list[CodeChunk]:
        """Chunk all files in a ProcessedRepository."""
        all_chunks: list[CodeChunk] = []
        for file in repository.files:
            all_chunks.extend(self.chunk_file(file))

        logger.info(
            "Generated %d chunks across %d files for project '%s'.",
            len(all_chunks),
            len(repository.files),
            repository.project_name,
        )
        return all_chunks
