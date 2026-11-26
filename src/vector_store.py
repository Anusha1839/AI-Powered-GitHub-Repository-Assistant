"""FAISS CPU vector store managing isolated per-project indexes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import faiss
import numpy as np

from src.chunking import CodeChunk
from utils.logger import get_logger

logger = get_logger(__name__)


class VectorStoreError(Exception):
    """Raised when FAISS index creation or search fails."""


@dataclass(slots=True)
class VectorSearchResult:
    """Single similarity search hit returned by FAISS."""

    chunk: CodeChunk
    similarity_score: float
    rank: int


@dataclass
class ProjectVectorIndex:
    """Isolated FAISS index and chunk mapping for one project."""

    project_id: str
    dimension: int
    index: faiss.Index
    chunks: list[CodeChunk]


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Ensure float32 contiguous array with unit L2 norm for cosine similarity via IP."""
    arr = np.ascontiguousarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


class FAISSVectorStoreManager:
    """Stores and queries isolated FAISS vector indexes keyed by project_id."""

    def __init__(self) -> None:
        self._indexes: dict[str, ProjectVectorIndex] = {}

    def has_project(self, project_id: str) -> bool:
        """Return True if a project already has a built FAISS index."""
        return project_id in self._indexes

    def get_chunk_count(self, project_id: str) -> int:
        """Return the number of indexed chunks for a project."""
        entry = self._indexes.get(project_id)
        return len(entry.chunks) if entry else 0

    def index_project(
        self,
        project_id: str,
        chunks: list[CodeChunk],
        embeddings: np.ndarray,
        force: bool = False,
    ) -> ProjectVectorIndex:
        """Build and store a FAISS IndexFlatIP cosine-similarity index for a project."""
        if not force and project_id in self._indexes:
            existing = self._indexes[project_id]
            if len(existing.chunks) == len(chunks):
                logger.info(
                    "Reusing existing FAISS index for unchanged project '%s' (%d chunks).",
                    project_id,
                    len(chunks),
                )
                return existing

        if not chunks:
            raise VectorStoreError("Cannot build a FAISS index from empty chunks.")

        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise VectorStoreError(
                f"Embedding shape mismatch: got {embeddings.shape} for {len(chunks)} chunks."
            )

        dimension = int(embeddings.shape[1])
        if dimension <= 0:
            raise VectorStoreError("Embedding vectors must have positive dimensionality.")

        try:
            normalized = _normalize_vectors(embeddings)
            index = faiss.IndexFlatIP(dimension)
            index.add(normalized)
        except Exception as exc:
            raise VectorStoreError(f"Failed to build FAISS index: {exc}") from exc

        project_index = ProjectVectorIndex(
            project_id=project_id,
            dimension=dimension,
            index=index,
            chunks=list(chunks),
        )
        self._indexes[project_id] = project_index
        logger.info(
            "Indexed %d chunks (dim=%d) in FAISS for project '%s'.",
            len(chunks),
            dimension,
            project_id,
        )
        return project_index

    def search(
        self,
        project_id: str,
        query_vector: np.ndarray,
        top_k: int = 15,
    ) -> list[VectorSearchResult]:
        """Search a specific project's FAISS index and return ranked chunks."""
        entry = self._indexes.get(project_id)
        if entry is None:
            raise VectorStoreError(
                f"No vector index found for project '{project_id}'. Analyze a project first."
            )

        if not entry.chunks or entry.index.ntotal == 0:
            return []

        k = max(1, min(int(top_k), len(entry.chunks)))
        query_norm = _normalize_vectors(query_vector)
        if query_norm.shape[1] != entry.dimension:
            raise VectorStoreError(
                f"Query vector dimension ({query_norm.shape[1]}) does not match "
                f"index dimension ({entry.dimension})."
            )

        try:
            scores, indices = entry.index.search(query_norm, k)
        except Exception as exc:
            raise VectorStoreError(f"FAISS similarity search failed: {exc}") from exc

        results: list[VectorSearchResult] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0 or idx >= len(entry.chunks):
                continue
            results.append(
                VectorSearchResult(
                    chunk=entry.chunks[int(idx)],
                    similarity_score=float(score),
                    rank=rank,
                )
            )
        return results

    def clear_project(self, project_id: str) -> None:
        """Remove a specific project's index from memory."""
        if project_id in self._indexes:
            del self._indexes[project_id]
            logger.info("Cleared FAISS index for project '%s'.", project_id)

    def clear_all(self) -> None:
        """Clear all project indexes from memory."""
        self._indexes.clear()
        logger.info("Cleared all FAISS project indexes.")
