"""Semantic retrieval module connecting query embeddings to the FAISS vector store."""

from __future__ import annotations

import config
from src.embeddings import EmbeddingError, EmbeddingManager
from src.vector_store import FAISSVectorStoreManager, VectorSearchResult, VectorStoreError
from utils.logger import get_logger

logger = get_logger(__name__)


class RetrievalError(Exception):
    """Raised when semantic retrieval cannot be executed."""


class SemanticRetriever:
    """Retrieves top-K semantically similar code chunks for a natural-language question."""

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        vector_store: FAISSVectorStoreManager,
    ) -> None:
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store

    def retrieve(
        self,
        project_id: str,
        question: str,
        top_k: int = config.DEFAULT_TOP_K_RETRIEVAL,
    ) -> list[VectorSearchResult]:
        """Embed the question and retrieve top-K chunks from the project's FAISS index."""
        cleaned = (question or "").strip()
        if not cleaned:
            raise RetrievalError("Please enter a non-empty question.")

        if not self.vector_store.has_project(project_id):
            raise RetrievalError(
                "No analyzed project index is loaded. Please analyze a repository first."
            )

        try:
            query_vector = self.embedding_manager.embed_query(cleaned)
            results = self.vector_store.search(
                project_id=project_id,
                query_vector=query_vector,
                top_k=top_k,
            )
        except (EmbeddingError, VectorStoreError) as exc:
            raise RetrievalError(str(exc)) from exc

        logger.info(
            "Retrieved %d candidate chunks for project '%s'.",
            len(results),
            project_id,
        )
        return results
