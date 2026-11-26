"""Tests for FAISS vector storage and semantic retrieval."""

from __future__ import annotations

import numpy as np
import pytest

from src.chunking import CodeChunk
from src.retriever import RetrievalError, SemanticRetriever
from src.vector_store import FAISSVectorStoreManager, VectorStoreError


class _StubEmbeddingManager:
    model_name = "stub-minilm"

    def embed_query(self, query: str) -> np.ndarray:
        q = query.lower()
        if "auth" in q:
            return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        if "database" in q:
            return np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        return np.array([[0.0, 0.0, 1.0]], dtype=np.float32)


def _chunk(cid: str, path: str, text: str) -> CodeChunk:
    return CodeChunk(
        chunk_id=cid,
        source_file_path=path,
        file_name=path.split("/")[-1],
        language="python",
        text=text,
        chunk_index=0,
        start_line=1,
        end_line=3,
    )


class TestVectorStoreAndRetriever:
    def test_index_search_and_project_isolation(self) -> None:
        store = FAISSVectorStoreManager()
        chunks_a = [
            _chunk("a1", "auth.py", "def login(): pass"),
            _chunk("a2", "db.py", "def connect_db(): pass"),
        ]
        vecs_a = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        store.index_project("proj_a", chunks_a, vecs_a)

        chunks_b = [_chunk("b1", "other.py", "def other(): pass")]
        vecs_b = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        store.index_project("proj_b", chunks_b, vecs_b)

        retriever = SemanticRetriever(_StubEmbeddingManager(), store)  # type: ignore[arg-type]
        hits_a = retriever.retrieve("proj_a", "How does authentication work?", top_k=2)
        assert hits_a[0].chunk.source_file_path == "auth.py"
        assert hits_a[0].similarity_score > hits_a[1].similarity_score

        hits_b = retriever.retrieve("proj_b", "How does authentication work?", top_k=2)
        assert len(hits_b) == 1
        assert hits_b[0].chunk.source_file_path == "other.py"

    def test_clear_project_removes_index(self) -> None:
        store = FAISSVectorStoreManager()
        store.index_project(
            "proj_x",
            [_chunk("x1", "app.py", "print(1)")],
            np.array([[1.0, 0.0]], dtype=np.float32),
        )
        assert store.has_project("proj_x")
        store.clear_project("proj_x")
        assert not store.has_project("proj_x")

        retriever = SemanticRetriever(_StubEmbeddingManager(), store)  # type: ignore[arg-type]
        with pytest.raises(RetrievalError):
            retriever.retrieve("proj_x", "What does app.py do?")

    def test_empty_question_raises(self) -> None:
        store = FAISSVectorStoreManager()
        retriever = SemanticRetriever(_StubEmbeddingManager(), store)  # type: ignore[arg-type]
        with pytest.raises(RetrievalError, match="non-empty"):
            retriever.retrieve("any", "   ")

    def test_dimension_mismatch_raises(self) -> None:
        store = FAISSVectorStoreManager()
        store.index_project(
            "p1",
            [_chunk("1", "a.py", "a=1")],
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        )
        with pytest.raises(VectorStoreError):
            store.search("p1", np.array([[1.0, 0.0]], dtype=np.float32))
