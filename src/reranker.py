"""Cross-Encoder reranking module using cross-encoder/ms-marco-MiniLM-L-6-v2."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import config
from src.chunking import CodeChunk
from src.vector_store import VectorSearchResult
from utils.helpers import ensure_windows_dll_paths
from utils.logger import get_logger

logger = get_logger(__name__)


class RerankerError(Exception):
    """Raised when Cross-Encoder model loading or reranking fails."""


@dataclass(slots=True)
class RerankedResult:
    """Candidate chunk enriched with Cross-Encoder relevance score."""

    chunk: CodeChunk
    similarity_score: float
    rerank_score: float
    rank: int


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    """Load CrossEncoder from local cache first, downloading only if needed."""
    ensure_windows_dll_paths()
    try:
        from sentence_transformers import CrossEncoder
    except Exception as exc:
        raise RerankerError(
            f"Failed to import CrossEncoder / PyTorch runtime: {exc}"
        ) from exc

    try:
        logger.info("Checking local cache for Cross-Encoder '%s'...", model_name)
        model = CrossEncoder(model_name, device="cpu", local_files_only=True)
        logger.info("Loaded Cross-Encoder '%s' from local cache.", model_name)
        return model
    except Exception:
        logger.info(
            "Cross-Encoder '%s' not found in local cache; downloading on CPU...",
            model_name,
        )

    try:
        return CrossEncoder(model_name, device="cpu")
    except Exception as exc:
        raise RerankerError(
            f"Failed to load or download Cross-Encoder model '{model_name}'. "
            f"Verify internet access for initial download or local cache availability. "
            f"Details: {exc}"
        ) from exc


class CrossEncoderReranker:
    """Reranks semantic retrieval candidates using a Cross-Encoder model."""

    def __init__(
        self,
        model_name: str = config.RERANKER_MODEL_NAME,
        max_candidates: int = config.MAX_RERANK_CANDIDATES,
        max_text_chars: int = config.MAX_RERANK_TEXT_CHARS,
    ) -> None:
        self.model_name = model_name
        self.max_candidates = max_candidates
        self.max_text_chars = max_text_chars

    def _get_model(self):
        return _load_cross_encoder(self.model_name)

    def warmup(self) -> None:
        """Explicitly load and cache the Cross-Encoder model."""
        self._get_model()

    def rerank(
        self,
        question: str,
        candidates: list[VectorSearchResult],
        top_k: int = config.DEFAULT_TOP_K_RERANK,
    ) -> list[RerankedResult]:
        """Score and reorder retrieved chunks against the user question."""
        cleaned_question = (question or "").strip()
        if not cleaned_question or not candidates:
            return []

        bounded_candidates = candidates[: self.max_candidates]
        pairs = [
            (
                cleaned_question,
                item.chunk.to_embedding_text()[: self.max_text_chars],
            )
            for item in bounded_candidates
        ]

        model = self._get_model()
        try:
            scores = model.predict(pairs)
        except Exception as exc:
            raise RerankerError(f"Cross-Encoder reranking failed: {exc}") from exc

        scored = [
            (candidate, float(score))
            for candidate, score in zip(bounded_candidates, scores)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)

        limit = max(1, min(int(top_k), len(scored)))
        reranked: list[RerankedResult] = []
        for rank, (candidate, r_score) in enumerate(scored[:limit], start=1):
            reranked.append(
                RerankedResult(
                    chunk=candidate.chunk,
                    similarity_score=candidate.similarity_score,
                    rerank_score=r_score,
                    rank=rank,
                )
            )

        logger.info(
            "Reranked %d candidates down to top %d results.",
            len(bounded_candidates),
            len(reranked),
        )
        return reranked
