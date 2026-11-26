"""Sentence Transformers embedding generation with local cache reuse and progress support."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Optional, Sequence

import numpy as np

import config
from utils.helpers import ensure_windows_dll_paths
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingError(Exception):
    """Raised when the embedding model fails to load or encode text."""


@lru_cache(maxsize=2)
def _load_sentence_transformer(model_name: str):
    """Load SentenceTransformer from local cache first, downloading only if needed."""
    ensure_windows_dll_paths()
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise EmbeddingError(
            f"Failed to import Sentence Transformers / PyTorch runtime: {exc}"
        ) from exc

    # Prefer local Hugging Face cache to avoid redundant network calls
    try:
        logger.info("Checking local cache for embedding model '%s'...", model_name)
        model = SentenceTransformer(model_name, device="cpu", local_files_only=True)
        logger.info("Loaded embedding model '%s' from local cache.", model_name)
        return model
    except Exception:
        logger.info(
            "Embedding model '%s' not found in local cache; downloading on CPU...",
            model_name,
        )

    try:
        return SentenceTransformer(model_name, device="cpu")
    except Exception as exc:
        raise EmbeddingError(
            f"Failed to load or download Sentence Transformer model '{model_name}'. "
            f"Verify internet access for initial model download or local cache availability. "
            f"Details: {exc}"
        ) from exc


class EmbeddingManager:
    """Generates L2-normalized dense embeddings for repository chunks and queries."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL_NAME) -> None:
        self.model_name = model_name

    def _get_model(self):
        return _load_sentence_transformer(self.model_name)

    def warmup(self) -> None:
        """Explicitly load and cache the embedding model."""
        self._get_model()

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = config.EMBEDDING_BATCH_SIZE,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> np.ndarray:
        """Encode texts in batches into a float32 L2-normalized numpy matrix."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        model = self._get_model()
        total = len(texts)
        eff_batch = max(1, batch_size)
        batches: list[np.ndarray] = []

        try:
            for start in range(0, total, eff_batch):
                slice_texts = list(texts[start : start + eff_batch])
                vectors = model.encode(
                    slice_texts,
                    batch_size=eff_batch,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )
                arr = np.asarray(vectors, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                batches.append(arr)
                if progress_callback is not None:
                    progress_callback(min(start + len(slice_texts), total), total)
            return np.vstack(batches) if len(batches) > 1 else batches[0]
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to generate embeddings for {total} text items: {exc}"
            ) from exc

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single natural-language query into a normalized 2D vector (1, dim)."""
        cleaned = (query or "").strip()
        if not cleaned:
            raise EmbeddingError("Cannot generate embedding for an empty question.")
        return self.embed_texts([cleaned], batch_size=1)
