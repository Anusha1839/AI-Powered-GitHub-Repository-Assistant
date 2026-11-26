"""Unified end-to-end RAG pipeline connecting all three input methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import config
from src.chunking import CodeChunk, CodeChunker
from src.embeddings import EmbeddingManager
from src.gemini_llm import GeminiClient
from src.github_loader import GitHubLoader
from src.local_project_loader import (
    load_from_local_directory,
    load_from_uploaded_files,
    load_from_zip,
)
from src.repository_processor import (
    ProcessedRepository,
    ProjectDocumentBundle,
    RepositoryProcessor,
)
from src.reranker import CrossEncoderReranker, RerankedResult
from src.retriever import SemanticRetriever
from src.vector_store import FAISSVectorStoreManager
from utils.logger import get_logger

logger = get_logger(__name__)

ProgressCallback = Optional[Callable[[str, float], None]]


@dataclass(slots=True)
class SourceReference:
    """Verified source file reference and code snippet attached to an answer."""

    file_path: str
    file_name: str
    language: str
    start_line: int
    end_line: int
    snippet: str
    similarity_score: float
    rerank_score: float


@dataclass(slots=True)
class IndexedProject:
    """Result of processing, chunking, embedding, and indexing a project."""

    repository: ProcessedRepository
    chunks: list[CodeChunk]
    chunk_count: int
    embedding_model: str
    reranker_model: str
    vector_db_status: str
    indexing_status: str
    reused_existing_index: bool = False


@dataclass(slots=True)
class RAGAnswer:
    """Complete RAG response with generated text and verified source snippets."""

    question: str
    answer: str
    sources: list[SourceReference]
    retrieved_count: int
    reranked_count: int
    project_id: str
    model_used: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGPipeline:
    """Orchestrates unified ingestion, retrieval, reranking, and Gemini generation."""

    def __init__(
        self,
        github_loader: Optional[GitHubLoader] = None,
        processor: Optional[RepositoryProcessor] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        vector_store: Optional[FAISSVectorStoreManager] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        gemini_client: Optional[GeminiClient] = None,
    ) -> None:
        self.github_loader = github_loader or GitHubLoader()
        self.processor = processor or RepositoryProcessor()
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.vector_store = vector_store or FAISSVectorStoreManager()
        self.retriever = SemanticRetriever(self.embedding_manager, self.vector_store)
        self.reranker = reranker or CrossEncoderReranker()
        self.gemini_client = gemini_client or GeminiClient(
            fallback_models=config.FALLBACK_GEMINI_MODELS
        )

    @staticmethod
    def _notify(callback: ProgressCallback, message: str, progress: float) -> None:
        if callback is not None:
            callback(message, max(0.0, min(1.0, progress)))

    def ingest_bundle(
        self,
        bundle: ProjectDocumentBundle,
        chunk_size: int = config.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
        progress_callback: ProgressCallback = None,
    ) -> IndexedProject:
        """Run any ProjectDocumentBundle through the unified RAG indexing pipeline."""
        self._notify(progress_callback, "Filtering and extracting source files...", 0.20)
        processed_repo = self.processor.process(bundle)

        self._notify(progress_callback, "Chunking source code and documentation...", 0.45)
        chunker = CodeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk_repository(processed_repo)

        if self.vector_store.has_project(processed_repo.project_id) and (
            self.vector_store.get_chunk_count(processed_repo.project_id) == len(chunks)
        ):
            self._notify(progress_callback, "Reusing existing FAISS vector index...", 1.0)
            return IndexedProject(
                repository=processed_repo,
                chunks=chunks,
                chunk_count=len(chunks),
                embedding_model=self.embedding_manager.model_name,
                reranker_model=self.reranker.model_name,
                vector_db_status="Ready (FAISS IndexFlatIP)",
                indexing_status="Indexed (cached)",
                reused_existing_index=True,
            )

        self._notify(
            progress_callback,
            f"Loading embedding model & encoding {len(chunks)} chunks in batches...",
            0.60,
        )
        texts_to_embed = [chunk.to_embedding_text() for chunk in chunks]

        if progress_callback is not None:
            def _on_embed_batch(done: int, total: int) -> None:
                if total > 0:
                    pct = 0.60 + 0.25 * (done / total)
                    self._notify(
                        progress_callback,
                        f"Generated embeddings for {done}/{total} chunks...",
                        pct,
                    )

            embeddings = self.embedding_manager.embed_texts(
                texts_to_embed, progress_callback=_on_embed_batch
            )
        else:
            embeddings = self.embedding_manager.embed_texts(texts_to_embed)

        self._notify(progress_callback, "Building FAISS vector index...", 0.90)
        self.vector_store.index_project(
            project_id=processed_repo.project_id,
            chunks=chunks,
            embeddings=embeddings,
        )

        self._notify(progress_callback, "Project analysis complete!", 1.0)
        return IndexedProject(
            repository=processed_repo,
            chunks=chunks,
            chunk_count=len(chunks),
            embedding_model=self.embedding_manager.model_name,
            reranker_model=self.reranker.model_name,
            vector_db_status="Ready (FAISS IndexFlatIP)",
            indexing_status="Indexed",
            reused_existing_index=False,
        )

    def ingest_github_url(
        self,
        repo_url: str,
        chunk_size: int = config.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
        progress_callback: ProgressCallback = None,
    ) -> IndexedProject:
        """Load a public GitHub repository and process it through the unified pipeline."""
        self._notify(progress_callback, "Fetching GitHub repository...", 0.10)
        bundle = self.github_loader.load(repo_url)
        return self.ingest_bundle(
            bundle,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            progress_callback=progress_callback,
        )

    def ingest_zip(
        self,
        zip_source: Any,
        filename: Optional[str] = None,
        chunk_size: int = config.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
        progress_callback: ProgressCallback = None,
    ) -> IndexedProject:
        """Load a local ZIP project archive and process it through the unified pipeline."""
        self._notify(progress_callback, "Extracting ZIP archive safely...", 0.10)
        bundle = load_from_zip(zip_source, filename=filename)
        return self.ingest_bundle(
            bundle,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            progress_callback=progress_callback,
        )

    def ingest_uploaded_folder(
        self,
        uploaded_files: Sequence[Any],
        fallback_project_name: str = "local-project",
        chunk_size: int = config.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
        progress_callback: ProgressCallback = None,
    ) -> IndexedProject:
        """Load browser folder/multi-file uploads and process via the unified pipeline."""
        self._notify(progress_callback, "Reading uploaded local project files...", 0.10)
        bundle = load_from_uploaded_files(
            uploaded_files, fallback_project_name=fallback_project_name
        )
        return self.ingest_bundle(
            bundle,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            progress_callback=progress_callback,
        )

    def ingest_local_directory(
        self,
        directory_path: str | Path,
        chunk_size: int = config.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
        progress_callback: ProgressCallback = None,
    ) -> IndexedProject:
        """Load a local directory path on disk and process via the unified pipeline."""
        self._notify(progress_callback, "Scanning local project directory...", 0.10)
        bundle = load_from_local_directory(directory_path)
        return self.ingest_bundle(
            bundle,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            progress_callback=progress_callback,
        )

    @staticmethod
    def _build_source_references(
        reranked_results: Sequence[RerankedResult],
    ) -> list[SourceReference]:
        return [
            SourceReference(
                file_path=item.chunk.source_file_path,
                file_name=item.chunk.file_name,
                language=item.chunk.language,
                start_line=item.chunk.start_line,
                end_line=item.chunk.end_line,
                snippet=item.chunk.text,
                similarity_score=item.similarity_score,
                rerank_score=item.rerank_score,
            )
            for item in reranked_results
        ]

    def answer_question(
        self,
        indexed_project: IndexedProject,
        question: str,
        top_k_retrieval: int = config.DEFAULT_TOP_K_RETRIEVAL,
        top_k_rerank: int = config.DEFAULT_TOP_K_RERANK,
        chat_history: Optional[Sequence[dict[str, str]]] = None,
    ) -> RAGAnswer:
        """Retrieve, rerank, and generate a grounded answer for an indexed project."""
        project_id = indexed_project.repository.project_id
        retrieved = self.retriever.retrieve(
            project_id=project_id,
            question=question,
            top_k=top_k_retrieval,
        )

        if not retrieved:
            return RAGAnswer(
                question=question,
                answer=(
                    "No matching code or documentation snippets were found in the indexed "
                    "project for that question."
                ),
                sources=[],
                retrieved_count=0,
                reranked_count=0,
                project_id=project_id,
                model_used=self.gemini_client.model_name,
            )

        reranked = self.reranker.rerank(
            question=question,
            candidates=retrieved,
            top_k=top_k_rerank,
        )

        answer_text = self.gemini_client.generate_answer(
            project_name=indexed_project.repository.project_name,
            input_source=indexed_project.repository.input_source,
            file_inventory=indexed_project.repository.file_paths,
            question=question,
            reranked_results=reranked,
            chat_history=chat_history,
        )

        sources = self._build_source_references(reranked)
        return RAGAnswer(
            question=question,
            answer=answer_text,
            sources=sources,
            retrieved_count=len(retrieved),
            reranked_count=len(reranked),
            project_id=project_id,
            model_used=getattr(
                self.gemini_client, "last_model_used", self.gemini_client.model_name
            ),
        )

    def reset_project(self, project_id: str) -> None:
        """Clear a single project's vector index."""
        self.vector_store.clear_project(project_id)

    def reset_all(self) -> None:
        """Clear all indexed projects from the vector store."""
        self.vector_store.clear_all()
