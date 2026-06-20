"""
backend/retrieval/engine.py
============================
Retrieval pipeline: embed -> hybrid_search -> rerank -> format_context.

store is typed as QdrantStore. weaviate_store.py no longer exists as an
importable module (weaviate-client was removed from pyproject.toml during
the migration), so this import is load-bearing -- this exact line is what
was crashing on startup.
"""

from __future__ import annotations

import asyncio
import logging

from backend.retrieval.embedder import Embedder
from backend.retrieval.reranker import Reranker, RankedResult
from backend.retrieval.qdrant_store import QdrantStore, SearchHit
from backend.shared.config import RetrievalSettings

logger = logging.getLogger(__name__)


class RetrievalEngine:
    def __init__(
        self,
        store: QdrantStore,
        embedder: Embedder,
        reranker: Reranker,
        settings: RetrievalSettings,
    ) -> None:
        self._store    = store
        self._embedder = embedder
        self._reranker = reranker
        self._settings = settings

    async def retrieve(
        self,
        query: str,
        alpha_override: float | None = None,
    ) -> list[RankedResult]:
        """
        Full embed -> search -> rerank pipeline. Fully async:
        encode_one runs in a thread pool, hybrid_search is awaited on
        QdrantStore's native AsyncQdrantClient, reranker runs in a thread pool.
        """
        if not query.strip():
            return []

        vector: list[float] = await asyncio.to_thread(self._embedder.encode_one, query)

        alpha = alpha_override if alpha_override is not None else self._settings.hybrid_alpha
        hits: list[SearchHit] = await self._store.hybrid_search(
            query_text=query,
            query_vector=vector,
            top_k=self._settings.top_k,
            alpha=alpha,
        )
        logger.debug("Qdrant search returned %d candidates.", len(hits))

        if not hits:
            return []

        passages = [
            {
                "content":     h.content,
                "document_id": h.document_id,
                "filename":    h.filename,
                "heading":     h.heading,
                "page":        h.page,
                "chunk_index": h.chunk_index,
                "weaviate_id": h.weaviate_id,
            }
            for h in hits
        ]

        ranked: list[RankedResult] = await asyncio.to_thread(
            self._reranker.rerank,
            query,
            passages,
            self._settings.rerank_top_n,
        )
        logger.debug("Reranker returned %d passages.", len(ranked))
        return ranked

    def format_context(self, results: list[RankedResult]) -> str:
        if not results:
            return ""
        parts: list[str] = ["<retrieved_context>"]
        for i, r in enumerate(results, start=1):
            source  = r.metadata.get("filename", "unknown")
            heading = r.metadata.get("heading", "")
            page    = r.metadata.get("page", "")
            header  = f"[{i}] {source}"
            if heading:
                header += f" - {heading}"
            if page:
                header += f" (p.{page})"
            parts.append(f"{header}\n{r.content}")
        parts.append("</retrieved_context>")
        return "\n\n".join(parts)
