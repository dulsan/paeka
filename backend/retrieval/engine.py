"""
backend/retrieval/engine.py
============================
Retrieval pipeline:

    query
      └─ embed (BGE-M3 dense vector)
      └─ hybrid_search (Weaviate: dense + BM25 fusion)  →  top_k candidates
      └─ rerank (BGE-Reranker-Large)                    →  top_n passages
      └─ format_context                                 →  str injected into prompt

All heavy models are singletons (loaded once via lru_cache).
"""

from __future__ import annotations

import logging

from backend.retrieval.embedder import Embedder
from backend.retrieval.reranker import Reranker, RankedResult
from backend.retrieval.weaviate_store import WeaviateStore, SearchHit
from backend.shared.config import RetrievalSettings

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    Orchestrates the full retrieval pipeline.

    Parameters
    ----------
    store:
        Connected WeaviateStore instance.
    embedder:
        Loaded Embedder instance.
    reranker:
        Loaded Reranker instance.
    settings:
        RetrievalSettings from the config.
    """

    def __init__(
        self,
        store: WeaviateStore,
        embedder: Embedder,
        reranker: Reranker,
        settings: RetrievalSettings,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, alpha_override: float | None = None) -> list[RankedResult]:
        """
        Run the full embed → search → rerank pipeline.

        Parameters
        ----------
        query:
            The user's natural-language query.

        Returns
        -------
        list[RankedResult]
            Top-N passages sorted by relevance score descending.
        """
        if not query.strip():
            return []

        # 1. Dense embedding
        vector = self._embedder.encode_one(query)

        # 2. Hybrid search
        alpha = alpha_override if alpha_override is not None else self._settings.hybrid_alpha
        hits: list[SearchHit] = self._store.hybrid_search(
            query_text=query,
            query_vector=vector,
            top_k=self._settings.top_k,
            alpha=alpha,
        )
        logger.debug("Hybrid search returned %d candidates.", len(hits))

        if not hits:
            return []

        # 3. Rerank
        passages = [
            {
                "content": h.content,
                "document_id": h.document_id,
                "filename": h.filename,
                "heading": h.heading,
                "page": h.page,
                "chunk_index": h.chunk_index,
                "weaviate_id": h.weaviate_id,
            }
            for h in hits
        ]
        ranked = self._reranker.rerank(
            query=query,
            passages=passages,
            top_n=self._settings.rerank_top_n,
        )
        logger.debug("Reranker returned %d passages.", len(ranked))
        return ranked

    def format_context(self, results: list[RankedResult]) -> str:
        """
        Format ranked results as a context block to inject into the LLM prompt.

        Each passage is wrapped with source metadata so the model can cite it.
        """
        if not results:
            return ""

        parts: list[str] = ["<retrieved_context>"]
        for i, r in enumerate(results, start=1):
            source = r.metadata.get("filename", "unknown")
            heading = r.metadata.get("heading", "")
            page = r.metadata.get("page", "")
            header = f"[{i}] {source}"
            if heading:
                header += f" — {heading}"
            if page:
                header += f" (p.{page})"
            parts.append(f"{header}\n{r.content}")

        parts.append("</retrieved_context>")
        return "\n\n".join(parts)
