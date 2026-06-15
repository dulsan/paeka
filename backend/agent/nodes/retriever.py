"""
backend/agent/nodes/retriever.py
=================================
Retriever node for the AgenticRAGPipeline.

Fixes applied:
  [FIX-PARALLEL] Sub-queries were dispatched sequentially. All independent
                 queries now run concurrently via asyncio.gather(), reducing
                 N×latency to ~1×latency for typical 4-6 sub-query plans.
  [FIX-ASYNC]    _run_vector() was def (synchronous). It called
                 engine.retrieve() which internally blocked the event loop.
                 Now declared async; engine.retrieve() is awaited (engine.py
                 is async after this review pass).
"""

from __future__ import annotations

import asyncio
import logging

from backend.agent.state import AgentState, RetrievalResult, SubQuery
from backend.knowledge.retriever import GraphRetriever
from backend.retrieval.engine import RetrievalEngine
from backend.tools.searxng import SearXNGClient

logger = logging.getLogger(__name__)


async def retriever_node(
    state: AgentState,
    engine: RetrievalEngine,
    graph_retriever: GraphRetriever | None = None,
    web_client: SearXNGClient | None = None,
) -> AgentState:
    """
    Dispatch all sub-queries from the planner concurrently and merge results.
    """
    sub_queries: list[SubQuery] = state.get("sub_queries", [])
    if not sub_queries:
        logger.warning("Retriever: no sub-queries in state — skipping.")
        return {**state, "retrieved_passages": [], "hop_count": 0}

    # [FIX-PARALLEL] Dispatch all sub-queries at once instead of sequentially.
    # Each query is independent I/O: vector search + rerank.
    # asyncio.gather runs them all concurrently; total latency ≈ max(individual).
    tasks = [
        _dispatch(sq, engine, graph_retriever, web_client)
        for sq in sorted(sub_queries, key=lambda x: x.get("priority", 0))
    ]
    per_query_results: list[list[RetrievalResult]] = await asyncio.gather(*tasks)

    # Merge and deduplicate by content hash
    seen: set[str] = set()
    merged: list[RetrievalResult] = []
    for results in per_query_results:
        for r in results:
            key = _dedup_key(r)
            if key not in seen:
                seen.add(key)
                merged.append(r)

    # Sort by score descending
    merged.sort(key=lambda r: r.get("score", 0.0), reverse=True)

    logger.info(
        "Retriever: %d sub-queries → %d unique passages (hop %d)",
        len(sub_queries), len(merged), state.get("hop_count", 0),
    )
    return {
        **state,
        "retrieved_passages": merged,
        "hop_count": state.get("hop_count", 0) + 1,
    }


async def _dispatch(
    sq: SubQuery,
    engine: RetrievalEngine,
    graph_retriever: GraphRetriever | None,
    web_client: SearXNGClient | None,
) -> list[RetrievalResult]:
    tool = sq.get("tool", "vector")
    q    = sq.get("query", "")
    try:
        if tool == "vector":
            return await _run_vector(q, engine)
        elif tool == "graph":
            return await _run_graph(q, graph_retriever)
        elif tool == "web":
            return await _run_web(q, web_client)
        elif tool == "keyword":
            # Keyword search: pure BM25 (alpha=0.0 → no vector weight)
            return await _run_vector(q, engine, alpha_override=0.0)
        else:
            logger.warning("Unknown retrieval tool '%s' — falling back to vector", tool)
            return await _run_vector(q, engine)
    except Exception as exc:
        logger.error("Retrieval dispatch failed for tool='%s' query='%s': %s", tool, q[:60], exc)
        return []


# [FIX-ASYNC] Was `def` — now `async def`. engine.retrieve() is awaited.
async def _run_vector(
    query: str,
    engine: RetrievalEngine,
    alpha_override: float | None = None,
) -> list[RetrievalResult]:
    ranked = await engine.retrieve(query, alpha_override=alpha_override)
    return [
        RetrievalResult(
            content=r.content,
            source=r.metadata.get("filename", "unknown"),
            heading=r.metadata.get("heading", ""),
            page=int(r.metadata.get("page", 0)),
            score=float(r.score),
            element_type=r.metadata.get("element_type", "text"),
            trust_tier="local",
        )
        for r in ranked
    ]


async def _run_graph(
    query: str,
    graph_retriever: GraphRetriever | None,
) -> list[RetrievalResult]:
    if graph_retriever is None:
        return []
    try:
        hits = await asyncio.to_thread(graph_retriever.retrieve, query)
        return [
            RetrievalResult(
                content=h.content,
                source=h.source,
                heading=h.heading if hasattr(h, "heading") else "",
                page=0,
                score=float(h.score) if hasattr(h, "score") else 0.5,
                element_type="graph_node",
                trust_tier="graph",
            )
            for h in hits
        ]
    except Exception as exc:
        logger.error("Graph retrieval failed: %s", exc)
        return []


async def _run_web(
    query: str,
    web_client: SearXNGClient | None,
) -> list[RetrievalResult]:
    if web_client is None:
        return []
    try:
        results = await web_client.search(query, num_results=3)
        return [
            RetrievalResult(
                content=r.content,
                source=r.url,
                heading=r.title,
                page=0,
                score=0.6,
                element_type="web",
                trust_tier="web",
            )
            for r in results
        ]
    except Exception as exc:
        logger.error("Web retrieval failed: %s", exc)
        return []


def _dedup_key(r: RetrievalResult) -> str:
    # Stable dedup key: first 120 chars of content (avoids full hash cost)
    return r.get("content", "")[:120]
