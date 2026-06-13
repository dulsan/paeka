"""
backend/agent/nodes/retriever.py
==================================
Retriever node — dispatches sub-queries to four tools:
  "vector"  — dense hybrid search (Weaviate + BGE-M3)
  "graph"   — knowledge graph entity lookup
  "keyword" — BM25-heavy hybrid search
  "web"     — SearXNG live web search

trust_tier values (used by Critic for confidence weighting):
  "local"   — retrieved from local vector store
  "graph"   — retrieved from knowledge graph
  "web"     — retrieved from SearXNG (lowest trust)
"""

from __future__ import annotations

import hashlib
import logging

from backend.agent.state import AgentState, RetrievalResult, SubQuery
from backend.retrieval.engine import RetrievalEngine
from backend.retrieval.reranker import RankedResult
from backend.knowledge.retriever import GraphRetriever

logger = logging.getLogger(__name__)

_MAX_PER_SUBQUERY = 5


async def retriever_node(
    state: AgentState,
    retrieval_engine: RetrievalEngine | None,
    graph_retriever: GraphRetriever | None,
    web_client=None,   # SearXNGClient | None
) -> AgentState:
    sub_queries: list[SubQuery] = state.get("sub_queries", [])
    existing: list[RetrievalResult] = state.get("retrieved_passages", [])
    hop = state.get("hop_count", 0)

    if not retrieval_engine and not graph_retriever and not web_client:
        logger.warning("Retriever: no tools available — skipping.")
        return {**state, "hop_count": hop + 1}

    seen: set[str] = {_hash(p["content"]) for p in existing}
    new_passages: list[RetrievalResult] = []

    for sq in sorted(sub_queries, key=lambda x: x["priority"]):
        results = await _dispatch(sq, retrieval_engine, graph_retriever, web_client)
        for r in results:
            h = _hash(r["content"])
            if h not in seen:
                seen.add(h)
                new_passages.append(r)

    all_passages = existing + new_passages
    logger.info(
        "Retriever hop %d: +%d new (%d total) | tools used: %s",
        hop + 1, len(new_passages), len(all_passages),
        list({sq["tool"] for sq in sub_queries}),
    )
    return {**state, "retrieved_passages": all_passages, "hop_count": hop + 1}


async def _dispatch(
    sq: SubQuery,
    engine: RetrievalEngine | None,
    graph: GraphRetriever | None,
    web: object | None,
) -> list[RetrievalResult]:
    tool = sq["tool"]

    if tool == "graph" and graph is not None:
        return await _run_graph(sq["query"], graph)

    if tool == "web" and web is not None:
        return await _run_web(sq["query"], web)

    if engine is not None:
        alpha = 0.1 if tool == "keyword" else None
        return _run_vector(sq["query"], engine, alpha)

    return []


def _run_vector(
    query: str,
    engine: RetrievalEngine,
    alpha: float | None,
) -> list[RetrievalResult]:
    try:
        ranked = engine.retrieve(query, alpha_override=alpha)
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
            for r in ranked[:_MAX_PER_SUBQUERY]
        ]
    except Exception as exc:  # noqa: BLE001
        logger.debug("Vector retrieval error: %s", exc)
        return []


async def _run_graph(query: str, graph: GraphRetriever) -> list[RetrievalResult]:
    try:
        gc = await graph.query(query)
        return [
            RetrievalResult(
                content=f"{e['label']} ({e['type']}): {e.get('description', '')}",
                source="knowledge_graph",
                heading=e["type"],
                page=0,
                score=1.0,
                element_type="graph_entity",
                trust_tier="graph",
            )
            for e in gc.entities[:_MAX_PER_SUBQUERY]
        ]
    except Exception as exc:  # noqa: BLE001
        logger.debug("Graph retrieval error: %s", exc)
        return []


async def _run_web(query: str, web_client) -> list[RetrievalResult]:
    try:
        results = await web_client.search(query, num_results=_MAX_PER_SUBQUERY)
        return [
            RetrievalResult(
                content=r.content or r.snippet,
                source=r.url,
                heading=r.title,
                page=0,
                score=0.7,           # fixed prior — web results lack relevance scores
                element_type="web",
                trust_tier="web",
            )
            for r in results
        ]
    except Exception as exc:  # noqa: BLE001
        logger.debug("Web retrieval error: %s", exc)
        return []


def _hash(text: str) -> str:
    return hashlib.md5(text[:200].encode(), usedforsecurity=False).hexdigest()
