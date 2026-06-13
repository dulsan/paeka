"""
backend/agent/nodes/synthesiser.py
=====================================
Synthesiser node — generates the final answer from approved passages
and knowledge graph context.

This is the last node in the agentic RAG loop. It:
  1. Formats all approved passages + graph context into a context block.
  2. Issues a single, non-streaming LLM call to compose the answer.
  3. Extracts citation metadata for the API response.

Note: The synthesiser produces the complete answer as a string.
Streaming to the client is handled by the chat route, which calls
the synthesiser and then streams the result token-by-token.
"""

from __future__ import annotations

import logging

from backend.agent.state import AgentState, RetrievalResult
from backend.llm.client import LLMClient

logger = logging.getLogger(__name__)


async def synthesiser_node(state: AgentState, llm: LLMClient) -> AgentState:
    """
    Compose the final answer from approved passages and graph context.

    Mutates and returns state with:
      - final_answer: the complete assistant reply
      - citations: list of source dicts for the API response
    """
    query = state["user_query"]
    approved: list[RetrievalResult] = state.get("approved_passages", [])
    graph_ctx = state.get("graph_context", "")
    system_prompt = state.get("system_prompt", "You are PAEKA, a helpful AI assistant.")

    context_block = _build_context(approved, graph_ctx)
    citations = _build_citations(approved)

    if context_block:
        user_message = (
            f"{context_block}\n\n"
            f"Using the retrieved context above, answer the following:\n\n{query}"
        )
    else:
        user_message = query

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]

    try:
        answer = await llm.complete(messages)
    except Exception as exc:  # noqa: BLE001
        logger.error("Synthesiser LLM call failed: %s", exc)
        answer = f"I encountered an error generating a response: {exc}"

    logger.info(
        "Synthesiser: produced %d-char answer from %d passages.",
        len(answer), len(approved),
    )

    return {
        **state,
        "final_answer": answer,
        "citations": citations,
    }


def _build_context(passages: list[RetrievalResult], graph_ctx: str) -> str:
    parts: list[str] = []

    if passages:
        parts.append("<retrieved_context>")
        for i, p in enumerate(passages, 1):
            source = p["source"]
            heading = p.get("heading", "")
            page = p.get("page", 0)
            header = f"[{i}] {source}"
            if heading:
                header += f" — {heading}"
            if page:
                header += f" (p.{page})"
            parts.append(f"{header}\n{p['content']}")
        parts.append("</retrieved_context>")

    if graph_ctx:
        parts.append(graph_ctx)

    return "\n\n".join(parts)


def _build_citations(passages: list[RetrievalResult]) -> list[dict]:
    seen: set[str] = set()
    citations: list[dict] = []
    for p in passages:
        key = f"{p['source']}:{p.get('page', 0)}"
        if key not in seen:
            seen.add(key)
            citations.append({
                "filename":     p["source"],
                "heading":      p.get("heading", ""),
                "page":         p.get("page", 0),
                "score":        round(p.get("score", 0.0), 4),
                "element_type": p.get("element_type", "text"),
            })
    return citations
