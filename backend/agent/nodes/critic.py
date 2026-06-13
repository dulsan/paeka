"""
backend/agent/nodes/critic.py
================================
Critic node — evaluates passage sufficiency with trust-tier awareness.

Trust tier weighting:
  local  (1.0) — highest confidence; comes from ingested, curated documents
  graph  (0.9) — structured knowledge; high confidence
  web    (0.6) — untrusted by default; lower weight, higher scrutiny

The critic is instructed to flag web passages that contradict local sources,
and to prefer local/graph passages when there is a conflict.
"""

from __future__ import annotations

import json
import logging

from backend.agent.state import AgentState, RetrievalResult
from backend.llm.client import LLMClient

logger = logging.getLogger(__name__)

_CRITIC_PROMPT = """\
You are a research critic evaluating retrieved evidence.

Trust tiers (higher = more reliable):
  local  (1.0) — passages from the user's curated local knowledge base
  graph  (0.9) — structured facts from a knowledge graph
  web    (0.6) — live web search results (treat with more scrutiny)

User question: {query}

Retrieved passages ({count} total):
{passages}

Evaluate:
1. Is there enough information to fully answer the question?
2. Are there contradictions between passages? (flag if web contradicts local)
3. What specific information is still missing?
4. Critically evaluate web passages — do they corroborate local sources?

Respond ONLY with valid JSON (no markdown, no preamble):
{{
  "sufficient": true/false,
  "missing_topics": ["topic1"],
  "contradictions": ["description if any"],
  "keep_passage_indices": [0, 1, 2],
  "reasoning": "one paragraph"
}}

Rules:
- "sufficient": true if the evidence collectively answers the question well.
- Prefer local/graph passages over web when both cover the same topic.
- If only web passages are available, be more conservative about "sufficient".
- "keep_passage_indices": 0-based indices of passages worth keeping.
"""


async def critic_node(state: AgentState, llm: LLMClient) -> AgentState:
    query = state["user_query"]
    passages: list[RetrievalResult] = state.get("retrieved_passages", [])
    hop_count = state.get("hop_count", 0)
    max_hops  = state.get("max_hops", 2)

    if not passages:
        return {**state,
                "critique": "No passages retrieved.",
                "needs_more_retrieval": False,
                "approved_passages": []}

    if hop_count >= max_hops:
        return {**state,
                "critique": f"Max hops ({max_hops}) reached.",
                "needs_more_retrieval": False,
                "approved_passages": passages}

    passages_text = _format_passages(passages)
    prompt = _CRITIC_PROMPT.format(
        query=query,
        count=len(passages),
        passages=passages_text,
    )

    try:
        raw = await llm.complete(
            [{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.1,
        )
        result = _parse_json(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Critic LLM failed (%s) — passing all passages.", exc)
        result = None

    if result is None:
        return {**state,
                "critique": "Critic unavailable.",
                "needs_more_retrieval": False,
                "approved_passages": passages}

    sufficient = bool(result.get("sufficient", True))
    missing    = result.get("missing_topics", [])
    reasoning  = result.get("reasoning", "")
    keep_idx   = result.get("keep_passage_indices", list(range(len(passages))))
    approved   = [passages[i] for i in keep_idx if 0 <= i < len(passages)] or passages

    needs_more = (not sufficient) and bool(missing) and (hop_count < max_hops)

    logger.info(
        "Critic: sufficient=%s missing=%s kept=%d/%d loop=%s tiers=%s",
        sufficient, missing, len(approved), len(passages), needs_more,
        {p["trust_tier"] for p in approved},
    )

    if needs_more:
        from backend.agent.state import SubQuery
        existing_queries = {sq["query"] for sq in state.get("sub_queries", [])}
        new_sqs = [
            SubQuery(query=topic, tool="vector", priority=2)
            for topic in missing[:3]
            if topic not in existing_queries
        ]
        return {**state,
                "critique": reasoning,
                "needs_more_retrieval": True,
                "approved_passages": approved,
                "sub_queries": state.get("sub_queries", []) + new_sqs}

    return {**state,
            "critique": reasoning,
            "needs_more_retrieval": False,
            "approved_passages": approved}


def _format_passages(passages: list[RetrievalResult]) -> str:
    lines = []
    for i, p in enumerate(passages[:15]):
        tier = p.get("trust_tier", "local")
        source = p["source"]
        snippet = p["content"][:300].replace("\n", " ")
        lines.append(f"[{i}] [{tier.upper()}] {source}\n{snippet}")
    return "\n\n".join(lines)


def _parse_json(raw: str) -> dict | None:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None
