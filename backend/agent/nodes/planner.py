"""
backend/agent/nodes/planner.py
================================
Planner node — decomposes query into sub-queries with tool assignments.

Tools available:
  "vector"  — semantic similarity search (local knowledge base)
  "graph"   — knowledge graph entity lookup
  "keyword" — BM25-heavy local search (exact terms, identifiers)
  "web"     — live web search (current info, recent events)

The planner only assigns "web" when the query explicitly requires
current information or is unlikely to be in the local knowledge base.
"""

from __future__ import annotations

import json
import logging

from backend.agent.state import AgentState, SubQuery
from backend.llm.client import LLMClient

logger = logging.getLogger(__name__)

_PLANNER_PROMPT = """\
You are a research planner for a personal AI assistant with access to:
  - A local knowledge base (vector + keyword search)
  - A knowledge graph of extracted entities and relations
  - Live web search (use ONLY when current/recent information is needed)

Given the user query below, decompose it into 2-6 focused sub-queries.
For each sub-query, assign the best retrieval tool:
  - "vector"  : semantic similarity (best for concepts, explanations, methods)
  - "graph"   : entity relationships (best for "what is X", "how does X relate to Y")
  - "keyword" : exact terms (best for specific names, identifiers, version numbers)
  - "web"     : live web search (ONLY for: recent events, current docs/releases,
                information clearly not in a local engineering knowledge base)

Respond ONLY with valid JSON (no markdown, no preamble):
{{
  "plan": "one sentence describing the research strategy",
  "sub_queries": [
    {{"query": "...", "tool": "vector|graph|keyword|web", "priority": 1}},
    ...
  ]
}}

Rules:
- Prioritise 1 (highest) to 6.
- Keep sub-queries specific and independent.
- Prefer local tools (vector/graph/keyword) over web when the information is
  likely in the knowledge base.
- Use "web" sparingly — only when timeliness is essential.
- If the query is simple, 2 sub-queries are sufficient.

User query: {query}
"""


async def planner_node(state: AgentState, llm: LLMClient) -> AgentState:
    query = state["user_query"]
    logger.info("Planner: decomposing '%s'", query[:60])

    prompt = _PLANNER_PROMPT.format(query=query)
    try:
        raw = await llm.complete(
            [{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.1,
        )
        plan_data = _parse_json(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Planner LLM failed (%s) — single passthrough.", exc)
        plan_data = None

    valid_tools = {"vector", "graph", "keyword", "web"}

    if plan_data and "sub_queries" in plan_data:
        sub_queries: list[SubQuery] = []
        for sq in plan_data["sub_queries"][:6]:
            if isinstance(sq, dict) and sq.get("query"):
                tool = sq.get("tool", "vector")
                sub_queries.append(SubQuery(
                    query=str(sq["query"]).strip(),
                    tool=tool if tool in valid_tools else "vector",
                    priority=int(sq.get("priority", 3)),
                ))
        research_plan = str(plan_data.get("plan", "")).strip()
    else:
        sub_queries = [SubQuery(query=query, tool="vector", priority=1)]
        research_plan = f"Direct retrieval for: {query}"

    # Ensure original query is always covered by at least one sub-query
    if not any(sq["query"].lower() == query.lower() for sq in sub_queries):
        sub_queries.insert(0, SubQuery(query=query, tool="vector", priority=0))

    logger.info(
        "Planner: %d sub-queries %s",
        len(sub_queries),
        [(sq["query"][:30], sq["tool"]) for sq in sub_queries],
    )
    return {**state, "sub_queries": sub_queries, "research_plan": research_plan}


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
