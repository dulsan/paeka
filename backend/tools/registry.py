"""
backend/tools/registry.py
==========================
Central registry of tools available to the self-healing tool calling agent.

Tools are async callables with a docstring describing their purpose.
The SelfHealingToolGraph reads tool names and docstrings to build
the tool schema presented to the LLM.

Registered tools:
  web_search(query)         — Web search
  lint_code(code)           — Ruff lint
  format_code(code)         — Ruff format
  typecheck_code(code)      — Pyright type check
  retrieve(query)           — Vector + graph retrieval
  graph_search(entity)      — Knowledge graph multi-hop traversal (FalkorDB)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

# Type for an async tool function
ToolFn = Callable[..., Awaitable[str]]


def get_registered_tools(request) -> dict[str, ToolFn]:
    """
    Build and return the tool registry for the current request.

    Each tool is a closure that captures the required app.state services.
    Returns only tools whose backing service is available.
    """
    tools: dict[str, ToolFn] = {}

    # ── Web search ────────────────────────────────────────────────────
    web_client = getattr(request.app.state, "web_client", None)
    if web_client is not None:
        async def web_search(query: str, num_results: int = 3) -> str:
            """Search the web. Returns top results as text."""
            results = await web_client.search(query, num_results=num_results)
            if not results:
                return "No results found."
            return "\n\n".join(
                f"[{r.title}] ({r.url})\n{r.content[:400]}"
                for r in results
            )
        tools["web_search"] = web_search

    # ── Code linting ──────────────────────────────────────────────────
    async def lint_code(code: str, filename: str = "snippet.py") -> str:
        """Run Ruff linting on Python code. Returns issues found or 'No issues'."""
        from backend.tools.verification import lint_python
        result = await lint_python(code, filename)
        return result.output

    tools["lint_code"] = lint_code

    async def format_code(code: str) -> str:
        """Auto-format Python code with Ruff. Returns the formatted code."""
        from backend.tools.verification import format_python
        result = await format_python(code)
        return result.fixed_code

    tools["format_code"] = format_code

    async def typecheck_code(code: str) -> str:
        """Run Pyright type checking on Python code. Returns type errors found."""
        from backend.tools.verification import typecheck_python
        result = await typecheck_python(code)
        return result.output

    tools["typecheck_code"] = typecheck_code

    # ── RAG retrieval ──────────────────────────────────────────────────
    retrieval = getattr(request.app.state, "retrieval", None)
    if retrieval is not None:
        async def retrieve(query: str) -> str:
            """Search the local knowledge base for relevant passages."""
            results = retrieval.retrieve(query)
            if not results:
                return "No relevant passages found."
            return "\n\n".join(
                f"[{r.metadata.get('filename', 'unknown')}] {r.content[:400]}"
                for r in results[:5]
            )
        tools["retrieve"] = retrieve

    # ── Knowledge graph multi-hop traversal ──────────────────────────────
    falkor = getattr(request.app.state, "kg_falkor", None)
    if falkor is not None and falkor.available:
        async def graph_search(entity: str, max_hops: int = 2) -> str:
            """Traverse the knowledge graph from an entity (1-4 hops) to find related entities."""
            hops = await falkor.traverse(label=entity, max_hops=max_hops)
            if not hops:
                return f"No graph relationships found for '{entity}'."
            return "\n".join(
                f"{entity} --[{' → '.join(h.relation_path)}]--> {h.label}"
                for h in hops
            )
        tools["graph_search"] = graph_search

    logger.debug("Registered tools: %s", list(tools.keys()))
    return tools
