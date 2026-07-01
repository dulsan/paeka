"""
backend/tools/schemas.py
=========================
Pydantic models for MCP tool arguments. Validated by backend/mcp/client.py
before every call_tool() dispatch -- this is the structural safety layer
(rejects malformed types/missing fields). It does NOT prevent redundant or
looping calls; that is handled separately by backend/agent/guardrails.py.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class WeaviateSearchArgs(BaseModel):
    """Args for qdrant_search. Class name kept for continuity with earlier drafts."""
    query: str = Field(..., min_length=1, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)
    collection: str = Field(default="chunks")


class WeaviateIngestArgs(BaseModel):
    """Args for qdrant_ingest."""
    content: str = Field(..., min_length=1)
    title: str = Field(default="")
    source: str = Field(default="agent")
    collection: str = Field(default="chunks")


class WebSearchArgs(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    num_results: int = Field(default=3, ge=1, le=10)


class GraphSearchArgs(BaseModel):
    """Args for graph_search -- multi-hop knowledge graph traversal."""
    entity: str = Field(..., min_length=1, max_length=200)
    max_hops: int = Field(default=2, ge=1, le=4)


class DiagnosticArgs(BaseModel):
    target: str = Field(default="all")


TOOL_ARG_SCHEMAS: dict[str, type[BaseModel]] = {
    "qdrant_search":   WeaviateSearchArgs,
    "qdrant_ingest":   WeaviateIngestArgs,
    "web_search":      WebSearchArgs,
    "graph_search":    GraphSearchArgs,
    "check_services":  DiagnosticArgs,
}


# ---------------------------------------------------------------------------
# Tool groups -- named subsets of the full MCP tool surface, so callers
# (deepagents sub-agents in particular, once wired in -- see
# backend/mcp/client.py's get_tool_schemas(groups=...)) can be scoped to
# only the tools relevant to their role instead of the entire registry.
# Schema-level filtering controls what the LLM is even told it can call;
# call_tool()'s optional allowed_tools param is the actual enforcement
# boundary, since a sub-agent could otherwise still attempt to invoke a
# tool name it was never shown.
# ---------------------------------------------------------------------------

TOOL_GROUPS: dict[str, list[str]] = {
    # Vector + graph retrieval -- the RAG sub-agent's toolset.
    "retrieval": ["qdrant_search", "qdrant_ingest", "graph_search"],
    # External web access -- kept separate from retrieval so a sub-agent
    # can be scoped to "only the local knowledge base" with no web egress.
    "web": ["web_search"],
    # Operational/diagnostic tools -- not generally exposed to a
    # user-facing chat sub-agent.
    "ops": ["check_services", "qdrant_snapshot"],
    # Self-discovery -- safe to expose alongside any other group.
    "meta": ["list_available_tools"],
}


def tools_in_groups(groups: list[str]) -> set[str]:
    """Return the union of tool names across the given group names.
    Unknown group names are silently ignored (return an empty set for
    that group) rather than raising, so a typo'd group degrades to "no
    extra tools" instead of crashing sub-agent construction."""
    names: set[str] = set()
    for g in groups:
        names.update(TOOL_GROUPS.get(g, []))
    return names


def validate_tool_args(tool_name: str, raw_args: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and coerce raw tool arguments against the tool's Pydantic schema.
    Tools without a registered schema (e.g. qdrant_snapshot, list_available_tools,
    which take only simple strings) pass through unchanged.
    """
    schema_cls = TOOL_ARG_SCHEMAS.get(tool_name)
    if schema_cls is None:
        return raw_args
    return schema_cls.model_validate(raw_args).model_dump()
