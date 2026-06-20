"""
backend/tools/schemas.py
=========================
Pydantic models for MCP tool arguments. Validated by backend/mcp/client.py
before every call_tool() dispatch -- this is the structural safety layer
(rejects malformed types/missing fields). It does NOT prevent redundant or
looping calls; that is handled separately by backend/agent/guardrails.py.
"""

from __future__ import annotations

from typing import Any, Literal

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


class CodeExecutionArgs(BaseModel):
    code: str = Field(..., min_length=1)
    language: Literal["python", "bash"] = Field(default="python")
    timeout: int = Field(default=30, ge=1, le=120)


class DiagnosticArgs(BaseModel):
    target: str = Field(default="all")


TOOL_ARG_SCHEMAS: dict[str, type[BaseModel]] = {
    "qdrant_search":   WeaviateSearchArgs,
    "qdrant_ingest":   WeaviateIngestArgs,
    "web_search":      WebSearchArgs,
    "execute_code":    CodeExecutionArgs,
    "check_services":  DiagnosticArgs,
}


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
