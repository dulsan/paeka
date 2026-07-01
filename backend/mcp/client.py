"""
backend/mcp/client.py
======================
Async MCP client bridge connecting ReActGraph (and any other caller) to
the FastMCP server mounted at /mcp inside the same FastAPI process.

Responsibilities:
  1. Discover tools via session.list_tools(), convert to OpenAI function-
     calling format for ChatOllama's bind_tools().
  2. call_tool(name, args) -- dispatch a single tool call.
  3. Cache schemas so repeated ReActGraph.run() calls don't re-fetch
     every time; force_refresh=True bypasses the cache (used by the
     list_available_tools self-discovery tool).
  4. Tool-group scoping -- get_tool_schemas(groups=[...]) and
     call_tool(..., allowed_tools=...) let a caller restrict itself to a
     named subset (see backend.tools.schemas.TOOL_GROUPS), so a deepagents
     sub-agent can be wired to e.g. only "retrieval" tools rather than the
     full registry. None (the default for both) means unscoped -- every
     existing caller (ReActGraph) is unaffected by this addition.

Transport: streamable HTTP (requires mcp>=1.1.0). Falls back to SSE for
mcp==1.0.x automatically.

[FIX] _DEFAULT_MCP_URL now ends in a trailing slash ("/mcp/" not "/mcp").
Confirmed against an official modelcontextprotocol/python-sdk GitHub issue
(#1168) describing exactly this symptom: FastMCP's streamable_http_app()
mounts on a Starlette Router with redirect_slashes=True by default, so a
POST to the bare mount path ("/mcp") gets a 307 redirect to "/mcp/" before
the real request is even handled. That roundtrip is exactly what was
showing up as two separate POSTs in the log (one to /mcp, one to /mcp/)
immediately before the "unhandled errors in a TaskGroup" failure --
several related SDK issues report the client's session/task-group setup
getting confused by the redirect hop itself. Requesting the trailing-slash
URL directly avoids the redirect entirely rather than trying to make the
client handle it gracefully.
"""

import asyncio
import logging
from typing import Any, AsyncIterator
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Trailing slash is required -- see module docstring [FIX] note above.
_DEFAULT_MCP_URL = "http://localhost:8000/mcp/"

_schema_cache: list[dict] | None = None
_cache_lock = asyncio.Lock()


def _get_local_tool_manager() -> Any | None:
    """
    Return the in-process FastMCP tool manager when the server is running in
    the same Python interpreter as the caller.

    This avoids an unnecessary HTTP round trip for the common in-app case and
    sidesteps MCP transport/session failures when the caller is already inside
    the FastAPI worker process.
    """
    try:
        from backend.mcp.server import mcp as local_mcp
    except Exception:
        return None
    return getattr(local_mcp, "_tool_manager", None)


def _log_unwrapped(context: str, exc: BaseException, _depth: int = 0) -> None:
    """
    Log the real cause of a failure, not just a generic wrapper message.

    asyncio.TaskGroup (used internally by mcp's streamablehttp_client for
    managing concurrent read/write streams) raises an ExceptionGroup on
    failure. Its default str() is just "unhandled errors in a TaskGroup
    (N sub-exception)" -- it says nothing about what actually went wrong.

    [FIX] This used to only unwrap ONE level. In practice the MCP SDK
    nests TaskGroups (one for the overall client session, another inside
    it for the read/write stream pump), so the "sub-exception" found at
    depth 1 was often itself ANOTHER ExceptionGroup, which just printed
    the same generic wrapper text one level down -- no more informative
    than not unwrapping at all. This now recurses until it reaches actual
    leaf exceptions, and prints a full traceback for each leaf (not just
    type+message), since at this nesting depth the leaf is often several
    frames away from anything obviously related to the original call.
    """
    import traceback

    indent = "  " * _depth
    if isinstance(exc, BaseExceptionGroup):
        logger.error("%s%s: %s (%d sub-exception(s))",
                     indent, context, type(exc).__name__, len(exc.exceptions))
        for i, sub in enumerate(exc.exceptions, start=1):
            _log_unwrapped(f"{context} -> [{i}/{len(exc.exceptions)}]", sub, _depth + 1)
    else:
        logger.error("%s%s: %s: %s", indent, context, type(exc).__name__, exc)
        tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
        for line in "".join(tb_lines).splitlines():
            logger.error("%s    %s", indent, line)


def invalidate_schema_cache() -> None:
    global _schema_cache
    _schema_cache = None


def _mcp_tool_to_openai(tool: Any) -> dict:
    schema = getattr(tool, "inputSchema", None) or getattr(tool, "parameters", None) or {}
    return {
        "type": "function",
        "function": {
            "name":        tool.name,
            "description": (tool.description or "").strip(),
            "parameters":  schema,
        },
    }


def _tool_result_to_text(result: Any) -> str:
    content = getattr(result, "content", None)
    if not content:
        return str(result)

    texts = [
        block.text for block in content
        if hasattr(block, "text") and block.text
    ]
    return "\n".join(texts) if texts else "(tool returned no output)"


@asynccontextmanager
async def _get_session(mcp_url: str) -> AsyncIterator[Any]:
    from mcp import ClientSession
    try:
        from mcp.client.streamable_http import streamablehttp_client
        async with streamablehttp_client(mcp_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
    except ImportError:
        from mcp.client.sse import sse_client  # type: ignore[import]
        async with sse_client(mcp_url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session


async def get_tool_schemas(
    mcp_url: str = _DEFAULT_MCP_URL,
    force_refresh: bool = False,
    groups: list[str] | None = None,
) -> list[dict]:
    """
    Fetch and cache all tool schemas from the MCP server in OpenAI format.

    Parameters
    ----------
    groups:
        Optional list of named tool groups (see backend.tools.schemas.
        TOOL_GROUPS) to scope the returned schemas to. None (default)
        returns every tool -- the existing, unscoped behaviour every
        current caller (ReActGraph) relies on. Pass e.g. ["retrieval"]
        to give a sub-agent only qdrant_search/qdrant_ingest/graph_search.
    """
    schemas = await _get_all_tool_schemas(mcp_url=mcp_url, force_refresh=force_refresh)
    if groups is None:
        return schemas

    from backend.tools.schemas import tools_in_groups
    allowed = tools_in_groups(groups)
    return [s for s in schemas if s["function"]["name"] in allowed]


async def _get_all_tool_schemas(
    mcp_url: str = _DEFAULT_MCP_URL,
    force_refresh: bool = False,
) -> list[dict]:
    """Unfiltered tool discovery (the actual fetch + cache logic)."""
    global _schema_cache
    async with _cache_lock:
        if _schema_cache is not None and not force_refresh:
            return _schema_cache

        local_manager = _get_local_tool_manager()
        if local_manager is not None:
            try:
                schemas = [_mcp_tool_to_openai(t) for t in local_manager.list_tools()]
                _schema_cache = schemas
                logger.info("MCP client: discovered %d local tools: %s",
                            len(schemas), [s["function"]["name"] for s in schemas])
                return schemas
            except Exception as exc:
                logger.warning("Local MCP tool discovery failed; falling back to HTTP transport")
                _log_unwrapped("Local MCP schema fetch failure", exc)

        try:
            async with _get_session(mcp_url) as session:
                result  = await session.list_tools()
                schemas = [_mcp_tool_to_openai(t) for t in result.tools]
            _schema_cache = schemas
            logger.info("MCP client: discovered %d tools: %s",
                        len(schemas), [s["function"]["name"] for s in schemas])
            return schemas
        except Exception as exc:
            logger.error("Failed to fetch MCP tool schemas from %s -- using cached/empty fallback", mcp_url)
            _log_unwrapped("MCP schema fetch failure", exc)
            return _schema_cache or []


async def call_tool(
    name: str,
    arguments: dict[str, Any],
    mcp_url: str = _DEFAULT_MCP_URL,
    allowed_tools: set[str] | None = None,
) -> str:
    """
    Call a named MCP tool and return its text output.

    Returns an error string prefixed with "[MCP ERROR]" on failure rather
    than raising, so the calling graph node can surface it to the LLM as
    a tool result instead of crashing the whole run.

    Parameters
    ----------
    allowed_tools:
        Optional set of tool names this caller is permitted to invoke
        (e.g. backend.tools.schemas.tools_in_groups(["retrieval"]) for a
        sub-agent scoped to retrieval-only). This is the actual
        enforcement boundary for sub-agent scoping -- get_tool_schemas()'s
        groups= filtering only controls what the LLM is told it can call;
        without this check a sub-agent could still attempt to call a tool
        name it was never shown. None (default) permits any tool, the
        existing unscoped behaviour.
    """
    if allowed_tools is not None and name not in allowed_tools:
        return f"[MCP ERROR] '{name}' is not in this agent's allowed tool set."

    from backend.tools.schemas import validate_tool_args
    try:
        arguments = validate_tool_args(name, arguments)
    except Exception as exc:
        return f"[MCP ERROR] Argument validation failed for '{name}': {exc}"

    try:
        local_manager = _get_local_tool_manager()
        if local_manager is not None:
            try:
                result = await local_manager.call_tool(name, arguments)
                return _tool_result_to_text(result)
            except Exception as exc:
                logger.warning("Local MCP tool call '%s' failed; falling back to HTTP transport", name)
                _log_unwrapped(f"Local MCP tool call '{name}' failed", exc)

        async with _get_session(mcp_url) as session:
            result = await session.call_tool(name, arguments)
            return _tool_result_to_text(result)
    except Exception as exc:
        _log_unwrapped(f"MCP tool call '{name}' failed", exc)
        return f"[MCP ERROR] {name}: {exc}"


async def list_tool_names(
    mcp_url: str = _DEFAULT_MCP_URL,
    groups: list[str] | None = None,
) -> list[str]:
    schemas = await get_tool_schemas(mcp_url=mcp_url, groups=groups)
    return [s["function"]["name"] for s in schemas]
