"""
backend/mcp/server.py
======================
FastMCP server exposing PAEKA's tools over the MCP protocol.

Mounted directly inside the FastAPI app at /mcp (same process, no subprocess).
ReActGraph discovers and calls these tools via backend/mcp/client.py.

Tool naming: qdrant_search / qdrant_ingest / qdrant_snapshot
  (renamed from the original weaviate_* naming used during the Weaviate
  prototype phase, since this is the first time these files are actually
  landing in the repository -- Qdrant has been the vector store since the
  migration, so the tool names should reflect that from day one.)

Service injection: configure() is called once from app.py's lifespan after
all services (Qdrant store, embedder, LLM) are initialised.
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

mcp = FastMCP(
    "PAEKA",
    # [FIX] Default is stateful session tracking (Mcp-Session-Id issued on
    # initialize(), expected on every subsequent request). Confirmed via
    # two independent sources that "McpError: Session terminated" on a
    # brand-new session's very first initialize() call is the documented
    # symptom this exact parameter resolves -- the session manager treats
    # every request as needing to match an existing tracked session, and
    # something about that tracking was failing here even with
    # session_manager.run() correctly active (a separate, necessary fix
    # applied earlier; this fixes a different layer of the same failure).
    #
    # This is also a clean fit for how this server is actually used: every
    # call in backend/mcp/client.py's _get_session() opens a brand new
    # streamablehttp_client + ClientSession + initialize() and tears it
    # down again within a single `async with` block -- nothing here ever
    # relies on a session persisting *across* separate calls. Stateless
    # mode (every request handled independently, no session ID tracking
    # at all) has no functional downside for that usage pattern.
    stateless_http=True,
    instructions=(
        "You are connected to the PAEKA local knowledge assistant. "
        "Use qdrant_search to retrieve information from the local knowledge base. "
        "Use qdrant_ingest to add new information. "
        "Use graph_search to explore relationships between entities multiple "
        "hops apart in the knowledge graph (e.g. 'how is X related to Y'). "
        "Use web_search when you need current information not in the knowledge base."
    ),
)

# Service handles injected by configure() at FastAPI startup
_store    = None   # QdrantStore
_embedder = None   # Embedder
_llm      = None   # LLMProvider
_web      = None   # WebSearchClient | None
_falkor   = None   # FalkorGraphStore | None


def configure(
    store=None,
    embedder=None,
    llm=None,
    web_client=None,
    falkor=None,
) -> None:
    """Inject lifespan-managed services into this module's tool functions."""
    global _store, _embedder, _llm, _web, _falkor
    _store    = store
    _embedder = embedder
    _llm      = llm
    _web      = web_client
    _falkor   = falkor
    logger.info(
        "MCP server configured: qdrant=%s embedder=%s llm=%s web=%s graph=%s",
        _store is not None, _embedder is not None, _llm is not None,
        _web is not None, _falkor is not None,
    )


# ---------------------------------------------------------------------------
# Tool: Qdrant semantic search
# ---------------------------------------------------------------------------

@mcp.tool()
async def qdrant_search(query: str, limit: int = 5, collection: str = "chunks") -> str:
    """
    Perform a semantic vector search across the local Qdrant knowledge base.

    Returns the most relevant text passages from ingested documents,
    ranked by cosine similarity to the query.

    Args:
        query: Natural language search query.
        limit: Number of results to return (1-20, default 5).
        collection: Qdrant collection name (default 'chunks').
    """
    if _store is None or _embedder is None:
        return "Qdrant is not available (retrieval disabled in settings)."

    from backend.tools.schemas import WeaviateSearchArgs
    try:
        args = WeaviateSearchArgs(query=query, limit=limit, collection=collection)
    except Exception as exc:
        return f"Invalid arguments: {exc}"

    try:
        import asyncio
        vector  = await asyncio.to_thread(_embedder.encode_one, args.query)
        results = await _store.search(
            vector=vector, limit=args.limit, collection_name=args.collection
        )
        if not results:
            return "No relevant documents found in the knowledge base."
        lines = []
        for i, r in enumerate(results, 1):
            src = r.filename or r.document_id or "unknown"
            lines.append(f"[{i}] {src}\n{r.content[:600]}")
        return "\n\n".join(lines)
    except Exception as exc:
        logger.error("qdrant_search failed: %s", exc)
        return f"Search error: {exc}"


# ---------------------------------------------------------------------------
# Tool: Knowledge graph multi-hop traversal
# ---------------------------------------------------------------------------

@mcp.tool()
async def graph_search(entity: str, max_hops: int = 2) -> str:
    """
    Traverse the knowledge graph from a named entity (1..max_hops relations
    away, either direction) and return the relationship chains found.

    Use this for "how is X related to Y" or "what connects to X" questions
    -- it finds multi-hop relationships that a single vector search over
    text chunks would miss. The graph is built from previously ingested
    documents, so unfamiliar entities simply return no results.

    Args:
        entity: Entity label to start from (e.g. 'Transformer', 'Qdrant').
        max_hops: How many relationship hops to traverse (1-4, default 2).
    """
    if _falkor is None or not _falkor.available:
        return "Graph search is not available (knowledge graph or FalkorDB disabled in settings)."

    from backend.tools.schemas import GraphSearchArgs
    try:
        args = GraphSearchArgs(entity=entity, max_hops=max_hops)
    except Exception as exc:
        return f"Invalid arguments: {exc}"

    try:
        hops = await _falkor.traverse(label=args.entity, max_hops=args.max_hops)
        if not hops:
            return f"No graph relationships found for '{args.entity}'."
        lines = []
        for hop in hops:
            chain = " → ".join(hop.relation_path)
            desc = f" ({hop.description})" if hop.description else ""
            lines.append(f"{args.entity} --[{chain}]--> {hop.label}{desc}")
        return "\n".join(lines)
    except Exception as exc:
        logger.error("graph_search failed: %s", exc)
        return f"Graph search error: {exc}"


# ---------------------------------------------------------------------------
# Tool: Qdrant ingest
# ---------------------------------------------------------------------------

@mcp.tool()
async def qdrant_ingest(
    content: str,
    title: str = "",
    source: str = "agent",
    collection: str = "chunks",
) -> str:
    """
    Ingest a piece of text into the local Qdrant knowledge base.

    Embeds the content with bge-m3 and stores it as a searchable vector.
    Use this when the user provides information they want PAEKA to remember.

    Args:
        content: Text content to ingest.
        title: Optional title or filename for the content.
        source: Source label (default 'agent').
        collection: Target Qdrant collection (default 'chunks').
    """
    if _store is None or _embedder is None:
        return "Qdrant is not available."

    from backend.tools.schemas import WeaviateIngestArgs
    try:
        args = WeaviateIngestArgs(content=content, title=title, source=source,
                                   collection=collection)
    except Exception as exc:
        return f"Invalid arguments: {exc}"

    try:
        import asyncio
        vector = await asyncio.to_thread(_embedder.encode_one, args.content)
        point_id = await _store.insert(
            content=args.content,
            vector=vector,
            metadata={"title": args.title, "source": args.source},
            collection_name=args.collection,
        )
        return f"Ingested successfully. Point ID: {point_id}"
    except Exception as exc:
        logger.error("qdrant_ingest failed: %s", exc)
        return f"Ingest error: {exc}"


# ---------------------------------------------------------------------------
# Tool: Qdrant snapshot (create / list)
# ---------------------------------------------------------------------------

@mcp.tool()
async def qdrant_snapshot(action: str, collection: str = "chunks") -> str:
    """
    Create or list Qdrant collection snapshots.

    Use 'create' before risky bulk ingestion. Use 'list' to see existing
    snapshots. Restore is not yet implemented through this tool -- use the
    Qdrant dashboard at http://localhost:6333/dashboard for manual recovery.

    Args:
        action: 'create' or 'list'.
        collection: Collection name (default 'chunks').
    """
    if _store is None:
        return "Qdrant is not available."
    if action not in ("create", "list"):
        return "Invalid action. Use 'create' or 'list'."

    try:
        client = _store._require_client()  # AsyncQdrantClient
        if action == "create":
            result = await client.create_snapshot(collection_name=collection)
            name = getattr(result, "name", str(result))
            return f"Snapshot created: {name}"
        else:
            snapshots = await client.list_snapshots(collection_name=collection)
            if not snapshots:
                return f"No snapshots found for collection '{collection}'."
            return "\n".join(
                f"- {getattr(s, 'name', s)} ({getattr(s, 'creation_time', '?')})"
                for s in snapshots
            )
    except Exception as exc:
        logger.error("qdrant_snapshot failed: %s", exc)
        return f"Snapshot error: {exc}"


# ---------------------------------------------------------------------------
# Tool: Web search
# ---------------------------------------------------------------------------

@mcp.tool()
async def web_search(query: str, num_results: int = 3) -> str:
    """
    Search the web (DuckDuckGo backend) for current information not
    available in the local knowledge base.

    Args:
        query: Search query string.
        num_results: Number of results to return (1-10, default 3).
    """
    if _web is None:
        return (
            "Web search is disabled. Set PAEKA_TOOLS__WEB_SEARCH_ENABLED=true "
            "in .env to enable it."
        )
    from backend.tools.schemas import WebSearchArgs
    try:
        args = WebSearchArgs(query=query, num_results=num_results)
    except Exception as exc:
        return f"Invalid arguments: {exc}"
    try:
        results = await _web.search(args.query, num_results=args.num_results)
        if not results:
            return "No web results found."
        return "\n\n".join(f"[{r.title}]\n{r.url}\n{r.content[:400]}" for r in results)
    except Exception as exc:
        logger.error("web_search failed: %s", exc)
        return f"Web search error: {exc}"


# ---------------------------------------------------------------------------
# Tool: Service health diagnostic
# ---------------------------------------------------------------------------

@mcp.tool()
async def check_services(target: str = "all") -> str:
    """
    Check the health of PAEKA's internal services: Ollama (:11434),
    Qdrant (:6333), and the PAEKA API itself (:8000).

    Args:
        target: 'all' or a name substring like 'ollama', 'qdrant'.
    """
    from backend.tools.diagnostics import check_services as _check
    return await _check(target=target)


# ---------------------------------------------------------------------------
# Tool: List available tools (self-discovery)
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_available_tools() -> str:
    """
    List all tools currently available to you, with a one-line description
    of what each does. Call this if you are unsure what capabilities you
    have, especially mid-task.
    """
    # Reuses the same client-side discovery path the ReAct loop itself uses
    # (backend/mcp/client.py), rather than guessing at FastMCP's internal
    # server-side registry API, which is not part of its stable public surface.
    from backend.mcp.client import get_tool_schemas
    schemas = await get_tool_schemas(force_refresh=True)
    if not schemas:
        return "No tools discovered (MCP client could not reach this server)."
    lines = []
    for s in schemas:
        fn   = s.get("function", {})
        name = fn.get("name", "?")
        desc = (fn.get("description") or "").splitlines()[0] if fn.get("description") else ""
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)
