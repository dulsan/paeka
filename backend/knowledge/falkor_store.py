"""
backend/knowledge/falkor_store.py
===================================
FalkorDB Cypher query layer over the knowledge graph.

SQLite (kg_nodes / kg_edges, see backend/knowledge/graph.py) remains the
system of record -- it's what extraction, refinement, and the REST API
read and write. FalkorDB is a derived, queryable *view* of the same data,
synced on demand, used for agentic multi-hop graph traversal that's
awkward to express as repeated single-hop SQLite round-trips:

    MATCH (a:Entity {label:$start})-[:RELATION*1..3]-(b:Entity)
    RETURN DISTINCT b.label, b.entity_type

falkordblite ships FalkorDB as an embedded Redis module (via redislite) --
no Docker, no server process, no network port. It forks a local subprocess
communicating over a unix socket and persists to a single file on disk.
This is the "no Docker, no sandboxing overhead" answer to agentic graph
RAG: a real graph database with full Cypher, running in-process.

The underlying redislite/FalkorDB client is synchronous. Every call here
is wrapped in asyncio.to_thread() so it doesn't block the event loop --
the same pattern already used for the synchronous langgraph compile/invoke
glue elsewhere in the agent layer.

Usage
-----
    store = FalkorGraphStore(settings.knowledge_graph)
    await store.connect()
    await store.sync_from_sqlite(kg_repo)          # full resync
    hits = await store.traverse(label="Transformer", max_hops=2)
    await store.close()
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_GRAPH_NAME = "paeka_kg"


@dataclass
class FalkorHop:
    """One node reached during a multi-hop traversal."""

    label: str
    entity_type: str
    description: str
    relation_path: list[str]   # relation types from the start node to this one


class FalkorGraphStore:
    """
    Embedded FalkorDB client wrapping a single graph (``paeka_kg``).

    Parameters
    ----------
    db_path:
        File path for the embedded Redis/FalkorDB instance. Created if
        missing. A sibling ``<name>.settings`` file is also written by
        redislite -- both belong together.
    """

    def __init__(self, db_path: str = "database/falkordb/paeka_kg.db") -> None:
        self._db_path = Path(db_path)
        self._db: Any = None     # redislite.falkordb_client.FalkorDB
        self._graph: Any = None  # redislite.falkordb_client.Graph

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """
        Start the embedded FalkorDB instance. Returns False (and disables
        this store) if falkordblite isn't installed -- callers should
        treat graph_search/traverse as unavailable rather than crash.
        """
        try:
            from redislite.falkordb_client import FalkorDB
        except ImportError:
            logger.warning(
                "falkordblite not installed -- FalkorDB graph queries "
                "disabled. Install with: uv add falkordblite"
            )
            return False

        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        def _open() -> Any:
            db = FalkorDB(str(self._db_path))
            return db, db.select_graph(_GRAPH_NAME)

        self._db, self._graph = await asyncio.to_thread(_open)
        logger.info("FalkorDB embedded graph ready: %s", self._db_path)
        return True

    async def close(self) -> None:
        if self._db is not None:
            await asyncio.to_thread(self._db.close)
            self._db, self._graph = None, None

    @property
    def available(self) -> bool:
        return self._graph is not None

    # ------------------------------------------------------------------
    # Sync from SQLite (system of record)
    # ------------------------------------------------------------------

    async def sync_from_sqlite(self, repo) -> dict:
        """
        Full resync: clear the FalkorDB graph and reload every node/edge
        currently in SQLite via *repo* (KnowledgeGraphRepository).

        Full resync rather than incremental is the right tradeoff here --
        this is a single-process, embedded, low-write-volume graph (KG
        extraction runs in batches after document ingestion, not on every
        request), so the simplicity of "SQLite is truth, Falkor is a
        rebuildable cache of it" beats the complexity of tracking deltas.

        Returns a small stats dict for logging/diagnostics.
        """
        if not self.available:
            return {"synced": False, "reason": "falkordb not available"}

        nodes = await repo.list_nodes()
        edges = await repo.list_edges()

        def _rebuild() -> None:
            self._graph.query("MATCH (n) DETACH DELETE n")
            for n in nodes:
                self._graph.query(
                    """
                    MERGE (e:Entity {id: $id})
                    SET e.label = $label, e.entity_type = $entity_type,
                        e.description = $description, e.confidence = $confidence
                    """,
                    {
                        "id": n.id, "label": n.label, "entity_type": n.entity_type,
                        "description": n.description or "", "confidence": n.confidence,
                    },
                )
            for e in edges:
                self._graph.query(
                    """
                    MATCH (a:Entity {id: $src}), (b:Entity {id: $tgt})
                    MERGE (a)-[r:RELATION {relation_type: $rtype}]->(b)
                    SET r.description = $description, r.confidence = $confidence
                    """,
                    {
                        "src": e.source_id, "tgt": e.target_id,
                        "rtype": e.relation_type,
                        "description": e.description or "", "confidence": e.confidence,
                    },
                )

        await asyncio.to_thread(_rebuild)
        logger.info(
            "FalkorDB synced from SQLite: %d nodes, %d edges", len(nodes), len(edges)
        )
        return {"synced": True, "nodes": len(nodes), "edges": len(edges)}

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def traverse(
        self,
        label: str,
        max_hops: int = 2,
        min_confidence: float = 0.0,
    ) -> list[FalkorHop]:
        """
        Multi-hop neighbourhood traversal from an entity label (1..max_hops
        relations away, either direction). This is the actual "agentic
        graph RAG" primitive -- a single Cypher variable-length-path query
        replaces what backend/knowledge/retriever.py previously did as a
        single-hop-only manual walk (its _MAX_HOPS constant was declared
        but never used -- this is the real multi-hop implementation).

        Returns nodes ordered by hop distance (closest first), each
        annotated with the relation-type chain that reached it.
        """
        if not self.available:
            return []

        max_hops = max(1, min(max_hops, 4))  # guard against pathological fan-out

        def _run() -> list:
            result = self._graph.query(
                f"""
                MATCH path = (start:Entity {{label: $label}})
                             -[r:RELATION*1..{max_hops}]-(b:Entity)
                WHERE ALL(rel IN relationships(path) WHERE rel.confidence >= $min_conf)
                RETURN DISTINCT b.label, b.entity_type, b.description,
                       [rel IN relationships(path) | rel.relation_type] AS rel_path,
                       length(path) AS hops
                ORDER BY hops ASC
                LIMIT 30
                """,
                {"label": label, "min_conf": min_confidence},
            )
            return result.result_set

        rows = await asyncio.to_thread(_run)
        return [
            FalkorHop(
                label=row[0], entity_type=row[1], description=row[2] or "",
                relation_path=row[3],
            )
            for row in rows
        ]

    async def query(self, cypher: str, params: dict | None = None) -> list[list]:
        """
        Run an arbitrary read-only-by-convention Cypher query. Used by the
        graph_search MCP tool to give the agent direct (but still
        SQLite-derived) graph access for cases traverse() doesn't cover.
        """
        if not self.available:
            return []

        def _run() -> list:
            result = self._graph.query(cypher, params or {})
            return result.result_set

        return await asyncio.to_thread(_run)
