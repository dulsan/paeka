"""
backend/knowledge/graph.py
===========================
Knowledge graph domain objects and SQLite repository.

The graph is stored in SQLite (kg_nodes + kg_edges tables defined in
database.py) and loaded into a networkx DiGraph for traversal.

Node  = entity (Concept, Algorithm, Person, …)
Edge  = directional relation (USES, IMPLEMENTS, IS_A, …)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import networkx as nx

from backend.shared.database import Database

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------


@dataclass
class KGNode:
    id: str
    label: str                  # canonical name, e.g. "Transformer"
    entity_type: str            # from ontology, e.g. "Algorithm"
    description: str
    source_doc: str             # document filename that introduced this node
    confidence: float
    created_at: str
    updated_at: str


@dataclass
class KGEdge:
    id: str
    source_id: str
    target_id: str
    relation_type: str          # from ontology, e.g. "USES"
    description: str
    confidence: float
    source_doc: str
    created_at: str


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class KnowledgeGraphRepository:
    """SQLite CRUD for the knowledge graph."""

    def __init__(self, db: Database) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    async def upsert_node(
        self,
        label: str,
        entity_type: str,
        description: str = "",
        source_doc: str = "",
        confidence: float = 1.0,
    ) -> KGNode:
        """
        Insert a new node or update description/confidence if label already exists.
        Returns the node (existing or newly created).
        """
        existing = await self.find_node_by_label(label)
        if existing:
            # Merge: keep highest confidence, append new description detail
            new_conf = max(existing.confidence, confidence)
            new_desc = existing.description
            if description and description not in existing.description:
                new_desc = f"{existing.description} | {description}".strip(" |")
            await self._db.execute(
                "UPDATE kg_nodes SET confidence=?, description=?, updated_at=? WHERE id=?",
                (new_conf, new_desc, _now(), existing.id),
            )
            existing.confidence = new_conf
            existing.description = new_desc
            return existing

        nid = str(uuid.uuid4())
        now = _now()
        await self._db.execute(
            """
            INSERT INTO kg_nodes
                (id, label, entity_type, description, source_doc, confidence, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (nid, label, entity_type, description, source_doc, confidence, now, now),
        )
        return KGNode(
            id=nid, label=label, entity_type=entity_type,
            description=description, source_doc=source_doc,
            confidence=confidence, created_at=now, updated_at=now,
        )

    async def find_node_by_label(self, label: str) -> KGNode | None:
        row = await self._db.fetchone(
            "SELECT * FROM kg_nodes WHERE lower(label)=lower(?)", (label,)
        )
        return KGNode(**dict(row)) if row else None

    async def get_node(self, node_id: str) -> KGNode | None:
        row = await self._db.fetchone(
            "SELECT * FROM kg_nodes WHERE id=?", (node_id,)
        )
        return KGNode(**dict(row)) if row else None

    async def list_nodes(
        self,
        entity_type: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[KGNode]:
        if entity_type:
            rows = await self._db.fetchall(
                "SELECT * FROM kg_nodes WHERE entity_type=? AND confidence>=? ORDER BY label",
                (entity_type, min_confidence),
            )
        else:
            rows = await self._db.fetchall(
                "SELECT * FROM kg_nodes WHERE confidence>=? ORDER BY label",
                (min_confidence,),
            )
        return [KGNode(**dict(r)) for r in rows]

    async def delete_node(self, node_id: str) -> None:
        """Delete node and all its edges (CASCADE in schema)."""
        await self._db.execute("DELETE FROM kg_nodes WHERE id=?", (node_id,))

    async def merge_nodes(self, keep_id: str, drop_id: str) -> None:
        """
        Merge *drop_id* into *keep_id* by re-pointing all edges,
        then deleting the dropped node.
        """
        await self._db.execute(
            "UPDATE kg_edges SET source_id=? WHERE source_id=?", (keep_id, drop_id)
        )
        await self._db.execute(
            "UPDATE kg_edges SET target_id=? WHERE target_id=?", (keep_id, drop_id)
        )
        await self._db.execute("DELETE FROM kg_nodes WHERE id=?", (drop_id,))
        logger.debug("Merged node %s into %s", drop_id, keep_id)

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    async def upsert_edge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        description: str = "",
        confidence: float = 1.0,
        source_doc: str = "",
    ) -> KGEdge:
        """
        Insert an edge or update confidence if it already exists.
        Self-loops are rejected.
        """
        if source_id == target_id:
            raise ValueError("Self-loops are not permitted in the knowledge graph.")

        existing = await self._find_edge(source_id, target_id, relation_type)
        if existing:
            new_conf = max(existing.confidence, confidence)
            await self._db.execute(
                "UPDATE kg_edges SET confidence=? WHERE id=?", (new_conf, existing.id)
            )
            existing.confidence = new_conf
            return existing

        eid = str(uuid.uuid4())
        now = _now()
        await self._db.execute(
            """
            INSERT INTO kg_edges
                (id, source_id, target_id, relation_type,
                 description, confidence, source_doc, created_at)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (eid, source_id, target_id, relation_type,
             description, confidence, source_doc, now),
        )
        return KGEdge(
            id=eid, source_id=source_id, target_id=target_id,
            relation_type=relation_type, description=description,
            confidence=confidence, source_doc=source_doc, created_at=now,
        )

    async def _find_edge(
        self, source_id: str, target_id: str, relation_type: str
    ) -> KGEdge | None:
        row = await self._db.fetchone(
            """
            SELECT * FROM kg_edges
            WHERE source_id=? AND target_id=? AND relation_type=?
            """,
            (source_id, target_id, relation_type),
        )
        return KGEdge(**dict(row)) if row else None

    async def list_edges(
        self, min_confidence: float = 0.0
    ) -> list[KGEdge]:
        rows = await self._db.fetchall(
            "SELECT * FROM kg_edges WHERE confidence>=? ORDER BY created_at",
            (min_confidence,),
        )
        return [KGEdge(**dict(r)) for r in rows]

    async def get_neighbours(
        self, node_id: str, min_confidence: float = 0.0
    ) -> tuple[list[KGEdge], list[KGEdge]]:
        """Return (outgoing_edges, incoming_edges) for *node_id*."""
        out_rows = await self._db.fetchall(
            "SELECT * FROM kg_edges WHERE source_id=? AND confidence>=?",
            (node_id, min_confidence),
        )
        in_rows = await self._db.fetchall(
            "SELECT * FROM kg_edges WHERE target_id=? AND confidence>=?",
            (node_id, min_confidence),
        )
        return (
            [KGEdge(**dict(r)) for r in out_rows],
            [KGEdge(**dict(r)) for r in in_rows],
        )

    async def delete_weak_edges(self, threshold: float) -> int:
        """Remove all edges below *threshold* confidence. Returns count deleted."""
        cursor = await self._db.execute(
            "DELETE FROM kg_edges WHERE confidence<?", (threshold,)
        )
        return cursor.rowcount

    # ------------------------------------------------------------------
    # Graph export
    # ------------------------------------------------------------------

    async def to_networkx(self, min_confidence: float = 0.0) -> nx.DiGraph:
        """Load the full graph into a networkx DiGraph for traversal."""
        g: nx.DiGraph = nx.DiGraph()

        nodes = await self.list_nodes(min_confidence=min_confidence)
        for n in nodes:
            g.add_node(n.id, label=n.label, entity_type=n.entity_type,
                       description=n.description, confidence=n.confidence)

        edges = await self.list_edges(min_confidence=min_confidence)
        for e in edges:
            g.add_edge(e.source_id, e.target_id,
                       relation=e.relation_type, confidence=e.confidence,
                       description=e.description)

        return g

    async def stats(self) -> dict:
        """Return a summary of the graph state."""
        n_row = await self._db.fetchone("SELECT count(*) as c FROM kg_nodes")
        e_row = await self._db.fetchone("SELECT count(*) as c FROM kg_edges")
        t_rows = await self._db.fetchall(
            "SELECT entity_type, count(*) as c FROM kg_nodes GROUP BY entity_type"
        )
        return {
            "nodes": n_row["c"] if n_row else 0,
            "edges": e_row["c"] if e_row else 0,
            "types": {r["entity_type"]: r["c"] for r in t_rows},
        }


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
