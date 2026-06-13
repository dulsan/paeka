"""
backend/knowledge/retriever.py
================================
Graph-aware retrieval: given a user query and a set of already-retrieved
RAG chunks, find related graph nodes and augment the context with
structured knowledge (neighbours, paths, entity descriptions).

This makes the context richer than raw chunks alone — the model sees
both the raw passage AND the graph neighbourhood of mentioned entities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import networkx as nx

from backend.knowledge.graph import KGNode, KnowledgeGraphRepository
from backend.retrieval.reranker import RankedResult
from backend.shared.config import KnowledgeGraphSettings

logger = logging.getLogger(__name__)

_MAX_HOPS = 2
_MAX_NEIGHBOURS = 10


@dataclass
class GraphContext:
    """Structured knowledge retrieved from the graph for one query."""

    entities: list[dict]   # matching nodes
    subgraph: str          # formatted text block for LLM injection


class GraphRetriever:
    """
    Augments retrieval results with knowledge graph context.

    Parameters
    ----------
    repo:
        KnowledgeGraphRepository.
    settings:
        KnowledgeGraphSettings.
    """

    def __init__(
        self,
        repo: KnowledgeGraphRepository,
        settings: KnowledgeGraphSettings,
    ) -> None:
        self._repo = repo
        self._settings = settings
        self._graph: nx.DiGraph | None = None

    async def load_graph(self) -> None:
        """Load the full graph into memory for fast traversal."""
        self._graph = await self._repo.to_networkx(
            min_confidence=self._settings.min_edge_confidence
        )
        stats = await self._repo.stats()
        logger.info(
            "Graph loaded: %d nodes, %d edges", stats["nodes"], stats["edges"]
        )

    async def query(
        self,
        query_text: str,
        rag_results: list[RankedResult] | None = None,
    ) -> GraphContext:
        """
        Find graph context relevant to *query_text*.

        Strategy:
          1. Extract entity mentions from the query text via simple keyword matching.
          2. For each matched node, walk ``_MAX_HOPS`` hops in the graph.
          3. Format the subgraph as a compact fact list for LLM injection.

        Parameters
        ----------
        query_text:
            The user query.
        rag_results:
            Already-retrieved RAG passages — entity labels in these passages
            are also used as graph entry points.

        Returns
        -------
        GraphContext
        """
        if self._graph is None:
            await self.load_graph()

        # Gather candidate entity labels from query + RAG passages
        search_text = query_text
        if rag_results:
            search_text += " " + " ".join(r.content[:200] for r in rag_results)

        matched_nodes = await self._match_entities(search_text)
        if not matched_nodes:
            return GraphContext(entities=[], subgraph="")

        # Walk neighbourhood
        facts: list[str] = []
        seen_edges: set[tuple] = set()

        for node in matched_nodes[:5]:  # cap to avoid context explosion
            outgoing, incoming = await self._repo.get_neighbours(
                node.id, min_confidence=self._settings.min_edge_confidence
            )

            for edge in (outgoing + incoming)[:_MAX_NEIGHBOURS]:
                key = (edge.source_id, edge.target_id, edge.relation_type)
                if key in seen_edges:
                    continue
                seen_edges.add(key)

                src = await self._repo.get_node(edge.source_id)
                tgt = await self._repo.get_node(edge.target_id)
                if src and tgt:
                    desc = f" ({edge.description})" if edge.description else ""
                    facts.append(
                        f"• {src.label} --[{edge.relation_type}]--> {tgt.label}{desc}"
                    )

        entity_list = [
            {
                "label": n.label,
                "type": n.entity_type,
                "description": n.description,
            }
            for n in matched_nodes
        ]

        subgraph_text = ""
        if facts:
            subgraph_text = (
                "<knowledge_graph>\n"
                + "\n".join(facts)
                + "\n</knowledge_graph>"
            )

        return GraphContext(entities=entity_list, subgraph=subgraph_text)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _match_entities(self, text: str) -> list[KGNode]:
        """
        Find graph nodes whose labels appear in *text* (case-insensitive).
        Returns nodes sorted by label length descending (longer = more specific).
        """
        nodes = await self._repo.list_nodes(
            min_confidence=self._settings.min_edge_confidence
        )
        text_lower = text.lower()
        matched = [n for n in nodes if n.label.lower() in text_lower]
        matched.sort(key=lambda n: len(n.label), reverse=True)
        return matched
