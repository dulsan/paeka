"""
backend/knowledge/refinement.py
=================================
Multi-pass knowledge graph refinement.

Each pass applies a targeted heuristic to improve graph quality.
Passes are defined as methods and selected by name in settings.toml
under ``[knowledge_graph] refinement_passes``.

Available passes:

  merge_duplicates
    Finds pairs of nodes with very similar labels (using the LLM as a
    semantic comparator) and merges the lower-confidence node into the
    higher-confidence one.

  prune_weak_edges
    Removes all edges below ``min_edge_confidence``.  Should be run after
    extraction since confidence is only estimated during extraction.

  validate_types
    Asks the LLM to review each node's entity_type and correct obviously
    wrong assignments (e.g. a person labelled as "Algorithm").

Passes are idempotent and safe to re-run.  Run order matters:
  merge_duplicates → validate_types → prune_weak_edges  is the recommended order.
"""

from __future__ import annotations

import json
import logging
from itertools import combinations

from backend.knowledge.graph import KGNode, KnowledgeGraphRepository
from backend.llm.client import LLMClient
from backend.shared.config import KnowledgeGraphSettings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_MERGE_PROMPT = """\
Are the following two entity labels referring to the same real-world concept?

Entity A: "{label_a}" (type: {type_a})
Entity B: "{label_b}" (type: {type_b})

Answer ONLY with a JSON object:
{{"same": true/false, "reason": "one sentence"}}
"""

_VALIDATE_TYPE_PROMPT = """\
Given this entity from a technical knowledge graph, is the assigned type correct?

Label: "{label}"
Assigned type: "{assigned_type}"
Description: "{description}"
Allowed types: {allowed_types}

Answer ONLY with a JSON object:
{{"correct_type": "<type from allowed list>", "changed": true/false}}
"""


# ---------------------------------------------------------------------------
# Refinement engine
# ---------------------------------------------------------------------------


class GraphRefiner:
    """
    Runs multi-pass refinement over the knowledge graph.

    Parameters
    ----------
    repo:
        KnowledgeGraphRepository.
    llm:
        LLMClient.
    settings:
        KnowledgeGraphSettings.
    """

    def __init__(
        self,
        repo: KnowledgeGraphRepository,
        llm: LLMClient,
        settings: KnowledgeGraphSettings,
        falkor=None,
    ) -> None:
        self._repo = repo
        self._llm = llm
        self._settings = settings
        # Optional FalkorGraphStore -- refinement passes merge/prune nodes
        # and edges, so the synced graph needs a fresh resync afterward
        # too, same reasoning as KnowledgeGraphExtractor.
        self._falkor = falkor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_all_passes(self) -> dict[str, int]:
        """
        Execute every pass listed in ``settings.refinement_passes``.

        Returns
        -------
        dict[str, int]
            Mapping of pass name → number of changes made.
        """
        results: dict[str, int] = {}
        for pass_name in self._settings.refinement_passes:
            method = getattr(self, f"_pass_{pass_name}", None)
            if method is None:
                logger.warning("Unknown refinement pass: '%s' — skipping.", pass_name)
                continue
            logger.info("Running refinement pass: %s", pass_name)
            changes = await method()
            results[pass_name] = changes
            logger.info("Pass '%s' complete — %d changes.", pass_name, changes)

        if self._falkor is not None and self._falkor.available:
            await self._falkor.sync_from_sqlite(self._repo)

        return results

    # ------------------------------------------------------------------
    # Pass: merge_duplicates
    # ------------------------------------------------------------------

    async def _pass_merge_duplicates(self) -> int:
        """
        Find nodes with similar labels and merge confirmed duplicates.

        Similarity check: normalised label overlap > 0.7 triggers LLM
        confirmation.  Only pairs within the same entity_type are compared
        (avoids cross-type false positives like "Python" (Language) vs
        "Python" (Framework)).
        """
        nodes = await self._repo.list_nodes()
        by_type: dict[str, list[KGNode]] = {}
        for n in nodes:
            by_type.setdefault(n.entity_type, []).append(n)

        merges = 0
        for entity_type, type_nodes in by_type.items():
            if len(type_nodes) < 2:
                continue
            for a, b in combinations(type_nodes, 2):
                if _label_similarity(a.label, b.label) < 0.70:
                    continue
                # LLM confirmation
                is_same = await self._llm_confirm_merge(a, b)
                if is_same:
                    # Keep the node with higher confidence
                    keep, drop = (a, b) if a.confidence >= b.confidence else (b, a)
                    await self._repo.merge_nodes(keep.id, drop.id)
                    logger.debug(
                        "Merged '%s' into '%s' (%s)", drop.label, keep.label, entity_type
                    )
                    merges += 1

        return merges

    # ------------------------------------------------------------------
    # Pass: prune_weak_edges
    # ------------------------------------------------------------------

    async def _pass_prune_weak_edges(self) -> int:
        """Remove edges below the configured confidence threshold."""
        deleted = await self._repo.delete_weak_edges(self._settings.min_edge_confidence)
        return deleted

    # ------------------------------------------------------------------
    # Pass: validate_types
    # ------------------------------------------------------------------

    async def _pass_validate_types(self) -> int:
        """Review and correct node entity_type assignments."""
        nodes = await self._repo.list_nodes()
        corrections = 0
        allowed = ", ".join(self._settings.entity_types)

        for node in nodes:
            prompt = _VALIDATE_TYPE_PROMPT.format(
                label=node.label,
                assigned_type=node.entity_type,
                description=node.description or "N/A",
                allowed_types=allowed,
            )
            raw = await self._llm.complete(
                [{"role": "user", "content": prompt}],
                max_tokens=128,
                temperature=0.0,
            )
            result = _parse_json(raw)
            if result is None:
                continue

            new_type = result.get("correct_type", "").strip()
            changed = result.get("changed", False)

            if (
                changed
                and new_type
                and new_type in self._settings.entity_types
                and new_type != node.entity_type
            ):
                await self._repo._db.execute(
                    "UPDATE kg_nodes SET entity_type=?, updated_at=datetime('now') WHERE id=?",
                    (new_type, node.id),
                )
                logger.debug(
                    "Type corrected: '%s' %s → %s",
                    node.label, node.entity_type, new_type,
                )
                corrections += 1

        return corrections

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    async def _llm_confirm_merge(self, a: KGNode, b: KGNode) -> bool:
        """Ask the LLM whether two nodes refer to the same entity."""
        prompt = _MERGE_PROMPT.format(
            label_a=a.label, type_a=a.entity_type,
            label_b=b.label, type_b=b.entity_type,
        )
        raw = await self._llm.complete(
            [{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.0,
        )
        result = _parse_json(raw)
        return bool(result and result.get("same"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _label_similarity(a: str, b: str) -> float:
    """Token-overlap Jaccard similarity between two label strings."""
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _parse_json(raw: str) -> dict | None:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None
