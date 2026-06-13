"""
backend/knowledge/extractor.py
================================
LLM-driven entity and relation extraction from document chunks.

For each batch of chunks the LLM is prompted to identify:
  - Entities: (label, type, short description)
  - Relations: (source_label, relation_type, target_label, confidence, description)

Only entity types and relation types defined in the ontology
(``KnowledgeGraphSettings``) are accepted.  The LLM is explicitly
told to ignore everything else.

Output format is JSON, parsed and validated before writing to the graph.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from backend.knowledge.graph import KnowledgeGraphRepository
from backend.llm.client import LLMClient
from backend.retrieval.chunker import TextChunk
from backend.shared.config import KnowledgeGraphSettings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
You are a knowledge graph extractor. Analyse the following text excerpts and
extract entities and relations that are EXPLICITLY stated or strongly implied.

ALLOWED ENTITY TYPES: {entity_types}
ALLOWED RELATION TYPES: {relation_types}

Rules:
- Only extract entities and relations you are confident about (confidence 0.0–1.0).
- Use the exact entity type and relation type strings from the allowed lists.
- Entity labels must be concise canonical names (e.g. "Transformer", not "the Transformer model").
- Do NOT invent relations not present in the text.
- Do NOT use entity or relation types not in the allowed lists.

Respond ONLY with a valid JSON object in exactly this format (no markdown, no preamble):
{{
  "entities": [
    {{"label": "...", "type": "...", "description": "one sentence max"}}
  ],
  "relations": [
    {{
      "source": "...",
      "relation": "...",
      "target": "...",
      "confidence": 0.0–1.0,
      "description": "one sentence max"
    }}
  ]
}}

TEXT EXCERPTS:
{text}
"""


# ---------------------------------------------------------------------------
# Extracted result types
# ---------------------------------------------------------------------------


@dataclass
class ExtractedEntity:
    label: str
    entity_type: str
    description: str


@dataclass
class ExtractedRelation:
    source: str
    relation_type: str
    target: str
    confidence: float
    description: str


@dataclass
class ExtractionResult:
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class KnowledgeGraphExtractor:
    """
    Extracts entities and relations from document chunks using the LLM.

    Parameters
    ----------
    repo:
        KnowledgeGraphRepository for persisting results.
    llm:
        LLMClient for extraction calls.
    settings:
        KnowledgeGraphSettings from config.
    """

    def __init__(
        self,
        repo: KnowledgeGraphRepository,
        llm: LLMClient,
        settings: KnowledgeGraphSettings,
    ) -> None:
        self._repo = repo
        self._llm = llm
        self._settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def extract_from_chunks(
        self,
        chunks: list[TextChunk],
        source_doc: str = "",
    ) -> ExtractionResult:
        """
        Run extraction over *chunks* in batches and persist to the graph.

        Parameters
        ----------
        chunks:
            Chunks from a single document or mixed sources.
        source_doc:
            Filename to attach to extracted nodes/edges as provenance.

        Returns
        -------
        ExtractionResult
            Aggregated entities and relations found across all batches.
        """
        s = self._settings
        batch_size = s.extraction_batch_size

        all_entities: list[ExtractedEntity] = []
        all_relations: list[ExtractedRelation] = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            result = await self._extract_batch(batch, source_doc)
            all_entities.extend(result.entities)
            all_relations.extend(result.relations)

        # Persist everything
        await self._persist(all_entities, all_relations, source_doc)

        logger.info(
            "Extraction complete for '%s': %d entities, %d relations",
            source_doc,
            len(all_entities),
            len(all_relations),
        )
        return ExtractionResult(entities=all_entities, relations=all_relations)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _extract_batch(
        self, chunks: list[TextChunk], source_doc: str
    ) -> ExtractionResult:
        """Send one batch to the LLM and parse the response."""
        s = self._settings
        text_block = "\n\n---\n\n".join(
            f"[{c.heading or 'Section'}]\n{c.content}" for c in chunks
        )

        prompt = _EXTRACTION_PROMPT.format(
            entity_types=", ".join(s.entity_types),
            relation_types=", ".join(s.relation_types),
            text=text_block,
        )

        raw = await self._llm.complete(
            [{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.1,
        )

        return self._parse_response(raw, s)

    def _parse_response(
        self, raw: str, s: KnowledgeGraphSettings
    ) -> ExtractionResult:
        """Parse and validate the LLM JSON response."""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
            if raw.endswith("```"):
                raw = raw[:-3].strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Extraction JSON parse error: %s | raw=%s", exc, raw[:120])
            return ExtractionResult(entities=[], relations=[])

        valid_entity_types = set(s.entity_types)
        valid_relation_types = set(s.relation_types)

        entities: list[ExtractedEntity] = []
        for e in data.get("entities", []):
            if (
                isinstance(e, dict)
                and e.get("label")
                and e.get("type") in valid_entity_types
            ):
                entities.append(ExtractedEntity(
                    label=str(e["label"]).strip(),
                    entity_type=str(e["type"]),
                    description=str(e.get("description", "")).strip(),
                ))

        relations: list[ExtractedRelation] = []
        for r in data.get("relations", []):
            if (
                isinstance(r, dict)
                and r.get("source")
                and r.get("relation") in valid_relation_types
                and r.get("target")
            ):
                try:
                    conf = float(r.get("confidence", 0.8))
                    conf = max(0.0, min(1.0, conf))
                except (TypeError, ValueError):
                    conf = 0.8
                if conf >= s.min_edge_confidence:
                    relations.append(ExtractedRelation(
                        source=str(r["source"]).strip(),
                        relation_type=str(r["relation"]),
                        target=str(r["target"]).strip(),
                        confidence=conf,
                        description=str(r.get("description", "")).strip(),
                    ))

        return ExtractionResult(entities=entities, relations=relations)

    async def _persist(
        self,
        entities: list[ExtractedEntity],
        relations: list[ExtractedRelation],
        source_doc: str,
    ) -> None:
        """Write extracted entities and relations to the graph repository."""
        # Upsert nodes first — relations reference node labels
        for ent in entities:
            await self._repo.upsert_node(
                label=ent.label,
                entity_type=ent.entity_type,
                description=ent.description,
                source_doc=source_doc,
            )

        # Upsert edges
        for rel in relations:
            src_node = await self._repo.find_node_by_label(rel.source)
            tgt_node = await self._repo.find_node_by_label(rel.target)

            # If either endpoint wasn't extracted, create a stub node
            if not src_node:
                src_node = await self._repo.upsert_node(
                    label=rel.source,
                    entity_type="Concept",
                    source_doc=source_doc,
                    confidence=0.5,
                )
            if not tgt_node:
                tgt_node = await self._repo.upsert_node(
                    label=rel.target,
                    entity_type="Concept",
                    source_doc=source_doc,
                    confidence=0.5,
                )

            try:
                await self._repo.upsert_edge(
                    source_id=src_node.id,
                    target_id=tgt_node.id,
                    relation_type=rel.relation_type,
                    description=rel.description,
                    confidence=rel.confidence,
                    source_doc=source_doc,
                )
            except ValueError as exc:
                logger.debug("Edge rejected: %s", exc)
