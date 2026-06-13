"""
backend/ingestion/pipeline.py
==============================
Document ingestion pipeline (v0.11.0).

Two chunking paths:
  1. Docling documents → HybridChunker (tokenisation-aware, structure-preserving)
     Each DoclingDocument is chunked by Docling's own HybridChunker which:
       a) Uses HierarchicalChunker to split along structural boundaries
          (section → subsection → paragraph, tables/equations atomic)
       b) Applies token-aware refinement so every chunk fits the embedding
          model's context window exactly
  2. Plain text / code / LaTeX / spreadsheets → layout-aware chunker
     (existing TextChunk-based path)

Version-aware re-ingestion: changed files retire old chunks and replace them.
Semantic deduplication: near-identical chunks (cosine ≥ 0.97) are skipped.
Content security: each chunk scanned for injection patterns before storage.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from backend.ingestion.parsers.dispatcher import content_hash, parse_file
from backend.ingestion.repository import DocumentRepository
from backend.retrieval.chunker import TextChunk, chunk_document, chunk_text
from backend.retrieval.embedder import Embedder
from backend.retrieval.weaviate_store import WeaviateStore
from backend.shared.config import IngestionSettings, RetrievalSettings
from backend.shared.database import Database

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        db: Database,
        store: WeaviateStore,
        embedder: Embedder,
        retrieval_settings: RetrievalSettings,
        ingestion_settings: IngestionSettings,
        scanner=None,
    ) -> None:
        self._repo      = DocumentRepository(db)
        self._db        = db
        self._store     = store
        self._embedder  = embedder
        self._rsettings = retrieval_settings
        self._isettings = ingestion_settings
        self._scanner   = scanner

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ingest_file(self, filepath: str | Path) -> str:
        """
        Ingest a file from disk.

        Idempotent on hash match. Changed files retire the old version.
        Returns document ID.
        """
        path = Path(filepath).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        file_hash = content_hash(path)
        existing  = await self._repo.get_document_by_path(str(path))

        if existing:
            if existing.content_hash == file_hash and existing.status == "ready":
                logger.info("Unchanged (hash match) — skipping: %s", path.name)
                return existing.id
            if self._isettings.versioning_enabled:
                await self._retire_version(existing)
            else:
                self._store.delete_document_chunks(existing.id)
                await self._repo.delete_document(existing.id)

        import mimetypes
        mime_type, _ = mimetypes.guess_type(str(path))
        mime_type    = mime_type or "application/octet-stream"

        parsed = parse_file(path, mode=self._isettings.default_parser)
        doc    = await self._repo.create_document(
            filename=path.name,
            filepath=str(path),
            mime_type=mime_type,
            content_hash=file_hash,
            parser_used=parsed.parser_used,
        )
        return await self._process(doc.id, doc.filename, parsed)

    async def ingest_text(self, text: str, filename: str, filepath: str = "") -> str:
        """Ingest pre-extracted text. Returns document ID."""
        import hashlib
        file_hash     = hashlib.sha256(text.encode()).hexdigest()
        virtual_path  = filepath or f"virtual://{filename}"
        existing      = await self._repo.get_document_by_path(virtual_path)

        if existing:
            if existing.content_hash == file_hash and existing.status == "ready":
                logger.info("Text unchanged — skipping: %s", filename)
                return existing.id
            await self._retire_version(existing)

        doc = await self._repo.create_document(
            filename=filename,
            filepath=virtual_path,
            mime_type="text/plain",
            content_hash=file_hash,
            parser_used="plaintext",
        )
        chunks = chunk_text(
            text,
            chunk_size=self._rsettings.chunk_size,
            chunk_overlap=self._rsettings.chunk_overlap,
        )
        if not chunks:
            await self._repo.set_status(doc.id, "failed")
            return doc.id
        return await self._embed_and_store(doc.id, filename, chunks)

    async def delete_document(self, doc_id: str) -> bool:
        doc = await self._repo.get_document(doc_id)
        if not doc:
            return False
        self._store.delete_document_chunks(doc_id)
        await self._repo.delete_document(doc_id)
        logger.info("Deleted document %s (%s)", doc_id, doc.filename)
        return True

    async def refresh_document(self, doc_id: str) -> str:
        doc = await self._repo.get_document(doc_id)
        if not doc:
            raise ValueError(f"Document not found: {doc_id}")
        path = Path(doc.filepath)
        if not path.exists():
            raise FileNotFoundError(f"Original file no longer on disk: {doc.filepath}")
        await self._retire_version(doc)
        return await self.ingest_file(path)

    # ------------------------------------------------------------------
    # Private: routing by parser used
    # ------------------------------------------------------------------

    async def _process(self, doc_id: str, filename: str, parsed) -> str:
        await self._repo.set_status(doc_id, "processing")
        try:
            # ── Path 1: Docling HybridChunker ────────────────────────
            if parsed.parser_used == "docling" and parsed.metadata.get("docling_doc"):
                return await self._process_docling(doc_id, filename, parsed)

            # ── Path 2: layout-aware TextChunk chunker ────────────────
            chunks = chunk_document(
                parsed.elements,
                chunk_size=self._rsettings.chunk_size,
                chunk_overlap=self._rsettings.chunk_overlap,
            )
            if not chunks:
                logger.warning("No chunks produced for %s", filename)
                await self._repo.set_status(doc_id, "failed")
                return doc_id
            return await self._embed_and_store(doc_id, filename, chunks)

        except Exception as exc:
            logger.error("Ingestion failed for %s: %s", filename, exc)
            await self._repo.set_status(doc_id, "failed")
            raise

    async def _process_docling(self, doc_id: str, filename: str, parsed) -> str:
        """
        Chunk with Docling's HybridChunker then embed and store.

        HybridChunker produces DocChunk objects with:
          .text      — the chunk text
          .meta      — DocMeta with headings, page refs, doc origin
        """
        from backend.ingestion.parsers.docling_parser import chunk_with_hybrid_chunker

        try:
            doc_chunks = chunk_with_hybrid_chunker(
                parsed,
                embed_model_name=self._rsettings.embed_model,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "HybridChunker failed for %s (%s) — falling back to layout chunker.",
                filename, exc,
            )
            text_chunks = chunk_document(parsed.elements,
                                         chunk_size=self._rsettings.chunk_size,
                                         chunk_overlap=self._rsettings.chunk_overlap)
            return await self._embed_and_store(doc_id, filename, text_chunks)

        # Convert DocChunk → TextChunk for the shared embed/store path
        text_chunks: list[TextChunk] = []
        for idx, dc in enumerate(doc_chunks):
            heading = ""
            page    = 0
            meta    = getattr(dc, "meta", None)
            if meta:
                # DocMeta exposes headings as a list of strings
                headings = getattr(meta, "headings", None) or []
                if headings:
                    heading = " › ".join(str(h) for h in headings)
                # Page reference
                doc_items = getattr(meta, "doc_items", None) or []
                for di in doc_items:
                    prov = getattr(di, "prov", None) or []
                    for p in prov:
                        pg = getattr(p, "page_no", None)
                        if isinstance(pg, int):
                            page = pg
                            break

            text_chunks.append(TextChunk(
                content=dc.text,
                chunk_index=idx,
                heading=heading,
                page=page,
                element_type="text",
            ))

        if not text_chunks:
            await self._repo.set_status(doc_id, "failed")
            return doc_id

        return await self._embed_and_store(doc_id, filename, text_chunks)

    async def _embed_and_store(
        self,
        doc_id: str,
        filename: str,
        chunks: list[TextChunk],
    ) -> str:
        await self._repo.set_status(doc_id, "processing")

        # Content security scan
        if self._scanner is not None:
            safe: list[TextChunk] = []
            for c in chunks:
                r = self._scanner.scan_input(c.content, source=filename)
                if not r.is_blocked:
                    safe.append(c)
                else:
                    logger.warning(
                        "Chunk %d from '%s' blocked: %s",
                        c.chunk_index, filename, r.findings,
                    )
            if len(safe) < len(chunks):
                logger.warning(
                    "Blocked %d/%d chunks from '%s'.",
                    len(chunks) - len(safe), len(chunks), filename,
                )
            chunks = safe

        if not chunks:
            await self._repo.set_status(doc_id, "failed")
            return doc_id

        texts   = [c.content for c in chunks]
        vectors = self._embedder.encode(texts)

        # Semantic deduplication
        if self._isettings.dedup_threshold > 0.0:
            chunks, vectors = _deduplicate(chunks, vectors, self._isettings.dedup_threshold)

        if not chunks:
            await self._repo.set_chunk_count(doc_id, 0)
            return doc_id

        props = [
            {
                "document_id":  doc_id,
                "filename":     filename,
                "heading":      c.heading,
                "page":         c.page,
                "chunk_index":  c.chunk_index,
                "content":      c.content,
                "element_type": c.element_type,
            }
            for c in chunks
        ]
        wids = self._store.upsert_chunks(props, vectors)
        for chunk, wid in zip(chunks, wids):
            await self._repo.add_chunk(
                document_id=doc_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                heading=chunk.heading,
                page=chunk.page,
                element_type=chunk.element_type,
                weaviate_id=wid,
            )

        await self._repo.set_chunk_count(doc_id, len(chunks))
        logger.info("Ingested %s → %d chunks", filename, len(chunks))
        return doc_id

    async def _retire_version(self, doc) -> None:
        import uuid
        await self._db.execute(
            """
            INSERT INTO document_versions
                (id, document_id, version, filename, content_hash, chunk_count, retired_at)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (uuid.uuid4().hex, doc.id, doc.version, doc.filename,
             doc.content_hash, doc.chunk_count),
        )
        self._store.delete_document_chunks(doc.id)
        await self._repo.delete_chunks(doc.id)
        logger.info("Retired v%d of document %s (%s)", doc.version, doc.id, doc.filename)


# ---------------------------------------------------------------------------
# Semantic deduplication
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _deduplicate(
    chunks: list[TextChunk],
    vectors: list[list[float]],
    threshold: float,
) -> tuple[list[TextChunk], list[list[float]]]:
    kept_c: list[TextChunk]       = []
    kept_v: list[list[float]]     = []
    for chunk, vec in zip(chunks, vectors):
        if not any(_cosine(vec, kv) >= threshold for kv in kept_v):
            kept_c.append(chunk)
            kept_v.append(vec)
    removed = len(chunks) - len(kept_c)
    if removed:
        logger.debug("Deduplication removed %d/%d chunks.", removed, len(chunks))
    return kept_c, kept_v
