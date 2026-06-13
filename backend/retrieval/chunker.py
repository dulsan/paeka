"""
backend/retrieval/chunker.py
=============================
Layout-aware chunker that operates on ``DocumentElement`` objects
produced by the Phase 4 parsers.

Chunking rules per element type:
  - HEADING       : never a standalone chunk; sets context for following elements
  - TABLE         : always its own chunk (never split across boundaries)
  - CODE          : always its own chunk
  - EQUATION      : always its own chunk
  - FIGURE/CAPTION: always its own chunk
  - TEXT/LIST_ITEM: accumulated up to chunk_size, split on sentence boundaries,
                    with overlap carried forward

Each TextChunk carries:
  - content      : the chunk text
  - chunk_index  : sequential index within the document
  - heading      : nearest ancestor section heading
  - page         : page number (0 if unknown)
  - element_type : type string for downstream filtering
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class TextChunk:
    content: str
    chunk_index: int
    heading: str = ""
    page: int = 0
    element_type: str = "text"
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Primary entry point  — operates on DocumentElement objects
# ---------------------------------------------------------------------------


def chunk_document(
    elements: list,                    # list[DocumentElement] — avoid circular import
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[TextChunk]:
    """
    Produce ``TextChunk`` objects from a list of ``DocumentElement`` objects.

    Structure-preserving rules:
      - Tables, code blocks, equations, and figures become individual chunks
        regardless of size (they must not be split).
      - Prose elements (TEXT, LIST_ITEM, CAPTION) are accumulated and split
        at sentence boundaries with overlap, inheriting the current heading.
      - A heading change always flushes the current prose accumulation.

    Parameters
    ----------
    elements:
        Ordered list of ``DocumentElement`` from a parser.
    chunk_size:
        Maximum character length for prose chunks.
    chunk_overlap:
        Character overlap between adjacent prose chunks.

    Returns
    -------
    list[TextChunk]
        Ordered, indexed chunks ready for embedding.
    """
    from backend.ingestion.parsers.base import ElementType

    chunks: list[TextChunk] = []
    prose_buf: list[str] = []
    prose_len: int = 0
    current_heading: str = ""
    current_page: int = 0
    idx: int = 0

    def flush_prose() -> None:
        nonlocal prose_buf, prose_len, idx
        if not prose_buf:
            return
        text = " ".join(prose_buf).strip()
        for chunk in _split_prose(text, chunk_size, chunk_overlap):
            if chunk.strip():
                chunks.append(TextChunk(
                    content=chunk,
                    chunk_index=idx,
                    heading=current_heading,
                    page=current_page,
                    element_type="text",
                ))
                idx += 1
        prose_buf = []
        prose_len = 0

    for el in elements:
        etype = el.element_type

        # ── Heading: flush prose, update context ─────────────────────
        if etype == ElementType.HEADING:
            flush_prose()
            current_heading = el.content
            current_page = el.page
            continue

        # ── Atomic elements: always individual chunks ─────────────────
        if etype in (ElementType.TABLE, ElementType.CODE,
                     ElementType.EQUATION, ElementType.FIGURE):
            flush_prose()
            if el.content.strip():
                chunks.append(TextChunk(
                    content=el.content.strip(),
                    chunk_index=idx,
                    heading=el.heading or current_heading,
                    page=el.page or current_page,
                    element_type=str(etype),
                ))
                idx += 1
            continue

        # ── Prose: accumulate ─────────────────────────────────────────
        if etype in (ElementType.TEXT, ElementType.LIST_ITEM, ElementType.CAPTION):
            content = el.content.strip()
            if not content:
                continue

            # Heading change flushes buffer
            if el.heading and el.heading != current_heading:
                flush_prose()
                current_heading = el.heading

            current_page = el.page or current_page

            if prose_len + len(content) > chunk_size and prose_buf:
                flush_prose()

            prose_buf.append(content)
            prose_len += len(content) + 1

    flush_prose()
    return chunks


# ---------------------------------------------------------------------------
# Legacy entry point  — operates on raw strings (used in tests / ingest_text)
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    heading: str = "",
    page: int = 0,
) -> list[TextChunk]:
    """
    Split a raw string into overlapping prose chunks.
    Kept for backward compatibility with ``ingest_text()``.
    """
    text = text.strip()
    if not text:
        return []

    sentences = _split_sentences(text)
    chunks: list[TextChunk] = []
    current: list[str] = []
    current_len = 0
    idx = 0

    for sent in sentences:
        sent_len = len(sent)

        if current_len + sent_len > chunk_size and current:
            chunk_str = " ".join(current).strip()
            if chunk_str:
                chunks.append(TextChunk(
                    content=chunk_str,
                    chunk_index=idx,
                    heading=heading,
                    page=page,
                ))
                idx += 1

            # Overlap carry-forward
            overlap_buf: list[str] = []
            overlap_chars = 0
            for tok in reversed(current):
                if overlap_chars + len(tok) > chunk_overlap:
                    break
                overlap_buf.insert(0, tok)
                overlap_chars += len(tok) + 1

            current = overlap_buf
            current_len = sum(len(t) + 1 for t in current)

        current.append(sent)
        current_len += sent_len + 1

    if current:
        chunk_str = " ".join(current).strip()
        if chunk_str:
            chunks.append(TextChunk(
                content=chunk_str,
                chunk_index=idx,
                heading=heading,
                page=page,
            ))

    return chunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_prose(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split a prose block into sized chunks using sentence boundaries."""
    sentences = _split_sentences(text)
    result: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len > chunk_size and current:
            result.append(" ".join(current).strip())

            overlap_buf: list[str] = []
            overlap_chars = 0
            for tok in reversed(current):
                if overlap_chars + len(tok) > chunk_overlap:
                    break
                overlap_buf.insert(0, tok)
                overlap_chars += len(tok) + 1

            current = overlap_buf
            current_len = sum(len(t) + 1 for t in current)

        current.append(sent)
        current_len += sent_len + 1

    if current:
        result.append(" ".join(current).strip())

    return result


def _split_sentences(text: str) -> list[str]:
    """Heuristic sentence splitter: paragraphs first, then sentence terminals."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sentences: list[str] = []
    for para in paragraphs:
        parts = re.split(r"(?<=[.!?])\s+", para)
        sentences.extend(p.strip() for p in parts if p.strip())
    return sentences
