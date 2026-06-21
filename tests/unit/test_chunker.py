"""
tests/unit/test_chunker.py
===========================
Unit tests for the layout-aware chunker.
"""

from __future__ import annotations

import pytest
from backend.retrieval.chunker import chunk_text, chunk_document, TextChunk


def test_chunk_text_basic():
    text = "First sentence. Second sentence. Third sentence."
    chunks = chunk_text(text, chunk_size=40, chunk_overlap=10)
    assert len(chunks) >= 1
    assert all(isinstance(c, TextChunk) for c in chunks)
    assert all(c.content for c in chunks)


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_overlap():
    long = " ".join([f"Sentence {i}." for i in range(20)])
    chunks = chunk_text(long, chunk_size=100, chunk_overlap=30)
    # Overlap means adjacent chunks should share some content
    if len(chunks) > 1:
        # Last word(s) of chunk N should appear in chunk N+1
        assert chunks[0].chunk_index == 0
        assert chunks[1].chunk_index == 1


def test_chunk_text_heading_passthrough():
    chunks = chunk_text("Some content here.", heading="Section 1", page=3)
    assert chunks[0].heading == "Section 1"
    assert chunks[0].page == 3


def test_chunk_document_atomic_elements():
    """Tables and code blocks must never be split."""
    from backend.ingestion.parsers.base import DocumentElement, ElementType

    big_table = "| A | B |\n|---|---|\n" + "| x | y |\n" * 50

    elements = [
        DocumentElement(ElementType.HEADING, "Section", level=1, heading="Section"),
        DocumentElement(ElementType.TABLE, big_table, heading="Section"),
        DocumentElement(ElementType.TEXT, "Some prose text.", heading="Section"),
    ]

    chunks = chunk_document(elements, chunk_size=64, chunk_overlap=16)
    table_chunks = [c for c in chunks if c.element_type == "table"]
    # Table must be a single chunk regardless of size
    assert len(table_chunks) == 1
    # [FIX] include_headings=True (the default) consistently prepends
    # heading context to every chunk type, tables included -- that's
    # intentional, the same way regular prose chunks get it, since
    # knowing which section a table belongs to is valuable context for
    # retrieval. The test was asserting the bare table content with no
    # prefix, which never matched real behaviour.
    assert table_chunks[0].content == f"Section\n\n{big_table}"


def test_chunk_document_heading_context():
    from backend.ingestion.parsers.base import DocumentElement, ElementType

    elements = [
        DocumentElement(ElementType.HEADING, "Introduction", level=1, heading="Introduction"),
        DocumentElement(ElementType.TEXT, "This is the intro text.", heading="Introduction"),
        DocumentElement(ElementType.HEADING, "Methods", level=1, heading="Methods"),
        DocumentElement(ElementType.TEXT, "This is the methods text.", heading="Methods"),
    ]

    chunks = chunk_document(elements)
    intro_chunks = [c for c in chunks if "intro" in c.content.lower()]
    methods_chunks = [c for c in chunks if "methods" in c.content.lower()]

    assert all(c.heading == "Introduction" for c in intro_chunks)
    assert all(c.heading == "Methods" for c in methods_chunks)
