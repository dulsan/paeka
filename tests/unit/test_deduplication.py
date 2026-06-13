"""
tests/unit/test_deduplication.py
==================================
Unit tests for the semantic deduplication in the ingestion pipeline.
"""

from __future__ import annotations

import pytest
from backend.ingestion.pipeline import _deduplicate, _cosine
from backend.retrieval.chunker import TextChunk


def _make_chunk(content: str, idx: int = 0) -> TextChunk:
    return TextChunk(content=content, chunk_index=idx)


def test_cosine_identical():
    v = [1.0, 0.0, 0.0]
    assert _cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert _cosine(a, b) == pytest.approx(0.0)


def test_cosine_zero_vector():
    assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_dedup_removes_identical():
    chunks = [_make_chunk("A", 0), _make_chunk("B", 1), _make_chunk("A again", 2)]
    # Use identical vectors for chunks 0 and 2
    vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    kept_c, kept_v = _deduplicate(chunks, vectors, threshold=0.99)
    assert len(kept_c) == 2
    assert kept_c[0].content == "A"
    assert kept_c[1].content == "B"


def test_dedup_keeps_distinct():
    chunks = [_make_chunk("A", 0), _make_chunk("B", 1), _make_chunk("C", 2)]
    vectors = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    kept_c, kept_v = _deduplicate(chunks, vectors, threshold=0.99)
    assert len(kept_c) == 3


def test_dedup_threshold_boundary():
    chunks = [_make_chunk("A", 0), _make_chunk("B", 1)]
    # Cosine of these two vectors is ~0.894
    vectors = [[1.0, 1.0], [1.0, 0.5]]
    # threshold 0.99 → both kept
    kept, _ = _deduplicate(chunks, vectors, threshold=0.99)
    assert len(kept) == 2
    # threshold 0.80 → second removed
    kept, _ = _deduplicate(chunks, vectors, threshold=0.80)
    assert len(kept) == 1
