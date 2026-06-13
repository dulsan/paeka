"""
tests/unit/test_config.py
==========================
Tests for settings loading — no external services required.
"""

from __future__ import annotations

import os
import pytest

from backend.shared.config import Settings, get_settings


def test_defaults_are_valid():
    s = Settings()
    assert s.app.name == "PAEKA"
    assert s.server.port == 8000
    assert s.llm.temperature >= 0.0
    assert s.retrieval.hybrid_alpha >= 0.0
    assert s.retrieval.hybrid_alpha <= 1.0
    assert s.memory.summary_threshold > 0


def test_env_override(monkeypatch):
    """Environment variables prefixed PAEKA_ should override TOML values."""
    monkeypatch.setenv("PAEKA_SERVER__PORT", "9999")
    # Clear cache so env var is picked up
    get_settings.cache_clear()
    s = get_settings.__wrapped__("config/settings.toml")  # type: ignore[attr-defined]
    assert s.server.port == 9999
    get_settings.cache_clear()


def test_retrieval_chunk_overlap_less_than_size():
    s = Settings()
    assert s.retrieval.chunk_overlap < s.retrieval.chunk_size
