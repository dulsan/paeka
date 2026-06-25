"""
tests/unit/test_diagnostics.py
================================
Tests for backend.tools.diagnostics._build_services -- no external
services required (only checks host/port resolution, not live probing).
"""

from __future__ import annotations

from backend.shared.config import get_settings
from backend.tools.diagnostics import _build_services, _split


def _names_to_hosts(services: list[dict]) -> dict[str, str]:
    return {s["name"]: s["host"] for s in services}


def test_split_extracts_host_and_port():
    assert _split("http://localhost:6333", "x", 1) == ("localhost", 6333)
    assert _split("http://paeka-ollama:11434/v1", "x", 1) == ("paeka-ollama", 11434)


def test_split_falls_back_on_garbage():
    assert _split("not a url", "fallback-host", 4242) == ("fallback-host", 4242)


def test_build_services_defaults_to_localhost():
    get_settings.cache_clear()
    try:
        services = _build_services()
    finally:
        get_settings.cache_clear()

    hosts = _names_to_hosts(services)
    assert hosts["Ollama (LLM)"] == "localhost"
    assert hosts["Qdrant (vector DB)"] == "localhost"
    # The API checking itself is always self-referential.
    assert hosts["PAEKA API"] == "localhost"


def test_build_services_respects_env_override(monkeypatch):
    """
    The whole point of this rewrite: once Ollama/Qdrant move into their own
    containers (reachable only by service name), diagnostics must follow
    the same env-var overrides the rest of the app already honours --
    not silently keep assuming localhost.
    """
    monkeypatch.setenv("PAEKA_LLM__BASE_URL", "http://paeka-ollama:11434/v1")
    monkeypatch.setenv("PAEKA_RETRIEVAL__QDRANT_URL", "http://paeka-qdrant:6333")
    get_settings.cache_clear()
    try:
        services = _build_services()
    finally:
        get_settings.cache_clear()

    hosts = _names_to_hosts(services)
    assert hosts["Ollama (LLM)"] == "paeka-ollama"
    assert hosts["Qdrant (vector DB)"] == "paeka-qdrant"


def test_build_services_falls_back_cleanly_if_settings_unavailable(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("settings file missing")

    monkeypatch.setattr("backend.shared.config.get_settings", _boom)
    services = _build_services()

    hosts = _names_to_hosts(services)
    assert hosts["Ollama (LLM)"] == "localhost"
    assert hosts["Qdrant (vector DB)"] == "localhost"
