"""
tests/unit/test_export.py
==========================
Unit tests for the conversation export formatters.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock

from backend.api.routes.export import _to_json, _to_markdown, _safe_name


def _mock_conv(id="c1", title="Test Conv"):
    m = MagicMock()
    m.id = id
    m.title = title
    m.created_at = "2025-01-01T00:00:00+00:00"
    m.updated_at = "2025-01-01T01:00:00+00:00"
    return m


def _mock_message(role: str, content: str):
    m = MagicMock()
    m.id = "m1"
    m.role = role
    m.content = content
    m.created_at = "2025-01-01T00:00:00+00:00"
    return m


def test_to_json_structure():
    conv = _mock_conv()
    messages = [
        _mock_message("user", "Hello"),
        _mock_message("assistant", "Hi there!"),
    ]
    data = _to_json(conv, messages, session_summary="Summary here.")

    assert data["id"] == "c1"
    assert data["title"] == "Test Conv"
    assert data["message_count"] == 2
    assert data["session_summary"] == "Summary here."
    assert len(data["messages"]) == 2
    assert data["messages"][0]["role"] == "user"
    assert data["messages"][1]["role"] == "assistant"


def test_to_json_no_summary():
    conv = _mock_conv()
    data = _to_json(conv, [], session_summary=None)
    assert data["session_summary"] is None
    assert data["messages"] == []


def test_to_markdown_contains_required_sections():
    conv = _mock_conv(title="My Conversation")
    messages = [
        _mock_message("user", "What is attention?"),
        _mock_message("assistant", "Attention is a mechanism..."),
    ]
    md = _to_markdown(conv, messages, session_summary="We discussed attention.")

    assert "# My Conversation" in md
    assert "**User**" in md
    assert "**PAEKA**" in md
    assert "What is attention?" in md
    assert "Attention is a mechanism" in md
    assert "## Session Summary" in md
    assert "We discussed attention." in md


def test_to_markdown_no_summary_omits_section():
    conv = _mock_conv()
    md = _to_markdown(conv, [], session_summary=None)
    assert "## Session Summary" not in md


def test_safe_name_removes_special_chars():
    assert _safe_name("My Conversation!") == "My_Conversation"
    assert _safe_name("Hello/World") == "HelloWorld"
    assert _safe_name("") == "conversation"
    assert len(_safe_name("a" * 100)) <= 40


def test_json_is_valid():
    conv = _mock_conv()
    messages = [_mock_message("user", "test")]
    data = _to_json(conv, messages, None)
    # Should be JSON-serialisable
    serialised = json.dumps(data)
    parsed = json.loads(serialised)
    assert parsed["id"] == "c1"
