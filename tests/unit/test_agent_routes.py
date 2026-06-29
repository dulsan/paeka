"""
tests/unit/test_agent_routes.py
=================================
Unit tests for ReactRequest.resolve_turn() in backend/api/routes/agent.py,
plus a route-level test exercising the actual react() handler end to end
(LLM and MCP calls mocked, no real FastAPI app/DB needed since the
handler only ever touches request.app.state).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage


def test_resolve_turn_with_messages_returns_last_as_new_turn_and_rest_as_history():
    from backend.api.routes.agent import ReactRequest

    req = ReactRequest(
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
    )

    user_message, history = req.resolve_turn()

    assert user_message == "second"
    assert history == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
    ]


def test_resolve_turn_with_messages_single_entry_has_empty_history():
    from backend.api.routes.agent import ReactRequest

    req = ReactRequest(messages=[{"role": "user", "content": "only this"}])
    user_message, history = req.resolve_turn()

    assert user_message == "only this"
    assert history == []


def test_resolve_turn_falls_back_to_legacy_message_field():
    from backend.api.routes.agent import ReactRequest

    req = ReactRequest(message="legacy single-turn call")
    user_message, history = req.resolve_turn()

    assert user_message == "legacy single-turn call"
    assert history == []


def test_resolve_turn_prefers_messages_when_both_are_present():
    from backend.api.routes.agent import ReactRequest

    req = ReactRequest(message="ignored", messages=[{"role": "user", "content": "used instead"}])
    user_message, _ = req.resolve_turn()

    assert user_message == "used instead"


def test_resolve_turn_raises_when_neither_field_is_set():
    from backend.api.routes.agent import ReactRequest

    req = ReactRequest()
    with pytest.raises(ValueError, match="message"):
        req.resolve_turn()


def test_resolve_turn_raises_on_empty_messages_list():
    from backend.api.routes.agent import ReactRequest

    req = ReactRequest(messages=[])
    with pytest.raises(ValueError):
        req.resolve_turn()


# ---------------------------------------------------------------------------
# Route-level: calls the actual react() handler, not just resolve_turn().
# request.app.state is the only thing the handler touches, so a minimal
# stub stands in for a real FastAPI app/lifespan (no DB, no real Ollama).
# ---------------------------------------------------------------------------


class _FakeProviderLLM:
    provider_name = "fake"

    async def health_check(self) -> bool:
        return True


class _FakeChatOllama:
    """Always returns a final answer with no tool calls -- this test is
    about the route's plumbing (request -> ReActGraph -> ReactResponse),
    not the agent loop itself, which test_react_graph.py already covers."""

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        return AIMessage(content="The answer is 4.")


class _StateStub:
    def __init__(self, llm=None, chat_ollama=None):
        self.llm = llm
        self.chat_ollama = chat_ollama


class _RequestStub:
    def __init__(self, llm=None, chat_ollama=None):
        self.app = type("AppStub", (), {"state": _StateStub(llm=llm, chat_ollama=chat_ollama)})()


async def test_react_route_returns_response_and_empty_tool_calls(monkeypatch):
    from backend.api.routes.agent import react, ReactRequest, ReactResponse

    monkeypatch.setattr(
        "backend.agent.react_graph.get_tool_schemas", AsyncMock(return_value=[])
    )

    request = _RequestStub(llm=_FakeProviderLLM(), chat_ollama=_FakeChatOllama())
    body = ReactRequest(messages=[{"role": "user", "content": "what is 2+2?"}])

    response = await react(body, request)

    assert isinstance(response, ReactResponse)
    assert response.response == "The answer is 4."
    assert response.tool_calls == []


async def test_react_route_returns_503_when_chat_ollama_not_initialised():
    from backend.api.routes.agent import react, ReactRequest
    from fastapi import HTTPException

    request = _RequestStub(llm=_FakeProviderLLM(), chat_ollama=None)
    body = ReactRequest(messages=[{"role": "user", "content": "hi"}])

    with pytest.raises(HTTPException) as exc_info:
        await react(body, request)

    assert exc_info.value.status_code == 503


async def test_react_route_returns_422_when_neither_message_field_is_set():
    from backend.api.routes.agent import react, ReactRequest
    from fastapi import HTTPException

    request = _RequestStub(llm=_FakeProviderLLM(), chat_ollama=_FakeChatOllama())
    body = ReactRequest()

    with pytest.raises(HTTPException) as exc_info:
        await react(body, request)

    assert exc_info.value.status_code == 422
