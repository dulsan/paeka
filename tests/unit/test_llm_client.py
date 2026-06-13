"""
tests/unit/test_llm_client.py
================================
Unit tests for the pure-httpx LLM client.
Uses httpx mock transport — no real SGLang needed.
"""

from __future__ import annotations

import json

import pytest
import httpx

from backend.shared.config import LLMSettings
from backend.llm.client import LLMClient


def _make_client(handler) -> LLMClient:
    settings = LLMSettings(
        base_url="http://test-sglang",
        api_key="test-key",
        model="test-model",
    )
    client = LLMClient(settings)
    # Replace the internal httpx client with one using a mock transport
    client._http = httpx.AsyncClient(
        base_url="http://test-sglang",
        transport=httpx.MockTransport(handler),
    )
    return client


@pytest.mark.anyio
async def test_complete_returns_content():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "Hello, world!"}}]
        })

    client = _make_client(handler)
    result = await client.complete([{"role": "user", "content": "hi"}])
    assert result == "Hello, world!"


@pytest.mark.anyio
async def test_complete_raises_on_4xx():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "Unauthorized"})

    client = _make_client(handler)
    with pytest.raises(httpx.HTTPStatusError):
        await client.complete([{"role": "user", "content": "hi"}])


@pytest.mark.anyio
async def test_health_check_ok():
    def handler(request: httpx.Request) -> httpx.Response:
        if "/v1/models" in str(request.url):
            return httpx.Response(200, json={"data": [{"id": "test-model"}]})
        return httpx.Response(404)

    client = _make_client(handler)
    assert await client.health_check() is True


@pytest.mark.anyio
async def test_health_check_fails_on_connection_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client = _make_client(handler)
    # Should return False, not raise
    assert await client.health_check() is False


@pytest.mark.anyio
async def test_system_prompt_injected():
    """System prompt must be prepended when not already present."""
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.extend(body["messages"])
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}]
        })

    client = _make_client(handler)
    await client.complete([{"role": "user", "content": "hello"}])

    assert captured[0]["role"] == "system"
    assert captured[1]["role"] == "user"


@pytest.mark.anyio
async def test_system_prompt_not_duplicated():
    """If caller already includes a system message, don't prepend another."""
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.extend(body["messages"])
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}]
        })

    client = _make_client(handler)
    await client.complete([
        {"role": "system", "content": "Custom system."},
        {"role": "user",   "content": "hello"},
    ])

    system_msgs = [m for m in captured if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert system_msgs[0]["content"] == "Custom system."
