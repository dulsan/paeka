"""
tests/unit/test_llm_providers.py
==================================
Unit tests for the LLM provider abstraction layer.
Uses httpx mock transport — no real inference server needed.
"""

from __future__ import annotations

import json
import pytest
import httpx

from backend.shared.config import LLMSettings


def _make_settings(**overrides) -> LLMSettings:
    defaults = dict(
        provider="llama_cpp",
        base_url="http://test-llm",
        api_key="test",
        model="test-model",
        model_path="/models/test.gguf",
        request_timeout=30,
    )
    defaults.update(overrides)
    return LLMSettings(**defaults)


def _patch_http(provider, handler):
    """Replace the internal httpx client with a mock transport."""
    provider._http = httpx.AsyncClient(
        base_url="http://test-llm",
        transport=httpx.MockTransport(handler),
    )
    return provider


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_factory_creates_llama_cpp():
    from backend.llm.factory import create_provider
    from backend.llm.llama_cpp import LlamaCppProvider

    settings = _make_settings(provider="llama_cpp")
    provider = create_provider(settings)
    assert isinstance(provider, LlamaCppProvider)
    assert provider.provider_name == "llama.cpp"


def test_factory_creates_ollama():
    from backend.llm.factory import create_provider
    from backend.llm.ollama import OllamaProvider

    settings = _make_settings(provider="ollama")
    provider = create_provider(settings)
    assert isinstance(provider, OllamaProvider)
    assert provider.provider_name == "ollama"


def test_factory_creates_sglang():
    from backend.llm.factory import create_provider
    from backend.llm.sglang import SGLangProvider

    settings = _make_settings(provider="sglang")
    provider = create_provider(settings)
    assert isinstance(provider, SGLangProvider)
    assert provider.provider_name == "sglang"


def test_factory_raises_on_unknown_provider():
    from backend.llm.factory import create_provider

    settings = _make_settings(provider="nonexistent_engine")
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        create_provider(settings)


# ---------------------------------------------------------------------------
# LlamaCppProvider
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_llama_cpp_complete():
    from backend.llm.llama_cpp import LlamaCppProvider

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "Hello from llama.cpp!"}}]
        })

    provider = LlamaCppProvider(_make_settings())
    _patch_http(provider, handler)
    result = await provider.complete([{"role": "user", "content": "hi"}])
    assert result == "Hello from llama.cpp!"
    await provider.close()


@pytest.mark.anyio
async def test_llama_cpp_health_check_ok():
    from backend.llm.llama_cpp import LlamaCppProvider

    def handler(request: httpx.Request) -> httpx.Response:
        if "/health" in str(request.url):
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(404)

    provider = LlamaCppProvider(_make_settings())
    _patch_http(provider, handler)
    assert await provider.health_check() is True
    await provider.close()


@pytest.mark.anyio
async def test_llama_cpp_health_check_loading():
    """llama.cpp returns 'loading model' while model is being loaded — should be True."""
    from backend.llm.llama_cpp import LlamaCppProvider

    def handler(request: httpx.Request) -> httpx.Response:
        if "/health" in str(request.url):
            return httpx.Response(200, json={"status": "loading model"})
        return httpx.Response(404)

    provider = LlamaCppProvider(_make_settings())
    _patch_http(provider, handler)
    assert await provider.health_check() is True
    await provider.close()


@pytest.mark.anyio
async def test_llama_cpp_health_check_fails():
    from backend.llm.llama_cpp import LlamaCppProvider

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    provider = LlamaCppProvider(_make_settings())
    _patch_http(provider, handler)
    assert await provider.health_check() is False
    await provider.close()


@pytest.mark.anyio
async def test_llama_cpp_stream():
    from backend.llm.llama_cpp import LlamaCppProvider

    sse_lines = [
        'data: {"choices": [{"delta": {"content": "Hello"}}]}',
        'data: {"choices": [{"delta": {"content": " world"}}]}',
        "data: [DONE]",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="\n".join(sse_lines) + "\n")

    provider = LlamaCppProvider(_make_settings())
    _patch_http(provider, handler)

    chunks = []
    async for chunk in provider.stream([{"role": "user", "content": "hi"}]):
        chunks.append(chunk)

    assert "".join(chunks) == "Hello world"
    await provider.close()


@pytest.mark.anyio
async def test_system_prompt_injected_once():
    """System prompt must be prepended exactly once."""
    from backend.llm.llama_cpp import LlamaCppProvider

    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.extend(body["messages"])
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}]
        })

    provider = LlamaCppProvider(_make_settings())
    _patch_http(provider, handler)
    await provider.complete([{"role": "user", "content": "hello"}])

    system_msgs = [m for m in captured if m["role"] == "system"]
    assert len(system_msgs) == 1
    await provider.close()


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


def test_registry_scan_empty_dir():
    from backend.models.registry import ModelRegistry
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        registry = ModelRegistry(tmp)
        models = registry.scan()
        assert models == []


def test_registry_scan_finds_gguf():
    from backend.models.registry import ModelRegistry
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        # Create a fake .gguf file
        model_dir = __import__("pathlib").Path(tmp) / "qwen"
        model_dir.mkdir()
        fake_gguf = model_dir / "Qwen3-35B-Q4_K_M.gguf"
        fake_gguf.write_bytes(b"GGUF" + b"\x00" * 1_000_000)  # 1MB fake model

        registry = ModelRegistry(tmp)
        models = registry.scan()
        assert len(models) == 1
        assert models[0].quantisation == "Q4_K_M"
        assert models[0].parameters == "35B"
        assert models[0].family == "qwen"


def test_registry_validate_path_missing():
    from backend.models.registry import ModelRegistry

    registry = ModelRegistry("models")
    ok, msg = registry.validate_path("/nonexistent/model.gguf")
    assert ok is False
    assert "not found" in msg.lower()


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------


def test_loader_sglang_skips_validation():
    from backend.models.loader import resolve_model
    from backend.models.registry import ModelRegistry

    settings = _make_settings(provider="sglang", model="Qwen/Qwen3-14B")
    registry = ModelRegistry("models")
    result = resolve_model(settings, registry)
    assert result.ok is True
    assert "SGLang" in result.message


def test_loader_ollama_skips_validation():
    from backend.models.loader import resolve_model
    from backend.models.registry import ModelRegistry

    settings = _make_settings(provider="ollama", model="qwen3:35b", model_path="")
    registry = ModelRegistry("models")
    result = resolve_model(settings, registry)
    assert result.ok is True
    assert "Ollama" in result.message


def test_loader_llama_cpp_missing_path_fails():
    from backend.models.loader import resolve_model
    from backend.models.registry import ModelRegistry

    settings = _make_settings(provider="llama_cpp", model_path="")
    registry = ModelRegistry("models")
    result = resolve_model(settings, registry)
    assert result.ok is False
    assert "model_path" in result.message


def test_backward_compat_llm_client_alias():
    """LLMClient must remain importable as an alias for LLMProvider."""
    from backend.llm.client import LLMClient
    from backend.llm.base import LLMProvider
    assert LLMClient is LLMProvider
