"""
backend/llm/ollama.py
======================
LLM provider for Ollama.

Ollama is an alternative local inference backend that also supports GGUF
(via its Modelfile system) and provides an OpenAI-compatible API.

Use Ollama when:
  - You want a GUI model manager (Ollama desktop)
  - You prefer `ollama pull` model management
  - You're running on a Mac (Ollama has excellent Metal support)

Use llama.cpp when:
  - You want direct GGUF file control
  - You need fine-grained GPU layer tuning
  - You're deploying in Docker on Linux with CUDA

Ollama's OpenAI-compatible endpoint: http://localhost:11434/v1

Set in settings.toml:
  [llm]
  provider = "ollama"
  base_url = "http://localhost:11434/v1"
  model    = "qwen3:35b"    # Ollama model tag
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from backend.llm.base import LLMProvider, Message
from backend.shared.config import LLMSettings

logger = logging.getLogger(__name__)

_COMPLETIONS = "/v1/chat/completions"
_MODELS      = "/v1/models"
_TAGS        = "/api/tags"   # Ollama-native endpoint for model listing


class OllamaProvider(LLMProvider):
    """Provider for a running Ollama instance."""

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        self._http = httpx.AsyncClient(
            base_url=settings.base_url,
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(
                connect=10.0,
                read=float(settings.request_timeout),
                write=30.0,
                pool=5.0,
            ),
        )

    @property
    def provider_name(self) -> str:
        return "ollama"

    async def complete(self, messages: list[Message], **kwargs: Any) -> str:
        payload = self._payload(messages, stream=False, **kwargs)
        try:
            resp = await self._http.post(_COMPLETIONS, json=payload)
            resp.raise_for_status()
            content: str = resp.json()["choices"][0]["message"]["content"] or ""
            logger.debug("Ollama complete: %d chars", len(content))
            return content
        except httpx.HTTPStatusError as exc:
            logger.error("Ollama HTTP %d: %s", exc.response.status_code, exc)
            raise
        except httpx.ConnectError as exc:
            logger.error("Ollama connection error: %s", exc)
            raise

    async def stream(
        self, messages: list[Message], **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        payload = self._payload(messages, stream=True, **kwargs)
        try:
            async with self._http.stream("POST", _COMPLETIONS, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        delta = json.loads(data_str)["choices"][0].get("delta", {}).get("content")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError):
                        continue
        except httpx.HTTPStatusError as exc:
            logger.error("Ollama stream HTTP %d: %s", exc.response.status_code, exc)
            raise

    async def health_check(self) -> bool:
        try:
            resp = await self._http.get(_MODELS, timeout=5.0)
            return resp.status_code == 200
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama health check failed: %s", exc)
            return False

    async def list_models(self) -> list[str]:
        """Use Ollama's native /api/tags endpoint for richer model info."""
        try:
            # Try OpenAI-compat first
            resp = await self._http.get(_MODELS, timeout=5.0)
            resp.raise_for_status()
            return [m["id"] for m in resp.json().get("data", [])]
        except Exception:  # noqa: BLE001
            return []

    async def close(self) -> None:
        await self._http.aclose()

    def _payload(
        self, messages: list[Message], stream: bool, **overrides: Any
    ) -> dict[str, Any]:
        s = self._settings
        return {
            "model":       overrides.pop("model",       s.model),
            "messages":    self._inject_system(messages),
            "max_tokens":  overrides.pop("max_tokens",  s.max_tokens),
            "temperature": overrides.pop("temperature", s.temperature),
            "top_p":       overrides.pop("top_p",       s.top_p),
            "stream":      stream,
            **overrides,
        }

    def _inject_system(self, messages: list[Message]) -> list[Message]:
        if messages and messages[0]["role"] == "system":
            return messages
        return [{"role": "system", "content": self._settings.system_prompt}, *messages]
