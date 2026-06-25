"""
backend/llm/llama_cpp.py
=========================
LLM provider for llama.cpp server.

Changes from v0.11.3:
  [FIX-A] Doubled /v1/v1 URL path corrected.

          httpx.AsyncClient with base_url="http://paeka-llamacpp:8080/v1"
          appends the path constant directly to the base_url.
          The original constants were:
              _COMPLETIONS = "/v1/chat/completions"
              _MODELS      = "/v1/models"
          which produced:
              http://paeka-llamacpp:8080/v1/v1/chat/completions  ← 503

          llama.cpp does not serve anything at that path — hence the 503.
          Fixed by removing the /v1 prefix from the path constants so
          httpx constructs the correct URL:
              http://paeka-llamacpp:8080/v1/chat/completions      ← 200
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

# FIX-A: paths relative to base_url which already contains /v1
_COMPLETIONS = "/chat/completions"
_MODELS      = "/models"
_HEALTH      = "/health"


class LlamaCppProvider(LLMProvider):
    """
    Provider for a running llama.cpp server instance.

    The llama.cpp server speaks the OpenAI /v1/chat/completions protocol,
    so the HTTP layer is identical to what we used with SGLang.
    The difference is entirely at the infrastructure layer:
      - model is a local .gguf file, not a HuggingFace repo id
      - GPU layers are controlled by --n-gpu-layers, not --mem-fraction-static
      - the container is ghcr.io/ggml-org/llama.cpp:server-cuda
    """

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings
        self._http = httpx.AsyncClient(
            base_url=settings.base_url,
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type":  "application/json",
            },
            timeout=httpx.Timeout(
                connect=10.0,
                read=float(settings.request_timeout),
                write=30.0,
                pool=5.0,
            ),
        )

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "llama.cpp"

    async def complete(self, messages: list[Message], **kwargs: Any) -> str:
        payload = self._payload(messages, stream=False, **kwargs)
        try:
            resp = await self._http.post(_COMPLETIONS, json=payload)
            resp.raise_for_status()
            content: str = resp.json()["choices"][0]["message"]["content"] or ""
            logger.debug("llama.cpp complete: %d chars", len(content))
            return content
        except httpx.HTTPStatusError as exc:
            logger.error("llama.cpp HTTP %d: %s", exc.response.status_code, exc)
            raise
        except httpx.ConnectError as exc:
            logger.error("llama.cpp connection error: %s", exc)
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
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0].get("delta", {}).get("content")
                        if delta:
                            yield delta
                    except (json.JSONDecodeError, KeyError):
                        continue
        except httpx.HTTPStatusError as exc:
            logger.error("llama.cpp stream HTTP %d: %s", exc.response.status_code, exc)
            raise
        except httpx.ConnectError as exc:
            logger.error("llama.cpp stream connection error: %s", exc)
            raise

    async def health_check(self) -> bool:
        """
        llama.cpp server exposes GET /health which returns:
          {"status": "ok"} when the model is loaded and ready.
        Note: /health is at the server root, not under /v1.
        We hit it by stripping the base_url's /v1 prefix.
        """
        # /health is not under /v1 — pass an absolute URL so it bypasses
        # self._http's base_url instead of spinning up a second, unmocked,
        # never-closed client just for this one probe.
        root_url = str(self._settings.base_url).rstrip("/").removesuffix("/v1")
        try:
            resp = await self._http.get(f"{root_url}/health", timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("status") in ("ok", "loading model")
        except Exception:  # noqa: BLE001
            pass
        # Fallback: try /v1/models via the regular client
        try:
            resp = await self._http.get(_MODELS, timeout=5.0)
            return resp.status_code == 200
        except Exception as exc:  # noqa: BLE001
            logger.warning("llama.cpp health check failed: %s", exc)
            return False

    async def list_models(self) -> list[str]:
        try:
            resp = await self._http.get(_MODELS, timeout=5.0)
            resp.raise_for_status()
            return [m["id"] for m in resp.json().get("data", [])]
        except Exception:  # noqa: BLE001
            return []

    async def close(self) -> None:
        await self._http.aclose()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

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
