"""
backend/llm/litellm_provider.py
================================
LiteLLM-backed LLMProvider implementation, used specifically by ReActGraph
for native function-calling support over Ollama's OpenAI-compatible API.

This is separate from the main chat provider (OllamaProvider, selected via
PAEKA_LLM__PROVIDER in .env) which handles plain conversational completion
for /v1/chat/completions and the existing AgenticRAGPipeline. LiteLLMProvider
is instantiated directly by ReActGraph because it needs raw access to
tool_calls and finish_reason on the response object, which the LLMProvider.complete()
str-only interface intentionally hides from the rest of the codebase.

Why LiteLLM specifically:
  - Normalises function-calling schemas across providers (Ollama, OpenAI,
    Anthropic) to one format, so MCP tool schemas need no per-provider mapping.
  - acompletion() is fully async with native httpx transport underneath,
    which logfire.instrument_httpx() can trace automatically.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from typing import Any

import litellm
from litellm import acompletion

from backend.llm.base import LLMProvider, Message

logger = logging.getLogger(__name__)

litellm.telemetry = os.environ.get("LITELLM_TELEMETRY", "false").lower() != "true"
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


class LiteLLMProvider(LLMProvider):
    """
    LiteLLM-backed provider configured to talk to Ollama by default.

    Configuration is read from environment variables, falling back to
    sensible defaults that match the Ollama setup already in .env:
      LITELLM_MODEL    e.g. "ollama/paeka-qwen" (LiteLLM's Ollama provider prefix)
      LITELLM_API_BASE e.g. "http://localhost:11434"
      LITELLM_API_KEY  unused by Ollama, kept for OpenAI-compatible providers
    """

    def __init__(
        self,
        model: str | None = None,
        api_base: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 180,
    ) -> None:
        configured_model = os.environ.get("PAEKA_LLM__MODEL", "paeka-qwen")
        self._model       = model    or os.environ.get("LITELLM_MODEL", f"ollama/{configured_model}")
        self._api_base    = api_base or os.environ.get("LITELLM_API_BASE", "http://localhost:11434")
        self._api_key     = api_key  or os.environ.get("LITELLM_API_KEY", "")
        self._temperature = temperature
        self._max_tokens  = max_tokens
        self._timeout     = timeout

        logger.info("LiteLLMProvider: model=%s base=%s", self._model, self._api_base)

    async def complete(
        self,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> str:
        call_kw: dict[str, Any] = {
            "model":       self._model,
            "api_base":    self._api_base,
            "messages":    messages,
            "max_tokens":  max_tokens  or self._max_tokens,
            "temperature": temperature if temperature is not None else self._temperature,
            "timeout":     self._timeout,
            **kwargs,
        }
        if self._api_key:
            call_kw["api_key"] = self._api_key
        if tools:
            call_kw["tools"]       = tools
            call_kw["tool_choice"] = "auto"

        try:
            response = await acompletion(**call_kw)
            msg = response.choices[0].message
            if msg.tool_calls:
                import json
                return json.dumps([
                    {"tool": tc.function.name, "arguments": json.loads(tc.function.arguments)}
                    for tc in msg.tool_calls
                ])
            return msg.content or ""
        except litellm.APIConnectionError as exc:
            logger.error("LiteLLM connection error (ollama serve running at %s?): %s",
                        self._api_base, exc)
            raise
        except Exception as exc:
            logger.error("LiteLLM complete() error: %s", exc)
            raise

    async def stream(
        self,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        call_kw: dict[str, Any] = {
            "model":       self._model,
            "api_base":    self._api_base,
            "messages":    messages,
            "max_tokens":  max_tokens  or self._max_tokens,
            "temperature": temperature if temperature is not None else self._temperature,
            "timeout":     self._timeout,
            "stream":      True,
            **kwargs,
        }
        if self._api_key:
            call_kw["api_key"] = self._api_key

        async def _gen() -> AsyncGenerator[str, None]:
            response = await acompletion(**call_kw)
            async for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content

        return _gen()

    async def health_check(self) -> bool:
        try:
            await acompletion(
                model=self._model, api_base=self._api_base,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1, timeout=5,
            )
            return True
        except Exception:
            return False

    async def close(self) -> None:
        pass

    @property
    def provider_name(self) -> str:
        return f"litellm({self._model})"

    # ------------------------------------------------------------------
    # Raw access needed by react_graph.py to inspect finish_reason/tool_calls
    # ------------------------------------------------------------------

    async def acompletion_raw(self, messages: list[dict], tools: list | None = None, **kwargs):
        """Return the full LiteLLM response object, not just text."""
        call_kw: dict[str, Any] = {
            "model":       self._model,
            "api_base":    self._api_base,
            "messages":    messages,
            "max_tokens":  self._max_tokens,
            "temperature": self._temperature,
            "timeout":     self._timeout,
            **kwargs,
        }
        if self._api_key:
            call_kw["api_key"] = self._api_key
        if tools:
            call_kw["tools"]       = tools
            call_kw["tool_choice"] = "auto"

        try:
            import logfire
            span_ctx = logfire.span(
                "litellm.acompletion",
                model=self._model,
                message_count=len(messages),
                tool_count=len(tools) if tools else 0,
            )
        except ImportError:
            span_ctx = None

        if span_ctx is None:
            return await acompletion(**call_kw)

        with span_ctx as span:
            response = await acompletion(**call_kw)
            choice = response.choices[0]
            span.set_attribute("finish_reason", choice.finish_reason)
            span.set_attribute("tool_calls_returned",
                               len(choice.message.tool_calls or []))
            span.set_attribute("response_chars", len(choice.message.content or ""))
            return response
