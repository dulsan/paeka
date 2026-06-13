"""
backend/llm/base.py
====================
Abstract LLM provider interface.

Every inference backend implements this contract.
The rest of PAEKA only ever imports and uses LLMProvider —
it never knows which engine is running underneath.

Implementations:
  LlamaCppProvider  — llama.cpp server (primary, GGUF-native)
  OllamaProvider    — Ollama (optional alternative)
  SGLangProvider    — SGLang (optional, for large GPU deployments)
  VLLMProvider      — vLLM (future)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

# Type alias used throughout the codebase
Message = dict[str, str]   # {"role": "...", "content": "..."}


class LLMProvider(ABC):
    """
    Abstract base for all LLM inference backends.

    All methods are async.  Implementations must be safe to call
    concurrently from multiple coroutines.
    """

    # ------------------------------------------------------------------
    # Required
    # ------------------------------------------------------------------

    @abstractmethod
    async def complete(self, messages: list[Message], **kwargs: Any) -> str:
        """
        Blocking chat completion.

        Parameters
        ----------
        messages:
            Conversation history in OpenAI message format.
        **kwargs:
            Optional overrides: model, temperature, max_tokens, top_p.

        Returns
        -------
        str
            Full assistant reply.
        """
        ...

    @abstractmethod
    async def stream(
        self, messages: list[Message], **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """
        Streaming chat completion.

        Yields text delta chunks as they arrive from the model.
        The caller accumulates them.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the backend is reachable and ready."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release resources (HTTP connections, etc.)."""
        ...

    # ------------------------------------------------------------------
    # Optional — providers may override
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        """Human-readable name shown in logs and health endpoint."""
        return self.__class__.__name__

    async def list_models(self) -> list[str]:
        """
        Return the names of models available on this backend.
        Returns empty list if the backend doesn't support model listing.
        """
        return []
