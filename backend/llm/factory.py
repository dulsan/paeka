"""
backend/llm/factory.py
=======================
LLM provider factory.

Reads [llm] provider from settings and returns the correct LLMProvider
implementation. The rest of PAEKA calls this once at startup and stores
the result in app.state.llm — no other code cares which provider is running.

Provider registry:
  "llama_cpp"   → LlamaCppProvider  (primary; GGUF-native)
  "ollama"      → OllamaProvider    (optional alternative)
  "sglang"      → SGLangProvider    (optional; large GPU deployments)

Adding a new provider:
  1. Create backend/llm/my_provider.py implementing LLMProvider
  2. Add an entry to _REGISTRY below
  3. Set [llm] provider = "my_provider" in settings.toml
  No other changes needed anywhere in PAEKA.
"""

from __future__ import annotations

import logging

from backend.llm.base import LLMProvider
from backend.shared.config import LLMSettings

logger = logging.getLogger(__name__)

# Registry: provider name → import path and class name
# Lazy imports — only the chosen provider is ever imported
_REGISTRY: dict[str, tuple[str, str]] = {
    "llama_cpp": ("backend.llm.llama_cpp", "LlamaCppProvider"),
    "ollama":    ("backend.llm.ollama",    "OllamaProvider"),
    "sglang":    ("backend.llm.sglang",    "SGLangProvider"),
}


def create_provider(settings: LLMSettings) -> LLMProvider:
    """
    Instantiate and return the configured LLM provider.

    Parameters
    ----------
    settings:
        LLMSettings containing ``provider``, ``base_url``, ``model``, etc.

    Returns
    -------
    LLMProvider
        The configured provider, ready to use.

    Raises
    ------
    ValueError
        If the provider name is not in the registry.
    """
    provider_name = settings.provider.lower().strip()

    if provider_name not in _REGISTRY:
        available = ", ".join(_REGISTRY.keys())
        raise ValueError(
            f"Unknown LLM provider: '{provider_name}'. "
            f"Available providers: {available}"
        )

    module_path, class_name = _REGISTRY[provider_name]

    import importlib
    module = importlib.import_module(module_path)
    provider_class = getattr(module, class_name)
    provider: LLMProvider = provider_class(settings)

    logger.info(
        "LLM provider: %s | base_url=%s | model=%s",
        provider.provider_name,
        settings.base_url,
        settings.model,
    )
    return provider
