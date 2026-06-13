"""
backend/models/loader.py
=========================
Model loader — resolves, validates, and reports on the configured model
before PAEKA starts accepting requests.

Called during app lifespan startup. If the model file is missing and
a download URL is configured, it can optionally trigger a download.

This layer keeps model-path logic out of the inference providers
themselves — every provider just receives a validated absolute path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from backend.models.registry import ModelEntry, ModelRegistry
from backend.shared.config import LLMSettings

logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    ok: bool
    model_path: str        # absolute path to the .gguf file (if ok)
    model_entry: ModelEntry | None
    message: str


def resolve_model(settings: LLMSettings, registry: ModelRegistry) -> LoadResult:
    """
    Validate the configured model path and return a LoadResult.

    For llama.cpp and ollama providers the model_path field is used.
    For sglang the model field (HF repo id) is used instead — this
    function is a no-op for sglang (returns ok=True without validation).

    Parameters
    ----------
    settings:
        LLMSettings from config.
    registry:
        ModelRegistry (should have been scanned already).

    Returns
    -------
    LoadResult
    """
    # SGLang uses HF repo ids, not local file paths
    if settings.provider == "sglang":
        return LoadResult(
            ok=True,
            model_path=settings.model,
            model_entry=None,
            message=f"SGLang provider — using HF model: {settings.model}",
        )

    # Ollama uses model tags, not file paths
    if settings.provider == "ollama":
        return LoadResult(
            ok=True,
            model_path=settings.model,
            model_entry=None,
            message=f"Ollama provider — using model tag: {settings.model}",
        )

    # llama.cpp needs a local .gguf file
    model_path = settings.model_path
    if not model_path:
        return LoadResult(
            ok=False,
            model_path="",
            model_entry=None,
            message=(
                "No model_path configured. Set [llm] model_path in settings.toml "
                "or PAEKA_LLM__MODEL_PATH environment variable."
            ),
        )

    # Resolve the path - try as-is first, then relative to models_dir.
    # Prevents double-prepending when model_path already starts with "models\"
    path = Path(model_path)
    if not path.is_absolute():
        if path.exists():
            path = path.resolve()
        else:
            candidate = Path("models") / path
            if candidate.exists():
                path = candidate.resolve()

    ok, msg = registry.validate_path(path)
    if not ok:
        return LoadResult(ok=False, model_path=str(path), model_entry=None, message=msg)

    entry = registry.get_model(path)
    if entry is None:
        # Not in registry yet — build a minimal entry
        from backend.models.registry import ModelEntry
        entry = ModelEntry(
            name=path.stem,
            path=path,
            size_gb=round(path.stat().st_size / 1e9, 2),
        )

    logger.info(
        "Model resolved: %s (%.1f GB, quant=%s)",
        entry.name, entry.size_gb, entry.quantisation or "unknown",
    )

    return LoadResult(ok=True, model_path=str(path), model_entry=entry, message=msg)
