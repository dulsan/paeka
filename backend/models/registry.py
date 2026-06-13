"""
backend/models/registry.py
===========================
GGUF model registry.

Scans the /models directory and maintains a catalogue of available
model files with their metadata. Models are discovered by convention:

  models/
  ├── qwen/
  │   ├── Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
  │   └── metadata.json      (optional)
  ├── deepseek/
  │   └── DeepSeek-R1-Q4.gguf
  └── mistral/
      └── Mistral-Small.gguf

The active model is set in settings.toml:

  [llm]
  model_path = "/models/qwen/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"

metadata.json (optional per model directory):
  {
    "name":        "Qwen3.6 35B Q4_K_M",
    "family":      "qwen",
    "parameters":  "35B",
    "quantisation": "Q4_K_M",
    "context":     8192,
    "chat_template": "chatml",
    "description": "Qwen3.6 35B MoE, 4-bit quantised"
  }
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_MODELS_DIR = Path("models")


@dataclass
class ModelEntry:
    """A single discovered GGUF model."""

    name: str                  # human-readable name (from metadata or filename)
    path: Path                 # absolute path to the .gguf file
    family: str = ""           # model family (qwen, deepseek, mistral, llama, …)
    parameters: str = ""       # e.g. "35B", "7B"
    quantisation: str = ""     # e.g. "Q4_K_M", "Q8_0"
    context: int = 4096        # max context window
    chat_template: str = ""    # chatml | llama3 | mistral | gemma | …
    description: str = ""
    size_gb: float = 0.0       # file size in GB
    metadata: dict = field(default_factory=dict)

    @property
    def filename(self) -> str:
        return self.path.name

    @property
    def exists(self) -> bool:
        return self.path.exists()


class ModelRegistry:
    """
    Scans the models directory and exposes a catalogue of available GGUF models.

    Parameters
    ----------
    models_dir:
        Root directory to scan (default: ./models).
    """

    def __init__(self, models_dir: str | Path = _MODELS_DIR) -> None:
        self._dir = Path(models_dir)
        self._models: dict[str, ModelEntry] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self) -> list[ModelEntry]:
        """
        Scan the models directory and return all discovered GGUF files.
        Re-scans on every call (cheap — filesystem glob).
        """
        self._models.clear()
        if not self._dir.exists():
            logger.debug("Models directory not found: %s", self._dir)
            return []

        for gguf_path in sorted(self._dir.rglob("*.gguf")):
            entry = self._build_entry(gguf_path)
            self._models[str(gguf_path)] = entry
            logger.debug("Registered model: %s", entry.name)

        logger.info(
            "Model registry: %d model(s) found in %s",
            len(self._models), self._dir,
        )
        return list(self._models.values())

    def list_models(self) -> list[ModelEntry]:
        """Return currently catalogued models (call scan() first)."""
        return list(self._models.values())

    def get_model(self, path: str | Path) -> ModelEntry | None:
        """Lookup a model by its file path."""
        return self._models.get(str(Path(path)))

    def find_by_name(self, name: str) -> ModelEntry | None:
        """Find a model by partial filename or display name match."""
        name_lower = name.lower()
        for entry in self._models.values():
            if (name_lower in entry.name.lower()
                    or name_lower in entry.filename.lower()):
                return entry
        return None

    def validate_path(self, model_path: str | Path) -> tuple[bool, str]:
        """
        Check whether a model path is usable.

        Returns (ok: bool, message: str).
        """
        path = Path(model_path)
        if not path.is_absolute():
            # Only prepend models_dir if path doesn't already exist as-is
            # Prevents double-prepending (models\models\...)
            if not path.exists():
                path = self._dir / path

        if not path.exists():
            return False, f"Model file not found: {path}"
        if path.suffix.lower() != ".gguf":
            return False, f"Expected a .gguf file, got: {path.suffix}"
        if path.stat().st_size < 1_000_000:
            return False, f"File too small to be a valid model: {path}"

        return True, f"OK ({path.stat().st_size / 1e9:.1f} GB)"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_entry(self, path: Path) -> ModelEntry:
        """Build a ModelEntry from a .gguf file path + optional metadata.json."""
        meta: dict = {}
        meta_file = path.parent / "metadata.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.debug("Could not read %s: %s", meta_file, exc)

        size_bytes = path.stat().st_size if path.exists() else 0

        # Infer fields from filename if not in metadata
        filename_stem = path.stem  # e.g. "Qwen3.6-35B-A3B-UD-Q4_K_M"
        family = meta.get("family", path.parent.name.lower())
        quant  = _infer_quantisation(filename_stem)

        return ModelEntry(
            name=meta.get("name", filename_stem),
            path=path,
            family=family,
            parameters=meta.get("parameters", _infer_parameters(filename_stem)),
            quantisation=meta.get("quantisation", quant),
            context=int(meta.get("context", 4096)),
            chat_template=meta.get("chat_template", _infer_template(family)),
            description=meta.get("description", ""),
            size_gb=round(size_bytes / 1e9, 2),
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _infer_quantisation(stem: str) -> str:
    """Extract quantisation type from filename stem."""
    import re
    # Match patterns like Q4_K_M, Q8_0, Q5_K_S, IQ3_M, etc.
    m = re.search(r"(IQ\d_\w+|Q\d+_\w+)", stem, re.I)
    return m.group(0).upper() if m else ""


def _infer_parameters(stem: str) -> str:
    """Extract parameter count from filename stem (e.g. 35B, 7B, 14B)."""
    import re
    m = re.search(r"(\d+\.?\d*)[Bb]", stem)
    return f"{m.group(1)}B" if m else ""


def _infer_template(family: str) -> str:
    """Infer likely chat template from model family name."""
    family = family.lower()
    if "qwen" in family:      return "chatml"
    if "llama" in family:     return "llama3"
    if "mistral" in family:   return "mistral"
    if "gemma" in family:     return "gemma"
    if "deepseek" in family:  return "chatml"
    if "phi" in family:       return "chatml"
    return ""
