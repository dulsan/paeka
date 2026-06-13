"""
backend/retrieval/embedder.py
==============================
Dense embedding using BAAI/bge-m3 via sentence-transformers.

Changes:
  [FIX-A] get_sentence_embedding_dimension() renamed to get_embedding_dimension()
          in sentence-transformers >= 3.x. Updated to try new name first,
          fall back to old name for backwards compatibility.
  [FIX-B] encode_async() wraps blocking encode() in asyncio.to_thread().
  [FIX-C] batch_size reduced to 8 on CPU for responsiveness.
"""

from __future__ import annotations

import asyncio
import logging
from functools import lru_cache

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_NORMALIZE = True


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu") -> None:
        self._device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        if self._device != device and device != "cpu":
            logger.warning(
                "CUDA not available -- falling back to CPU for embeddings. "
                "Set embed_device = \"cpu\" in config to suppress this warning."
            )
        logger.info("Loading embedding model %s on %s ...", model_name, self._device)
        self._model = SentenceTransformer(model_name, device=self._device)

        # FIX-A: handle renamed method across sentence-transformers versions
        if hasattr(self._model, "get_embedding_dimension"):
            self._dim: int = self._model.get_embedding_dimension()
        else:
            self._dim = self._model.get_sentence_embedding_dimension()  # type: ignore[assignment]

        self._batch_size = 32 if self._device == "cuda" else 8
        logger.info("Embedder ready -- dim=%d  batch_size=%d  device=%s",
                    self._dim, self._batch_size, self._device)

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            normalize_embeddings=_NORMALIZE,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    async def encode_async(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self.encode, texts)

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]

    async def encode_one_async(self, text: str) -> list[float]:
        result = await self.encode_async([text])
        return result[0]


@lru_cache(maxsize=1)
def get_embedder(model_name: str = "BAAI/bge-m3", device: str = "cpu") -> Embedder:
    return Embedder(model_name, device)
