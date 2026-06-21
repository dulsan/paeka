"""
backend/retrieval/reranker.py
==============================
Cross-encoder reranking using BAAI/bge-reranker-large via FlagEmbedding.

Given a query and a list of candidate passages, the reranker assigns
a relevance score to each (query, passage) pair and returns the top-N
passages sorted by score descending.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import torch
from FlagEmbedding import FlagReranker

logger = logging.getLogger(__name__)


@dataclass
class RankedResult:
    """A passage with its reranker relevance score."""

    content: str
    score: float
    metadata: dict


class Reranker:
    """Wraps BGE-Reranker-Large for cross-encoder scoring."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        device: str = "cuda",
    ) -> None:
        _device = device if torch.cuda.is_available() else "cpu"
        if _device != device:
            logger.warning("CUDA not available — reranker running on CPU.")
        logger.info("Loading reranker %s on %s …", model_name, _device)
        self._reranker = FlagReranker(model_name, use_fp16=(_device == "cuda"))
        logger.info("Reranker ready.")

    def rerank(
        self,
        query: str,
        passages: list[dict],
        top_n: int = 5,
        content_key: str = "content",
    ) -> list[RankedResult]:
        """
        Score each passage against the query and return the top_n highest.

        Parameters
        ----------
        query:
            The user query string.
        passages:
            List of dicts, each must have at least ``content_key``.
        top_n:
            Maximum number of results to return.
        content_key:
            Key in each passage dict that holds the text to score.

        Returns
        -------
        list[RankedResult]
            Sorted descending by relevance score.
        """
        if not passages:
            return []

        pairs = [[query, p[content_key]] for p in passages]
        scores: list[float] = self._reranker.compute_score(pairs, normalize=True)

        ranked = sorted(
            zip(scores, passages, strict=True),
            key=lambda x: x[0],
            reverse=True,
        )

        return [
            RankedResult(
                content=p[content_key],
                score=score,
                metadata={k: v for k, v in p.items() if k != content_key},
            )
            for score, p in ranked[:top_n]
        ]


@lru_cache(maxsize=1)
def get_reranker(
    model_name: str = "BAAI/bge-reranker-large",
    device: str = "cuda",
) -> Reranker:
    """Return the singleton Reranker (loaded once per process)."""
    return Reranker(model_name, device)
