# backend/retrieval/__init__.py
# QdrantStore is the active vector store. Do not add WeaviateStore back here
# -- weaviate-client is not in pyproject.toml and was uninstalled during
# the migration to Qdrant.
#
# [FIX] These used to be eager imports at module level. Because Python always
# runs a package's __init__.py before any of its submodules, that meant
# ANYTHING importing just backend.retrieval.chunker (e.g.
# `from backend.retrieval.chunker import chunk_text`, which only needs pure-
# Python text-splitting logic) was forced to also import engine.py,
# embedder.py (torch), qdrant_store.py (qdrant-client), and reranker.py
# (FlagEmbedding) -- none of which chunker.py needs or uses. Those are
# known to be multi-second-to-import libraries even before loading any
# actual model. This is almost certainly why tests/unit/test_chunker.py
# looked like it had "hung" -- it never got the chance to run a single
# trivial assertion before the import chain finished pulling in the
# entire heavy ML/vector-store stack.
#
# PEP 562 module-level __getattr__ makes these lazy: `from backend.retrieval
# import RetrievalEngine` (or any other name below) still works exactly as
# before, resolved on first access -- but `from backend.retrieval.chunker
# import chunk_text` no longer pays for any of it, since nothing in this
# file touches engine/embedder/qdrant_store/reranker unless one of their
# names is actually requested through this package's namespace.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.retrieval.engine import RetrievalEngine
    from backend.retrieval.embedder import Embedder, get_embedder
    from backend.retrieval.reranker import Reranker, get_reranker, RankedResult
    from backend.retrieval.qdrant_store import QdrantStore, SearchHit
    from backend.retrieval.chunker import TextChunk, chunk_text

__all__ = [
    "RetrievalEngine",
    "Embedder",
    "get_embedder",
    "Reranker",
    "get_reranker",
    "RankedResult",
    "QdrantStore",
    "SearchHit",
    "TextChunk",
    "chunk_text",
]

_LAZY = {
    "RetrievalEngine": ("backend.retrieval.engine", "RetrievalEngine"),
    "Embedder":        ("backend.retrieval.embedder", "Embedder"),
    "get_embedder":    ("backend.retrieval.embedder", "get_embedder"),
    "Reranker":        ("backend.retrieval.reranker", "Reranker"),
    "get_reranker":    ("backend.retrieval.reranker", "get_reranker"),
    "RankedResult":    ("backend.retrieval.reranker", "RankedResult"),
    "QdrantStore":     ("backend.retrieval.qdrant_store", "QdrantStore"),
    "SearchHit":       ("backend.retrieval.qdrant_store", "SearchHit"),
    "TextChunk":       ("backend.retrieval.chunker", "TextChunk"),
    "chunk_text":      ("backend.retrieval.chunker", "chunk_text"),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr_name = target
    import importlib
    module = importlib.import_module(module_path)
    value = getattr(module, attr_name)
    globals()[name] = value  # cache on the module so repeat access is free
    return value


def __dir__():
    return sorted(__all__)
