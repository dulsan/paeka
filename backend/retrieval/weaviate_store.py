"""
backend/retrieval/weaviate_store.py
====================================
Weaviate v4 client wrapper.

Changes from v0.10.0:
  [FIX-A] URL parsing: replaced fragile string splitting with urllib.parse.
          The original code split on ":" which breaks for any URL that
          has a path component after the port (e.g. http://host:8080/path).

  [FIX-B] connect() and _ensure_collection() are async but called
          weaviate.connect_to_local() — a blocking synchronous function —
          directly on the event loop. Under Docker's slower DNS this stalls
          all coroutines for multiple seconds. Both blocking calls are now
          wrapped in asyncio.to_thread() so they run in a thread pool.

Nothing else changed. All method signatures, return types, and Weaviate
query logic are identical to the original.
"""

from __future__ import annotations

import asyncio
import logging
import uuid as _uuid
from dataclasses import dataclass
from urllib.parse import urlparse

import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import HybridFusion, MetadataQuery

logger = logging.getLogger(__name__)

_COLLECTION = "Chunk"


@dataclass
class SearchHit:
    weaviate_id: str
    content: str
    document_id: str
    filename: str
    heading: str
    page: int
    chunk_index: int
    score: float


class WeaviateStore:
    """Manages the Weaviate connection and Chunk collection."""

    def __init__(self, url: str = "http://localhost:8080", vector_dim: int = 1024) -> None:
        self._url = url
        self._vector_dim = vector_dim
        self._client: weaviate.WeaviateClient | None = None

        # FIX-A: proper URL parsing instead of string splitting
        parsed = urlparse(url)
        self._host = parsed.hostname or "localhost"
        self._port = parsed.port or 8080

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open connection and ensure schema exists."""
        # FIX-B: blocking I/O moved off the event loop
        def _connect_sync() -> weaviate.WeaviateClient:
            return weaviate.connect_to_local(
                host=self._host,
                port=self._port,
            )

        self._client = await asyncio.to_thread(_connect_sync)
        await self._ensure_collection()
        logger.info("Weaviate connected at %s", self._url)

    async def close(self) -> None:
        if self._client:
            await asyncio.to_thread(self._client.close)
            self._client = None

    @property
    def _col(self):
        assert self._client, "WeaviateStore.connect() not called"
        return self._client.collections.get(_COLLECTION)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    async def _ensure_collection(self) -> None:
        # FIX-B: blocking schema I/O moved off the event loop
        def _sync() -> None:
            assert self._client is not None
            if self._client.collections.exists(_COLLECTION):
                logger.debug("Weaviate collection '%s' already exists.", _COLLECTION)
                return

            logger.info("Creating Weaviate collection '%s' …", _COLLECTION)
            self._client.collections.create(
                name=_COLLECTION,
                description="Text chunks from ingested documents",
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="document_id",  data_type=DataType.TEXT),
                    Property(name="filename",      data_type=DataType.TEXT),
                    Property(name="heading",       data_type=DataType.TEXT),
                    Property(name="page",          data_type=DataType.INT),
                    Property(name="chunk_index",   data_type=DataType.INT),
                    Property(name="content",       data_type=DataType.TEXT),
                ],
            )
            logger.info("Collection '%s' created.", _COLLECTION)

        await asyncio.to_thread(_sync)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: list[dict],
        vectors: list[list[float]],
    ) -> list[str]:
        assert len(chunks) == len(vectors), "chunks and vectors must be same length"
        col = self._col
        ids: list[str] = []

        with col.batch.dynamic() as batch:
            for props, vec in zip(chunks, vectors):
                wid = str(_uuid.uuid4())
                batch.add_object(properties=props, vector=vec, uuid=wid)
                ids.append(wid)

        return ids

    def delete_document_chunks(self, document_id: str) -> int:
        col = self._col
        result = col.data.delete_many(
            where=wvc.query.Filter.by_property("document_id").equal(document_id)
        )
        deleted = result.successful if result else 0
        logger.info("Deleted %d chunks for document %s", deleted, document_id)
        return deleted

    # ------------------------------------------------------------------
    # Read — hybrid search
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 20,
        alpha: float = 0.75,
    ) -> list[SearchHit]:
        col = self._col
        response = col.query.hybrid(
            query=query_text,
            vector=query_vector,
            alpha=alpha,
            limit=top_k,
            fusion_type=HybridFusion.RELATIVE_SCORE,
            return_metadata=MetadataQuery(score=True),
        )

        hits: list[SearchHit] = []
        for obj in response.objects:
            p = obj.properties
            hits.append(
                SearchHit(
                    weaviate_id=str(obj.uuid),
                    content=p.get("content", ""),
                    document_id=p.get("document_id", ""),
                    filename=p.get("filename", ""),
                    heading=p.get("heading", ""),
                    page=int(p.get("page", 0)),
                    chunk_index=int(p.get("chunk_index", 0)),
                    score=obj.metadata.score if obj.metadata else 0.0,
                )
            )
        return hits
