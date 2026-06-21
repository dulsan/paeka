"""
backend/retrieval/qdrant_store.py
===================================
Qdrant vector store replacing Weaviate.

Why Qdrant instead of Weaviate:
  - Single binary (qdrant.exe) with no RAFT cluster config
  - Native async Python client (AsyncQdrantClient) - no asyncio.to_thread needed
  - No Docker required for local development
  - Zero startup RAFT issues
  - Simple healthcheck: GET /healthz returns 200 immediately when ready

Download qdrant.exe from:
  https://github.com/qdrant/qdrant/releases
  Look for: qdrant-x86_64-pc-windows-msvc.zip

Place at: bin/qdrant.exe
Config at: config/qdrant.yaml (sets data dir and ports)

API reference: https://qdrant.tech/documentation/
Python client: https://github.com/qdrant/qdrant-client

SearchHit maintains the same field names as the old WeaviateStore to avoid
cascading changes in RetrievalEngine, IngestionPipeline, and agent nodes.
The field weaviate_id holds the Qdrant point UUID string.
"""

from __future__ import annotations

import logging
import uuid as _uuid
from dataclasses import dataclass, field

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

logger = logging.getLogger(__name__)

_COLLECTION = "chunks"


@dataclass
class SearchHit:
    """Search result from Qdrant. Field names kept compatible with old WeaviateStore."""
    weaviate_id: str      # holds Qdrant point UUID string
    content: str
    document_id: str
    filename: str
    heading: str
    page: int
    chunk_index: int
    score: float
    metadata: dict = field(default_factory=dict)


class QdrantStore:
    """
    Async Qdrant vector store.

    All operations use the native AsyncQdrantClient - no blocking calls
    and no asyncio.to_thread() wrappers needed.
    """

    def __init__(self, url: str = "http://localhost:6333", vector_dim: int = 1024) -> None:
        self._url        = url
        self._vector_dim = vector_dim
        self._client: AsyncQdrantClient | None = None

    async def connect(self) -> None:
        """
        Open connection and create the chunks collection if it does not exist.

        [FIX] check_compatibility=False added below. qdrant-client's own
        AsyncQdrantClient tries to fetch the server version on first use to
        verify client/server compatibility; when that check itself fails
        (independent of whether the main connection succeeds), it raises a
        UserWarning -- "Failed to obtain server version... Set
        check_compatibility=False to skip version check." That's a
        recommendation straight from the SDK's own warning message for a
        check we don't need: this is a single local binary we control and
        know the version of, not a fleet of servers at unknown versions.

        If you see "Retrieval init failed: All connection attempts failed"
        in the startup log, that is a separate, more fundamental issue --
        it means Qdrant isn't reachable at self._url at all, almost always
        because bin\\qdrant.exe isn't running yet. Confirm with:
            curl http://localhost:6333/healthz
        in a separate terminal before starting PAEKA. check_compatibility
        has no effect on that failure mode; it only silences the separate
        version-check warning that can fire on its own schedule.
        """
        # [FIX] Use a local variable for the calls below, not self._client
        # directly. pyright doesn't carry narrowing forward across
        # statements for instance attributes the way it does for local
        # variables (a deliberate, standard type-checker conservatism --
        # nothing here is actually unsafe at runtime since this is
        # sequential code, but pyright can't prove that statically for
        # `self._client`). This is the same reason _require_client()
        # exists and is used at every other call site in this file below;
        # connect() just hadn't been brought in line with that pattern.
        client = AsyncQdrantClient(url=self._url, check_compatibility=False)
        self._client = client

        exists = await client.collection_exists(_COLLECTION)
        if not exists:
            logger.info("Creating Qdrant collection '%s' (dim=%d)", _COLLECTION, self._vector_dim)
            await client.create_collection(
                collection_name=_COLLECTION,
                vectors_config=VectorParams(
                    size=self._vector_dim,
                    distance=Distance.COSINE,
                ),
            )
            # Index key payload fields for fast filter-based deletes
            await client.create_payload_index(
                collection_name=_COLLECTION,
                field_name="document_id",
                field_schema="keyword",
            )
        else:
            logger.info("Qdrant collection '%s' exists.", _COLLECTION)

        logger.info("Qdrant connected at %s", self._url)

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    def _require_client(self) -> AsyncQdrantClient:
        if self._client is None:
            raise RuntimeError(
                "QdrantStore.connect() was not called. "
                "Await store.connect() during the FastAPI lifespan."
            )
        return self._client

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def upsert_chunks(
        self,
        chunks: list[dict],
        vectors: list[list[float]],
    ) -> list[str]:
        """
        Batch-upsert document chunks with their pre-computed embedding vectors.

        Parameters
        ----------
        chunks:
            List of property dicts. Required keys: content, document_id,
            filename, heading, page, chunk_index.
        vectors:
            Corresponding embedding vectors (must match len(chunks)).

        Returns
        -------
        list[str]
            UUID strings of the upserted points.
        """
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")

        client = self._require_client()
        point_ids: list[str] = []
        points: list[PointStruct] = []

        for props, vec in zip(chunks, vectors, strict=True):
            pid = str(_uuid.uuid4())
            points.append(PointStruct(id=pid, vector=vec, payload=props))
            point_ids.append(pid)

        await client.upsert(collection_name=_COLLECTION, points=points)
        logger.debug("Upserted %d chunks into Qdrant", len(points))
        return point_ids

    async def insert(
        self,
        content: str,
        vector: list[float],
        metadata: dict | None = None,
        collection_name: str = _COLLECTION,
    ) -> str:
        """
        Insert a single object into any named collection.

        Used by the MCP weaviate_ingest tool and ConversationMemory archival.
        Creates the collection with the configured vector dim if it does not exist.
        """
        client = self._require_client()

        if collection_name != _COLLECTION:
            exists = await client.collection_exists(collection_name)
            if not exists:
                await client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=len(vector), distance=Distance.COSINE
                    ),
                )

        pid    = str(_uuid.uuid4())
        payload = {"content": content, **(metadata or {})}
        await client.upsert(
            collection_name=collection_name,
            points=[PointStruct(id=pid, vector=vector, payload=payload)],
        )
        return pid

    async def delete_document_chunks(self, document_id: str) -> int:
        """Delete all chunks belonging to a document, matched by document_id payload."""
        client = self._require_client()
        result = await client.delete(
            collection_name=_COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
        )
        # Qdrant returns an UpdateResult; deleted count isn't directly available
        # but the operation is idempotent. Log for visibility.
        logger.info("Deleted chunks for document_id=%s (status=%s)", document_id, result.status)
        return 0   # count not returned by Qdrant delete API

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def search(
        self,
        vector: list[float],
        limit: int = 5,
        collection_name: str = _COLLECTION,
        where_filter: dict | None = None,
    ) -> list[SearchHit]:
        """
        Vector similarity search over any named collection.

        Used by MCP weaviate_search tool and ConversationMemory.search().

        Parameters
        ----------
        vector:
            Query embedding vector.
        limit:
            Maximum results.
        collection_name:
            Target collection.
        where_filter:
            Optional filter dict with keys: path (list[str]), operator, valueText.
            Example: {"path": ["session_id"], "operator": "Equal", "valueText": "abc"}
        """
        client = self._require_client()

        qdrant_filter = None
        if where_filter:
            prop = (where_filter.get("path") or [""])[0]
            val  = where_filter.get("valueText") or where_filter.get("valueString", "")
            if prop and val:
                qdrant_filter = Filter(
                    must=[FieldCondition(key=prop, match=MatchValue(value=val))]
                )

        results = await client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=limit,
            with_payload=True,
            query_filter=qdrant_filter,
        )

        hits: list[SearchHit] = []
        for point in results:
            p = point.payload or {}
            hits.append(SearchHit(
                weaviate_id=str(point.id),
                content=p.get("content", ""),
                document_id=p.get("document_id", ""),
                filename=p.get("filename", ""),
                heading=p.get("heading", ""),
                page=int(p.get("page", 0)),
                chunk_index=int(p.get("chunk_index", 0)),
                score=float(point.score),
                metadata={k: v for k, v in p.items() if k != "content"},
            ))
        return hits

    async def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 20,
        alpha: float = 0.75,
    ) -> list[SearchHit]:
        """
        Hybrid search. Currently uses dense vector search only.

        Qdrant supports sparse vectors for BM25-style hybrid search via
        fastembed integration. Dense-only is used here because:
          - bge-m3 is already an excellent semantic retrieval model
          - Avoids the fastembed dependency for the initial daily-driver setup
          - alpha parameter reserved for when sparse vectors are added

        To enable true hybrid search later:
          1. Add sparse vector config to the collection
          2. Encode query_text with a sparse encoder (e.g. BM25 via fastembed)
          3. Use client.search() with named_vectors for both dense and sparse
        """
        return await self.search(vector=query_vector, limit=top_k)

    async def ensure_collection(
        self,
        name: str,
        description: str = "",
        properties: list[dict] | None = None,
    ) -> None:
        """
        Idempotent collection creation. Used by ConversationMemory bootstrap.

        Qdrant does not use typed property schemas like Weaviate.
        Payload fields are schemaless - any key/value pairs can be stored.
        The description parameter is accepted but not stored (Qdrant has no
        built-in collection descriptions; add to your own metadata if needed).
        """
        client = self._require_client()
        if await client.collection_exists(name):
            logger.debug("Qdrant collection '%s' already exists.", name)
            return

        # Determine vector size from the default collection's config
        # or fall back to the configured dim
        vec_size = self._vector_dim
        await client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vec_size, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s'.", name)
