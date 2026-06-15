"""
backend/retrieval/weaviate_store.py
====================================
Weaviate v4 client wrapper.

Changes applied in this review pass:
  [FIX-ASSERT]  _col property used assert — removed in -O mode. Now raises RuntimeError.
  [FIX-ASYNC]   hybrid_search(), upsert_chunks(), delete_document_chunks() were
                synchronous. All Weaviate SDK calls moved to asyncio.to_thread() so
                they never block the event loop.
  [ADD-SEARCH]  search(vector, limit, collection_name) — generic vector search used
                by MCP server and ConversationMemory.
  [ADD-INSERT]  insert(content, vector, metadata, collection_name) — single-object
                write used by MCP weaviate_ingest tool and ConversationMemory.
  [ADD-ENSURE]  ensure_collection(name, description, properties) — idempotent
                collection creation used by ConversationMemory bootstrap.
"""

from __future__ import annotations

import asyncio
import logging
import uuid as _uuid
from dataclasses import dataclass, field
from urllib.parse import urlparse

import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import HybridFusion, MetadataQuery

logger = logging.getLogger(__name__)

_COLLECTION = "Chunk"

# Map generic string type names → Weaviate DataType for ensure_collection()
_DTYPE_MAP: dict[str, DataType] = {
    "text":    DataType.TEXT,
    "int":     DataType.INT,
    "boolean": DataType.BOOL,
    "number":  DataType.NUMBER,
    "date":    DataType.DATE,
}


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
    metadata: dict = field(default_factory=dict)


class WeaviateStore:
    """Manages the Weaviate connection and document chunk collection."""

    def __init__(self, url: str = "http://localhost:8090", vector_dim: int = 1024) -> None:
        self._url        = url
        self._vector_dim = vector_dim
        self._client: weaviate.WeaviateClient | None = None

        parsed      = urlparse(url)
        self._host  = parsed.hostname or "localhost"
        self._port  = parsed.port or 8090

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open connection and ensure schema exists."""
        def _connect_sync() -> weaviate.WeaviateClient:
            return weaviate.connect_to_local(host=self._host, port=self._port)

        self._client = await asyncio.to_thread(_connect_sync)
        await self._ensure_collection_internal()
        logger.info("Weaviate connected at %s", self._url)

    async def close(self) -> None:
        if self._client:
            await asyncio.to_thread(self._client.close)
            self._client = None

    @property
    def _col(self):
        # [FIX-ASSERT] assert is stripped by python -O; raise explicitly instead.
        if self._client is None:
            raise RuntimeError(
                "WeaviateStore.connect() was not called before use. "
                "Call await store.connect() in the FastAPI lifespan."
            )
        return self._client.collections.get(_COLLECTION)

    # ------------------------------------------------------------------
    # Schema — internal default collection
    # ------------------------------------------------------------------

    async def _ensure_collection_internal(self) -> None:
        """Create the default Chunk collection if it doesn't exist."""
        def _sync() -> None:
            assert self._client is not None
            if self._client.collections.exists(_COLLECTION):
                return
            logger.info("Creating Weaviate collection '%s'...", _COLLECTION)
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

    # [ADD-ENSURE]
    async def ensure_collection(
        self,
        name: str,
        description: str = "",
        properties: list[dict] | None = None,
    ) -> None:
        """
        Idempotent: create a named collection if it does not already exist.

        Parameters
        ----------
        name:
            Collection name (e.g. "ConversationMemory").
        description:
            Human-readable description stored in the schema.
        properties:
            List of property dicts: {"name": str, "dataType": list[str]}.
            dataType values: "text", "int", "boolean", "number", "date".
        """
        def _sync() -> None:
            assert self._client is not None
            if self._client.collections.exists(name):
                logger.debug("Weaviate collection '%s' already exists.", name)
                return
            wv_props = []
            for p in (properties or []):
                raw_type = p.get("dataType", ["text"])
                type_str = raw_type[0] if isinstance(raw_type, list) else raw_type
                dtype    = _DTYPE_MAP.get(str(type_str).lower(), DataType.TEXT)
                wv_props.append(Property(name=p["name"], data_type=dtype))
            self._client.collections.create(
                name=name,
                description=description,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=wv_props,
            )
            logger.info("Created Weaviate collection '%s'.", name)

        await asyncio.to_thread(_sync)

    # ------------------------------------------------------------------
    # Write — document chunks
    # ------------------------------------------------------------------

    async def upsert_chunks(
        self,
        chunks: list[dict],
        vectors: list[list[float]],
    ) -> list[str]:
        """
        Batch-upsert chunks with their pre-computed vectors.
        [FIX-ASYNC] Moved off event loop via asyncio.to_thread().
        """
        assert len(chunks) == len(vectors), "chunks and vectors must be same length"

        def _sync() -> list[str]:
            col = self._col
            ids: list[str] = []
            with col.batch.dynamic() as batch:
                for props, vec in zip(chunks, vectors):
                    wid = str(_uuid.uuid4())
                    batch.add_object(properties=props, vector=vec, uuid=wid)
                    ids.append(wid)
            return ids

        return await asyncio.to_thread(_sync)

    async def delete_document_chunks(self, document_id: str) -> int:
        """
        Delete all chunks belonging to a document.
        [FIX-ASYNC] Moved off event loop.
        """
        def _sync() -> int:
            col = self._col
            result = col.data.delete_many(
                where=wvc.query.Filter.by_property("document_id").equal(document_id)
            )
            deleted = result.successful if result else 0
            logger.info("Deleted %d chunks for document %s", deleted, document_id)
            return deleted

        return await asyncio.to_thread(_sync)

    # [ADD-INSERT]
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

        Parameters
        ----------
        content:
            The primary text payload stored in the "content" property.
        vector:
            Pre-computed embedding vector.
        metadata:
            Additional properties merged alongside "content".
        collection_name:
            Target collection (default: "Chunk").

        Returns
        -------
        str
            UUID of the created object.
        """
        def _sync() -> str:
            assert self._client is not None
            col  = self._client.collections.get(collection_name)
            wid  = str(_uuid.uuid4())
            props = {"content": content, **(metadata or {})}
            col.data.insert(properties=props, vector=vector, uuid=wid)
            return wid

        return await asyncio.to_thread(_sync)

    # ------------------------------------------------------------------
    # Read — hybrid search (default Chunk collection)
    # ------------------------------------------------------------------

    async def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 20,
        alpha: float = 0.75,
    ) -> list[SearchHit]:
        """
        Hybrid dense+BM25 search over the Chunk collection.
        [FIX-ASYNC] Moved off event loop.
        """
        def _sync() -> list[SearchHit]:
            col      = self._col
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
                hits.append(SearchHit(
                    weaviate_id=str(obj.uuid),
                    content=p.get("content", ""),
                    document_id=p.get("document_id", ""),
                    filename=p.get("filename", ""),
                    heading=p.get("heading", ""),
                    page=int(p.get("page", 0)),
                    chunk_index=int(p.get("chunk_index", 0)),
                    score=obj.metadata.score if obj.metadata else 0.0,
                ))
            return hits

        return await asyncio.to_thread(_sync)

    # [ADD-SEARCH]
    async def search(
        self,
        vector: list[float],
        limit: int = 5,
        collection_name: str = _COLLECTION,
        where_filter: dict | None = None,
    ) -> list[SearchHit]:
        """
        Generic vector-only (near_vector) search over any named collection.

        Used by MCP weaviate_search tool, ConversationMemory.search(), and any
        code that needs to query a collection other than "Chunk".

        Parameters
        ----------
        vector:
            Query embedding vector.
        limit:
            Maximum results to return.
        collection_name:
            Collection to search (default: "Chunk").
        where_filter:
            Optional Weaviate-format where filter dict (passed as-is to
            Filter.by_property(). For internal use; callers should use the
            structured filter helpers when possible.

        Returns
        -------
        list[SearchHit]
            Results ordered by vector similarity descending. Fields not present
            in the target collection are returned as empty string / 0.
        """
        def _sync() -> list[SearchHit]:
            assert self._client is not None
            col = self._client.collections.get(collection_name)

            kwargs: dict = dict(
                near_vector=vector,
                limit=limit,
                return_metadata=MetadataQuery(distance=True),
            )
            if where_filter:
                prop   = where_filter.get("path", [""])[0] or ""
                op     = where_filter.get("operator", "Equal")
                val    = where_filter.get("valueText") or where_filter.get("valueString", "")
                if prop and val:
                    kwargs["filters"] = wvc.query.Filter.by_property(prop).equal(val)

            response = col.query.near_vector(**kwargs)
            hits: list[SearchHit] = []
            for obj in response.objects:
                p = obj.properties
                dist  = obj.metadata.distance if obj.metadata else 1.0
                score = max(0.0, 1.0 - (dist or 0.0))
                hits.append(SearchHit(
                    weaviate_id=str(obj.uuid),
                    content=p.get("content", ""),
                    document_id=p.get("document_id", ""),
                    filename=p.get("filename", ""),
                    heading=p.get("heading", ""),
                    page=int(p.get("page", 0)),
                    chunk_index=int(p.get("chunk_index", 0)),
                    score=score,
                    metadata={k: v for k, v in p.items() if k != "content"},
                ))
            return hits

        return await asyncio.to_thread(_sync)
