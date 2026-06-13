"""
backend/ingestion/repository.py
================================
SQLite CRUD for documents and chunks.
Extended in Phase 4 to support versioning fields.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from backend.shared.database import Database


@dataclass
class Document:
    id: str
    filename: str
    filepath: str
    mime_type: str
    status: str
    chunk_count: int
    parser_used: str | None
    version: int
    content_hash: str | None
    created_at: str
    updated_at: str


@dataclass
class Chunk:
    id: str
    document_id: str
    weaviate_id: str | None
    chunk_index: int
    content: str
    heading: str
    page: int
    element_type: str
    created_at: str


class DocumentRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    async def create_document(
        self,
        filename: str,
        filepath: str,
        mime_type: str = "application/octet-stream",
        content_hash: str | None = None,
        parser_used: str | None = None,
        version: int = 1,
    ) -> Document:
        did = str(uuid.uuid4())
        now = _now()
        await self._db.execute(
            """
            INSERT INTO documents
                (id, filename, filepath, mime_type, status,
                 chunk_count, parser_used, version, content_hash,
                 created_at, updated_at)
            VALUES (?,?,?,?,'pending',0,?,?,?,?,?)
            """,
            (did, filename, filepath, mime_type,
             parser_used, version, content_hash, now, now),
        )
        return Document(
            id=did, filename=filename, filepath=filepath,
            mime_type=mime_type, status="pending", chunk_count=0,
            parser_used=parser_used, version=version,
            content_hash=content_hash, created_at=now, updated_at=now,
        )

    async def get_document(self, doc_id: str) -> Document | None:
        row = await self._db.fetchone(
            "SELECT * FROM documents WHERE id=?", (doc_id,)
        )
        return Document(**dict(row)) if row else None

    async def get_document_by_path(self, filepath: str) -> Document | None:
        row = await self._db.fetchone(
            "SELECT * FROM documents WHERE filepath=?", (filepath,)
        )
        return Document(**dict(row)) if row else None

    async def list_documents(self) -> list[Document]:
        rows = await self._db.fetchall(
            "SELECT * FROM documents ORDER BY created_at DESC"
        )
        return [Document(**dict(r)) for r in rows]

    async def set_status(self, doc_id: str, status: str) -> None:
        await self._db.execute(
            "UPDATE documents SET status=?, updated_at=? WHERE id=?",
            (status, _now(), doc_id),
        )

    async def set_chunk_count(self, doc_id: str, count: int) -> None:
        await self._db.execute(
            "UPDATE documents SET chunk_count=?, status='ready', updated_at=? WHERE id=?",
            (count, _now(), doc_id),
        )

    async def delete_document(self, doc_id: str) -> None:
        await self._db.execute("DELETE FROM documents WHERE id=?", (doc_id,))

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------

    async def add_chunk(
        self,
        document_id: str,
        chunk_index: int,
        content: str,
        heading: str = "",
        page: int = 0,
        element_type: str = "text",
        weaviate_id: str | None = None,
    ) -> Chunk:
        cid = str(uuid.uuid4())
        now = _now()
        await self._db.execute(
            """
            INSERT INTO chunks
                (id, document_id, weaviate_id, chunk_index,
                 content, heading, page, element_type, created_at)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (cid, document_id, weaviate_id, chunk_index,
             content, heading, page, element_type, now),
        )
        return Chunk(
            id=cid, document_id=document_id, weaviate_id=weaviate_id,
            chunk_index=chunk_index, content=content,
            heading=heading, page=page, element_type=element_type,
            created_at=now,
        )

    async def delete_chunks(self, document_id: str) -> None:
        await self._db.execute(
            "DELETE FROM chunks WHERE document_id=?", (document_id,)
        )

    async def get_chunks(self, document_id: str) -> list[Chunk]:
        rows = await self._db.fetchall(
            "SELECT * FROM chunks WHERE document_id=? ORDER BY chunk_index",
            (document_id,),
        )
        return [Chunk(**dict(r)) for r in rows]


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
