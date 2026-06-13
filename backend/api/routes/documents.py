"""
backend/api/routes/documents.py
================================
Document management endpoints.

Changes from v0.11.3:
  [FIX-A] MIME type detection now falls back to filename extension.
          curl and .NET HttpClient send Content-Type: application/octet-stream
          for the file part of a multipart upload when no explicit MIME type
          is set. The original code rejected these with HTTP 415 before
          reading the request body, causing the TCP connection to stall
          (client waiting to send body, server waiting for client to close).

  [FIX-B] file.read() replaced with chunked async streaming write.
          await file.read() loads the entire file into memory before
          doing anything — and it blocks the event loop while doing it.
          Now streams the file to disk in 1 MB chunks, then ingests from disk.
          This keeps memory usage flat regardless of file size.

  [FIX-C] Upload directory created at startup not per-request.

Routes
------
POST   /api/documents/upload           — upload + ingest a file
POST   /api/documents/ingest-text      — ingest raw text directly
GET    /api/documents                  — list all documents
GET    /api/documents/{id}             — get document + chunk metadata
DELETE /api/documents/{id}             — delete document (Weaviate + SQLite)
POST   /api/documents/{id}/refresh     — force re-ingest from disk
GET    /api/documents/{id}/chunks      — list chunks for a document
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from backend.ingestion.repository import DocumentRepository
from backend.shared.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["documents"])

_UPLOAD_DIR = Path("data/uploads")
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Mapping of file extensions to MIME types for robust fallback detection
_EXT_MIME: dict[str, str] = {
    ".pdf":  "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".txt":  "text/plain",
    ".md":   "text/markdown",
    ".html": "text/html",
    ".htm":  "text/html",
    ".tex":  "text/x-latex",
}


def _resolve_mime(file: UploadFile) -> str:
    """
    FIX-A: Resolve MIME type from content_type header first, then filename
    extension. curl and .NET HttpClient often send application/octet-stream
    for all file parts — fall back to extension-based detection so valid
    files aren't rejected with 415.
    """
    ct = (file.content_type or "").lower().split(";")[0].strip()

    # If the client sent a real MIME type, use it
    if ct and ct != "application/octet-stream" and "/" in ct:
        return ct

    # Fall back to extension
    fname = file.filename or ""
    ext = Path(fname).suffix.lower()
    if ext in _EXT_MIME:
        return _EXT_MIME[ext]

    # Try stdlib mimetypes as last resort
    guessed, _ = mimetypes.guess_type(fname)
    return guessed or "application/octet-stream"


class DocumentOut(BaseModel):
    id: str
    filename: str
    mime_type: str
    status: str
    chunk_count: int
    parser_used: str | None
    version: int
    created_at: str
    updated_at: str


class ChunkOut(BaseModel):
    id: str
    chunk_index: int
    content: str
    heading: str
    page: int
    element_type: str


class DocumentDetailOut(DocumentOut):
    chunks: list[ChunkOut]


class IngestTextRequest(BaseModel):
    text: str
    filename: str


class IngestResponse(BaseModel):
    document_id: str
    status: str


def _require_ingestion(request: Request):
    pipeline = getattr(request.app.state, "ingestion", None)
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Ingestion not enabled. Set [retrieval] enabled = true in settings.toml.",
        )
    return pipeline


@router.post("/documents/upload", response_model=IngestResponse, status_code=202)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
) -> IngestResponse:
    """Upload and ingest a file. Idempotent on hash match."""
    settings = get_settings()
    pipeline = _require_ingestion(request)

    # FIX-A: resolve MIME type with extension fallback
    mime_type = _resolve_mime(file)

    if mime_type not in settings.ingestion.supported_types:
        # Drain the body before raising so the TCP connection closes cleanly
        await file.read()
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported media type: {mime_type} "
                f"(file: {file.filename}). "
                f"Supported: {', '.join(settings.ingestion.supported_types)}"
            ),
        )

    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = _UPLOAD_DIR / (file.filename or "upload")

    # FIX-B: stream file to disk in chunks — avoids loading entire file into
    # memory and keeps the event loop responsive during the write
    total_bytes = 0
    chunk_size  = 1024 * 1024  # 1 MB chunks

    def _open_dest():
        return open(dest, "wb")

    loop = asyncio.get_event_loop()
    fh   = await loop.run_in_executor(None, _open_dest)
    try:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > settings.ingestion.max_file_bytes:
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File too large")
            await loop.run_in_executor(None, fh.write, chunk)
    finally:
        await loop.run_in_executor(None, fh.close)

    logger.info("Uploaded %s (%d bytes) → %s", file.filename, total_bytes, dest)

    try:
        doc_id = await pipeline.ingest_file(dest)
    except Exception as exc:
        logger.error("Ingestion error for %s: %s", file.filename, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    repo = DocumentRepository(request.app.state.db)
    doc  = await repo.get_document(doc_id)
    return IngestResponse(document_id=doc_id, status=doc.status if doc else "unknown")


@router.post("/documents/ingest-text", response_model=IngestResponse, status_code=202)
async def ingest_text(body: IngestTextRequest, request: Request) -> IngestResponse:
    """Ingest pre-extracted text directly."""
    pipeline = _require_ingestion(request)
    try:
        doc_id = await pipeline.ingest_text(body.text, body.filename)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    repo = DocumentRepository(request.app.state.db)
    doc  = await repo.get_document(doc_id)
    return IngestResponse(document_id=doc_id, status=doc.status if doc else "unknown")


@router.get("/documents", response_model=list[DocumentOut])
async def list_documents(request: Request) -> list[DocumentOut]:
    repo = DocumentRepository(request.app.state.db)
    docs = await repo.list_documents()
    return [
        DocumentOut(
            id=d.id, filename=d.filename, mime_type=d.mime_type,
            status=d.status, chunk_count=d.chunk_count,
            parser_used=d.parser_used, version=d.version,
            created_at=d.created_at, updated_at=d.updated_at,
        )
        for d in docs
    ]


@router.get("/documents/{document_id}", response_model=DocumentDetailOut)
async def get_document(document_id: str, request: Request) -> DocumentDetailOut:
    repo = DocumentRepository(request.app.state.db)
    doc  = await repo.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    chunks = await repo.get_chunks(document_id)
    return DocumentDetailOut(
        id=doc.id, filename=doc.filename, mime_type=doc.mime_type,
        status=doc.status, chunk_count=doc.chunk_count,
        parser_used=doc.parser_used, version=doc.version,
        created_at=doc.created_at, updated_at=doc.updated_at,
        chunks=[
            ChunkOut(
                id=c.id, chunk_index=c.chunk_index, content=c.content,
                heading=c.heading or "", page=c.page or 0,
                element_type=c.element_type,
            )
            for c in chunks
        ],
    )


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(document_id: str, request: Request) -> None:
    pipeline = _require_ingestion(request)
    deleted  = await pipeline.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")


@router.post("/documents/{document_id}/refresh", response_model=IngestResponse)
async def refresh_document(document_id: str, request: Request) -> IngestResponse:
    """Force re-ingest from original file path, retiring the current version."""
    pipeline = _require_ingestion(request)
    try:
        new_id = await pipeline.refresh_document(document_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    repo = DocumentRepository(request.app.state.db)
    doc  = await repo.get_document(new_id)
    return IngestResponse(document_id=new_id, status=doc.status if doc else "unknown")


@router.get("/documents/{document_id}/chunks", response_model=list[ChunkOut])
async def get_chunks(document_id: str, request: Request) -> list[ChunkOut]:
    repo   = DocumentRepository(request.app.state.db)
    doc    = await repo.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    chunks = await repo.get_chunks(document_id)
    return [
        ChunkOut(
            id=c.id, chunk_index=c.chunk_index, content=c.content,
            heading=c.heading or "", page=c.page or 0,
            element_type=c.element_type,
        )
        for c in chunks
    ]
