"""
backend/api/routes/memory.py
=============================
Memory inspection and management endpoints.

GET    /api/memory/global               — list all global memory entries
DELETE /api/memory/global               — clear all global memories
GET    /api/memory/session/{conv_id}    — get session summary for a conversation
DELETE /api/memory/session/{conv_id}    — clear session summary
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["memory"])


class MemoryEntryOut(BaseModel):
    content: str


class ClearResponse(BaseModel):
    cleared: bool


def _require_memory(request: Request):
    svc = request.app.state.memory
    if svc is None:
        raise HTTPException(
            status_code=503,
            detail="Memory not enabled. Set [memory] enabled = true in settings.toml.",
        )
    return svc


@router.get("/memory/global", response_model=list[MemoryEntryOut])
async def list_global_memory(request: Request) -> list[MemoryEntryOut]:
    svc = _require_memory(request)
    entries = await svc._mem_repo.list_global(limit=100)
    return [MemoryEntryOut(content=e) for e in entries]


@router.delete("/memory/global", response_model=ClearResponse)
async def clear_global_memory(request: Request) -> ClearResponse:
    svc = _require_memory(request)
    await svc._mem_repo.clear_global()
    return ClearResponse(cleared=True)


@router.get("/memory/session/{conversation_id}", response_model=MemoryEntryOut | None)
async def get_session_summary(
    conversation_id: str, request: Request
) -> MemoryEntryOut | None:
    svc = _require_memory(request)
    summary = await svc._mem_repo.get_session_summary(conversation_id)
    return MemoryEntryOut(content=summary) if summary else None


@router.delete("/memory/session/{conversation_id}", response_model=ClearResponse)
async def clear_session_memory(
    conversation_id: str, request: Request
) -> ClearResponse:
    svc = _require_memory(request)
    await svc._mem_repo.delete_session(conversation_id)
    return ClearResponse(cleared=True)
