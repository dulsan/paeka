"""
backend/api/routes/conversations.py
=====================================
CRUD endpoints for conversations.

GET    /api/conversations
POST   /api/conversations
GET    /api/conversations/{id}
PATCH  /api/conversations/{id}
DELETE /api/conversations/{id}
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.memory.repository import ConversationRepository

logger = logging.getLogger(__name__)
router = APIRouter(tags=["conversations"])


class ConversationOut(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    created_at: str


class ConversationDetailOut(ConversationOut):
    messages: list[MessageOut]


class CreateConversationRequest(BaseModel):
    title: str = "New Conversation"


class RenameConversationRequest(BaseModel):
    title: str


@router.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(request: Request) -> list[ConversationOut]:
    repo = ConversationRepository(request.app.state.db)
    convs = await repo.list_conversations()
    return [ConversationOut(**c.__dict__) for c in convs]


@router.post("/conversations", response_model=ConversationOut, status_code=201)
async def create_conversation(
    body: CreateConversationRequest, request: Request
) -> ConversationOut:
    repo = ConversationRepository(request.app.state.db)
    conv = await repo.create_conversation(body.title)
    return ConversationOut(**conv.__dict__)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailOut)
async def get_conversation(
    conversation_id: str, request: Request
) -> ConversationDetailOut:
    repo = ConversationRepository(request.app.state.db)
    conv = await repo.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = await repo.get_messages(conversation_id)
    return ConversationDetailOut(
        **conv.__dict__,
        messages=[MessageOut(**m.__dict__) for m in messages],
    )


@router.patch("/conversations/{conversation_id}", response_model=ConversationOut)
async def rename_conversation(
    conversation_id: str,
    body: RenameConversationRequest,
    request: Request,
) -> ConversationOut:
    repo = ConversationRepository(request.app.state.db)
    conv = await repo.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await repo.update_conversation_title(conversation_id, body.title)
    conv.title = body.title
    return ConversationOut(**conv.__dict__)


@router.delete("/conversations/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str, request: Request) -> None:
    repo = ConversationRepository(request.app.state.db)
    conv = await repo.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await repo.delete_conversation(conversation_id)
