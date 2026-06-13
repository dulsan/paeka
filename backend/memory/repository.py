"""
backend/memory/repository.py
=============================
Data-access layer for conversations and messages.
All I/O is async via the shared Database wrapper.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from backend.shared.database import Database


# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------


@dataclass
class Conversation:
    id: str
    title: str
    created_at: str
    updated_at: str


@dataclass
class ChatMessage:
    id: str
    conversation_id: str
    role: str
    content: str
    created_at: str


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class ConversationRepository:
    """CRUD operations for conversations and their messages."""

    def __init__(self, db: Database) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------

    async def create_conversation(self, title: str = "New Conversation") -> Conversation:
        cid = str(uuid.uuid4())
        now = _now()
        await self._db.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?,?,?,?)",
            (cid, title, now, now),
        )
        return Conversation(id=cid, title=title, created_at=now, updated_at=now)

    async def list_conversations(self) -> list[Conversation]:
        rows = await self._db.fetchall(
            "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
        )
        return [Conversation(**dict(r)) for r in rows]

    async def get_conversation(self, cid: str) -> Conversation | None:
        row = await self._db.fetchone(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id=?", (cid,)
        )
        return Conversation(**dict(row)) if row else None

    async def update_conversation_title(self, cid: str, title: str) -> None:
        await self._db.execute(
            "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
            (title, _now(), cid),
        )

    async def delete_conversation(self, cid: str) -> None:
        await self._db.execute("DELETE FROM conversations WHERE id=?", (cid,))

    async def touch_conversation(self, cid: str) -> None:
        await self._db.execute(
            "UPDATE conversations SET updated_at=? WHERE id=?", (_now(), cid)
        )

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    async def add_message(self, conversation_id: str, role: str, content: str) -> ChatMessage:
        mid = str(uuid.uuid4())
        now = _now()
        await self._db.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?,?,?,?,?)",
            (mid, conversation_id, role, content, now),
        )
        await self.touch_conversation(conversation_id)
        return ChatMessage(
            id=mid, conversation_id=conversation_id, role=role, content=content, created_at=now
        )

    async def get_messages(
        self, conversation_id: str, limit: int | None = None
    ) -> list[ChatMessage]:
        sql = "SELECT id, conversation_id, role, content, created_at FROM messages WHERE conversation_id=? ORDER BY created_at ASC"
        params: tuple = (conversation_id,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (conversation_id, limit)
        rows = await self._db.fetchall(sql, params)
        return [ChatMessage(**dict(r)) for r in rows]

    async def delete_messages(self, conversation_id: str) -> None:
        await self._db.execute(
            "DELETE FROM messages WHERE conversation_id=?", (conversation_id,)
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
