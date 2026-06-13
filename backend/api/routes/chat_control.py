"""
backend/api/routes/chat_control.py
====================================
Chat session management and context-window reset.

Provides endpoints for:
  POST   /api/chat/reset                    – erase llama.cpp KV-cache slot + wipe session history
  POST   /api/chat/sessions                 – create a new session (makes it active)
  GET    /api/chat/sessions                 – list all sessions
  GET    /api/chat/sessions/{session_id}    – get a specific session
  DELETE /api/chat/sessions/{session_id}    – delete a session + erase its slot
  POST   /api/chat/sessions/{session_id}/activate – switch active session

How context reset works
-----------------------
llama.cpp exposes a slot-erase endpoint (requires --slots flag on the server):

    POST http://localhost:8080/slots/{slot_id}
    Content-Type: application/json
    {"action": "erase"}

This clears the KV-cache for that slot without touching the loaded weights on
the GPU — the model stays warm and the next request starts with a blank context.
It is strictly faster than restarting the server.

How to wire this into main.py
------------------------------
Add these two lines somewhere in your lifespan or startup:

    from backend.api.routes.chat_control import router as chat_control_router, configure as configure_chat
    app.include_router(chat_control_router)

And after your LLM config is resolved:

    configure_chat(llama_base_url="http://localhost:8080")

Or rely on the PAEKA_LLM__BASE_URL / PAEKA_LLM__LLAMA_PORT environment
variable — this module reads it automatically if configure() is never called.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat-control"])


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------

@dataclass
class ChatSession:
    """
    Lightweight in-memory conversation session.

    messages contains the raw list of {"role": ..., "content": ...} dicts
    that you pass directly to the llama.cpp /v1/chat/completions endpoint.
    """
    session_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    messages:   list[dict] = field(default_factory=list)
    llama_slot: int = 0   # the llama.cpp slot id this session is pinned to

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            "session_id":    self.session_id,
            "created_at":    self.created_at.isoformat(),
            "updated_at":    self.updated_at.isoformat(),
            "message_count": len(self.messages),
            "llama_slot":    self.llama_slot,
        }


# ---------------------------------------------------------------------------
# Session store
# ---------------------------------------------------------------------------

class SessionStore:
    """
    Thread-safe* in-memory session registry.

    (* FastAPI runs on a single asyncio event loop, so dict operations are
    effectively thread-safe for typical usage. If you add background threads
    wrap mutations in asyncio.Lock.)
    """

    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}
        self._active_id: Optional[str] = None

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(self, slot: int = 0) -> ChatSession:
        session = ChatSession(session_id=str(uuid.uuid4()), llama_slot=slot)
        self._sessions[session.session_id] = session
        self._active_id = session.session_id
        logger.info("Session created: %s  slot=%d", session.session_id, slot)
        return session

    def get(self, session_id: str) -> Optional[ChatSession]:
        return self._sessions.get(session_id)

    def get_active(self) -> Optional[ChatSession]:
        if self._active_id:
            return self._sessions.get(self._active_id)
        return None

    def set_active(self, session_id: str) -> None:
        if session_id not in self._sessions:
            raise KeyError(f"Unknown session: {session_id}")
        self._active_id = session_id

    def delete(self, session_id: str) -> bool:
        if session_id not in self._sessions:
            return False
        del self._sessions[session_id]
        if self._active_id == session_id:
            # Fall back to the most recently created remaining session
            remaining = sorted(
                self._sessions.values(), key=lambda s: s.created_at, reverse=True
            )
            self._active_id = remaining[0].session_id if remaining else None
        logger.info("Session deleted: %s", session_id)
        return True

    def list_all(self) -> list[ChatSession]:
        return sorted(self._sessions.values(), key=lambda s: s.created_at)

    # ------------------------------------------------------------------
    # Message helpers (convenience wrappers used by the chat pipeline)
    # ------------------------------------------------------------------

    def append_message(self, session_id: str, role: str, content: str) -> None:
        s = self._sessions.get(session_id)
        if s:
            s.messages.append({"role": role, "content": content})
            s.touch()

    def get_history(self, session_id: str) -> list[dict]:
        s = self._sessions.get(session_id)
        return list(s.messages) if s else []

    def clear_history(self, session_id: str) -> None:
        s = self._sessions.get(session_id)
        if s:
            s.messages.clear()
            s.touch()
            logger.info("Session history cleared: %s", session_id)


# Module-level singleton — import `session_store` wherever you need history.
session_store = SessionStore()


# ---------------------------------------------------------------------------
# llama.cpp slot erase
# ---------------------------------------------------------------------------

# Resolved at startup via configure() or from env-var fallback.
_LLAMA_BASE_URL: str = ""


def _get_llama_url() -> str:
    if _LLAMA_BASE_URL:
        return _LLAMA_BASE_URL
    # Env-var fallback so this works even if configure() was never called
    port = os.environ.get("PAEKA_LLM__LLAMA_PORT", "8080")
    base = os.environ.get("PAEKA_LLM__BASE_URL", f"http://localhost:{port}")
    return base.rstrip("/v1").rstrip("/")


def configure(llama_base_url: str) -> None:
    """
    Wire in the actual llama.cpp server URL at startup.

    Call from your FastAPI lifespan after LLM config is resolved:

        from backend.api.routes.chat_control import configure as configure_chat
        configure_chat(settings.llm.base_url)   # e.g. "http://localhost:8080"
    """
    global _LLAMA_BASE_URL
    _LLAMA_BASE_URL = llama_base_url.rstrip("/v1").rstrip("/")
    logger.info("chat_control: llama.cpp base URL → %s", _LLAMA_BASE_URL)


async def _erase_llama_slot(slot_id: int) -> tuple[bool, str]:
    """
    POST /slots/{slot_id}  body={"action":"erase"}

    Requires llama-server to be started with the --slots flag.
    Returns (success: bool, detail: str).
    """
    url = f"{_get_llama_url()}/slots/{slot_id}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json={"action": "erase"})
        if resp.status_code in (200, 204):
            logger.info("llama.cpp slot %d erased", slot_id)
            return True, f"slot {slot_id} erased"
        else:
            msg = f"slot {slot_id}: HTTP {resp.status_code} — {resp.text[:200]}"
            logger.warning("Slot erase warning: %s", msg)
            # Non-fatal: still clear in-memory history even if slot erase fails
            return False, msg
    except httpx.RequestError as exc:
        msg = f"Could not reach llama.cpp at {url}: {exc}"
        logger.error(msg)
        return False, msg


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    session_id: Optional[str] = None  # None → use active session
    slot_id:    int = 0


class ResetResponse(BaseModel):
    ok:           bool
    session_id:   str
    slot_erased:  bool
    slot_detail:  str
    message:      str


class SessionCreateRequest(BaseModel):
    slot_id: int = 0


class SessionInfo(BaseModel):
    session_id:    str
    created_at:    str
    updated_at:    str
    message_count: int
    llama_slot:    int
    active:        bool


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _to_info(session: ChatSession, active_id: Optional[str]) -> SessionInfo:
    return SessionInfo(**session.to_dict(), active=(session.session_id == active_id))


def _active_id() -> Optional[str]:
    s = session_store.get_active()
    return s.session_id if s else None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/reset", response_model=ResetResponse)
async def reset_chat(req: ResetRequest = ResetRequest()) -> ResetResponse:
    """
    Clear the active (or specified) session's message history and erase the
    corresponding llama.cpp KV-cache slot.

    The loaded model stays on the GPU — only the context state is wiped.
    This is equivalent to starting a fresh conversation without restarting
    the server.

    Requires llama-server to have been started with the `--slots` flag.
    """
    # Resolve session -------------------------------------------------------
    if req.session_id:
        session = session_store.get(req.session_id)
        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {req.session_id}",
            )
    else:
        session = session_store.get_active()
        if session is None:
            # No sessions yet — auto-create one so callers always get a clean slate
            session = session_store.create(slot=req.slot_id)

    # Erase llama.cpp slot --------------------------------------------------
    slot_erased, slot_detail = await _erase_llama_slot(session.llama_slot)

    # Clear in-memory history -----------------------------------------------
    session_store.clear_history(session.session_id)

    return ResetResponse(
        ok=True,
        session_id=session.session_id,
        slot_erased=slot_erased,
        slot_detail=slot_detail,
        message=(
            f"Context cleared for session {session.session_id}. "
            f"KV-cache slot {session.llama_slot}: erased={slot_erased}."
        ),
    )


@router.post(
    "/sessions",
    response_model=SessionInfo,
    status_code=status.HTTP_201_CREATED,
)
async def create_session(req: SessionCreateRequest = SessionCreateRequest()) -> SessionInfo:
    """
    Create a new chat session.

    The new session immediately becomes the active session. The previous
    session (if any) is preserved in memory — switch back with `/activate`.
    """
    session = session_store.create(slot=req.slot_id)
    return _to_info(session, session.session_id)


@router.get("/sessions", response_model=list[SessionInfo])
async def list_sessions() -> list[SessionInfo]:
    """List all sessions, oldest first. The active session has `active: true`."""
    aid = _active_id()
    return [_to_info(s, aid) for s in session_store.list_all()]


@router.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str) -> SessionInfo:
    """Get metadata for a single session."""
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    return _to_info(session, _active_id())


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str) -> None:
    """
    Delete a session and erase its llama.cpp KV-cache slot.

    If this was the active session, the next most-recently-created session
    becomes active automatically.
    """
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    await _erase_llama_slot(session.llama_slot)
    session_store.delete(session_id)


@router.post("/sessions/{session_id}/activate", response_model=SessionInfo)
async def activate_session(session_id: str) -> SessionInfo:
    """Switch the active session to an existing one (without erasing anything)."""
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    session_store.set_active(session_id)
    return _to_info(session, session_id)
