"""
backend/api/routes/export.py
=============================
Conversation export endpoints.

GET /api/conversations/{id}/export?format=json     — full JSON archive
GET /api/conversations/{id}/export?format=markdown — readable Markdown
GET /api/export/all?format=json                    — all conversations in one archive
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

from backend.memory.repository import ConversationRepository

logger = logging.getLogger(__name__)
router = APIRouter(tags=["export"])


@router.get("/conversations/{conversation_id}/export")
async def export_conversation(
    conversation_id: str,
    request: Request,
    format: str = Query("json", description="Export format: json | markdown"),
) -> Response:
    """
    Export a single conversation.

    JSON export includes full message history, metadata, and any
    available memory summaries.

    Markdown export produces a human-readable chat log suitable for
    sharing or archiving outside of PAEKA.
    """
    repo = ConversationRepository(request.app.state.db)
    conv = await repo.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = await repo.get_messages(conversation_id)

    # Fetch session memory summary if available
    session_summary: str | None = None
    memory_svc = request.app.state.memory
    if memory_svc is not None:
        session_summary = await memory_svc._mem_repo.get_session_summary(conversation_id)

    if format.lower() == "markdown":
        content = _to_markdown(conv, messages, session_summary)
        filename = f"paeka_{_safe_name(conv.title)}_{conversation_id[:8]}.md"
        return Response(
            content=content,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    if format.lower() == "json":
        data = _to_json(conv, messages, session_summary)
        filename = f"paeka_{_safe_name(conv.title)}_{conversation_id[:8]}.json"
        return Response(
            content=json.dumps(data, indent=2, ensure_ascii=False),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    raise HTTPException(
        status_code=400,
        detail=f"Unknown format '{format}'. Use 'json' or 'markdown'.",
    )


@router.get("/export/all")
async def export_all_conversations(
    request: Request,
    format: str = Query("json", description="Export format: json | markdown"),
) -> Response:
    """
    Export all conversations in a single file.

    JSON: array of conversation objects.
    Markdown: all conversations concatenated with separators.
    """
    repo = ConversationRepository(request.app.state.db)
    conversations = await repo.list_conversations()
    memory_svc = request.app.state.memory

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    if format.lower() == "json":
        all_data = []
        for conv in conversations:
            messages = await repo.get_messages(conv.id)
            summary = None
            if memory_svc:
                summary = await memory_svc._mem_repo.get_session_summary(conv.id)
            all_data.append(_to_json(conv, messages, summary))

        filename = f"paeka_export_{timestamp}.json"
        return Response(
            content=json.dumps({"exported_at": timestamp, "conversations": all_data},
                               indent=2, ensure_ascii=False),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    if format.lower() == "markdown":
        parts = []
        for conv in conversations:
            messages = await repo.get_messages(conv.id)
            summary = None
            if memory_svc:
                summary = await memory_svc._mem_repo.get_session_summary(conv.id)
            parts.append(_to_markdown(conv, messages, summary))

        filename = f"paeka_export_{timestamp}.md"
        separator = "\n\n---\n\n"
        return Response(
            content=f"# PAEKA Export — {timestamp}\n\n" + separator.join(parts),
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    raise HTTPException(status_code=400, detail=f"Unknown format '{format}'.")


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _to_json(conv, messages, session_summary: str | None) -> dict:
    return {
        "id":             conv.id,
        "title":          conv.title,
        "created_at":     conv.created_at,
        "updated_at":     conv.updated_at,
        "message_count":  len(messages),
        "session_summary": session_summary,
        "messages": [
            {
                "id":         m.id,
                "role":       m.role,
                "content":    m.content,
                "created_at": m.created_at,
            }
            for m in messages
        ],
    }


def _to_markdown(conv, messages, session_summary: str | None) -> str:
    lines = [
        f"# {conv.title}",
        f"",
        f"**ID:** `{conv.id}`",
        f"**Created:** {conv.created_at}",
        f"**Updated:** {conv.updated_at}",
        f"**Messages:** {len(messages)}",
    ]

    if session_summary:
        lines += ["", "## Session Summary", "", session_summary]

    lines += ["", "## Messages", ""]

    for msg in messages:
        role_label = "**User**" if msg.role == "user" else "**PAEKA**"
        timestamp  = msg.created_at.split("T")[1][:8] if "T" in msg.created_at else ""
        lines.append(f"### {role_label} _{timestamp}_")
        lines.append("")
        lines.append(msg.content)
        lines.append("")

    return "\n".join(lines)


def _safe_name(title: str) -> str:
    """Convert a conversation title to a safe filename segment."""
    import re
    safe = re.sub(r"[^\w\s-]", "", title)
    safe = re.sub(r"\s+", "_", safe.strip())
    return safe[:40] or "conversation"
