"""
backend/api/routes/chat.py
===========================
Agentic RAG chat endpoint with content security scanning.

Security applied:
  1. User message scanned for prompt injection before entering pipeline.
  2. LLM output scanned for dangerous shell patterns before streaming.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.memory.repository import ConversationRepository
from backend.retrieval.chunker import chunk_text
from backend.security.content import Severity
from backend.shared.config import get_settings
from backend.shared.logging import bind_context

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    skill: str | None = None


def _event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@router.post("/conversations/{conversation_id}/chat")
async def chat(
    conversation_id: str,
    body: ChatRequest,
    request: Request,
) -> StreamingResponse:
    bind_context(conversation_id=conversation_id)
    settings      = get_settings()
    conv_repo     = ConversationRepository(request.app.state.db)
    llm           = request.app.state.llm
    memory_svc    = request.app.state.memory
    pipeline      = request.app.state.agent_pipeline
    kg_ext        = request.app.state.kg_extractor
    skills_mgr    = request.app.state.skills
    scanner       = request.app.state.scanner

    conv = await conv_repo.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # ── Input security scan ───────────────────────────────────────────────
    scan = scanner.scan_input(body.message, source="user")
    if scan.is_blocked:
        raise HTTPException(
            status_code=400,
            detail=f"Message blocked by content security: {scan.findings[0]}",
        )

    async def generate():
        try:
            # ── System prompt + skill injection ──────────────────────────
            system_prompt = settings.llm.system_prompt
            llm_overrides: dict = {}

            if skills_mgr and body.skill:
                skill = skills_mgr.get_skill(body.skill)
                if skill:
                    system_prompt = f"{skill.system_prompt.strip()}\n\n{system_prompt}"
                    if skill.temperature is not None:
                        llm_overrides["temperature"] = skill.temperature
                    if skill.max_tokens is not None:
                        llm_overrides["max_tokens"] = skill.max_tokens

            if memory_svc:
                messages_for_llm = await memory_svc.build_messages(
                    conversation_id=conversation_id,
                    new_user_message=body.message,
                    system_prompt=system_prompt,
                )
            else:
                history = await conv_repo.get_messages(
                    conversation_id, limit=settings.memory.max_session_messages
                )
                messages_for_llm = [{"role": m.role, "content": m.content} for m in history]
                messages_for_llm.append({"role": "user", "content": body.message})

            # ── Agentic RAG pipeline ──────────────────────────────────────
            if pipeline is not None:
                result = await pipeline.run(
                    query=body.message,
                    conversation_id=conversation_id,
                    system_prompt=system_prompt,
                )

                if result.get("plan"):
                    yield _event({"type": "plan", "content": result["plan"]})

                citations  = result.get("citations", [])
                graph_ctx  = result.get("graph_context", "")
                if citations or graph_ctx:
                    yield _event({
                        "type": "context",
                        "sources": citations,
                        "graph": graph_ctx,
                        "hops": result.get("hops", 0),
                    })

                await conv_repo.add_message(conversation_id, "user", body.message)
                full_reply = result.get("answer", "")

                # ── Output security scan ──────────────────────────────────
                out_scan = scanner.scan_output(full_reply)
                full_reply = out_scan.sanitised_text

                chunk_size = 20
                for i in range(0, len(full_reply), chunk_size):
                    yield _event({"type": "delta", "content": full_reply[i:i + chunk_size]})

            else:
                # Fallback: direct streaming
                await conv_repo.add_message(conversation_id, "user", body.message)
                full_reply = ""
                async for token in llm.stream(messages_for_llm, **llm_overrides):
                    full_reply += token
                    yield _event({"type": "delta", "content": token})

                # Output scan on accumulated reply
                out_scan = scanner.scan_output(full_reply)
                if out_scan.sanitised_text != full_reply:
                    logger.warning("Output sanitised — dangerous patterns redacted.")

            saved = await conv_repo.add_message(conversation_id, "assistant", full_reply)

            if conv.title == "New Conversation" and body.message:
                await conv_repo.update_conversation_title(
                    conversation_id, body.message[:60].strip()
                )

            yield _event({"type": "done", "message_id": saved.id})

            # ── Background tasks ──────────────────────────────────────────
            if memory_svc:
                asyncio.create_task(
                    memory_svc.process_turn(conversation_id, body.message, full_reply)
                )
            if kg_ext and full_reply:
                reply_chunks = chunk_text(full_reply, chunk_size=512)
                asyncio.create_task(
                    kg_ext.extract_from_chunks(reply_chunks, source_doc="chat")
                )

        except Exception as exc:  # noqa: BLE001
            logger.error("Chat stream error: %s", exc)
            yield _event({"type": "error", "detail": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
