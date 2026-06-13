"""
backend/memory/service.py
==========================
Phase 3 Memory System.

Three memory tiers are managed here:

  Session memory
    The rolling window of recent messages kept in-context.
    Bounded by ``settings.memory.max_session_messages``.

  Session summary
    When the session window exceeds ``settings.memory.summary_threshold``
    new messages, the LLM is asked to summarise the conversation so far.
    Summaries are persisted in SQLite (scope='session').

  Global memory
    Important facts, preferences, and context extracted from the
    conversation by the LLM and stored permanently in SQLite (scope='global').
    The most recent global memories are injected at the top of every prompt.

Prompt injection order:
    1. System prompt  (from settings)
    2. Global memories  (compact factual bullets)
    3. Session summary  (if one exists and session is long)
    4. Recent messages  (rolling window)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from backend.llm.client import LLMClient, Message
from backend.memory.repository import ChatMessage, ConversationRepository
from backend.shared.config import MemorySettings
from backend.shared.database import Database

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompts used for extraction / summarisation
# ---------------------------------------------------------------------------

_EXTRACT_GLOBAL_PROMPT = """\
Review the following conversation excerpt and extract any important facts,
user preferences, or domain knowledge worth remembering long-term.
Return ONLY a JSON array of concise strings (max 20 words each).
If there is nothing worth extracting, return an empty array: []

Conversation:
{conversation}

JSON array:"""

_SUMMARISE_SESSION_PROMPT = """\
Summarise the following conversation in 3–5 sentences, preserving the key
technical details and any conclusions reached. Be concise and factual.

Conversation:
{conversation}

Summary:"""


# ---------------------------------------------------------------------------
# Memory repository — SQLite layer for memory entries
# ---------------------------------------------------------------------------


class MemoryRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    async def add(
        self,
        content: str,
        scope: str,
        conversation_id: str | None = None,
    ) -> str:
        mid = str(uuid.uuid4())
        now = datetime.now(tz=timezone.utc).isoformat()
        await self._db.execute(
            """
            INSERT INTO memory (id, conversation_id, scope, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (mid, conversation_id, scope, content, now),
        )
        return mid

    async def list_global(self, limit: int = 10) -> list[str]:
        rows = await self._db.fetchall(
            "SELECT content FROM memory WHERE scope='global' ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [r["content"] for r in rows]

    async def get_session_summary(self, conversation_id: str) -> str | None:
        row = await self._db.fetchone(
            """
            SELECT content FROM memory
            WHERE scope='session' AND conversation_id=?
            ORDER BY created_at DESC LIMIT 1
            """,
            (conversation_id,),
        )
        return row["content"] if row else None

    async def delete_session(self, conversation_id: str) -> None:
        await self._db.execute(
            "DELETE FROM memory WHERE scope='session' AND conversation_id=?",
            (conversation_id,),
        )

    async def clear_global(self) -> None:
        await self._db.execute("DELETE FROM memory WHERE scope='global'")


# ---------------------------------------------------------------------------
# Memory service
# ---------------------------------------------------------------------------


class MemoryService:
    """
    Manages session-window management, summarisation, and global memory.

    Parameters
    ----------
    db:
        Open Database connection.
    llm:
        LLMClient instance used for extraction/summarisation calls.
    settings:
        MemorySettings from the config.
    """

    def __init__(self, db: Database, llm: LLMClient, settings: MemorySettings) -> None:
        self._conv_repo = ConversationRepository(db)
        self._mem_repo = MemoryRepository(db)
        self._llm = llm
        self._settings = settings

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    async def build_messages(
        self,
        conversation_id: str,
        new_user_message: str,
        system_prompt: str,
    ) -> list[Message]:
        """
        Assemble the full message list to send to the LLM:

          system → global_memory_block → session_summary → recent_messages → new_user_msg

        Parameters
        ----------
        conversation_id:
            Active conversation ID.
        new_user_message:
            The message the user just sent (not yet persisted).
        system_prompt:
            Base system prompt from settings.

        Returns
        -------
        list[Message]
            Ready to pass directly to LLMClient.
        """
        s = self._settings

        # 1. Load recent history
        recent = await self._conv_repo.get_messages(
            conversation_id, limit=s.max_session_messages
        )

        # 2. Build system block
        system_parts = [system_prompt.strip()]

        # 3. Inject global memory
        globals_ = await self._mem_repo.list_global(limit=s.global_memory_limit)
        if globals_:
            bullet_block = "\n".join(f"- {g}" for g in globals_)
            system_parts.append(f"Known context about the user:\n{bullet_block}")

        # 4. Inject session summary (if session is getting long)
        if len(recent) >= s.summary_threshold:
            summary = await self._mem_repo.get_session_summary(conversation_id)
            if summary:
                system_parts.append(f"Earlier in this conversation:\n{summary}")

        combined_system = "\n\n".join(system_parts)

        messages: list[Message] = [{"role": "system", "content": combined_system}]
        messages.extend({"role": m.role, "content": m.content} for m in recent)
        messages.append({"role": "user", "content": new_user_message})
        return messages

    # ------------------------------------------------------------------
    # Post-turn processing
    # ------------------------------------------------------------------

    async def process_turn(
        self,
        conversation_id: str,
        user_message: str,
        assistant_reply: str,
    ) -> None:
        """
        Run memory extraction/summarisation after a completed turn.

        Should be called after the assistant reply has been persisted.
        This is a background operation — failures are logged, not raised.
        """
        s = self._settings
        if not s.enabled:
            return

        try:
            recent = await self._conv_repo.get_messages(conversation_id)
            total = len(recent)

            # Trigger session summarisation when threshold crossed
            if total > 0 and total % s.summary_threshold == 0:
                await self._summarise_session(conversation_id, recent)

            # Extract global facts every 10 messages
            if total > 0 and total % 10 == 0:
                await self._extract_global(conversation_id, user_message, assistant_reply)

        except Exception as exc:  # noqa: BLE001
            logger.warning("Memory processing error (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _summarise_session(
        self,
        conversation_id: str,
        messages: list[ChatMessage],
    ) -> None:
        """Ask the LLM to summarise the conversation and store it."""
        convo_text = _format_messages(messages[-self._settings.summary_threshold :])
        prompt = _SUMMARISE_SESSION_PROMPT.format(conversation=convo_text)

        summary = await self._llm.complete(
            [{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.3,
        )
        summary = summary.strip()
        if summary:
            await self._mem_repo.delete_session(conversation_id)  # replace old summary
            await self._mem_repo.add(summary, scope="session", conversation_id=conversation_id)
            logger.info("Session summary stored for conversation %s", conversation_id)

    async def _extract_global(
        self,
        conversation_id: str,
        user_message: str,
        assistant_reply: str,
    ) -> None:
        """Extract long-term facts from the last exchange and store globally."""
        import json  # local to avoid top-level noise

        snippet = f"User: {user_message}\nAssistant: {assistant_reply}"
        prompt = _EXTRACT_GLOBAL_PROMPT.format(conversation=snippet)

        raw = await self._llm.complete(
            [{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.1,
        )
        raw = raw.strip()

        try:
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1].lstrip("json").strip()
            facts: list[str] = json.loads(raw)
            for fact in facts:
                if isinstance(fact, str) and fact.strip():
                    await self._mem_repo.add(
                        fact.strip(), scope="global", conversation_id=conversation_id
                    )
            if facts:
                logger.info("Extracted %d global memories.", len(facts))
        except (json.JSONDecodeError, ValueError) as exc:
            logger.debug("Global memory extraction parse error: %s | raw=%s", exc, raw[:80])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_messages(messages: list[ChatMessage]) -> str:
    return "\n".join(f"{m.role.capitalize()}: {m.content}" for m in messages)
