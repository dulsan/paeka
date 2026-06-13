"""
backend/shared/database.py
==========================
Async SQLite wrapper. Schema covers all phases idempotently.

Phase 1: conversations, messages
Phase 2: documents, chunks
Phase 3: memory
Phase 4: document_versions (versioning layer)
Phase 5: kg_nodes, kg_edges (knowledge graph)
"""

from __future__ import annotations

import logging
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

_DDL: list[str] = [
    # ── Phase 1 ──────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS conversations (
        id         TEXT PRIMARY KEY,
        title      TEXT NOT NULL DEFAULT 'New Conversation',
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id              TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        role            TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
        content         TEXT NOT NULL,
        created_at      TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,

    # ── Phase 2 ──────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS documents (
        id          TEXT PRIMARY KEY,
        filename    TEXT NOT NULL,
        filepath    TEXT NOT NULL,
        mime_type   TEXT NOT NULL DEFAULT 'application/octet-stream',
        status      TEXT NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending','processing','ready','failed')),
        chunk_count INTEGER NOT NULL DEFAULT 0,
        parser_used TEXT,
        version     INTEGER NOT NULL DEFAULT 1,
        content_hash TEXT,
        created_at  TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks (
        id          TEXT PRIMARY KEY,
        document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        weaviate_id TEXT,
        chunk_index INTEGER NOT NULL,
        content     TEXT NOT NULL,
        heading     TEXT,
        page        INTEGER,
        element_type TEXT NOT NULL DEFAULT 'text',
        created_at  TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,

    # ── Phase 3 ──────────────────────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS memory (
        id              TEXT PRIMARY KEY,
        conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
        scope           TEXT NOT NULL DEFAULT 'global'
                            CHECK(scope IN ('session','global')),
        content         TEXT NOT NULL,
        created_at      TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,

    # ── Phase 4: versioning ───────────────────────────────────────────────
    # Retired document versions — keeps old chunk records for audit / rollback
    """
    CREATE TABLE IF NOT EXISTS document_versions (
        id           TEXT PRIMARY KEY,
        document_id  TEXT NOT NULL,
        version      INTEGER NOT NULL,
        filename     TEXT NOT NULL,
        content_hash TEXT,
        chunk_count  INTEGER NOT NULL DEFAULT 0,
        retired_at   TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,

    # ── Phase 5: knowledge graph ──────────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS kg_nodes (
        id          TEXT PRIMARY KEY,
        label       TEXT NOT NULL,
        entity_type TEXT NOT NULL,
        description TEXT,
        source_doc  TEXT,
        confidence  REAL NOT NULL DEFAULT 1.0,
        created_at  TEXT NOT NULL DEFAULT (datetime('now')),
        updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS kg_edges (
        id            TEXT PRIMARY KEY,
        source_id     TEXT NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
        target_id     TEXT NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
        relation_type TEXT NOT NULL,
        description   TEXT,
        confidence    REAL NOT NULL DEFAULT 1.0,
        source_doc    TEXT,
        created_at    TEXT NOT NULL DEFAULT (datetime('now'))
    )
    """,

    # ── Indexes ───────────────────────────────────────────────────────────
    "CREATE INDEX IF NOT EXISTS idx_messages_conv     ON messages(conversation_id)",
    "CREATE INDEX IF NOT EXISTS idx_chunks_doc        ON chunks(document_id)",
    "CREATE INDEX IF NOT EXISTS idx_memory_scope      ON memory(scope)",
    "CREATE INDEX IF NOT EXISTS idx_memory_conv       ON memory(conversation_id)",
    "CREATE INDEX IF NOT EXISTS idx_documents_status  ON documents(status)",
    "CREATE INDEX IF NOT EXISTS idx_documents_hash    ON documents(content_hash)",
    "CREATE INDEX IF NOT EXISTS idx_kg_nodes_label    ON kg_nodes(label)",
    "CREATE INDEX IF NOT EXISTS idx_kg_nodes_type     ON kg_nodes(entity_type)",
    "CREATE INDEX IF NOT EXISTS idx_kg_edges_source   ON kg_edges(source_id)",
    "CREATE INDEX IF NOT EXISTS idx_kg_edges_target   ON kg_edges(target_id)",
    "CREATE INDEX IF NOT EXISTS idx_kg_edges_relation ON kg_edges(relation_type)",
]


class Database:
    """Async SQLite gateway. Single connection, WAL mode, foreign keys ON."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._migrate()
        logger.info("SQLite connected: %s", self._path)

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> "Database":
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def _migrate(self) -> None:
        assert self._conn is not None
        for stmt in _DDL:
            await self._conn.execute(stmt)
        await self._conn.commit()
        logger.debug("Database schema applied.")

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database.connect() has not been called.")
        return self._conn

    async def execute(self, sql: str, params: tuple = ()) -> aiosqlite.Cursor:
        cursor = await self.conn.execute(sql, params)
        await self.conn.commit()
        return cursor

    async def fetchall(self, sql: str, params: tuple = ()) -> list[aiosqlite.Row]:
        cursor = await self.conn.execute(sql, params)
        return await cursor.fetchall()

    async def fetchone(self, sql: str, params: tuple = ()) -> aiosqlite.Row | None:
        cursor = await self.conn.execute(sql, params)
        return await cursor.fetchone()
