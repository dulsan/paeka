"""
tests/integration/test_api.py
==============================
Integration tests — FastAPI test client with mocked LLM and in-memory SQLite.
No GPU, no Weaviate, no SGLang required to run these.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from backend.shared.database import Database


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    """Async test client with mocked LLM and in-memory DB."""
    # Patch get_settings before importing the app
    with patch("backend.shared.config.get_settings") as mock_settings:
        from backend.shared.config import (
            AppSettings, ServerSettings, LLMSettings, DatabaseSettings,
            MemorySettings, RetrievalSettings, IngestionSettings,
            KnowledgeGraphSettings, SkillsSettings, ToolsSettings, LoggingSettings,
            SecuritySettings, DeploymentSettings,
        )
        settings = MagicMock()
        settings.app = AppSettings()
        settings.server = ServerSettings()
        settings.llm = LLMSettings()
        settings.database = DatabaseSettings(sqlite_path=":memory:")
        settings.memory = MemorySettings(enabled=False)
        settings.retrieval = RetrievalSettings(enabled=False)
        settings.ingestion = IngestionSettings()
        settings.knowledge_graph = KnowledgeGraphSettings(enabled=False)
        settings.skills = SkillsSettings(enabled=False)
        settings.tools = ToolsSettings()
        settings.logging = LoggingSettings()
        # [FIX] These two were missing entirely. settings is a bare
        # MagicMock(), so any attribute NOT explicitly assigned here --
        # like settings.security -- auto-generates its own fresh MagicMock
        # on access instead of raising. That meant settings.security.
        # default_rpm (read by RateLimitMiddleware on every single request)
        # was silently a MagicMock instead of a real int, and `rpm / 60.0`
        # then `min(float(rpm), bucket.tokens + ...)` blew up with
        # "'<' not supported between instances of 'MagicMock' and 'float'"
        # -- on literally every request, which is why all 9 tests in this
        # file failed identically. SecuritySettings() defaults to
        # rate_limit_enabled=False already, so just instantiating it for
        # real (instead of leaving it as an auto-mock) is enough to fix
        # this -- explicit here for clarity since middleware behaviour
        # shouldn't depend on silently-unconfigured mock defaults.
        # settings.deploy was also missing -- create_app() reads
        # settings.deploy.mode for CORS origin selection; an unconfigured
        # MagicMock there wouldn't crash (mode == "development" just
        # evaluates False against a MagicMock) but would silently produce
        # a nonsense CORS origin string rather than the intended behaviour.
        settings.security = SecuritySettings(rate_limit_enabled=False)
        settings.deploy = DeploymentSettings()
        mock_settings.return_value = settings

        from backend.api.app import create_app
        app = create_app()

        # Wire state manually (bypass lifespan)
        db = Database(":memory:")
        await db.connect()

        mock_llm = MagicMock()
        mock_llm.health_check = AsyncMock(return_value=True)
        # [FIX] Same MagicMock gotcha as settings.security above, different
        # spot: health.py does getattr(llm, "provider_name", <default>).
        # On a bare MagicMock(), accessing .provider_name doesn't raise
        # AttributeError -- it auto-generates a fresh MagicMock as the
        # value, so the attribute access *succeeds* and getattr's default
        # is never used at all. Pydantic's ComponentStatus model then
        # rejects that MagicMock when validating llm_provider as a string.
        mock_llm.provider_name = "ollama"

        async def fake_stream(messages, **_):
            for word in ["Hello", " world", "!"]:
                yield word

        mock_llm.stream = fake_stream

        app.state.db = db
        app.state.llm = mock_llm
        app.state.memory = None
        app.state.retrieval = None
        app.state.ingestion = None
        app.state.weaviate = None
        app.state.kg_repo = None
        app.state.kg_extractor = None
        app.state.kg_refiner = None
        app.state.kg_retriever = None
        app.state.skills = None

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            yield ac

        await db.close()


@pytest.mark.anyio
async def test_health(client):
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "components" in data


@pytest.mark.anyio
async def test_create_list_conversations(client):
    resp = await client.post("/api/conversations", json={"title": "Test"})
    assert resp.status_code == 201
    cid = resp.json()["id"]

    resp = await client.get("/api/conversations")
    assert resp.status_code == 200
    assert any(c["id"] == cid for c in resp.json())


@pytest.mark.anyio
async def test_get_conversation_detail(client):
    resp = await client.post("/api/conversations", json={"title": "Detail test"})
    cid = resp.json()["id"]

    resp = await client.get(f"/api/conversations/{cid}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == cid
    assert "messages" in data


@pytest.mark.anyio
async def test_rename_conversation(client):
    resp = await client.post("/api/conversations", json={"title": "Old"})
    cid = resp.json()["id"]
    resp = await client.patch(f"/api/conversations/{cid}", json={"title": "New"})
    assert resp.status_code == 200
    assert resp.json()["title"] == "New"


@pytest.mark.anyio
async def test_delete_conversation(client):
    resp = await client.post("/api/conversations", json={"title": "Delete me"})
    cid = resp.json()["id"]
    resp = await client.delete(f"/api/conversations/{cid}")
    assert resp.status_code == 204
    resp = await client.get(f"/api/conversations/{cid}")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_get_nonexistent_conversation(client):
    resp = await client.get("/api/conversations/does-not-exist")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_documents_503_when_disabled(client):
    resp = await client.get("/api/documents")
    # Repository is always available even without ingestion pipeline
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_knowledge_503_when_disabled(client):
    resp = await client.get("/api/knowledge/stats")
    assert resp.status_code == 503


@pytest.mark.anyio
async def test_skills_503_when_disabled(client):
    resp = await client.get("/api/skills")
    assert resp.status_code == 503
