"""
backend/api/app.py
==================
FastAPI application factory.

Startup order:
  0. Observability (Logfire, local-only -- must be first so it can trace
     everything after it, including Pydantic validation during settings load)
  1. Database
  2. Model registry
  3. LLM provider (Ollama by default via factory)
  4. Content scanner
  5. Web search client
  6. Retrieval + Agentic RAG pipeline (Qdrant-backed)
  7. Memory
  8. Knowledge graph
  9. Skills
  10. MCP server: inject services, mount at /mcp
  11. Deep orchestrator (deepagents: planning + HITL + sub-agents)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.shared.config import get_settings
from backend.shared.database import Database
from backend.shared.logging import setup_logging
from backend.llm.factory import create_provider
from backend.llm.base import LLMProvider
from backend.security.content import ContentScanner
from backend.security.auth import AuthMiddleware
from backend.security.ratelimit import RateLimitMiddleware
from backend.security.request_context import RequestContextMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # 0. Observability -- first, so everything after this point is traced.
    from backend.observability.logfire_setup import configure_observability
    observability_active = configure_observability()

    settings = get_settings()
    setup_logging(settings.logging.level, settings.logging.format)
    logger.info("Starting PAEKA v%s [mode=%s] [observability=%s]",
                settings.app.version, settings.deploy.mode, observability_active)

    # 1. Database
    db = Database(settings.database.sqlite_path)
    await db.connect()
    app.state.db = db

    # 2. Model registry
    from backend.models.registry import ModelRegistry
    from backend.models.loader import resolve_model
    registry = ModelRegistry(settings.models.models_dir)
    if settings.models.auto_scan:
        models = registry.scan()
        if models:
            logger.info("Available models: %s",
                        [f"{m.name} ({m.size_gb}GB)" for m in models])

    load_result = resolve_model(settings.llm, registry)
    if not load_result.ok:
        logger.warning("Model validation: %s", load_result.message)
    else:
        logger.info("Model: %s", load_result.message)
    app.state.model_registry = registry

    # 3. LLM provider (main conversational provider, e.g. Ollama)
    llm: LLMProvider = create_provider(settings.llm)
    app.state.llm = llm

    from backend.api.routes.chat_control import configure as configure_chat
    configure_chat(llama_base_url=settings.llm.base_url)

    if await llm.health_check():
        logger.info("%s reachable at %s", llm.provider_name, settings.llm.base_url)
    else:
        logger.warning("%s NOT reachable at %s", llm.provider_name, settings.llm.base_url)
        if settings.llm.provider == "ollama":
            logger.warning("Start Ollama with: ollama serve")
            logger.warning("Import model with: ollama create paeka-qwen -f models\\qwen\\Modelfile")

    # 3b. ChatOllama (used only by ReActGraph for native function calling)
    # [MIGRATION] Replaces LiteLLMProvider entirely -- see react_graph.py's
    # module docstring for what was verified before making this change.
    #
    # settings.llm.base_url is "http://localhost:11434/v1" -- that "/v1"
    # suffix is specifically for OllamaProvider's OpenAI-compat HTTP calls
    # (backend/llm/ollama.py) and openai_compat.py's pass-through route.
    # ChatOllama talks to Ollama's NATIVE /api/chat endpoint instead (not
    # the OpenAI-compat layer), which doesn't use a "/v1" prefix at all --
    # confirmed this is langchain-ollama's actual behavior, not an
    # OpenAI-compat wrapper, before relying on it. Strip the suffix here
    # rather than introduce a second, separately-configured base URL.
    from langchain_ollama import ChatOllama
    _ollama_native_base = settings.llm.base_url.rstrip("/")
    if _ollama_native_base.endswith("/v1"):
        _ollama_native_base = _ollama_native_base[: -len("/v1")]

    app.state.chat_ollama = ChatOllama(
        model=settings.llm.model,
        base_url=_ollama_native_base,
        temperature=0.7,
        num_predict=4096,  # Ollama's native name for max output tokens
        client_kwargs={"timeout": 180.0},
    )
    logger.info("ChatOllama: model=%s base=%s", settings.llm.model, _ollama_native_base)

    # 4. Content scanner
    scanner = ContentScanner(
        enabled=settings.security.content_scan_enabled,
        strict_mode=settings.security.strict_mode,
    )
    app.state.scanner = scanner
    logger.info("Content scanner: enabled=%s strict=%s",
                settings.security.content_scan_enabled,
                settings.security.strict_mode)

    # 5. Web search client
    app.state.web_client = None
    if settings.tools.web_search_enabled:
        from backend.tools.websearch import WebSearchClient
        web_client = WebSearchClient(settings.tools, scanner=scanner)
        app.state.web_client = web_client
        logger.info("Web search client ready (DuckDuckGo backend).")
    else:
        logger.info("Web search disabled (set PAEKA_TOOLS__WEB_SEARCH_ENABLED=true to enable)")

    # 6. Retrieval + Agentic RAG pipeline (Qdrant)
    app.state.retrieval      = None
    app.state.ingestion      = None
    app.state.vector_store   = None
    app.state.weaviate       = None
    app.state.agent_pipeline = None
    app.state.embedder       = None

    if settings.retrieval.enabled:
        try:
            from backend.retrieval.embedder import get_embedder
            from backend.retrieval.reranker import get_reranker
            from backend.retrieval.qdrant_store import QdrantStore
            from backend.retrieval.engine import RetrievalEngine
            from backend.ingestion.pipeline import IngestionPipeline
            from backend.agent.graph import AgenticRAGPipeline

            embedder = get_embedder(settings.retrieval.embed_model,
                                    settings.retrieval.embed_device)
            app.state.embedder = embedder
            reranker = get_reranker(settings.retrieval.reranker_model,
                                    settings.retrieval.reranker_device)

            qdrant_url = getattr(settings.retrieval, "qdrant_url", "http://localhost:6333")
            store = QdrantStore(url=qdrant_url, vector_dim=embedder.dim)
            await store.connect()

            app.state.vector_store = store
            app.state.weaviate     = store
            app.state.retrieval    = RetrievalEngine(store, embedder, reranker, settings.retrieval)
            app.state.ingestion = IngestionPipeline(
                db, store, embedder, settings.retrieval, settings.ingestion, scanner=scanner,
            )
            app.state.agent_pipeline = AgenticRAGPipeline(
                llm=llm,
                retrieval_engine=app.state.retrieval,
                graph_retriever=None,
                web_client=app.state.web_client,
                max_hops=settings.retrieval.max_hops,
            )
            logger.info("Retrieval + Agentic RAG pipeline ready (Qdrant).")
        except Exception as exc:
            logger.error("Retrieval init failed: %s", exc)
            logger.warning("Continuing without retrieval.")

    # 7. Memory
    app.state.memory = None
    if settings.memory.enabled:
        from backend.memory.service import MemoryService
        app.state.memory = MemoryService(db, llm, settings.memory)
        logger.info("Memory service ready.")

    # 8. Knowledge graph
    app.state.kg_repo      = None
    app.state.kg_extractor = None
    app.state.kg_refiner   = None
    app.state.kg_retriever = None
    app.state.kg_falkor    = None

    if settings.knowledge_graph.enabled:
        try:
            from backend.knowledge.graph import KnowledgeGraphRepository
            from backend.knowledge.extractor import KnowledgeGraphExtractor
            from backend.knowledge.refinement import GraphRefiner
            from backend.knowledge.retriever import GraphRetriever
            from backend.knowledge.falkor_store import FalkorGraphStore

            kg_repo = KnowledgeGraphRepository(db)
            app.state.kg_repo      = kg_repo

            # FalkorDB query layer (Cypher multi-hop traversal). SQLite via
            # kg_repo above remains the system of record -- this is a
            # derived, rebuildable view synced from it on startup and
            # after every extraction/refinement pass.
            falkor = None
            if settings.knowledge_graph.falkor_enabled:
                falkor = FalkorGraphStore(settings.knowledge_graph.falkor_db_path)
                if await falkor.connect():
                    await falkor.sync_from_sqlite(kg_repo)
                    app.state.kg_falkor = falkor
                else:
                    falkor = None

            app.state.kg_extractor = KnowledgeGraphExtractor(kg_repo, llm, settings.knowledge_graph, falkor=falkor)
            app.state.kg_refiner   = GraphRefiner(kg_repo, llm, settings.knowledge_graph, falkor=falkor)
            kg_ret = GraphRetriever(kg_repo, settings.knowledge_graph)
            await kg_ret.load_graph()
            kg_ret.falkor = falkor  # opt-in multi-hop path, see retriever.py
            app.state.kg_retriever = kg_ret

            if app.state.agent_pipeline is not None:
                app.state.agent_pipeline._graph_retriever = kg_ret
                logger.info("Knowledge graph wired into agentic pipeline.")

            logger.info("Knowledge graph ready.")
        except Exception as exc:
            logger.error("Knowledge graph init failed: %s", exc)

    # 9. Skills
    app.state.skills = None
    if settings.skills.enabled:
        from backend.skills.manager import SkillsManager
        mgr = SkillsManager(settings.skills.skills_dir)
        mgr.load()
        app.state.skills = mgr
        logger.info("Skills loaded: %d", len(mgr.list_skills()))

    # 10. MCP server: inject services, mount at /mcp
    mcp_server = None
    try:
        from backend.mcp.server import mcp as mcp_server
        from backend.mcp.server import configure as configure_mcp
        configure_mcp(
            store=app.state.vector_store,
            embedder=app.state.embedder,
            llm=llm,
            web_client=app.state.web_client,
            falkor=app.state.kg_falkor,
        )
        logger.info("MCP server configured.")
    except Exception as exc:
        logger.error("MCP server configuration failed: %s", exc)

    # 11. Deep orchestrator (deepagents harness: planning + HITL + sub-agents)
    # Mounted after the MCP server so configure_mcp() has already injected
    # the services the orchestrator's MCP tool wrappers will call.
    app.state.deep_orchestrator = None
    try:
        from backend.agent.deep_orchestrator import DeepOrchestrator
        mcp_url = f"http://localhost:{settings.server.port}/mcp"
        app.state.deep_orchestrator = DeepOrchestrator.create(
            app_state=app.state,
            settings=settings,
            mcp_url=mcp_url,
        )
        logger.info("Deep orchestrator ready.")
    except Exception as exc:
        logger.error("Deep orchestrator init failed (non-fatal): %s", exc)

    # [FIX] Confirmed root cause of "McpError: Session terminated" on every
    # MCP client call, including PAEKA's own self-call from react_graph.py:
    # app.mount("/mcp", mcp_server.streamable_http_app()) in create_app()
    # does NOT automatically run streamable_http_app()'s own lifespan --
    # FastAPI/Starlette does not propagate lifespan execution into mounted
    # sub-applications just because they're mounted (confirmed against an
    # official modelcontextprotocol/python-sdk GitHub issue, #1467,
    # describing this exact symptom, plus several independent working
    # examples using the same fix). Without mcp_server.session_manager.run()
    # actually active, the session manager's internal task group never
    # initializes, so it has no concept of an "active" session at all --
    # every session immediately looks terminated because nothing was ever
    # tracking it as alive in the first place.
    #
    # This context needs to stay open for the server's entire serving
    # lifetime, so it wraps the yield (and shutdown) rather than just being
    # entered and exited around step 11 alone.
    if mcp_server is not None:
        async with mcp_server.session_manager.run():
            logger.info("MCP session manager started.")
            yield
    else:
        # MCP server failed to import/configure above -- still yield so the
        # rest of the app serves normally, just without MCP tool-calling.
        yield

    logger.info("Shutting down PAEKA...")
    if app.state.vector_store:
        await app.state.vector_store.close()
    if app.state.web_client:
        await app.state.web_client.close()
    if app.state.kg_falkor:
        await app.state.kg_falkor.close()
    await llm.close()
    await db.close()


def create_app() -> FastAPI:
    settings = get_settings()
    sec = settings.security
    dep = settings.deploy

    app = FastAPI(
        title=settings.app.name,
        description=settings.app.description,
        version=settings.app.version,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    cors_origins = ["*"] if dep.mode == "development" else [f"https://{dep.domain}"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Authorization", "X-API-Key", "Content-Type"],
    )
    app.add_middleware(AuthMiddleware, token=sec.token, enabled=sec.enabled)
    app.add_middleware(
        RateLimitMiddleware,
        enabled=sec.rate_limit_enabled,
        chat_rpm=sec.chat_rpm,
        upload_rpm=sec.upload_rpm,
        default_rpm=sec.default_rpm,
    )
    # Added last so it becomes the OUTERMOST middleware (Starlette wraps in
    # reverse-add order) -- request_id must be bound before auth/rate-limit
    # run so their own log lines are tagged with it too.
    app.add_middleware(RequestContextMiddleware)

    from backend.api.routes import (
        health, conversations, chat, documents, memory, knowledge,
        skills, code, export, models, agent,
        openai_compat, chat_control,
    )
    app.include_router(health.router,        prefix="/api")
    app.include_router(conversations.router, prefix="/api")
    app.include_router(chat.router,          prefix="/api")
    app.include_router(documents.router,     prefix="/api")
    app.include_router(memory.router,        prefix="/api")
    app.include_router(knowledge.router,     prefix="/api")
    app.include_router(skills.router,        prefix="/api")
    app.include_router(code.router,          prefix="/api")
    app.include_router(export.router,        prefix="/api")
    app.include_router(models.router,        prefix="/api")
    app.include_router(agent.router,         prefix="/api")
    app.include_router(openai_compat.router, prefix="/v1")
    app.include_router(chat_control.router,  prefix="/api")

    # Mount MCP server at /mcp. streamable_http_app() requires mcp>=1.1.0.
    try:
        from backend.mcp.server import mcp as mcp_server
        app.mount("/mcp", mcp_server.streamable_http_app())
        logger.debug("MCP server mounted at /mcp")
    except Exception as exc:
        logger.warning("MCP server mount failed: %s", exc)

    return app
