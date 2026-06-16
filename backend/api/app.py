"""
backend/api/app.py
==================
FastAPI application factory.

Changes from original:
  [QDRANT]   WeaviateStore replaced with QdrantStore.
             app.state.weaviate renamed to app.state.vector_store for clarity.
             settings.retrieval.weaviate_url -> settings.retrieval.qdrant_url
  [LLM-MSG]  LLM not-reachable warning no longer says "docker compose up paeka-llamacpp"
             since we use Ollama now.

Startup order:
  1. Logging
  2. SQLite
  3. LLM provider (ollama by default via factory)
  4. Content scanner
  5. SearXNG web client
  6. Retrieval + Agentic RAG pipeline (Qdrant-backed)
  7. Memory
  8. Knowledge graph
  9. Skills
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

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    setup_logging(settings.logging.level, settings.logging.format)
    logger.info("Starting PAEKA v%s [mode=%s]", settings.app.version, settings.deploy.mode)

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

    # 3. LLM provider
    llm: LLMProvider = create_provider(settings.llm)
    app.state.llm = llm

    from backend.api.routes.chat_control import configure as configure_chat
    configure_chat(llama_base_url=settings.llm.base_url)

    if await llm.health_check():
        logger.info("%s reachable at %s", llm.provider_name, settings.llm.base_url)
    else:
        logger.warning(
            "%s NOT reachable at %s",
            llm.provider_name, settings.llm.base_url,
        )
        if settings.llm.provider == "ollama":
            logger.warning("Start Ollama with: ollama serve")
            logger.warning("Import model with: ollama create paeka-qwen -f models\\qwen\\Modelfile")

    # 4. Content scanner
    scanner = ContentScanner(
        enabled=settings.security.content_scan_enabled,
        strict_mode=settings.security.strict_mode,
    )
    app.state.scanner = scanner
    logger.info("Content scanner: enabled=%s strict=%s",
                settings.security.content_scan_enabled,
                settings.security.strict_mode)

    # 5. SearXNG web client
    app.state.web_client = None
    if settings.tools.web_search_enabled:
        from backend.tools.searxng import SearXNGClient
        web_client = SearXNGClient(settings.tools, scanner=scanner)
        app.state.web_client = web_client
        logger.info("SearXNG web client ready at %s", settings.tools.searxng_url)
    else:
        logger.info("Web search disabled (set PAEKA_TOOLS__WEB_SEARCH_ENABLED=true to enable)")

    # 6. Retrieval + Agentic RAG pipeline (Qdrant)
    app.state.retrieval      = None
    app.state.ingestion      = None
    app.state.vector_store   = None   # renamed from app.state.weaviate
    app.state.weaviate       = None   # kept as alias for any code that still references it
    app.state.agent_pipeline = None

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
            reranker = get_reranker(settings.retrieval.reranker_model,
                                    settings.retrieval.reranker_device)

            qdrant_url = getattr(settings.retrieval, "qdrant_url",
                                 "http://localhost:6333")
            store = QdrantStore(url=qdrant_url, vector_dim=embedder.dim)
            await store.connect()

            app.state.vector_store = store
            app.state.weaviate     = store   # alias
            app.state.retrieval    = RetrievalEngine(
                store, embedder, reranker, settings.retrieval
            )
            app.state.ingestion = IngestionPipeline(
                db, store, embedder,
                settings.retrieval,
                settings.ingestion,
                scanner=scanner,
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

    if settings.knowledge_graph.enabled:
        try:
            from backend.knowledge.graph import KnowledgeGraphRepository
            from backend.knowledge.extractor import KnowledgeGraphExtractor
            from backend.knowledge.refinement import GraphRefiner
            from backend.knowledge.retriever import GraphRetriever

            kg_repo = KnowledgeGraphRepository(db)
            app.state.kg_repo      = kg_repo
            app.state.kg_extractor = KnowledgeGraphExtractor(
                kg_repo, llm, settings.knowledge_graph)
            app.state.kg_refiner = GraphRefiner(kg_repo, llm, settings.knowledge_graph)
            kg_ret = GraphRetriever(kg_repo, settings.knowledge_graph)
            await kg_ret.load_graph()
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

    yield

    logger.info("Shutting down PAEKA...")
    if app.state.vector_store:
        await app.state.vector_store.close()
    if app.state.web_client:
        await app.state.web_client.close()
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

    cors_origins = (
        ["*"] if dep.mode == "development"
        else [f"https://{dep.domain}"]
    )
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

    from backend.api.routes import (
        health, conversations, chat, documents, memory, knowledge,
        skills, code, export, models, sandbox, agent,
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
    app.include_router(sandbox.router,       prefix="/api")
    app.include_router(agent.router,         prefix="/api")
    app.include_router(openai_compat.router, prefix="/v1")
    app.include_router(chat_control.router,  prefix="/api")

    return app
