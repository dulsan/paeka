"""
backend/api/routes/health.py
=============================
GET /api/health — liveness + component readiness probe.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from backend.shared.config import get_settings

router = APIRouter(tags=["health"])


class ComponentStatus(BaseModel):
    llm: bool
    llm_provider: str
    llm_model: str
    retrieval: bool
    memory: bool
    knowledge_graph: bool
    skills: bool


class HealthResponse(BaseModel):
    status: str
    version: str
    components: ComponentStatus


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    settings = get_settings()
    llm = request.app.state.llm
    llm_ok = await llm.health_check()

    return HealthResponse(
        status="ok",
        version=settings.app.version,
        components=ComponentStatus(
            llm=llm_ok,
            llm_provider=getattr(llm, "provider_name", settings.llm.provider),
            llm_model=settings.llm.model,
            retrieval=request.app.state.retrieval is not None,
            memory=request.app.state.memory is not None,
            knowledge_graph=request.app.state.kg_repo is not None,
            skills=request.app.state.skills is not None,
        ),
    )
