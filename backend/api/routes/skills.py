"""
backend/api/routes/skills.py
=============================
Skills management endpoints.

GET  /api/skills            — list all available skills
GET  /api/skills/{name}     — get a specific skill (name + description + prompt)
POST /api/skills/reload     — hot-reload skill folders from disk
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["skills"])


class SkillOut(BaseModel):
    name: str
    description: str
    tags: list[str]
    temperature: float | None
    max_tokens: int | None
    system_prompt: str
    has_scripts: bool
    has_references: bool


class ReloadResponse(BaseModel):
    loaded: int


def _require_skills(request: Request):
    mgr = request.app.state.skills
    if mgr is None:
        raise HTTPException(
            status_code=503,
            detail="Skills not enabled. Set [skills] enabled = true in settings.toml.",
        )
    return mgr


def _skill_to_out(s) -> SkillOut:
    return SkillOut(
        name=s.name,
        description=s.description,
        tags=s.tags,
        temperature=s.temperature,
        max_tokens=s.max_tokens,
        system_prompt=s.system_prompt,
        has_scripts=bool(s.list_scripts()),
        has_references=bool(s.list_references()),
    )


@router.get("/skills", response_model=list[SkillOut])
async def list_skills(request: Request) -> list[SkillOut]:
    """List all available skills."""
    mgr = _require_skills(request)
    return [_skill_to_out(s) for s in mgr.list_skills()]


@router.get("/skills/{skill_name}", response_model=SkillOut)
async def get_skill(skill_name: str, request: Request) -> SkillOut:
    """Get a specific skill by name."""
    mgr = _require_skills(request)
    skill = mgr.get_skill(skill_name)
    if not skill:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found")
    return _skill_to_out(skill)


@router.post("/skills/reload", response_model=ReloadResponse)
async def reload_skills(request: Request) -> ReloadResponse:
    """
    Hot-reload all skill definitions from disk without restarting.
    Add or edit a skill folder, then call this endpoint.
    """
    mgr = _require_skills(request)
    mgr.reload()
    return ReloadResponse(loaded=len(mgr.list_skills()))
