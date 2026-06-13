"""
backend/api/routes/agent.py
============================
Agentic feature endpoints.

POST /api/agent/iterate       — autonomous iteration (generate → evaluate → reflect loop)
POST /api/agent/tools/execute — self-healing tool calling pipeline
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["agent"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class IterateRequest(BaseModel):
    task: str
    context: str = ""
    max_iterations: int = 4
    score_threshold: float = 0.85
    skill: str | None = None    # optional skill to apply


class IterateResponse(BaseModel):
    final_output: str
    iterations: int
    final_score: float
    converged: bool
    critique_history: list[str]


class ToolExecuteRequest(BaseModel):
    request: str                # natural language request for the tool agent
    tools: list[str]            # tool names to make available (from registered tools)
    max_retries: int = 3


class ToolExecuteResponse(BaseModel):
    final_response: str
    succeeded: bool
    iterations: int
    reflections: list[str]
    results: list[dict]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/agent/iterate", response_model=IterateResponse)
async def iterate(body: IterateRequest, request: Request) -> IterateResponse:
    """
    Run the autonomous iteration loop on a task.

    The LLM generates an output, evaluates it, reflects on weaknesses,
    and regenerates — up to max_iterations times or until score_threshold
    is reached.

    Example use cases:
      - "Write a Python function that does X, make it as clean as possible"
      - "Summarise this document, ensuring all key findings are covered"
      - "Draft a technical explanation of Y for a graduate student"
    """
    from backend.agent.iteration_graph import AutonomousIterationGraph
    from backend.shared.config import get_settings

    settings   = get_settings()
    llm        = request.app.state.llm
    skills_mgr = request.app.state.skills

    system_prompt = settings.llm.system_prompt
    if skills_mgr and body.skill:
        skill = skills_mgr.get_skill(body.skill)
        if skill:
            system_prompt = f"{skill.system_prompt.strip()}\n\n{system_prompt}"

    graph = AutonomousIterationGraph(
        llm=llm,
        score_threshold=body.score_threshold,
        max_iterations=min(body.max_iterations, 6),   # hard cap
    )

    result = await graph.run(
        task=body.task,
        system_prompt=system_prompt,
        context=body.context,
    )

    return IterateResponse(
        final_output=result["final_output"],
        iterations=result["iterations"],
        final_score=result["final_score"],
        converged=result["converged"],
        critique_history=result["critique_history"],
    )


@router.post("/agent/tools/execute", response_model=ToolExecuteResponse)
async def execute_tools(
    body: ToolExecuteRequest, request: Request
) -> ToolExecuteResponse:
    """
    Run the self-healing tool calling pipeline.

    The LLM plans tool calls, executes them, evaluates results,
    and retries with corrections on failure — up to max_retries times.

    Available tools are passed by name. Only registered tools can be invoked.
    """
    from backend.agent.tool_graph import SelfHealingToolGraph
    from backend.tools.registry import get_registered_tools

    llm              = request.app.state.llm
    all_tools        = get_registered_tools(request)
    selected_tools   = {k: v for k, v in all_tools.items() if k in body.tools}

    if not selected_tools:
        raise HTTPException(
            status_code=400,
            detail=f"None of the requested tools are registered: {body.tools}",
        )

    graph = SelfHealingToolGraph(
        llm=llm,
        tools=selected_tools,
        max_retries=min(body.max_retries, 5),
    )

    result = await graph.run(request=body.request)

    return ToolExecuteResponse(
        final_response=result["final_response"],
        succeeded=result["succeeded"],
        iterations=result["iterations"],
        reflections=result["reflections"],
        results=[
            {
                "tool":    r["tool_name"],
                "success": r["success"],
                "output":  r["output"][:500],
                "error":   r["error"],
            }
            for r in result["tool_results"]
        ],
    )
