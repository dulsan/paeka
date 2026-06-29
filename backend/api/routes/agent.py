"""
backend/api/routes/agent.py
============================
Agentic feature endpoints.

POST /api/agent/iterate       — autonomous iteration (generate → evaluate → reflect loop)
POST /api/agent/tools/execute — self-healing tool calling pipeline (legacy, JSON-text parsing)
POST /api/agent/react         — ReAct loop with native function calling (MCP-backed, Phase 1 litmus test)
"""

from __future__ import annotations

import logging
from typing import Literal

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


class ChatTurnIn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ReactRequest(BaseModel):
    # `message` (single string, no history) is kept only for backward
    # compatibility with callers that predate multi-turn support
    # (test-paeka-api.ps1, the README curl example). `messages` (full
    # history, last entry is the new turn) is the shape the rwp chat UI
    # actually sends, and is preferred when both are present.
    message: str | None = None
    messages: list[ChatTurnIn] | None = None
    max_rounds: int = 10

    def resolve_turn(self) -> tuple[str, list[dict[str, str]]]:
        """Returns (latest_user_message, prior_history) regardless of which shape the caller used."""
        if self.messages:
            history = [m.model_dump() for m in self.messages[:-1]]
            return self.messages[-1].content, history
        if self.message is not None:
            return self.message, []
        raise ValueError("Either 'message' or 'messages' is required.")


class ToolCallTrace(BaseModel):
    id: str
    name: str
    args: dict
    result: str
    ok: bool


class ReactResponse(BaseModel):
    response: str
    tool_calls: list[ToolCallTrace] = []


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
    Run the self-healing tool calling pipeline (legacy).

    Superseded by /agent/react for new development -- this endpoint uses
    the older JSON-text-parsing tool selection pattern. Kept for now since
    nothing has migrated off it yet.
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


@router.post("/agent/react", response_model=ReactResponse)
async def react(body: ReactRequest, request: Request) -> ReactResponse:
    """
    Run the ReAct tool-calling loop (Phase 1 litmus test).

    Uses ChatOllama (langchain-ollama) native function calling against
    Ollama, MCP-discovered tools (qdrant_search, qdrant_ingest, web_search,
    execute_code, check_services, qdrant_snapshot, list_available_tools),
    and the orchestration guardrails (call memoization + circuit breaker).

    With Logfire configured (local-only, no account needed), every round
    of this loop -- the prompt sent, the raw completion, every tool call
    and its latency, and any guardrail trips -- is visible in the trace.
    This is the endpoint to hit to verify the loop end-to-end.

    Example (single-turn, no history):
        curl -X POST http://localhost:8000/api/agent/react \\
             -H "Content-Type: application/json" \\
             -d '{"message": "What tools do you have available?"}'

    Example (multi-turn, preferred -- mirrors /v1/chat/completions' shape):
        curl -X POST http://localhost:8000/api/agent/react \\
             -H "Content-Type: application/json" \\
             -d '{"messages": [{"role": "user", "content": "What tools do you have available?"}]}'
    """
    from backend.agent.react_graph import ReActGraph

    try:
        user_message, history = body.resolve_turn()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    llm = getattr(request.app.state, "llm", None)
    if llm is None:
        raise HTTPException(
            status_code=503,
            detail="LLM provider not initialised. Check app startup logs.",
        )
    if not await llm.health_check():
        raise HTTPException(
            status_code=503,
            detail=(
                f"{llm.provider_name} backend is unreachable. "
                "Start the backend and retry."
            ),
        )

    chat_ollama = getattr(request.app.state, "chat_ollama", None)
    if chat_ollama is None:
        raise HTTPException(
            status_code=503,
            detail="ChatOllama not initialised. Check app startup logs.",
        )

    graph = ReActGraph(llm=chat_ollama, max_rounds=min(body.max_rounds, 15))
    result = await graph.run(user_message=user_message, history=history)

    if not result["response"]:
        raise HTTPException(
            status_code=502,
            detail="ReAct loop completed with no final text response. "
                   "Check logs/logfire trace for what happened in each round.",
        )

    return ReactResponse(response=result["response"], tool_calls=result["tool_calls"])
