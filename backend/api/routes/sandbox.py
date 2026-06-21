"""
backend/api/routes/sandbox.py
==============================
Secure code execution endpoints.

POST /api/sandbox/execute     — run code in an isolated Docker container
GET  /api/sandbox/languages   — list supported languages
GET  /api/sandbox/status      — check Docker availability
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.agent.sandbox import get_sandbox
from backend.shared.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["sandbox"])


class ExecuteRequest(BaseModel):
    code: str
    language: str = "python"
    timeout: int | None = None


class ExecuteResponse(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    success: bool
    output: str              # combined stdout+stderr (truncated)


class SandboxStatusResponse(BaseModel):
    docker_available: bool
    supported_languages: list[str]


@router.post("/sandbox/execute", response_model=ExecuteResponse)
async def execute_code(
    body: ExecuteRequest,
    request: Request,
) -> ExecuteResponse:
    """
    Execute code in an isolated Docker container.

    Security: --network=none, --read-only, --cap-drop=ALL,
              --memory=256m, --pids-limit=64, tmpfs /tmp
    """
    scanner = request.app.state.scanner
    if scanner:
        scan = scanner.scan_input(body.code, source="sandbox")
        if scan.is_blocked:
            raise HTTPException(
                status_code=400,
                detail=f"Code blocked by content security: {scan.findings[0]}",
            )

    sandbox = get_sandbox()

    # Check Docker is available before attempting
    if not await sandbox.is_available():
        raise HTTPException(
            status_code=503,
            detail="Docker is not available. Ensure Docker is running on the host.",
        )

    # [FIX] settings was fetched and never used -- max_timeout below is a
    # hardcoded constant, not read from settings. Dead code.
    max_timeout = 60   # hard cap regardless of user request
    timeout = min(body.timeout or 30, max_timeout)

    try:
        result = await sandbox.execute(
            code=body.code,
            language=body.language,
            timeout=timeout,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ExecuteResponse(
        stdout=result.stdout,
        stderr=result.stderr,
        exit_code=result.exit_code,
        timed_out=result.timed_out,
        success=result.success,
        output=result.output,
    )


@router.get("/sandbox/languages", response_model=list[str])
async def list_languages() -> list[str]:
    """List languages supported by the sandbox."""
    return ["python", "bash", "javascript"]


@router.get("/sandbox/status", response_model=SandboxStatusResponse)
async def sandbox_status() -> SandboxStatusResponse:
    """Check whether Docker is available for sandbox execution."""
    sandbox = get_sandbox()
    available = await sandbox.is_available()
    return SandboxStatusResponse(
        docker_available=available,
        supported_languages=["python", "bash", "javascript"] if available else [],
    )
