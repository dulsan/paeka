"""
tests/unit/test_sandbox.py
===========================
Unit tests for the Docker sandbox.
Docker availability is mocked — no real container is needed.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.anyio
async def test_sandbox_unavailable_raises():
    from backend.agent.sandbox import CodeSandbox

    sandbox = CodeSandbox(docker_available=False)
    with pytest.raises(NotImplementedError):
        await sandbox.execute("print('hello')", language="python")


@pytest.mark.anyio
async def test_sandbox_unsupported_language():
    from backend.agent.sandbox import CodeSandbox

    sandbox = CodeSandbox(docker_available=True)
    with pytest.raises(ValueError, match="Unsupported language"):
        await sandbox.execute("code", language="cobol")


@pytest.mark.anyio
async def test_sandbox_docker_cmd_structure():
    """Verify the Docker command includes all security flags."""
    from backend.agent.sandbox import CodeSandbox, SandboxResult

    captured_cmd: list = []

    async def mock_run(cmd, cid, timeout):
        captured_cmd.extend(cmd)
        return SandboxResult(
            stdout="hello", stderr="", exit_code=0, timed_out=False, container_id=cid
        )

    sandbox = CodeSandbox(docker_available=True)
    with patch.object(sandbox, "_run_docker", side_effect=mock_run):
        await sandbox.execute("print('hello')", language="python")

    cmd_str = " ".join(captured_cmd)
    assert "--network" in cmd_str
    assert "none" in cmd_str
    assert "--read-only" in cmd_str
    assert "--cap-drop" in cmd_str
    assert "ALL" in cmd_str
    assert "--memory" in cmd_str
    assert "--pids-limit" in cmd_str
    assert "no-new-privileges" in cmd_str


@pytest.mark.anyio
async def test_sandbox_successful_execution():
    from backend.agent.sandbox import CodeSandbox, SandboxResult

    async def mock_run(cmd, cid, timeout):
        return SandboxResult(
            stdout="Hello, World!\n", stderr="", exit_code=0,
            timed_out=False, container_id=cid,
        )

    sandbox = CodeSandbox(docker_available=True)
    with patch.object(sandbox, "_run_docker", side_effect=mock_run):
        result = await sandbox.execute("print('Hello, World!')", language="python")

    assert result.success is True
    assert "Hello, World!" in result.stdout


@pytest.mark.anyio
async def test_sandbox_timeout_detected():
    from backend.agent.sandbox import CodeSandbox, SandboxResult

    async def mock_run(cmd, cid, timeout):
        return SandboxResult(
            stdout="", stderr="Execution timed out", exit_code=0,
            timed_out=True, container_id=cid,
        )

    sandbox = CodeSandbox(docker_available=True)
    with patch.object(sandbox, "_run_docker", side_effect=mock_run):
        result = await sandbox.execute("import time; time.sleep(1000)")

    assert result.timed_out is True
    assert result.success is False


@pytest.mark.anyio
async def test_sandbox_output_truncated():
    from backend.agent.sandbox import SandboxResult

    large_output = "x" * 20_000
    result = SandboxResult(
        stdout=large_output, stderr="", exit_code=0,
        timed_out=False, container_id="test",
    )
    assert len(result.output) <= 10_100   # 10KB + "truncated" message
    assert "truncated" in result.output


def test_sandbox_result_properties():
    from backend.agent.sandbox import SandboxResult

    success = SandboxResult("out", "", 0, False, "c1")
    assert success.success is True

    failed = SandboxResult("", "err", 1, False, "c2")
    assert failed.success is False

    timed = SandboxResult("", "", 0, True, "c3")
    assert timed.success is False


@pytest.mark.anyio
async def test_is_available_false_without_docker():
    from backend.agent.sandbox import CodeSandbox
    import asyncio

    sandbox = CodeSandbox(docker_available=True)

    async def mock_exec(*args, **kwargs):
        proc = MagicMock()
        proc.communicate = AsyncMock(return_value=(b"", b""))
        proc.returncode = 1
        return proc

    with patch("asyncio.create_subprocess_exec", side_effect=mock_exec):
        available = await sandbox.is_available()

    assert available is False
