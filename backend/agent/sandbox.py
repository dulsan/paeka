"""
backend/agent/sandbox.py
=========================
Secure code execution sandbox using Docker containers.

Security model (defence-in-depth):
  --network=none          no outbound internet access
  --read-only             immutable root filesystem
  --cap-drop=ALL          all Linux capabilities dropped
  --security-opt=no-new-privileges  prevent privilege escalation
  --memory / --cpus       resource hard limits
  --pids-limit            prevent fork bombs
  tmpfs /tmp              writable scratch space, in-memory only

Why Docker over Firecracker:
  Firecracker requires KVM access on the host, which is unavailable
  inside Docker (our own containers) without --privileged, negating
  the security benefit. For a single-user self-hosted system, hardened
  Docker containers (all caps dropped, no network, read-only FS) provide
  strong practical isolation without needing a microVM control plane.
  The community consensus is: Firecracker for multi-tenant cloud,
  hardened Docker for single-user/trusted environments.

Supported languages:
  Python   → python:3.12-slim image
  Bash     → alpine image
  (extensible via SandboxConfig)
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Resource limits for sandbox containers
_DEFAULT_MEMORY  = "256m"
_DEFAULT_CPUS    = "1.0"
_DEFAULT_TIMEOUT = 30       # seconds before SIGKILL
_DEFAULT_PIDS    = 64       # max processes inside container


@dataclass
class SandboxConfig:
    """Per-language sandbox configuration."""
    image: str
    run_cmd: list[str]      # command to execute the script (without the script path)
    file_ext: str           # source file extension
    memory: str     = _DEFAULT_MEMORY
    cpus: str       = _DEFAULT_CPUS
    timeout: int    = _DEFAULT_TIMEOUT
    pids_limit: int = _DEFAULT_PIDS
    env: dict       = field(default_factory=dict)


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    container_id: str

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def output(self) -> str:
        """Combined stdout + stderr, truncated to 10KB."""
        combined = (self.stdout + "\n" + self.stderr).strip()
        if len(combined) > 10_000:
            combined = combined[:10_000] + "\n... (output truncated)"
        return combined


# Language → sandbox config
_CONFIGS: dict[str, SandboxConfig] = {
    "python": SandboxConfig(
        image="python:3.12-slim",
        run_cmd=["python3", "/sandbox/script.py"],
        file_ext=".py",
    ),
    "bash": SandboxConfig(
        image="alpine:latest",
        run_cmd=["sh", "/sandbox/script.sh"],
        file_ext=".sh",
    ),
    "javascript": SandboxConfig(
        image="node:20-alpine",
        run_cmd=["node", "/sandbox/script.js"],
        file_ext=".js",
    ),
}


class CodeSandbox:
    """
    Executes untrusted code in isolated Docker containers.

    Each execution gets a fresh container that is automatically removed
    after completion or timeout.

    Parameters
    ----------
    docker_available:
        Set False to disable sandbox (raises NotImplementedError on execute).
        Useful for testing or environments without Docker.
    """

    def __init__(self, docker_available: bool = True) -> None:
        self._available = docker_available

    async def execute(
        self,
        code: str,
        language: str = "python",
        timeout: int | None = None,
    ) -> SandboxResult:
        """
        Execute *code* in an isolated Docker container.

        Parameters
        ----------
        code:
            Source code to execute.
        language:
            Language identifier: "python" | "bash" | "javascript"
        timeout:
            Override default timeout in seconds.

        Returns
        -------
        SandboxResult

        Raises
        ------
        NotImplementedError
            If docker_available is False.
        ValueError
            If language is not supported.
        RuntimeError
            If Docker is not reachable on the host.
        """
        if not self._available:
            raise NotImplementedError("Sandbox is disabled (docker_available=False)")

        if language not in _CONFIGS:
            raise ValueError(
                f"Unsupported language: {language}. "
                f"Supported: {sorted(_CONFIGS.keys())}"
            )

        config  = _CONFIGS[language]
        t       = timeout or config.timeout
        cid     = f"paeka-sandbox-{uuid.uuid4().hex[:8]}"

        # Write code to a temporary file on the host, mount into container
        with tempfile.TemporaryDirectory(prefix="paeka_sandbox_") as tmpdir:
            script_path = Path(tmpdir) / f"script{config.file_ext}"
            script_path.write_text(code, encoding="utf-8")

            docker_cmd = [
                "docker", "run",
                "--rm",
                "--name",            cid,
                "--network",         "none",
                "--read-only",
                "--cap-drop",        "ALL",
                "--security-opt",    "no-new-privileges",
                "--memory",          config.memory,
                "--cpus",            config.cpus,
                "--pids-limit",      str(config.pids_limit),
                "--tmpfs",           "/tmp:size=64m,noexec",
                "--volume",          f"{tmpdir}:/sandbox:ro",
                "--workdir",         "/sandbox",
            ]

            # Inject environment variables
            for k, v in config.env.items():
                docker_cmd += ["--env", f"{k}={v}"]

            docker_cmd += [config.image] + config.run_cmd

            return await self._run_docker(docker_cmd, cid, t)

    async def is_available(self) -> bool:
        """Return True if Docker daemon is reachable."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "info",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()
            return proc.returncode == 0
        except FileNotFoundError:
            return False

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    async def _run_docker(
        self, cmd: list[str], cid: str, timeout: int
    ) -> SandboxResult:
        timed_out = False
        proc: asyncio.subprocess.Process | None = None

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(), timeout=float(timeout)
                )
            except asyncio.TimeoutError:
                timed_out = True
                stdout_b, stderr_b = b"", b"Execution timed out"
                logger.warning("Sandbox %s timed out after %ds", cid, timeout)
                # Kill the container
                await asyncio.create_subprocess_exec(
                    "docker", "kill", cid,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )

            exit_code = proc.returncode or 0
            stdout    = stdout_b.decode("utf-8", errors="replace")
            stderr    = stderr_b.decode("utf-8", errors="replace")

        except FileNotFoundError as exc:
            raise RuntimeError(
                "Docker not found. Install Docker and ensure it's running."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("Sandbox execution error: %s", exc)
            stdout, stderr, exit_code = "", str(exc), 1

        logger.debug(
            "Sandbox %s: exit=%d timed_out=%s stdout=%d stderr=%d",
            cid, exit_code, timed_out, len(stdout), len(stderr),
        )

        return SandboxResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            timed_out=timed_out,
            container_id=cid,
        )


# ---------------------------------------------------------------------------
# Singleton for use in routes
# ---------------------------------------------------------------------------

_sandbox: CodeSandbox | None = None


def get_sandbox() -> CodeSandbox:
    global _sandbox
    if _sandbox is None:
        _sandbox = CodeSandbox(docker_available=True)
    return _sandbox
