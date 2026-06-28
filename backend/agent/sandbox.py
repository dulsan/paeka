"""
backend/agent/sandbox.py
=========================
Secure code execution sandbox using Docker containers.

Security model (defence-in-depth, all enforced independently -- a gap
in one doesn't remove the others):
  --network=none          no outbound internet access
  --read-only             immutable root filesystem
  --cap-drop=ALL          all Linux capabilities dropped
  --security-opt=no-new-privileges  prevent privilege escalation
  --security-opt=seccomp=sandbox-seccomp.json  trimmed syscall allowlist
                          (see below)
  --user 65534:65534     runs as an unprivileged UID inside the
                          container (the conventional "nobody" UID on
                          most Linux distros), not root-in-container
  --memory / --cpus       resource hard limits
  --pids-limit / --ulimit nproc  two independent process-count limits
                          (cgroup-enforced and rlimit-enforced)
  --ulimit nofile         caps open file descriptors
  tmpfs /tmp              writable scratch space, in-memory only

[FIX] sandbox-seccomp.json: Docker's own default seccomp profile
(verified against github.com/moby/profiles/seccomp/default.json) is
itself already an allowlist of ~440 syscalls -- this file is that exact
profile with the 14 groups gated behind `includes: {caps: [...]}`
removed (CAP_SYS_ADMIN, CAP_SYS_PTRACE, CAP_SYS_MODULE, CAP_SYS_BOOT,
CAP_SYS_CHROOT, CAP_SYS_RAWIO, CAP_SYS_TIME, CAP_SYS_NICE,
CAP_SYS_TTY_CONFIG, CAP_SYS_PACCT, CAP_SYSLOG, CAP_BPF, CAP_PERFMON,
CAP_DAC_READ_SEARCH -- ~53 syscalls total, including mount/umount2/
unshare/setns/ptrace/process_vm_readv/init_module/reboot/chroot).
Since --cap-drop=ALL means none of those capabilities are ever held,
every one of those syscalls was already unreachable -- removing them
from the seccomp allowlist too changes nothing about what a normal
python3/node/sh script can do, but adds a second, independent
enforcement layer: a future change that accidentally re-adds a
capability (e.g. someone debugging adds --cap-add SYS_PTRACE and
forgets to remove it) still can't reach these syscalls, because
seccomp blocks them regardless of what capabilities are held. Built
mechanically from the verified upstream file, not hand-written, to
avoid guessing at a from-scratch allowlist and accidentally breaking
normal script execution.

[FIX] --user: nothing previously set this, so sandboxed code ran as
root *inside* the container (still fully constrained by everything
else above, but root-in-container is a meaningfully larger attack
surface than an unprivileged UID for the same escape attempt). 65534 is
the conventional "nobody" UID/GID on Debian/Alpine alike -- used as a
raw number rather than a name, so it doesn't depend on each base image
defining that name in its own /etc/passwd.

[FIX] scratch_dir / scratch_volume: when paeka-api itself runs inside
Docker Compose (sibling containers via the Docker socket -- see
SETUP_DOCKER.md section 4), its own filesystem isn't the real Docker
host's filesystem. A plain tempfile.TemporaryDirectory() path can't be
bind-mounted into the sandbox container in that setup, because the
daemon resolves that path against the actual host, not against
paeka-api's own container, and finds nothing there. Setting both of
these together switches to writing scripts into a Docker *named
volume* instead, referenced by name (not path) -- resolved by the
daemon itself, identically regardless of host OS or path translation.
Leaving both unset (the default) keeps the simpler, original behaviour
for native deployments, where this problem doesn't exist.

Supported languages:
  Python   -> python:3.12-slim image
  Bash     -> alpine image
  Node     -> node:20-alpine image
  (extensible via SandboxConfig)
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Resource limits for sandbox containers
_DEFAULT_MEMORY  = "256m"
_DEFAULT_CPUS    = "1.0"
_DEFAULT_TIMEOUT = 30       # seconds before SIGKILL
_DEFAULT_PIDS    = 64       # max processes inside container (cgroup-enforced)

# [FIX] Resolved relative to this file, not cwd -- correct in both
# deployment modes: native (file lives in the real checkout) and Docker
# (Dockerfile's `COPY backend/ ./backend/` preserves the same relative
# layout). Docker's CLI reads this file's *content* client-side and
# sends it inline to the daemon -- no daemon-side filesystem access
# needed, so this works through the docker-socket-proxy too.
_SECCOMP_PROFILE_PATH = Path(__file__).parent / "sandbox-seccomp.json"

# Conventional "nobody" UID/GID on Debian and Alpine alike. Used as a
# raw number (not a name) so it doesn't depend on each base image
# defining that name in its own /etc/passwd.
_SANDBOX_UID_GID = "65534:65534"


@dataclass
class SandboxConfig:
    """Per-language sandbox configuration."""
    image: str
    run_cmd: list[str]      # interpreter command only -- the resolved
                             # script path is appended at call time, since
                             # it differs between the native tempdir path
                             # and the named-volume (DooD) path
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
        run_cmd=["python3"],
        file_ext=".py",
    ),
    "bash": SandboxConfig(
        image="alpine:latest",
        run_cmd=["sh"],
        file_ext=".sh",
    ),
    "javascript": SandboxConfig(
        image="node:20-alpine",
        run_cmd=["node"],
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
    memory, cpus, default_timeout:
        Override the hardcoded per-language defaults in _CONFIGS. None
        (the default) keeps each language's own default. get_sandbox()
        passes these from [sandbox] settings (memory_limit, cpu_limit,
        default_timeout) so PAEKA_SANDBOX__* env vars actually take
        effect -- previously these settings existed but nothing read them.
    scratch_dir, scratch_volume:
        Set both together to use the named-volume path (for paeka-api
        running inside Docker Compose -- see module docstring). Leave
        both None (the default) for native deployments.
    """

    def __init__(
        self,
        docker_available: bool = True,
        memory: str | None = None,
        cpus: str | None = None,
        default_timeout: int | None = None,
        scratch_dir: str | None = None,
        scratch_volume: str | None = None,
    ) -> None:
        self._available = docker_available
        self._memory = memory
        self._cpus = cpus
        self._default_timeout = default_timeout
        self._scratch_dir = scratch_dir
        self._scratch_volume = scratch_volume

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

        config   = _CONFIGS[language]
        memory   = self._memory or config.memory
        cpus     = self._cpus or config.cpus
        t        = timeout or self._default_timeout or config.timeout
        cid      = f"paeka-sandbox-{uuid.uuid4().hex[:8]}"

        if self._scratch_dir and self._scratch_volume:
            # Sibling-container-safe path -- see module docstring. Each
            # execution gets its own subdirectory so concurrent runs
            # don't see each other's scripts, explicitly removed in the
            # finally block once the container has exited (the volume
            # itself is long-lived and shared, unlike a TemporaryDirectory).
            run_id  = uuid.uuid4().hex[:12]
            run_dir = Path(self._scratch_dir) / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            try:
                (run_dir / f"script{config.file_ext}").write_text(code, encoding="utf-8")
                volume_arg    = f"{self._scratch_volume}:/sandbox:ro"
                script_in_box = f"/sandbox/{run_id}/script{config.file_ext}"
                return await self._run_in_container(
                    volume_arg, script_in_box, config, memory, cpus, cid, t,
                )
            finally:
                shutil.rmtree(run_dir, ignore_errors=True)

        # Native deployment: paeka-api's own tempdir *is* the real host
        # filesystem, so the simple bind-mount-by-path approach just works.
        with tempfile.TemporaryDirectory(prefix="paeka_sandbox_") as tmpdir:
            (Path(tmpdir) / f"script{config.file_ext}").write_text(code, encoding="utf-8")
            volume_arg    = f"{tmpdir}:/sandbox:ro"
            script_in_box = f"/sandbox/script{config.file_ext}"
            return await self._run_in_container(
                volume_arg, script_in_box, config, memory, cpus, cid, t,
            )

    async def _run_in_container(
        self,
        volume_arg: str,
        script_in_box: str,
        config: SandboxConfig,
        memory: str,
        cpus: str,
        cid: str,
        timeout: int,
    ) -> SandboxResult:
        docker_cmd = [
            "docker", "run",
            "--rm",
            "--name",            cid,
            "--network",         "none",
            "--read-only",
            "--cap-drop",        "ALL",
            "--security-opt",    "no-new-privileges",
            "--user",            _SANDBOX_UID_GID,
            "--memory",          memory,
            "--cpus",            cpus,
            "--pids-limit",      str(config.pids_limit),
            "--ulimit",          f"nproc={config.pids_limit}",
            "--ulimit",          "nofile=256:512",
            "--tmpfs",           "/tmp:size=64m,noexec",
            "--volume",          volume_arg,
            "--workdir",         "/sandbox",
        ]

        if _SECCOMP_PROFILE_PATH.is_file():
            docker_cmd += ["--security-opt", f"seccomp={_SECCOMP_PROFILE_PATH}"]
        else:
            logger.warning(
                "Sandbox: seccomp profile not found at %s -- using Docker's "
                "default profile instead. Sandbox still runs, just with a "
                "slightly wider syscall allowlist.",
                _SECCOMP_PROFILE_PATH,
            )

        # Inject environment variables
        for k, v in config.env.items():
            docker_cmd += ["--env", f"{k}={v}"]

        docker_cmd += [config.image] + config.run_cmd + [script_in_box]

        return await self._run_docker(docker_cmd, cid, timeout)

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
        # Lazy import: keeps sandbox.py's own top-level imports stdlib-only
        # (see backend/agent/__init__.py for why that matters), while still
        # letting the one real caller path -- actually running code --
        # pick up [sandbox] settings from settings.toml / PAEKA_SANDBOX__*.
        from backend.shared.config import get_settings
        cfg = get_settings().sandbox
        _sandbox = CodeSandbox(
            docker_available=True,
            memory=cfg.memory_limit,
            cpus=cfg.cpu_limit,
            default_timeout=cfg.default_timeout,
            scratch_dir=cfg.scratch_dir,
            scratch_volume=cfg.scratch_volume,
        )
    return _sandbox
