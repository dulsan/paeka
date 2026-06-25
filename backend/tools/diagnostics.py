"""
backend/tools/diagnostics.py
==============================
Service health diagnostic tool, updated for the Ollama + Qdrant stack
(was originally written against llama-server + Weaviate ports).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


def _split(url: str, default_host: str, default_port: int) -> tuple[str, int]:
    """Best-effort host/port extraction; falls back to the given default."""
    try:
        parsed = urlparse(url)
        return parsed.hostname or default_host, parsed.port or default_port
    except Exception:  # noqa: BLE001
        return default_host, default_port


def _build_services() -> list[dict]:
    """
    Resolve actual configured hosts/ports rather than assuming everything
    is on localhost -- true for the native dev setup, false as soon as
    Ollama/Qdrant/SearXNG move into their own containers reachable only by
    service name.
    """
    try:
        from backend.shared.config import get_settings
        settings = get_settings()
        llm_host, llm_port   = _split(settings.llm.base_url, "localhost", 11434)
        qd_host,  qd_port    = _split(settings.retrieval.qdrant_url, "localhost", 6333)
        sx_host,  sx_port    = _split(settings.tools.searxng_url, "localhost", 8888)
        api_port             = settings.server.port
    except Exception as exc:  # noqa: BLE001
        logger.warning("diagnostics: falling back to localhost defaults (%s)", exc)
        llm_host, llm_port = "localhost", 11434
        qd_host,  qd_port  = "localhost", 6333
        sx_host,  sx_port  = "localhost", 8888
        api_port            = 8000

    return [
        {"name": "Ollama (LLM)",         "host": llm_host, "port": llm_port, "http_path": "/api/tags",  "critical": True},
        {"name": "Qdrant (vector DB)",   "host": qd_host,  "port": qd_port,  "http_path": "/healthz",   "critical": True},
        # The API checking itself is always self-referential -- localhost
        # is correct here even inside a container.
        {"name": "PAEKA API",            "host": "localhost", "port": api_port, "http_path": "/api/health", "critical": False},
        {"name": "SearXNG (web search)", "host": sx_host,  "port": sx_port,  "http_path": "/healthz",   "critical": False},
    ]


Status = Literal["ok", "degraded", "down", "unknown"]


@dataclass
class ServiceStatus:
    name: str
    host: str
    port: int
    status: Status
    latency_ms: float | None = None
    http_code: int | None = None
    error: str | None = None
    critical: bool = False


@dataclass
class DiagnosticReport:
    services: list[ServiceStatus] = field(default_factory=list)

    @property
    def all_critical_ok(self) -> bool:
        return all(s.status == "ok" for s in self.services if s.critical)

    def to_text(self) -> str:
        lines = ["=== PAEKA Service Diagnostic ==="]
        for s in self.services:
            icon = {"ok": "[OK]", "degraded": "[WARN]", "down": "[DOWN]", "unknown": "[?]"}[s.status]
            crit = " (CRITICAL)" if s.critical else ""
            lat  = f" {s.latency_ms:.0f}ms" if s.latency_ms is not None else ""
            err  = f" -- {s.error}" if s.error else ""
            lines.append(f"  {icon} {s.name} (:{s.port}){crit}{lat}{err}")
        if self.all_critical_ok:
            lines.append("\nAll critical services are reachable.")
        else:
            down = [s.name for s in self.services if s.critical and s.status != "ok"]
            lines.append(f"\nCritical services DOWN: {', '.join(down)}")
        return "\n".join(lines)


async def _check_http(host: str, port: int, path: str, timeout: float = 3.0):
    import time
    url = f"http://{host}:{port}{path}"
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url)
        latency = (time.monotonic() - start) * 1000
        if resp.status_code == 200:
            return "ok", resp.status_code, latency, None
        return "degraded", resp.status_code, latency, f"unexpected status {resp.status_code}"
    except httpx.ConnectError as exc:
        return "down", None, None, f"connection refused: {exc}"
    except httpx.TimeoutException:
        return "down", None, None, "timeout"
    except Exception as exc:
        return "unknown", None, None, str(exc)


async def run_diagnostics(timeout: float = 3.0) -> DiagnosticReport:
    async def _probe(svc: dict) -> ServiceStatus:
        status, code, latency, error = await _check_http(
            svc["host"], svc["port"], svc["http_path"], timeout=timeout
        )
        return ServiceStatus(
            name=svc["name"], host=svc["host"], port=svc["port"],
            status=status, latency_ms=latency, http_code=code,
            error=error, critical=svc.get("critical", False),
        )
    services = _build_services()
    results = await asyncio.gather(*[_probe(s) for s in services])
    return DiagnosticReport(services=list(results))


async def check_services(target: str = "all") -> str:
    report = await run_diagnostics()
    if target != "all":
        tgt = target.lower()
        filtered = [s for s in report.services if tgt in s.name.lower()]
        if filtered:
            report.services = filtered
    return report.to_text()
