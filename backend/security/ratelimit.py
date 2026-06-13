"""
backend/security/ratelimit.py
==============================
Simple in-process token-bucket rate limiter.

Protects against:
  - Accidental runaway clients hammering the chat endpoint
  - Brute-force token guessing against the auth layer
  - Abuse on Mode 3 (internet-facing) deployments

Limits are per client IP, configurable per endpoint group:
  - /api/chat endpoints    : tighter (LLM calls are expensive)
  - /api/documents/upload  : tight   (disk + ingestion cost)
  - everything else        : loose   (metadata reads, health)

Implementation:
  - Token bucket per IP, refilled at a fixed rate.
  - State is in-process (not Redis). This is single-user, single-process —
    a distributed rate limiter would be over-engineering.
  - No dependencies beyond the standard library + FastAPI.

Disabled when rate_limit_enabled = false in settings.toml (default for
localhost mode where rate limiting would be annoying during development).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


@dataclass
class _Bucket:
    """Token bucket for one IP address."""
    tokens: float
    last_refill: float = field(default_factory=time.monotonic)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Per-IP token bucket rate limiter.

    Parameters
    ----------
    app:
        The ASGI application.
    enabled:
        False = completely bypassed (dev/localhost mode).
    chat_rpm:
        Max requests-per-minute to /api/conversations/*/chat.
    upload_rpm:
        Max requests-per-minute to /api/documents/upload.
    default_rpm:
        Max requests-per-minute for all other routes.
    """

    def __init__(
        self,
        app,
        enabled: bool = False,
        chat_rpm: int = 20,
        upload_rpm: int = 10,
        default_rpm: int = 120,
    ) -> None:
        super().__init__(app)
        self._enabled     = enabled
        self._chat_rpm    = chat_rpm
        self._upload_rpm  = upload_rpm
        self._default_rpm = default_rpm
        # {ip: {route_group: Bucket}}
        self._buckets: dict[str, dict[str, _Bucket]] = defaultdict(dict)
        self._lock = asyncio.Lock()

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self._enabled:
            return await call_next(request)

        ip = _client_ip(request)
        group, rpm = self._classify(request)

        allowed = await self._check(ip, group, rpm)
        if not allowed:
            logger.warning(
                "Rate limit exceeded: %s %s from %s",
                request.method, request.url.path, ip,
            )
            return Response(
                content='{"detail":"Rate limit exceeded. Please slow down."}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": "60"},
            )

        return await call_next(request)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify(self, request: Request) -> tuple[str, int]:
        path = request.url.path
        if "/chat" in path and request.method == "POST":
            return "chat", self._chat_rpm
        if "/documents/upload" in path and request.method == "POST":
            return "upload", self._upload_rpm
        return "default", self._default_rpm

    async def _check(self, ip: str, group: str, rpm: int) -> bool:
        """Return True if the request is within the rate limit."""
        rate_per_second = rpm / 60.0
        now = time.monotonic()

        async with self._lock:
            if group not in self._buckets[ip]:
                self._buckets[ip][group] = _Bucket(tokens=float(rpm))

            bucket = self._buckets[ip][group]
            elapsed = now - bucket.last_refill
            bucket.tokens = min(float(rpm), bucket.tokens + elapsed * rate_per_second)
            bucket.last_refill = now

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True
            return False


def _client_ip(request: Request) -> str:
    """Extract real client IP, respecting X-Forwarded-For from Caddy."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"
