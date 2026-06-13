"""
backend/security/auth.py
=========================
Request authentication for PAEKA.

Two modes controlled by [deployment] auth_enabled in settings.toml:

  auth_enabled = false  (default, Mode 1 — localhost dev)
    All requests pass through. No credentials required.

  auth_enabled = true   (Mode 2/3 — LAN or internet-facing)
    Every request must include one of:
      - Header:  Authorization: Bearer <PAEKA_AUTH_TOKEN>
      - Header:  X-API-Key: <PAEKA_AUTH_TOKEN>

The token is set via environment variable PAEKA_AUTH__TOKEN.
It is never stored in settings.toml to avoid accidental exposure in git.

For Mode 3 (internet-facing), auth_enabled should always be true.
Caddy sits in front and handles TLS; this layer handles identity.
"""

from __future__ import annotations

import logging
import secrets

from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Paths that bypass authentication (health check must always be reachable
# so the reverse proxy can verify the backend is up)
_PUBLIC_PATHS: frozenset[str] = frozenset({
    "/api/health",
})


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Token-based authentication middleware.

    Checks Authorization: Bearer <token> or X-API-Key: <token> headers.
    Returns 401 on missing credentials, 403 on wrong token.
    """

    def __init__(self, app, token: str, enabled: bool = True) -> None:
        super().__init__(app)
        self._token   = token
        self._enabled = enabled

        if enabled and not token:
            raise ValueError(
                "Auth is enabled but PAEKA_AUTH__TOKEN is not set. "
                "Set it as an environment variable before starting the server."
            )

    async def dispatch(self, request: Request, call_next) -> Response:
        if not self._enabled:
            return await call_next(request)

        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        token = _extract_token(request)

        if token is None:
            logger.warning(
                "Unauthenticated request: %s %s from %s",
                request.method, request.url.path,
                request.client.host if request.client else "unknown",
            )
            return Response(
                content='{"detail":"Authentication required"}',
                status_code=401,
                media_type="application/json",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Constant-time comparison to prevent timing attacks
        if not secrets.compare_digest(token, self._token):
            logger.warning(
                "Invalid token: %s %s from %s",
                request.method, request.url.path,
                request.client.host if request.client else "unknown",
            )
            return Response(
                content='{"detail":"Invalid authentication token"}',
                status_code=403,
                media_type="application/json",
            )

        return await call_next(request)


def _extract_token(request: Request) -> str | None:
    """Extract bearer token from Authorization or X-API-Key header."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()

    api_key = request.headers.get("X-API-Key", "").strip()
    if api_key:
        return api_key

    return None
