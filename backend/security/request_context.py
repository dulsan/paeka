"""
backend/security/request_context.py
=====================================
Binds a per-request structlog context so every log line emitted while
handling a request -- stdlib logging.getLogger() or structlog-native,
anywhere in the call stack -- automatically carries request_id without
threading it through every function signature.

Mounted as the outermost middleware in app.py (before auth/rate-limit) so
their own log lines are tagged too.

Route handlers that know a more specific identifier (e.g. conversation_id
in chat.py) call backend.shared.logging.bind_context(conversation_id=...)
themselves once it's available -- this middleware only owns request_id,
which is universal to every request regardless of route.
"""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from backend.shared.logging import bind_context, clear_context, get_logger

logger = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Generates a request_id, binds it for the request's lifetime, logs
    a single structured access line on completion."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = uuid.uuid4().hex[:12]
        clear_context()
        bind_context(request_id=request_id)
        request.state.request_id = request_id

        start = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "request_failed",
                method=request.method,
                path=request.url.path,
                duration_ms=round((time.monotonic() - start) * 1000, 1),
            )
            clear_context()
            raise

        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round((time.monotonic() - start) * 1000, 1),
        )
        response.headers["X-Request-ID"] = request_id
        clear_context()
        return response
