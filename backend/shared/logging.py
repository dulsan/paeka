"""
backend/shared/logging.py
=========================
Structured logging via structlog, with full stdlib compatibility.

Call ``setup_logging()`` once at application start (backend/api/app.py,
step 0). After that:

  - Existing code is untouched: every `logging.getLogger(__name__)` call
    site across the codebase keeps working exactly as before -- stdlib
    LogRecords are routed through structlog's ProcessorFormatter so they
    get the same structured rendering as everything else.
  - New code can call `get_logger(__name__)` from this module instead for
    a structlog-native logger that supports `.bind()` and keyword-argument
    structured fields directly: `log.info("tool_call", tool=name, ms=42)`.
  - Request-scoped context (conversation_id, request_id, etc.) is bound via
    `bind_context(**kw)` / `clear_context()` at request boundaries (see
    backend/security/auth.py's RequestContextMiddleware) using structlog's
    contextvars support, so it automatically attaches to every log line --
    stdlib or structlog-native -- emitted while handling that request,
    without threading extra parameters through every function signature.
"""

from __future__ import annotations

import logging
import sys

import structlog

_configured = False


def setup_logging(level: str = "INFO", fmt: str = "console") -> None:
    """
    Configure structlog + stdlib logging to work as a single pipeline.

    Parameters
    ----------
    level:
        Standard log level string (DEBUG, INFO, WARNING, ERROR).
    fmt:
        ``"console"`` for human-readable coloured dev output (structlog's
        own ConsoleRenderer -- no extra dependency), ``"json"`` for
        structured single-line JSON (log aggregators, production).
        ``"rich"`` is accepted as a legacy alias for ``"console"``.
    """
    global _configured
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    if fmt == "rich":  # legacy config value from the pre-structlog setup
        fmt = "console"

    # Shared pre-processing chain -- runs for BOTH stdlib LogRecords (via
    # ProcessorFormatter.wrap_for_formatter) and native structlog loggers.
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if fmt == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    # -- stdlib side: every `logging.getLogger(__name__)` call site in the
    # codebase keeps working unchanged. ProcessorFormatter intercepts the
    # final formatting step so stdlib records get the same renderer.
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric_level)

    # -- structlog side: native structlog.get_logger() callers.
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # [FIX] huggingface_hub emits the "unauthenticated requests" warning via
    # TWO independent paths for the same condition: a direct warnings.warn()
    # (prints raw to stderr, no prefix -- not something this logger config
    # can touch, since it bypasses the logging module entirely) AND its own
    # internal logger.warning() call (the second, structured copy seen in
    # the terminal). Setting this logger's level to ERROR removes the
    # second copy, leaving one instead of two.
    #
    # The actual root-cause fix -- as the warning itself states -- is
    # providing a real HF_TOKEN (a free, anonymous "read" token is enough;
    # bge-m3 and bge-reranker-large are public models, no paid access
    # needed). That removes the underlying condition entirely rather than
    # just quieting one of its two output paths. Add HF_TOKEN=hf_... to
    # .env if you want it gone outright rather than just deduplicated.
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Return a structlog-native logger for new code.

    Supports `.bind()` and structured keyword fields:
        log = get_logger(__name__)
        log.info("tool_call", tool="qdrant_search", latency_ms=42)

    Safe to call before setup_logging() runs (e.g. at module import time) --
    structlog buffers via its default config until configure() is called.
    """
    return structlog.get_logger(name)


def bind_context(**kwargs: object) -> None:
    """
    Bind key/value pairs into the current request's context. Every log
    line emitted on this thread/task afterward (stdlib or structlog-native)
    automatically includes these fields, until clear_context() is called.

    Call at request boundaries, e.g.:
        bind_context(request_id=request_id)
        bind_context(conversation_id=conversation_id)
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all bound context vars. Call at the end of a request."""
    structlog.contextvars.clear_contextvars()
