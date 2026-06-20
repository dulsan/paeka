"""
backend/shared/logging.py
=========================
Configures structured logging using the standard library + Rich handler.
Call ``setup_logging()`` once at application start.
"""

from __future__ import annotations

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True)


def setup_logging(level: str = "INFO", fmt: str = "rich") -> None:
    """
    Configure root logger.

    Parameters
    ----------
    level:
        Standard log level string (DEBUG, INFO, WARNING, ERROR).
    fmt:
        ``"rich"`` for coloured console output, ``"json"`` for structured
        single-line JSON (useful when piping to log aggregators).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    if fmt == "rich":
        handler: logging.Handler = RichHandler(
            console=_console,
            rich_tracebacks=True,
            markup=True,
            show_path=False,
        )
        formatter = logging.Formatter("%(message)s", datefmt="[%X]")
    else:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '{"time":"%(asctime)s","level":"%(levelname)s",'
            '"logger":"%(name)s","msg":"%(message)s"}',
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric_level)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # [FIX] huggingface_hub emits the "unauthenticated requests" warning via
    # TWO independent paths for the same condition: a direct warnings.warn()
    # (prints raw to stderr, no prefix -- not something this logger config
    # can touch, since it bypasses the logging module entirely) AND its own
    # internal logger.warning() call (the second, [HH:MM:SS]-prefixed copy
    # seen in the terminal). Setting this logger's level to ERROR removes
    # the second copy, leaving one instead of two.
    #
    # The actual root-cause fix -- as the warning itself states -- is
    # providing a real HF_TOKEN (a free, anonymous "read" token is enough;
    # bge-m3 and bge-reranker-large are public models, no paid access
    # needed). That removes the underlying condition entirely rather than
    # just quieting one of its two output paths. Add HF_TOKEN=hf_... to
    # .env if you want it gone outright rather than just deduplicated.
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
