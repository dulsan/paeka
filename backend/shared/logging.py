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
