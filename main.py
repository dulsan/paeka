"""
main.py
=======
Application entrypoint.

Run development server:
    uv run python main.py

Or via uvicorn directly (recommended — start.ps1 does this automatically):
    uv run uvicorn main:app --host 0.0.0.0 --port 8000 --ws wsproto --http httptools

Changes:
  [FIX-WS]  Suppress websockets.legacy DeprecationWarning emitted at import
             time by uvicorn[standard] even when --ws wsproto is active.
             websockets 14.0 deprecated the legacy ServerProtocol API; uvicorn
             hasn't removed the import yet so the warning fires unconditionally.
  [FIX-HF]  Suppress HuggingFace Hub unauthenticated-request noise for public
             models (bge-m3, bge-reranker-large). Set via env in .env; this
             is a belt-and-suspenders guard for when main.py is invoked directly.
"""

from __future__ import annotations

# ── Warning filters — must come before any other imports ────────────────────
import warnings
import os

# [FIX-WS] websockets.legacy is deprecated in websockets>=14.0.
# uvicorn.protocols.websockets.websockets_impl imports it at module load time
# even when --ws wsproto is selected. This silences the noise without
# downgrading websockets or patching uvicorn.
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"websockets.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"uvicorn\.protocols\.websockets\.websockets_impl")

# [FIX-HF] Silence HuggingFace Hub "unauthenticated request" warnings.
# Public models don't need a token. If you set HF_TOKEN in .env the warning
# disappears automatically; these env vars are a fallback for bare invocations.
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "true")

# ── Application ──────────────────────────────────────────────────────────────
import uvicorn
import typer

from backend.api.app import create_app
from backend.shared.config import get_settings

# WSGI/ASGI application object (used by uvicorn when passed as module:attr)
app = create_app()

# ── CLI ──────────────────────────────────────────────────────────────────────

cli = typer.Typer(name="paeka", add_completion=False)


@cli.command()
def serve(
    host: str = typer.Option(None, help="Override server host"),
    port: int = typer.Option(None, help="Override server port"),
    reload: bool = typer.Option(None, help="Enable hot-reload (dev mode)"),
) -> None:
    """Start the PAEKA FastAPI server."""
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=host or settings.server.host,
        port=port or settings.server.port,
        reload=reload if reload is not None else settings.server.reload,
        log_config=None,  # logging configured by setup_logging() in lifespan
        ws="wsproto",     # avoid websockets.legacy deprecation warning
        http="httptools",
    )


def app_cli() -> None:
    cli()


if __name__ == "__main__":
    serve()
