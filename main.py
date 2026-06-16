"""
main.py
=======
PAEKA application entrypoint.

Server: Granian (Rust-based ASGI, native WebSocket support)
  - No 'websockets' package dependency -> no websockets.legacy warning at all
  - No wsproto, no httptools needed as separate installs
  - Single dependency replaces uvicorn + wsproto + httptools + websockets

Run:
    uv run python main.py
    uv run granian --interface asgi --host 0.0.0.0 --port 8000 main:app

HuggingFace warnings (symlinks, unauthenticated requests) are set in .env
and loaded by start_fixed.ps1 before this process starts. No env var
overrides in Python code needed.
"""

from __future__ import annotations

import typer

from backend.api.app import create_app
from backend.shared.config import get_settings

# ASGI application object - referenced by granian as "main:app"
app = create_app()

cli = typer.Typer(name="paeka", add_completion=False)


@cli.command()
def serve(
    host: str = typer.Option(None, help="Override server host"),
    port: int = typer.Option(None, help="Override server port"),
) -> None:
    """Start the PAEKA server via Granian."""
    from granian import Granian

    settings = get_settings()

    Granian(
        target="main:app",
        address=host or settings.server.host,
        port=port or settings.server.port,
        interface="asgi",
        workers=1,
        websockets=True,
        log_enabled=True,
        log_level="info",
    ).serve()


def app_cli() -> None:
    cli()


if __name__ == "__main__":
    serve()
