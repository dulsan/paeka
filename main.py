"""
main.py
=======
PAEKA application entrypoint.

Server: Granian (Rust-based ASGI, native WebSocket support, no dependency
on the 'websockets' Python package -- see pyproject.toml for why uvicorn
was dropped from our own deps; it's still present in the venv only because
the mcp package declares it as its own hard dependency for internal tooling,
unrelated to how PAEKA actually serves requests).

Run:
    uv run python main.py
    uv run python main.py serve --host 0.0.0.0 --port 8000   (CLI form, via Typer)

[FIX] Bug: calling a @cli.command()-decorated function directly as a plain
Python function (serve()) bypasses Typer/Click's argument resolution
entirely. The function's parameters keep their raw, unresolved defaults --
which are typer.Option(...) calls themselves (OptionInfo instances), not
the values they're supposed to resolve to (None). Typer only substitutes
the real default in when the function is invoked through cli()'s actual
CLI dispatch, never on a bare direct call.

`if __name__ == "__main__": serve()` was exactly that direct call, so
`host` was literally an OptionInfo object, not None. `host or
settings.server.host` then evaluated to the OptionInfo (truthy, so `or`
never reached the real fallback), and Granian's SocketSpec construction
failed trying to cast it to a string.

Fix: the actual serving logic lives in _run_server(), a plain function
with ordinary Python defaults (None). Both the __main__ block and the
Typer CLI command call into it -- the CLI command still declares its
typer.Option() parameters (so `paeka serve --host ...` keeps working via
proper Click dispatch), but immediately hands off real resolved values to
_run_server() rather than ever using the unresolved defaults itself.
"""

from __future__ import annotations

import typer

from backend.api.app import create_app
from backend.shared.config import get_settings

# ASGI application object - referenced by Granian as "main:app"
app = create_app()

cli = typer.Typer(name="paeka", add_completion=False)


def _run_server(host: str | None = None, port: int | None = None) -> None:
    """
    Actual serving logic, with ordinary Python defaults.

    Safe to call directly (e.g. from `if __name__ == "__main__":`) without
    going through Typer/Click at all -- host=None and port=None here are
    real None values, not OptionInfo sentinels, so `host or settings...`
    falls through to the configured default correctly either way.
    """
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


@cli.command()
def serve(
    host: str = typer.Option(None, help="Override server host"),
    port: int = typer.Option(None, help="Override server port"),
) -> None:
    """Start the PAEKA server via Granian (CLI entrypoint: `paeka serve`)."""
    # By the time Click/Typer calls this, host/port are already resolved
    # real values (str/int or None) -- never OptionInfo objects. Safe to
    # hand straight to _run_server().
    _run_server(host=host, port=port)


def app_cli() -> None:
    cli()


if __name__ == "__main__":
    # Direct call, no Typer/Click dispatch involved -- _run_server()'s own
    # plain defaults (None, None) are used here, not serve()'s OptionInfo
    # defaults. This is the line that was broken before.
    _run_server()
