#!/usr/bin/env python3
"""
scripts/gen_token.py
=====================
Generate a cryptographically secure API token and write it to .env.

Usage:
    uv run python scripts/gen_token.py
    uv run python scripts/gen_token.py --length 48
    uv run python scripts/gen_token.py --dry-run   (print only, don't write)
"""

from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

_ENV_FILE = Path(".env")
_EXAMPLE  = Path(".env.example")
_TOKEN_KEY = "PAEKA_AUTH__TOKEN"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a PAEKA API token")
    parser.add_argument("--length", type=int, default=32,
                        help="Token byte length (default: 32 → 43-char URL-safe string)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print token without writing to .env")
    args = parser.parse_args()

    token = secrets.token_urlsafe(args.length)

    if args.dry_run:
        print(f"{_TOKEN_KEY}={token}")
        return

    # Create .env from example if it doesn't exist
    if not _ENV_FILE.exists():
        if _EXAMPLE.exists():
            _ENV_FILE.write_text(_EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"Created .env from .env.example")
        else:
            _ENV_FILE.write_text("", encoding="utf-8")
            print(f"Created empty .env")

    env_text = _ENV_FILE.read_text(encoding="utf-8")

    if f"{_TOKEN_KEY}=" in env_text:
        # Replace existing token line
        lines = env_text.splitlines()
        updated = []
        for line in lines:
            if line.startswith(f"{_TOKEN_KEY}="):
                updated.append(f"{_TOKEN_KEY}={token}")
                print(f"Updated {_TOKEN_KEY} in .env")
            else:
                updated.append(line)
        _ENV_FILE.write_text("\n".join(updated) + "\n", encoding="utf-8")
    else:
        # Append token
        _ENV_FILE.write_text(env_text.rstrip() + f"\n{_TOKEN_KEY}={token}\n", encoding="utf-8")
        print(f"Added {_TOKEN_KEY} to .env")

    print(f"\nToken: {token}")
    print(f"\nTo authenticate API requests:")
    print(f'  curl -H "Authorization: Bearer {token}" http://localhost/api/health')
    print(f"\nRemember to set PAEKA_AUTH__ENABLED=true in .env")


if __name__ == "__main__":
    main()
