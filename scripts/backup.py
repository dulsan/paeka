#!/usr/bin/env python3
"""
scripts/backup.py
==================
Backup all PAEKA data to a timestamped .tar.gz archive.

What is backed up:
  - database/sqlite/paeka.db          (conversations, documents, memory, KG)
  - database/weaviate/                (Weaviate vector data)
  - config/settings.toml              (configuration, NOT .env)
  - backend/skills/definitions/       (custom skill definitions)

What is NOT backed up:
  - .env                              (contains secrets — back up separately)
  - data/hf_cache/                    (model weights — re-downloadable)
  - data/uploads/                     (source files — keep originals elsewhere)
  - .venv/                            (recreatable with uv sync)

Usage:
    uv run python scripts/backup.py
    uv run python scripts/backup.py --output /path/to/backups/
    uv run python scripts/backup.py --list        (list existing backups)
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

_DEFAULT_BACKUP_DIR = Path("backups")

_INCLUDE_PATHS = [
    Path("database/sqlite"),
    Path("database/weaviate"),
    Path("config/settings.toml"),
    Path("backend/skills/definitions"),
]


def backup(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive = output_dir / f"paeka_backup_{ts}.tar.gz"

    print(f"Creating backup: {archive}")

    with tarfile.open(archive, "w:gz") as tar:
        for path in _INCLUDE_PATHS:
            if path.exists():
                tar.add(path, arcname=str(path))
                print(f"  + {path}")
            else:
                print(f"  - {path} (not found, skipping)")

    size_mb = archive.stat().st_size / 1_048_576
    print(f"\nBackup complete: {archive} ({size_mb:.1f} MB)")
    return archive


def list_backups(backup_dir: Path) -> None:
    if not backup_dir.exists():
        print(f"No backup directory found at {backup_dir}")
        return
    archives = sorted(backup_dir.glob("paeka_backup_*.tar.gz"))
    if not archives:
        print("No backups found.")
        return
    print(f"Backups in {backup_dir}:")
    for a in archives:
        size_mb = a.stat().st_size / 1_048_576
        print(f"  {a.name}  ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backup PAEKA data")
    parser.add_argument("--output", type=Path, default=_DEFAULT_BACKUP_DIR,
                        help=f"Backup directory (default: {_DEFAULT_BACKUP_DIR})")
    parser.add_argument("--list", action="store_true",
                        help="List existing backups and exit")
    args = parser.parse_args()

    if args.list:
        list_backups(args.output)
        return

    backup(args.output)


if __name__ == "__main__":
    main()
