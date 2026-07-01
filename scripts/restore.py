#!/usr/bin/env python3
"""
scripts/restore.py
===================
Restore PAEKA data from a backup archive created by backup.py.

Usage:
    uv run python scripts/restore.py backups/paeka_backup_20250101_120000.tar.gz
    uv run python scripts/restore.py backups/paeka_backup_20250101_120000.tar.gz --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
from pathlib import Path


def restore(archive: Path, dry_run: bool = False) -> None:
    if not archive.exists():
        print(f"Archive not found: {archive}", file=sys.stderr)
        sys.exit(1)

    print(f"{'[DRY RUN] ' if dry_run else ''}Restoring from: {archive}")
    print()

    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        print(f"Contents ({len(members)} files):")
        for m in members[:20]:
            print(f"  {m.name}")
        if len(members) > 20:
            print(f"  ... and {len(members) - 20} more")
        print()

        if dry_run:
            print("[DRY RUN] No files written.")
            return

        # Prompt before overwriting
        confirm = input("Restore will overwrite existing data. Continue? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return

        # Backup current data before overwriting
        dest_backup = Path("database/_pre_restore_backup")
        if Path("database/sqlite").exists() or Path("database/weaviate").exists():
            print(f"Saving current database to {dest_backup}/ ...")
            dest_backup.mkdir(parents=True, exist_ok=True)
            for src in [Path("database/sqlite"), Path("database/weaviate")]:
                if src.exists():
                    shutil.copytree(src, dest_backup / src.name, dirs_exist_ok=True)

        tar.extractall(path=Path("."))
        print("\nRestore complete.")
        print(f"Previous data saved to: {dest_backup}/")
        print("\nRestart PAEKA to apply the restored data:")
        print("  uv run python main.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore PAEKA from backup")
    parser.add_argument("archive", type=Path, help="Path to backup .tar.gz archive")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be restored without writing files")
    args = parser.parse_args()
    restore(args.archive, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
