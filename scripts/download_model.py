#!/usr/bin/env python3
"""
scripts/download_model.py
==========================
Download a GGUF model from HuggingFace into the local models directory.

Usage:
    # Download the recommended model for 8GB VRAM + 32GB RAM
    uv run python scripts/download_model.py \\
        --repo  unsloth/Qwen3.6-35B-A3B-MTP-GGUF \\
        --file  Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \\
        --dest  models/qwen

    # With hash verification
    uv run python scripts/download_model.py \\
        --repo  unsloth/Qwen3.6-35B-A3B-MTP-GGUF \\
        --file  Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \\
        --dest  models/qwen \\
        --sha256 <hash-from-model-card>

    # List models already in ./models
    uv run python scripts/download_model.py --list
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path


def _progress_bar(downloaded: int, total: int) -> None:
    if total == 0:
        return
    pct = downloaded / total * 100
    filled = int(pct / 2)
    bar = "█" * filled + "░" * (50 - filled)
    dl_gb = downloaded / 1e9
    tot_gb = total / 1e9
    print(
        f"\r  [{bar}] {pct:5.1f}%  {dl_gb:.2f}/{tot_gb:.2f} GB",
        end="", flush=True,
    )
    if downloaded >= total:
        print()


async def _download(repo_id: str, filename: str, dest: str, sha256: str) -> None:
    from backend.models.downloader import download_gguf

    print(f"\nDownloading {filename}")
    print(f"  from: {repo_id}")
    print(f"  to:   {dest}/\n")

    start = time.monotonic()
    path = await download_gguf(
        repo_id=repo_id,
        filename=filename,
        dest_dir=dest,
        expected_sha256=sha256 or None,
        progress_callback=_progress_bar,
    )
    elapsed = time.monotonic() - start
    size_gb = path.stat().st_size / 1e9
    print(f"\nDone in {elapsed:.0f}s — {size_gb:.1f} GB saved to {path}")
    print(
        "\nNext steps:"
        "\n  1. Update PAEKA_LLM__MODEL_PATH in .env"
        "\n  2. Restart PAEKA: uv run python main.py"
        "\n  3. curl http://localhost/api/models/active"
    )


def _list_models(models_dir: str) -> None:
    from backend.models.registry import ModelRegistry
    registry = ModelRegistry(models_dir)
    models = registry.scan()
    if not models:
        print(f"No models found in {models_dir}/")
        return
    print(f"Models in {models_dir}/:")
    for m in models:
        print(
            f"  {m.name:<50} {m.size_gb:>6.1f} GB  "
            f"quant={m.quantisation or '?':<8} "
            f"ctx={m.context}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a GGUF model for PAEKA")
    parser.add_argument("--repo",   help="HuggingFace repo id")
    parser.add_argument("--file",   help="Filename to download (.gguf)")
    parser.add_argument("--dest",   default="models/qwen", help="Destination directory")
    parser.add_argument("--sha256", default="", help="Expected SHA-256 hash (optional)")
    parser.add_argument("--list",   action="store_true", help="List available models")
    parser.add_argument("--models-dir", default="models", help="Models root directory")
    args = parser.parse_args()

    if args.list:
        _list_models(args.models_dir)
        return

    if not args.repo or not args.file:
        parser.error("--repo and --file are required (or use --list)")

    asyncio.run(_download(args.repo, args.file, args.dest, args.sha256))


if __name__ == "__main__":
    main()
