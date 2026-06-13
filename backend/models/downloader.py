"""
backend/models/downloader.py
=============================
GGUF model downloader with progress reporting and hash verification.

Downloads directly from HuggingFace using their raw file URLs —
no HuggingFace Hub library required, just httpx.

Usage:
    uv run python scripts/download_model.py \\
        --repo unsloth/Qwen3.6-35B-A3B-MTP-GGUF \\
        --file Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \\
        --dest models/qwen/

Hash verification:
  If a SHA-256 hash is provided, the downloaded file is verified.
  HuggingFace provides hashes in the model card or via the API.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path
from typing import Callable

import httpx

logger = logging.getLogger(__name__)

_HF_ENDPOINT = "https://huggingface.co"
_CHUNK_SIZE  = 8 * 1024 * 1024   # 8 MB


async def download_gguf(
    repo_id: str,
    filename: str,
    dest_dir: str | Path,
    expected_sha256: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """
    Download a GGUF file from HuggingFace.

    Parameters
    ----------
    repo_id:
        HuggingFace repository id, e.g. "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
    filename:
        File to download, e.g. "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    dest_dir:
        Local directory to save into.
    expected_sha256:
        Optional SHA-256 hex digest for verification.
    progress_callback:
        Optional callable(bytes_downloaded, total_bytes) for progress reporting.

    Returns
    -------
    Path
        Path to the downloaded file.

    Raises
    ------
    httpx.HTTPStatusError
        If the download request fails.
    ValueError
        If hash verification fails.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / filename

    url = f"{_HF_ENDPOINT}/{repo_id}/resolve/main/{filename}"

    # Skip if already downloaded and verified
    if dest_file.exists():
        if expected_sha256:
            logger.info("Verifying existing file: %s", dest_file.name)
            actual = _sha256(dest_file)
            if actual == expected_sha256.lower():
                logger.info("Hash verified — skipping download: %s", filename)
                return dest_file
            else:
                logger.warning(
                    "Hash mismatch for existing %s — re-downloading.", filename
                )
        else:
            logger.info("File already exists — skipping download: %s", filename)
            return dest_file

    logger.info("Downloading: %s → %s", url, dest_file)

    tmp_file = dest_file.with_suffix(".gguf.part")
    hasher = hashlib.sha256() if expected_sha256 else None
    downloaded = 0
    total = 0

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                if total:
                    logger.info("File size: %.1f GB", total / 1e9)

                with tmp_file.open("wb") as fh:
                    async for chunk in resp.aiter_bytes(_CHUNK_SIZE):
                        fh.write(chunk)
                        if hasher:
                            hasher.update(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total:
                            progress_callback(downloaded, total)
                        elif total:
                            pct = downloaded / total * 100
                            logger.info(
                                "Download progress: %.1f%% (%.1f / %.1f GB)",
                                pct, downloaded / 1e9, total / 1e9,
                            )

    except Exception:
        tmp_file.unlink(missing_ok=True)
        raise

    # Verify hash
    if expected_sha256 and hasher:
        actual = hasher.hexdigest()
        if actual != expected_sha256.lower():
            tmp_file.unlink(missing_ok=True)
            raise ValueError(
                f"SHA-256 mismatch for {filename}:\n"
                f"  expected: {expected_sha256}\n"
                f"  actual:   {actual}"
            )
        logger.info("SHA-256 verified: %s", filename)

    tmp_file.rename(dest_file)
    logger.info("Download complete: %s (%.1f GB)", dest_file.name, dest_file.stat().st_size / 1e9)
    return dest_file


def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
