"""
backend/api/routes/models.py
=============================
Model management endpoints.

GET  /api/models                — list all discovered GGUF models
POST /api/models/scan           — re-scan models directory
GET  /api/models/active         — currently configured model
POST /api/models/download       — trigger async GGUF download from HuggingFace
GET  /api/models/download/status — check download progress
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel

from backend.shared.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["models"])

# In-process download status tracker (single user — no persistence needed)
_download_status: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ModelOut(BaseModel):
    name: str
    filename: str
    path: str
    family: str
    parameters: str
    quantisation: str
    context: int
    chat_template: str
    size_gb: float
    exists: bool
    description: str


class ActiveModelOut(BaseModel):
    provider: str
    model: str
    model_path: str
    base_url: str


class ScanResponse(BaseModel):
    found: int
    models: list[ModelOut]


class DownloadRequest(BaseModel):
    repo_id: str       # e.g. "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
    filename: str      # e.g. "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    dest_subdir: str   # subdirectory under models/, e.g. "qwen"
    sha256: str = ""   # optional hash verification


class DownloadStatusOut(BaseModel):
    key: str
    status: str        # "pending" | "downloading" | "done" | "error"
    filename: str
    bytes_downloaded: int
    total_bytes: int
    percent: float
    message: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/models", response_model=list[ModelOut])
async def list_models(request: Request) -> list[ModelOut]:
    """List all GGUF models discovered in the models directory."""
    registry = request.app.state.model_registry
    models = registry.list_models()
    return [_to_out(m) for m in models]


@router.post("/models/scan", response_model=ScanResponse)
async def scan_models(request: Request) -> ScanResponse:
    """Re-scan the models directory and refresh the registry."""
    registry = request.app.state.model_registry
    models = registry.scan()
    return ScanResponse(found=len(models), models=[_to_out(m) for m in models])


@router.get("/models/active", response_model=ActiveModelOut)
async def active_model() -> ActiveModelOut:
    """Return the currently configured model and provider."""
    settings = get_settings()
    return ActiveModelOut(
        provider=settings.llm.provider,
        model=settings.llm.model,
        model_path=settings.llm.model_path,
        base_url=settings.llm.base_url,
    )


@router.post("/models/download", response_model=DownloadStatusOut, status_code=202)
async def download_model(
    body: DownloadRequest,
    background_tasks: BackgroundTasks,
    request: Request,
) -> DownloadStatusOut:
    """
    Trigger an async GGUF model download from HuggingFace.

    The download runs in the background. Poll /api/models/download/status
    to check progress.

    Note: If using the llama.cpp provider, restart the llama-server
    process after downloading a new model to load it.
    """
    settings = get_settings()
    dest_dir = Path(settings.models.models_dir) / body.dest_subdir
    key = f"{body.repo_id}/{body.filename}"

    if key in _download_status and _download_status[key]["status"] == "downloading":
        raise HTTPException(
            status_code=409,
            detail=f"Download already in progress: {body.filename}",
        )

    _download_status[key] = {
        "key": key,
        "status": "pending",
        "filename": body.filename,
        "bytes_downloaded": 0,
        "total_bytes": 0,
        "percent": 0.0,
        "message": "Queued",
    }

    background_tasks.add_task(
        _run_download,
        key=key,
        repo_id=body.repo_id,
        filename=body.filename,
        dest_dir=dest_dir,
        sha256=body.sha256,
    )

    return DownloadStatusOut(**_download_status[key])


@router.get("/models/download/status", response_model=list[DownloadStatusOut])
async def download_status() -> list[DownloadStatusOut]:
    """Return status of all download operations (current session only)."""
    return [DownloadStatusOut(**v) for v in _download_status.values()]


# ---------------------------------------------------------------------------
# Background download task
# ---------------------------------------------------------------------------


async def _run_download(
    key: str,
    repo_id: str,
    filename: str,
    dest_dir: Path,
    sha256: str,
) -> None:
    from backend.models.downloader import download_gguf

    _download_status[key]["status"] = "downloading"
    _download_status[key]["message"] = "Downloading…"

    def _progress(downloaded: int, total: int) -> None:
        pct = (downloaded / total * 100) if total else 0.0
        _download_status[key].update({
            "bytes_downloaded": downloaded,
            "total_bytes":      total,
            "percent":          round(pct, 1),
            "message":          f"{downloaded / 1e9:.2f} / {total / 1e9:.2f} GB",
        })

    try:
        await download_gguf(
            repo_id=repo_id,
            filename=filename,
            dest_dir=dest_dir,
            expected_sha256=sha256 or None,
            progress_callback=_progress,
        )
        _download_status[key].update({
            "status":  "done",
            "percent": 100.0,
            "message": f"Downloaded to {dest_dir / filename}",
        })
        logger.info("Download complete: %s", filename)
    except Exception as exc:  # noqa: BLE001
        _download_status[key].update({
            "status":  "error",
            "message": str(exc),
        })
        logger.error("Download failed for %s: %s", filename, exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_out(m) -> ModelOut:
    return ModelOut(
        name=m.name,
        filename=m.filename,
        path=str(m.path),
        family=m.family,
        parameters=m.parameters,
        quantisation=m.quantisation,
        context=m.context,
        chat_template=m.chat_template,
        size_gb=m.size_gb,
        exists=m.exists,
        description=m.description,
    )
