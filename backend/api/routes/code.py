"""
backend/api/routes/code.py
===========================
Code-specific endpoints.

POST /api/code/ingest         — ingest a source file via tree-sitter
POST /api/code/verify         — run static verification on a code snippet
POST /api/code/format         — auto-format Python code via Ruff
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from backend.tools.verification import lint_python, format_python, typecheck_python

logger = logging.getLogger(__name__)
router = APIRouter(tags=["code"])

_SUPPORTED_EXTENSIONS = {".py", ".ts", ".tsx", ".c", ".h", ".cpp", ".cxx", ".cc", ".hpp"}
_CODE_UPLOAD_DIR = Path("data/uploads/code")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class VerifyRequest(BaseModel):
    code: str
    language: str = "python"       # python | typescript | c | cpp
    filename: str = "snippet.py"


class VerifyResponse(BaseModel):
    tool: str
    passed: bool
    output: str
    issues: list[dict]


class FormatRequest(BaseModel):
    code: str


class FormatResponse(BaseModel):
    fixed_code: str
    changed: bool
    output: str


class IngestCodeResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    chunks: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/code/ingest", response_model=IngestCodeResponse, status_code=202)
async def ingest_code_file(
    request: Request,
    file: UploadFile = File(...),
) -> IngestCodeResponse:
    """
    Upload a source code file and ingest it via Tree-sitter.

    Extracts class and function-level chunks into the knowledge base.
    Requires retrieval to be enabled.
    """
    pipeline = request.app.state.ingestion
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Ingestion not enabled. Set [retrieval] enabled = true.",
        )

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported code extension: {suffix}. Supported: {sorted(_SUPPORTED_EXTENSIONS)}",
        )

    content = await file.read()
    _CODE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = _CODE_UPLOAD_DIR / (file.filename or "upload")
    dest.write_bytes(content)

    try:
        doc_id = await pipeline.ingest_file(dest)
    except Exception as exc:
        logger.error("Code ingestion error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    from backend.ingestion.repository import DocumentRepository
    repo = DocumentRepository(request.app.state.db)
    doc = await repo.get_document(doc_id)

    return IngestCodeResponse(
        document_id=doc_id,
        filename=file.filename or "",
        status=doc.status if doc else "unknown",
        chunks=doc.chunk_count if doc else 0,
    )


@router.post("/code/verify", response_model=VerifyResponse)
async def verify_code(body: VerifyRequest) -> VerifyResponse:
    """
    Run static linting/type-checking on a code snippet.

    Currently supports Python (Ruff + Pyright).
    Returns structured issues with line numbers and codes.
    """
    lang = body.language.lower()

    if lang == "python":
        result = await lint_python(body.code, body.filename)
        # Also run Pyright if linting passes
        if result.passed:
            pyright_result = await typecheck_python(body.code)
            if not pyright_result.passed:
                result = pyright_result
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Static verification for '{lang}' not yet implemented. Supported: python",
        )

    return VerifyResponse(
        tool=result.tool,
        passed=result.passed,
        output=result.output,
        issues=result.issues,
    )


@router.post("/code/format", response_model=FormatResponse)
async def format_code(body: FormatRequest) -> FormatResponse:
    """
    Auto-format Python code using Ruff.

    Returns the formatted code and whether any changes were made.
    """
    result = await format_python(body.code)
    return FormatResponse(
        fixed_code=result.fixed_code,
        changed=result.fixed_code != body.code,
        output=result.output,
    )
