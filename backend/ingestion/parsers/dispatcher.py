"""
backend/ingestion/parsers/dispatcher.py
========================================
Selects and runs the correct parser for any supported file.

File type → parser:
  .pdf / .docx / .pptx / .html / images  → docling_parser (primary)
  .tex / .latex                           → latex_parser
  .md / .markdown                         → marker_parser (Markdown structural)
  .txt / .rst                             → plaintext
  .xlsx / .xls / .csv                     → spreadsheet_parser
  .py / .ts / .c / .cpp / …              → code_parser (Tree-sitter, optional)

Docling is the primary parser for all rich-format documents.
PyMuPDF has been removed — Docling 2.9.0+ handles Python 3.14 natively.
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
from pathlib import Path

from backend.ingestion.parsers.base import ParsedDocument

logger = logging.getLogger(__name__)

_SPREADSHEET_EXTS = {".xlsx", ".xls", ".xlsm", ".csv"}
_PLAINTEXT_EXTS   = {".txt", ".rst", ".log"}
_MARKDOWN_EXTS    = {".md", ".markdown"}
_LATEX_EXTS       = {".tex", ".latex"}
_CODE_EXTS        = {
    ".py", ".ts", ".tsx", ".js", ".c", ".h",
    ".cpp", ".cxx", ".cc", ".hpp",
}
# Formats Docling handles natively
_DOCLING_EXTS = {
    ".pdf", ".docx", ".pptx", ".html", ".htm",
    ".png", ".jpg", ".jpeg", ".tiff", ".bmp",
}


def parse_file(path: Path, mode: str = "auto") -> ParsedDocument:
    """
    Parse *path* and return a ParsedDocument.

    Parameters
    ----------
    path:   File to parse (must exist).
    mode:   "auto" | "docling" | "marker" — force a specific parser.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix   = path.suffix.lower()
    mime, _  = mimetypes.guess_type(str(path))
    mime     = mime or "application/octet-stream"

    # ── Spreadsheet ──────────────────────────────────────────────────────
    if suffix in _SPREADSHEET_EXTS:
        from backend.ingestion.parsers import spreadsheet_parser
        return spreadsheet_parser.parse(path)

    # ── LaTeX ─────────────────────────────────────────────────────────────
    if suffix in _LATEX_EXTS:
        from backend.ingestion.parsers import latex_parser
        return latex_parser.parse(path)

    # ── Markdown ──────────────────────────────────────────────────────────
    if suffix in _MARKDOWN_EXTS:
        from backend.ingestion.parsers import marker_parser
        return marker_parser.parse_markdown_text(_read_text(path), path.name)

    # ── Plain text ────────────────────────────────────────────────────────
    if suffix in _PLAINTEXT_EXTS or (
        mime.startswith("text/")
        and suffix not in _MARKDOWN_EXTS | _LATEX_EXTS
    ):
        return _parse_plaintext(path, mime)

    # ── Source code ───────────────────────────────────────────────────────
    if suffix in _CODE_EXTS:
        try:
            from backend.ingestion.parsers import code_parser
            return code_parser.parse(path)
        except (ImportError, ValueError) as exc:
            logger.warning(
                "Code parser unavailable for %s (%s) — plaintext fallback.",
                path.name, exc,
            )
            return _parse_plaintext(path, "text/plain")

    # ── Force marker ─────────────────────────────────────────────────────
    if mode == "marker":
        from backend.ingestion.parsers import marker_parser
        return marker_parser.parse_markdown_text(_read_text(path), path.name)

    # ── Primary: Docling ─────────────────────────────────────────────────
    # Handles PDF, DOCX, PPTX, HTML, images and more.
    # Falls back to plaintext extraction on failure.
    try:
        from backend.ingestion.parsers import docling_parser
        return docling_parser.parse(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Docling failed for %s (%s) — plaintext fallback.",
            path.name, exc,
        )
        return _parse_plaintext(path, mime)


def content_hash(path: Path) -> str:
    """SHA-256 hex digest of file contents (for change detection)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _parse_plaintext(path: Path, mime_type: str) -> ParsedDocument:
    import re
    from backend.ingestion.parsers.base import DocumentElement, ElementType, ParsedDocument as PD
    text = _read_text(path)
    elements = [
        DocumentElement(element_type=ElementType.TEXT, content=para.strip())
        for para in re.split(r"\n\s*\n", text)
        if para.strip()
    ]
    return PD(filename=path.name, mime_type=mime_type, elements=elements, parser_used="plaintext")
