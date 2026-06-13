"""
backend/ingestion/parsers/pymupdf_parser.py
============================================
Primary document parser using PyMuPDF (fitz).

PyMuPDF ships pre-built wheels for Python 3.12–3.14 on Windows, Linux,
and macOS — no zlib, no C compilation, no Pillow version conflicts.

Capabilities:
  - PDF: text blocks with page numbers, headings detected via font size,
    tables via pymupdf4llm helper (if available), embedded images skipped
  - DOCX/HTML: via pymupdf's unified Document interface

Replaces docling which pulls Pillow 10.4.0 (no Py3.14 Windows wheel).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from backend.ingestion.parsers.base import (
    DocumentElement,
    ElementType,
    ParsedDocument,
)

logger = logging.getLogger(__name__)

# Relative font-size multiplier above which a text block is treated as a heading
_HEADING_SIZE_RATIO = 1.15


def parse(path: Path) -> ParsedDocument:
    """
    Parse a PDF (or DOCX/HTML) with PyMuPDF.

    Parameters
    ----------
    path:
        File to parse.

    Returns
    -------
    ParsedDocument

    Raises
    ------
    ImportError
        If pymupdf is not installed.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "pymupdf is not installed. Run: uv add pymupdf"
        ) from exc

    logger.info("PyMuPDF parsing: %s", path.name)

    doc = fitz.open(str(path))
    elements: list[DocumentElement] = []
    current_heading = ""

    # --- Pass 1: collect all text blocks and compute median font size -----
    all_spans: list[dict] = []
    for page in doc:
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block.get("type") != 0:          # 0 = text, 1 = image
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    size = span.get("size", 0.0)
                    if text and size > 0:
                        all_spans.append({
                            "text": text,
                            "size": size,
                            "page": page.number + 1,
                            "flags": span.get("flags", 0),  # bold=16, italic=2
                        })

    if not all_spans:
        doc.close()
        return ParsedDocument(
            filename=path.name,
            mime_type=_guess_mime(path),
            elements=[],
            parser_used="pymupdf",
        )

    # Median font size used as baseline for heading detection
    sizes = sorted(s["size"] for s in all_spans)
    median_size = sizes[len(sizes) // 2]
    heading_threshold = median_size * _HEADING_SIZE_RATIO

    # --- Pass 2: classify spans into elements ----------------------------
    prose_buf: list[str] = []
    prose_page = 0

    def flush_prose() -> None:
        nonlocal prose_buf
        text = " ".join(prose_buf).strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) > 15:
            elements.append(DocumentElement(
                element_type=ElementType.TEXT,
                content=text,
                page=prose_page,
                heading=current_heading,
            ))
        prose_buf.clear()

    for span in all_spans:
        text = span["text"]
        size = span["size"]
        page = span["page"]

        # Detect heading: larger font or bold short line
        is_large  = size >= heading_threshold
        is_bold   = bool(span["flags"] & 16)
        is_short  = len(text.split()) <= 12
        is_heading = (is_large or (is_bold and is_short)) and not text.endswith(",")

        if is_heading:
            flush_prose()
            current_heading = text
            # Approximate heading level from font size ratio
            ratio = size / median_size
            level = 1 if ratio >= 1.5 else 2 if ratio >= 1.3 else 3
            elements.append(DocumentElement(
                element_type=ElementType.HEADING,
                content=text,
                page=page,
                level=level,
                heading=text,
            ))
        else:
            prose_buf.append(text)
            prose_page = page

    flush_prose()
    doc.close()

    # --- Optional: try pymupdf4llm for richer Markdown output -----------
    # pymupdf4llm is an optional companion package that produces better
    # table extraction. If installed, we run a second pass to pick up tables.
    try:
        import pymupdf4llm  # type: ignore[import]
        md_text = pymupdf4llm.to_markdown(str(path))
        from backend.ingestion.parsers.marker_parser import parse_markdown_text
        md_doc = parse_markdown_text(md_text, path.name)
        table_elements = [
            e for e in md_doc.elements if e.element_type == ElementType.TABLE
        ]
        if table_elements:
            elements.extend(table_elements)
            logger.debug(
                "pymupdf4llm added %d table elements to %s",
                len(table_elements),
                path.name,
            )
    except ImportError:
        pass    # pymupdf4llm is optional; base parsing is sufficient

    logger.info(
        "PyMuPDF parsed %s → %d elements (%d headings)",
        path.name,
        len(elements),
        sum(1 for e in elements if e.element_type == ElementType.HEADING),
    )

    import mimetypes
    mime_type, _ = mimetypes.guess_type(str(path))
    return ParsedDocument(
        filename=path.name,
        mime_type=mime_type or "application/pdf",
        elements=elements,
        parser_used="pymupdf",
    )


def _guess_mime(path: Path) -> str:
    import mimetypes
    mt, _ = mimetypes.guess_type(str(path))
    return mt or "application/octet-stream"
