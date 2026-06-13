"""
backend/ingestion/parsers/marker_parser.py
===========================================
Fallback document parser using marker-pdf.

Marker is a PDF-to-Markdown converter that handles:
  - Complex multi-column academic PDFs
  - Scanned documents (via OCR)
  - LaTeX equations → MathML/LaTeX strings
  - Code blocks and tables

It produces clean Markdown which we then parse structurally into
our ``DocumentElement`` objects.

Used when:
  - Docling conversion fails
  - [ingestion] default_parser = "marker"
  - The file is a scanned/image-based PDF
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

# Heading patterns in Marker's Markdown output
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
_CODE_FENCE_RE = re.compile(r"^```")
_TABLE_ROW_RE = re.compile(r"^\|.+\|$")
_EQUATION_RE = re.compile(r"^\$\$.+\$\$$", re.DOTALL)


def parse(path: Path) -> ParsedDocument:
    """
    Parse a PDF with marker-pdf and return a ``ParsedDocument``.

    Parameters
    ----------
    path:
        Path to the PDF file.

    Returns
    -------
    ParsedDocument

    Raises
    ------
    ImportError
        If marker-pdf is not installed.
    """
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
    except ImportError as exc:
        raise ImportError(
            "marker-pdf is not installed. Run: uv add marker-pdf"
        ) from exc

    logger.info("Marker parsing: %s", path.name)

    model_dict = create_model_dict()
    converter = PdfConverter(artifact_dict=model_dict)
    rendered = converter(str(path))
    markdown_text, _, _ = text_from_rendered(rendered)

    elements = _parse_markdown(markdown_text)

    logger.info(
        "Marker parsed %s → %d elements",
        path.name,
        len(elements),
    )

    return ParsedDocument(
        filename=path.name,
        mime_type="application/pdf",
        elements=elements,
        parser_used="marker",
    )


def parse_markdown_text(text: str, filename: str) -> ParsedDocument:
    """
    Parse a raw Markdown string (e.g. from any upstream converter).

    Useful when another tool has already produced Markdown and you want
    to run it through the same structural splitter.
    """
    elements = _parse_markdown(text)
    return ParsedDocument(
        filename=filename,
        mime_type="text/markdown",
        elements=elements,
        parser_used="markdown",
    )


# ---------------------------------------------------------------------------
# Markdown structural parser
# ---------------------------------------------------------------------------


def _parse_markdown(text: str) -> list[DocumentElement]:
    """
    Walk Markdown line-by-line and classify each section into elements.

    Handles:
      - ATX headings (#, ##, …)
      - Fenced code blocks (``` … ```)
      - Tables (| col | col |)
      - Display equations ($$ … $$)
      - Paragraphs (everything else)
    """
    elements: list[DocumentElement] = []
    lines = text.splitlines()
    current_heading = ""
    current_level = 0

    i = 0
    while i < len(lines):
        line = lines[i]

        # ── Heading ─────────────────────────────────────────────────
        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            heading_text = m.group(2).strip()
            current_heading = heading_text
            current_level = level
            elements.append(DocumentElement(
                element_type=ElementType.HEADING,
                content=heading_text,
                level=level,
                heading=heading_text,
            ))
            i += 1
            continue

        # ── Fenced code block ────────────────────────────────────────
        if _CODE_FENCE_RE.match(line):
            code_lines = []
            i += 1
            while i < len(lines) and not _CODE_FENCE_RE.match(lines[i]):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            code_content = "\n".join(code_lines).strip()
            if code_content:
                elements.append(DocumentElement(
                    element_type=ElementType.CODE,
                    content=code_content,
                    heading=current_heading,
                ))
            continue

        # ── Table ────────────────────────────────────────────────────
        if _TABLE_ROW_RE.match(line):
            table_lines = []
            while i < len(lines) and _TABLE_ROW_RE.match(lines[i]):
                table_lines.append(lines[i])
                i += 1
            table_text = "\n".join(table_lines)
            elements.append(DocumentElement(
                element_type=ElementType.TABLE,
                content=table_text,
                heading=current_heading,
                metadata={"has_table": True},
            ))
            continue

        # ── Display equation ─────────────────────────────────────────
        if line.strip().startswith("$$"):
            eq_lines = [line]
            if not line.strip().endswith("$$") or line.strip() == "$$":
                i += 1
                while i < len(lines) and not lines[i].strip().endswith("$$"):
                    eq_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    eq_lines.append(lines[i])
            i += 1
            eq_text = "\n".join(eq_lines).strip()
            elements.append(DocumentElement(
                element_type=ElementType.EQUATION,
                content=eq_text,
                heading=current_heading,
            ))
            continue

        # ── Paragraph accumulation ───────────────────────────────────
        if line.strip():
            para_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not _HEADING_RE.match(lines[i]):
                para_lines.append(lines[i])
                i += 1
            para_text = " ".join(para_lines).strip()
            if len(para_text) > 10:
                elements.append(DocumentElement(
                    element_type=ElementType.TEXT,
                    content=para_text,
                    heading=current_heading,
                ))
            continue

        i += 1

    return elements
