"""
backend/ingestion/parsers/docling_parser.py
============================================
Docling-backed document parser.

Fixes applied:
  [FIX-LOCK]   _get_converter() had a double-checked locking race. Two
               concurrent ingestion requests both seeing _converter is None
               would both construct a DocumentConverter, loading models
               twice. Added threading.Lock around the check-and-assign.

  [FIX-ASYNC]  parse() is a blocking CPU-bound call (5–60s for dense PDFs).
               Called from async ingest_file() without asyncio.to_thread(),
               stalling the event loop for the full parse duration.
               Added async parse_async() wrapper for use in the pipeline.

  [FIX-EQ]     Equations were exported via export_to_markdown() which wraps
               LaTeX in backtick code fences (`$E=mc^2$`). Stored in Weaviate
               as code strings, rendered as code blocks in the UI. Now uses
               item.text directly (raw LaTeX) when available.

  [FIX-SCAN]   No detection of scanned PDFs. After parsing, if zero text
               elements are found, the document is flagged as "likely scanned"
               and re-parsed with OCR enabled, with a clear log warning.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from pathlib import Path

from backend.ingestion.parsers.base import (
    DocumentElement,
    ElementType,
    ParsedDocument,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Converter singleton — thread-safe
# ---------------------------------------------------------------------------

_converter      = None
_converter_lock = threading.Lock()   # [FIX-LOCK]


def _get_converter(force_ocr: bool = False):
    """
    Return the cached DocumentConverter singleton.
    Thread-safe: uses a lock to prevent concurrent construction races.
    """
    global _converter

    # Fast path — already built (read without lock is safe because the
    # variable is only ever written under the lock once).
    if _converter is not None and not force_ocr:
        return _converter

    with _converter_lock:
        # Re-check inside the lock (classic double-checked locking, now safe).
        if _converter is not None and not force_ocr:
            return _converter

        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.datamodel.base_models import InputFormat
        except ImportError as exc:
            raise ImportError(
                "docling is not installed. Run: uv add 'docling>=2.9.0'"
            ) from exc

        fast_mode = os.environ.get("PAEKA_INGESTION__FAST_MODE", "false").lower() == "true"
        pipeline_options = PdfPipelineOptions()

        if force_ocr:
            pipeline_options.do_ocr            = True
            pipeline_options.do_table_structure = True
            logger.info("Docling converter: OCR mode (scanned document detected)")
        else:
            pipeline_options.do_ocr = False
            if fast_mode:
                pipeline_options.do_table_structure = False
                logger.info("Docling converter: fast mode (OCR=off, tables=off)")
            else:
                pipeline_options.do_table_structure = True
                logger.info("Docling converter: full mode (OCR=off, tables=on)")

        pipeline_options.generate_picture_images = False
        pipeline_options.images_scale            = 1.0

        conv = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        if not force_ocr:
            _converter = conv
        return conv


# ---------------------------------------------------------------------------
# Async wrapper for pipeline use  [FIX-ASYNC]
# ---------------------------------------------------------------------------

async def parse_async(path: Path, force_ocr: bool = False) -> ParsedDocument:
    """
    Async wrapper around parse(). Use this from async ingestion pipeline
    code instead of calling parse() directly.

    Runs the blocking Docling conversion in a thread pool worker so it
    does not stall the FastAPI event loop for the 5–60s parse duration.
    """
    return await asyncio.to_thread(parse, path, force_ocr)


# ---------------------------------------------------------------------------
# Synchronous parse (called via asyncio.to_thread from parse_async)
# ---------------------------------------------------------------------------

def parse(path: Path, force_ocr: bool = False) -> ParsedDocument:
    """
    Parse any Docling-supported file (PDF, DOCX, PPTX, HTML, MD, images).

    The raw DoclingDocument is stored in metadata["docling_doc"] so the
    HybridChunker can operate on it directly without re-parsing.
    """
    converter = _get_converter(force_ocr=force_ocr)

    logger.info("Docling parsing: %s (ocr=%s)", path.name, force_ocr)
    result = converter.convert(str(path))

    if result is None or result.document is None:
        raise RuntimeError(f"Docling returned no document for {path.name}")

    doc      = result.document
    elements = _extract_elements(doc)

    # [FIX-SCAN] Detect scanned PDFs: if no text elements were found and
    # this is a PDF, retry with OCR enabled and a visible warning.
    text_count = sum(
        1 for e in elements
        if e.element_type in (ElementType.TEXT, ElementType.LIST_ITEM,
                               ElementType.HEADING, ElementType.CODE)
        and len(e.content.strip()) > 10
    )
    suffix = path.suffix.lower()
    if text_count == 0 and suffix == ".pdf" and not force_ocr:
        logger.warning(
            "Docling: zero text elements found in '%s'. "
            "This is likely a scanned PDF. Re-parsing with OCR enabled. "
            "Expect significantly longer parse time (30-300s).",
            path.name,
        )
        ocr_converter = _get_converter(force_ocr=True)
        ocr_result    = ocr_converter.convert(str(path))
        if ocr_result and ocr_result.document:
            doc      = ocr_result.document
            elements = _extract_elements(doc)

    meta: dict = {"docling_doc": doc}
    if hasattr(doc, "name") and doc.name:
        meta["title"] = doc.name

    logger.info(
        "Docling parsed %s → %d elements (%dH %dT %dEQ)",
        path.name, len(elements),
        sum(1 for e in elements if e.element_type == ElementType.HEADING),
        sum(1 for e in elements if e.element_type == ElementType.TABLE),
        sum(1 for e in elements if e.element_type == ElementType.EQUATION),
    )

    import mimetypes
    mime_type, _ = mimetypes.guess_type(str(path))
    return ParsedDocument(
        filename=path.name,
        mime_type=mime_type or "application/octet-stream",
        elements=elements,
        parser_used="docling",
        metadata=meta,
    )


def _extract_elements(doc) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    current_heading = ""

    for item, _level in doc.iterate_items():
        item_type = type(item).__name__.lower()

        if "sectionheader" in item_type or "title" in item_type:
            text = _text(item)
            current_heading = text
            elements.append(DocumentElement(
                element_type=ElementType.HEADING,
                content=text,
                page=_page(item),
                level=getattr(item, "level", 1),
                heading=text,
            ))

        elif "table" in item_type:
            md = _table_md(item)
            if md:
                elements.append(DocumentElement(
                    element_type=ElementType.TABLE,
                    content=md,
                    page=_page(item),
                    heading=current_heading,
                    metadata={"has_table": True},
                ))

        elif "figure" in item_type or "picture" in item_type:
            caption = _caption(item)
            elements.append(DocumentElement(
                element_type=ElementType.FIGURE,
                content=caption or "[Figure]",
                page=_page(item),
                heading=current_heading,
            ))

        elif "equation" in item_type or "formula" in item_type:
            # [FIX-EQ] Prefer raw text (LaTeX) over export_to_markdown()
            # which wraps equations in backtick code fences.
            text = _equation_text(item)
            if text:
                elements.append(DocumentElement(
                    element_type=ElementType.EQUATION,
                    content=text,
                    page=_page(item),
                    heading=current_heading,
                    metadata={"is_latex": True},
                ))

        elif "code" in item_type or "listing" in item_type:
            text = _text(item)
            if text:
                elements.append(DocumentElement(
                    element_type=ElementType.CODE,
                    content=text,
                    page=_page(item),
                    heading=current_heading,
                ))

        elif "listitem" in item_type or "list_item" in item_type:
            text = _text(item)
            if text:
                elements.append(DocumentElement(
                    element_type=ElementType.LIST_ITEM,
                    content=text,
                    page=_page(item),
                    heading=current_heading,
                ))

        else:
            text = _text(item)
            if text and len(text.strip()) > 10:
                etype = (
                    ElementType.CAPTION
                    if "caption" in item_type
                    else ElementType.TEXT
                )
                elements.append(DocumentElement(
                    element_type=etype,
                    content=text,
                    page=_page(item),
                    heading=current_heading,
                ))

    return elements


# ---------------------------------------------------------------------------
# HybridChunker interface (unchanged)
# ---------------------------------------------------------------------------

def chunk_with_hybrid_chunker(
    parsed: ParsedDocument,
    embed_model_name: str = "BAAI/bge-m3",
    max_tokens: int | None = None,
) -> list:
    try:
        from docling.chunking import HybridChunker
    except ImportError:
        try:
            from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
        except ImportError as exc:
            raise ImportError("docling-core not installed") from exc

    doc = parsed.metadata.get("docling_doc")
    if doc is None:
        raise ValueError(
            "No DoclingDocument in metadata. Was this ParsedDocument created by docling_parser.parse()?"
        )
    kwargs: dict = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    chunker = HybridChunker(tokenizer=embed_model_name, **kwargs)
    chunks  = list(chunker.chunk(doc))
    logger.debug("HybridChunker: %d chunks from %s", len(chunks), parsed.filename)
    return chunks


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def _text(item: object) -> str:
    for attr in ("text", "content", "value"):
        val = getattr(item, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    if callable(getattr(item, "export_to_markdown", None)):
        try:
            return item.export_to_markdown().strip()  # type: ignore[union-attr]
        except Exception:
            pass
    return ""


def _equation_text(item: object) -> str:
    """
    Extract equation text preferring raw LaTeX over markdown export.
    [FIX-EQ] export_to_markdown() wraps math in backticks → code strings.
    We want the raw LaTeX string for proper math rendering (MathJax/KaTeX).
    """
    # Try direct .text attribute first (raw LaTeX in most Docling versions)
    for attr in ("text", "content", "value", "latex"):
        val = getattr(item, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    # Fall back to export_to_markdown as last resort (will have code fences)
    if callable(getattr(item, "export_to_markdown", None)):
        try:
            md = item.export_to_markdown().strip()  # type: ignore[union-attr]
            # Strip backtick fences that docling adds around math
            md = md.strip("`").strip()
            return md
        except Exception:
            pass
    return ""


def _page(item: object) -> int:
    prov = getattr(item, "prov", None)
    if prov and hasattr(prov, "__iter__"):
        for p in prov:
            page = getattr(p, "page_no", None) or getattr(p, "page", None)
            if isinstance(page, int):
                return page
    return 0


def _caption(item: object) -> str:
    captions = getattr(item, "captions", [])
    if captions:
        return " ".join(_text(c) for c in captions).strip()
    return _text(item)


def _table_md(item: object) -> str:
    if callable(getattr(item, "export_to_markdown", None)):
        try:
            md = item.export_to_markdown()  # type: ignore[union-attr]
            if md and md.strip():
                return md.strip()
        except Exception:
            pass
    return _text(item)
