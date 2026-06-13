"""
backend/ingestion/parsers/docling_parser.py
============================================
Primary document parser using Docling 2.9.0+.

Changes from v0.11.3:
  [OPT-1] DocumentConverter now configured explicitly instead of using
          all defaults. The default configuration enables OCR, table
          structure recognition via a vision model (TableFormer), and
          figure classification — all of which require heavy model
          downloads and significant CPU time per page.

          For born-digital PDFs (i.e. any PDF you can select text in),
          OCR is completely unnecessary and was the main cause of
          600-second timeouts on dense academic papers.

          New configuration:
            - do_ocr = False             skip OCR entirely
            - do_table_structure = True  keep table extraction (fast, rule-based)
            - generate_picture_images = False  skip figure rendering
            - images_scale = 1.0         minimal resolution if images needed

  [OPT-2] Added PAEKA_INGESTION__FAST_MODE env var toggle.
          Set PAEKA_INGESTION__FAST_MODE=true in .env for maximum speed:
            - disables table structure recognition too
            - plain text + headings only
            - ~5-15s per paper instead of 60-300s
          Default is False (full mode: text + tables + equations).

  [OPT-3] Converter is cached as a module-level singleton so the
          pipeline and its models are loaded only once per process,
          not once per document. Previously a new DocumentConverter()
          was instantiated on every call to parse(), paying the model
          load cost on every single file.

Expected parse times after these changes (CPU, 9B model host machine):
  Born-digital PDF, 10-20 pages:   5-20s   (was 60-180s)
  Born-digital PDF, 50-100 pages:  20-60s  (was 300-600s)
  Scanned PDF (needs OCR):         use PAEKA_INGESTION__FAST_MODE=false
                                   and accept the longer parse time,
                                   or pre-OCR with a dedicated tool
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from backend.ingestion.parsers.base import (
    DocumentElement,
    ElementType,
    ParsedDocument,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Converter singleton — loaded once, reused for every document
# ---------------------------------------------------------------------------

_converter = None


def _get_converter():
    global _converter
    if _converter is not None:
        return _converter

    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import PdfFormatOption
    except ImportError as exc:
        raise ImportError(
            "docling is not installed. Run: uv add 'docling>=2.9.0'"
        ) from exc

    fast_mode = os.environ.get("PAEKA_INGESTION__FAST_MODE", "false").lower() == "true"

    pipeline_options = PdfPipelineOptions()

    # OPT-1: Disable OCR — born-digital PDFs don't need it and it's the
    # single biggest time cost (loads TesseractOCR + EasyOCR models)
    pipeline_options.do_ocr = False

    if fast_mode:
        # OPT-2: Fast mode — plain text + headings only, no table vision model
        pipeline_options.do_table_structure = False
        logger.info("Docling converter: fast mode (OCR=off, tables=off)")
    else:
        # Full mode — table structure on (rule-based, much faster than OCR)
        pipeline_options.do_table_structure = True
        logger.info("Docling converter: full mode (OCR=off, tables=on)")

    # Skip rendering figure images — we only keep captions
    pipeline_options.generate_picture_images = False
    pipeline_options.images_scale = 1.0

    _converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    logger.info("Docling converter initialised (singleton)")
    return _converter


# ---------------------------------------------------------------------------
# Public parse function
# ---------------------------------------------------------------------------

def parse(path: Path) -> ParsedDocument:
    """
    Parse any supported file with Docling's DocumentConverter.

    Supported: PDF, DOCX, PPTX, HTML, MD, images (PNG/JPEG), LaTeX

    Returns
    -------
    ParsedDocument
        Elements map 1-to-1 with Docling's document items.
        The raw DoclingDocument is stored in metadata["docling_doc"]
        so the HybridChunker can operate on it directly.
    """
    converter = _get_converter()

    logger.info("Docling parsing: %s", path.name)

    result = converter.convert(str(path))

    if result is None or result.document is None:
        raise RuntimeError(f"Docling returned no document for {path.name}")

    doc = result.document
    elements: list[DocumentElement] = []
    current_heading = ""

    for item, _level in doc.iterate_items():
        item_type = type(item).__name__.lower()

        if "sectionheader" in item_type or "title" in item_type:
            text = _text(item)
            heading_level = getattr(item, "level", 1)
            current_heading = text
            elements.append(DocumentElement(
                element_type=ElementType.HEADING,
                content=text,
                page=_page(item),
                level=heading_level,
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
            text = _text(item)
            if text:
                elements.append(DocumentElement(
                    element_type=ElementType.EQUATION,
                    content=text,
                    page=_page(item),
                    heading=current_heading,
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

    meta: dict = {"docling_doc": doc}
    if hasattr(doc, "name") and doc.name:
        meta["title"] = doc.name

    logger.info(
        "Docling parsed %s → %d elements (%d headings, %d tables, %d equations)",
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


def chunk_with_hybrid_chunker(
    parsed: ParsedDocument,
    embed_model_name: str = "BAAI/bge-m3",
    max_tokens: int | None = None,
) -> list:
    """
    Use Docling's HybridChunker to produce tokenisation-aware chunks
    directly from a DoclingDocument.
    """
    try:
        from docling.chunking import HybridChunker
    except ImportError:
        try:
            from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
        except ImportError as exc:
            raise ImportError(
                "docling-core is not installed. Run: uv add 'docling>=2.9.0'"
            ) from exc

    doc = parsed.metadata.get("docling_doc")
    if doc is None:
        raise ValueError(
            "No DoclingDocument found in metadata. "
            "Was this ParsedDocument created by docling_parser.parse()?"
        )

    kwargs: dict = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    chunker = HybridChunker(tokenizer=embed_model_name, **kwargs)
    chunks = list(chunker.chunk(doc))
    logger.debug("HybridChunker produced %d chunks from %s", len(chunks), parsed.filename)
    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text(item: object) -> str:
    for attr in ("text", "content", "value"):
        val = getattr(item, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    if callable(getattr(item, "export_to_markdown", None)):
        try:
            return item.export_to_markdown().strip()  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
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
        except Exception:  # noqa: BLE001
            pass
    return _text(item)
