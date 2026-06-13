"""
backend/ingestion/parsers/latex_parser.py
==========================================
Parser for LaTeX (.tex) source files.

LaTeX is the primary format for academic papers and technical documents
in engineering and science contexts. This parser extracts:
  - Section hierarchy (\\section, \\subsection, \\subsubsection)
  - Body text (paragraphs between commands)
  - Equations (\\begin{equation}, \\begin{align}, inline $...$)
  - Tables (\\begin{tabular})
  - Code listings (\\begin{lstlisting}, \\begin{verbatim})
  - Figure captions (\\caption{})
  - Abstract
  - Bibliography references (\\bibitem — stored as metadata)

No LaTeX compilation is required. This is a regex/structural parser
operating on the raw .tex source.

Limitations:
  - Does not expand macros or custom commands
  - Multi-file projects (\\input, \\include) are not followed automatically
    (ingest each file individually)
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

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_SECTION_RE   = re.compile(
    r"\\(chapter|section|subsection|subsubsection)\*?\{([^}]+)\}"
)
_EQUATION_ENV = re.compile(
    r"\\begin\{(equation|align|gather|multline|eqnarray)\*?\}(.*?)\\end\{\1\*?\}",
    re.DOTALL,
)
_INLINE_EQ    = re.compile(r"\$\$(.+?)\$\$|\$([^$\n]+?)\$", re.DOTALL)
_TABLE_ENV    = re.compile(
    r"\\begin\{(tabular|table)\*?\}(.*?)\\end\{\1\*?\}",
    re.DOTALL,
)
_LISTING_ENV  = re.compile(
    r"\\begin\{(lstlisting|verbatim|minted)\*?\}(.*?)\\end\{\1\*?\}",
    re.DOTALL,
)
_ABSTRACT_ENV = re.compile(
    r"\\begin\{abstract\}(.*?)\\end\{abstract\}", re.DOTALL
)
_CAPTION_RE   = re.compile(r"\\caption\{([^}]+)\}")
_COMMENT_RE   = re.compile(r"(?<!\\)%.*")
_COMMAND_RE   = re.compile(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})*")

# Section command → heading level
_SECTION_LEVELS = {
    "chapter": 1,
    "section": 1,
    "subsection": 2,
    "subsubsection": 3,
}


def parse(path: Path) -> ParsedDocument:
    """
    Parse a LaTeX .tex file into a ParsedDocument.

    Parameters
    ----------
    path:
        Path to the .tex file.

    Returns
    -------
    ParsedDocument
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")

    logger.info("LaTeX parser: %s", path.name)
    elements = _parse_latex(text)

    logger.info(
        "LaTeX parsed %s → %d elements (%d equations, %d headings)",
        path.name,
        len(elements),
        sum(1 for e in elements if e.element_type == ElementType.EQUATION),
        sum(1 for e in elements if e.element_type == ElementType.HEADING),
    )

    return ParsedDocument(
        filename=path.name,
        mime_type="text/x-latex",
        elements=elements,
        parser_used="latex",
    )


def _parse_latex(text: str) -> list[DocumentElement]:
    elements: list[DocumentElement] = []
    current_heading = ""

    # Strip comments
    text = _COMMENT_RE.sub("", text)

    # Extract abstract first (special block)
    abstract_match = _ABSTRACT_ENV.search(text)
    if abstract_match:
        abstract = _clean_text(abstract_match.group(1))
        if abstract:
            elements.append(DocumentElement(
                element_type=ElementType.TEXT,
                content=abstract,
                heading="Abstract",
                metadata={"is_abstract": True},
            ))
        text = text[:abstract_match.start()] + text[abstract_match.end():]

    # Mark positions of all structural elements
    positions: list[tuple[int, str, str, int]] = []  # (pos, type, content, level)

    for m in _SECTION_RE.finditer(text):
        level = _SECTION_LEVELS.get(m.group(1), 2)
        positions.append((m.start(), "heading", m.group(2).strip(), level))

    for m in _EQUATION_ENV.finditer(text):
        eq = m.group(2).strip()
        if eq:
            positions.append((m.start(), "equation", f"\\begin{{{m.group(1)}}}\n{eq}\n\\end{{{m.group(1)}}}", 0))

    for m in _TABLE_ENV.finditer(text):
        tbl = m.group(2).strip()
        if tbl:
            positions.append((m.start(), "table", tbl, 0))

    for m in _LISTING_ENV.finditer(text):
        code = m.group(2).strip()
        if code:
            positions.append((m.start(), "code", code, 0))

    for m in _CAPTION_RE.finditer(text):
        positions.append((m.start(), "caption", m.group(1).strip(), 0))

    positions.sort(key=lambda x: x[0])

    # Fill gaps between structural elements with prose text
    last_pos = 0
    for pos, etype, content, level in positions:
        # Extract prose between last element and this one
        gap = text[last_pos:pos]
        prose = _extract_prose(gap)
        if prose and len(prose) > 20:
            elements.append(DocumentElement(
                element_type=ElementType.TEXT,
                content=prose,
                heading=current_heading,
            ))

        if etype == "heading":
            current_heading = content
            elements.append(DocumentElement(
                element_type=ElementType.HEADING,
                content=content,
                level=level,
                heading=content,
            ))
        elif etype == "equation":
            elements.append(DocumentElement(
                element_type=ElementType.EQUATION,
                content=content,
                heading=current_heading,
            ))
        elif etype == "table":
            elements.append(DocumentElement(
                element_type=ElementType.TABLE,
                content=content,
                heading=current_heading,
                metadata={"has_table": True},
            ))
        elif etype == "code":
            elements.append(DocumentElement(
                element_type=ElementType.CODE,
                content=content,
                heading=current_heading,
            ))
        elif etype == "caption":
            elements.append(DocumentElement(
                element_type=ElementType.CAPTION,
                content=content,
                heading=current_heading,
            ))

        # Move past the matched region — approximate by content length
        last_pos = pos + len(content) + 20

    # Remaining text after last element
    remainder = _extract_prose(text[last_pos:])
    if remainder and len(remainder) > 20:
        elements.append(DocumentElement(
            element_type=ElementType.TEXT,
            content=remainder,
            heading=current_heading,
        ))

    return elements


def _extract_prose(text: str) -> str:
    """Strip LaTeX commands from a block and return readable plain text."""
    # Remove environments we've already handled
    text = _EQUATION_ENV.sub(" ", text)
    text = _TABLE_ENV.sub(" ", text)
    text = _LISTING_ENV.sub(" ", text)
    # Remove inline equations but preserve content conceptually
    text = _INLINE_EQ.sub(lambda m: f" {(m.group(1) or m.group(2)).strip()} ", text)
    # Remove LaTeX commands
    text = _COMMAND_RE.sub(" ", text)
    # Remove braces
    text = re.sub(r"[{}]", " ", text)
    # Clean whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_text(text: str) -> str:
    """Clean text extracted from a LaTeX environment."""
    text = _COMMAND_RE.sub(" ", text)
    text = re.sub(r"[{}]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
