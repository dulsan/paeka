"""
backend/ingestion/parsers/base.py
==================================
Domain objects produced by every parser.

All parsers return a ``ParsedDocument`` containing a flat list of
``DocumentElement`` objects.  The hierarchy (chapter → section →
subsection) is encoded via ``level`` and ``parent_heading`` so the
chunker can respect document structure without needing a tree walker.

Element types mirror the SRS hierarchy:
    text | heading | table | figure | caption | equation | code | list_item
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ElementType(StrEnum):
    TEXT       = "text"
    HEADING    = "heading"
    TABLE      = "table"
    FIGURE     = "figure"
    CAPTION    = "caption"
    EQUATION   = "equation"
    CODE       = "code"
    LIST_ITEM  = "list_item"


@dataclass
class DocumentElement:
    """A single structural unit extracted from a document."""

    element_type: ElementType
    content: str                    # raw text / markdown representation
    page: int = 0                   # 0 = unknown / not applicable
    level: int = 0                  # heading depth (1 = H1, 0 = body text)
    heading: str = ""               # nearest ancestor heading label
    metadata: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """All elements extracted from one file."""

    filename: str
    mime_type: str
    elements: list[DocumentElement] = field(default_factory=list)
    parser_used: str = "unknown"
    metadata: dict = field(default_factory=dict)  # title, author, etc.

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def text_elements(self) -> list[DocumentElement]:
        """Return all non-heading prose elements."""
        return [e for e in self.elements if e.element_type != ElementType.HEADING]

    def headings(self) -> list[DocumentElement]:
        return [e for e in self.elements if e.element_type == ElementType.HEADING]

    def full_text(self) -> str:
        """Flat concatenation of all element content — used for hashing."""
        return "\n".join(e.content for e in self.elements)
