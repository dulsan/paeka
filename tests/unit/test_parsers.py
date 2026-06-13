"""
tests/unit/test_parsers.py
===========================
Unit tests for the Markdown / plaintext parsers (no heavy deps required).
"""

from __future__ import annotations

import pytest
from backend.ingestion.parsers.marker_parser import parse_markdown_text
from backend.ingestion.parsers.base import ElementType


def test_markdown_headings():
    md = "# Title\n\nSome text.\n\n## Section\n\nMore text."
    doc = parse_markdown_text(md, "test.md")
    headings = [e for e in doc.elements if e.element_type == ElementType.HEADING]
    assert len(headings) == 2
    assert headings[0].content == "Title"
    assert headings[0].level == 1
    assert headings[1].content == "Section"
    assert headings[1].level == 2


def test_markdown_code_block():
    md = "## Code\n\n```python\ndef hello():\n    pass\n```"
    doc = parse_markdown_text(md, "test.md")
    code_blocks = [e for e in doc.elements if e.element_type == ElementType.CODE]
    assert len(code_blocks) == 1
    assert "def hello" in code_blocks[0].content


def test_markdown_table():
    md = "## Data\n\n| A | B |\n|---|---|\n| 1 | 2 |"
    doc = parse_markdown_text(md, "test.md")
    tables = [e for e in doc.elements if e.element_type == ElementType.TABLE]
    assert len(tables) == 1


def test_markdown_equation():
    md = "## Math\n\n$$E = mc^2$$"
    doc = parse_markdown_text(md, "test.md")
    equations = [e for e in doc.elements if e.element_type == ElementType.EQUATION]
    assert len(equations) == 1
    assert "E = mc" in equations[0].content


def test_heading_context_propagation():
    md = "# Chapter\n\nFirst paragraph.\n\n## Sub\n\nSecond paragraph."
    doc = parse_markdown_text(md, "test.md")
    texts = [e for e in doc.elements if e.element_type == ElementType.TEXT]
    assert texts[0].heading == "Chapter"
    assert texts[1].heading == "Sub"


def test_parser_used_field():
    doc = parse_markdown_text("Hello world.", "test.md")
    assert doc.parser_used == "markdown"
    assert doc.filename == "test.md"
