"""
tests/unit/test_latex_parser.py
================================
Unit tests for the LaTeX structural parser.
"""

from __future__ import annotations

import pytest
import tempfile
from pathlib import Path

from backend.ingestion.parsers.latex_parser import parse
from backend.ingestion.parsers.base import ElementType


def _write_tex(content: str) -> Path:
    f = tempfile.NamedTemporaryFile(suffix=".tex", mode="w",
                                    encoding="utf-8", delete=False)
    f.write(content)
    f.close()
    return Path(f.name)


def test_section_heading_extracted():
    path = _write_tex(r"""
\section{Introduction}
This is the introduction text.
""")
    doc = parse(path)
    headings = [e for e in doc.elements if e.element_type == ElementType.HEADING]
    assert any("Introduction" in h.content for h in headings)
    path.unlink()


def test_subsection_hierarchy():
    path = _write_tex(r"""
\section{Methods}
\subsection{Data Collection}
We collected data from 100 participants.
\subsubsection{Survey Design}
The survey consisted of 20 questions.
""")
    doc = parse(path)
    headings = [e for e in doc.elements if e.element_type == ElementType.HEADING]
    assert len(headings) >= 3
    path.unlink()


def test_equation_environment_extracted():
    path = _write_tex(r"""
\section{Theory}
The loss function is defined as:
\begin{equation}
    L = -\sum_{i} y_i \log(\hat{y}_i)
\end{equation}
""")
    doc = parse(path)
    equations = [e for e in doc.elements if e.element_type == ElementType.EQUATION]
    assert len(equations) >= 1
    assert "equation" in equations[0].content.lower() or "L" in equations[0].content
    path.unlink()


def test_abstract_extracted():
    path = _write_tex(r"""
\begin{abstract}
We present a novel method for machine learning.
The method achieves state-of-the-art results.
\end{abstract}
\section{Introduction}
""")
    doc = parse(path)
    abstract_elements = [e for e in doc.elements if e.metadata.get("is_abstract")]
    assert len(abstract_elements) == 1
    assert "novel method" in abstract_elements[0].content
    path.unlink()


def test_code_listing_extracted():
    path = _write_tex(r"""
\section{Implementation}
\begin{lstlisting}[language=Python]
def train(model, data):
    optimizer.zero_grad()
    loss.backward()
\end{lstlisting}
""")
    doc = parse(path)
    code_elements = [e for e in doc.elements if e.element_type == ElementType.CODE]
    assert len(code_elements) >= 1
    path.unlink()


def test_comments_stripped():
    path = _write_tex(r"""
\section{Intro} % this is a comment
This is real text. % another comment
""")
    doc = parse(path)
    text_elements = [e for e in doc.elements if e.element_type == ElementType.TEXT]
    assert all("%" not in e.content for e in text_elements)
    path.unlink()


def test_parser_used_field():
    path = _write_tex(r"\section{Test}\nHello world.")
    doc = parse(path)
    assert doc.parser_used == "latex"
    assert doc.mime_type == "text/x-latex"
    path.unlink()
