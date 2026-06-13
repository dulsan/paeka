"""
backend/ingestion/parsers/code_parser.py
=========================================
Parses source code files into structured DocumentElements using Tree-sitter.

Supported languages: Python, TypeScript, C, C++

Hierarchy extracted:
  Repository → Module → Class → Function/Method

Each function/method becomes a TEXT element with:
  - heading  : "ClassName.method_name" or "function_name"
  - content  : the full function source text
  - metadata : {"language": "python", "type": "function|class|method"}

This allows retrieval to find specific code units rather than arbitrary
line-range chunks, and the knowledge graph extractor to build accurate
dependency and call graphs from code.

Requires: uv add tree-sitter tree-sitter-python tree-sitter-typescript
          tree-sitter-c tree-sitter-cpp
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path

from backend.ingestion.parsers.base import (
    DocumentElement,
    ElementType,
    ParsedDocument,
)

logger = logging.getLogger(__name__)

# Mapping from file extension to (language_name, tree-sitter module name)
_LANG_MAP: dict[str, tuple[str, str]] = {
    ".py":   ("python",     "tree_sitter_python"),
    ".ts":   ("typescript", "tree_sitter_typescript"),
    ".tsx":  ("typescript", "tree_sitter_typescript"),
    ".c":    ("c",          "tree_sitter_c"),
    ".h":    ("c",          "tree_sitter_c"),
    ".cpp":  ("cpp",        "tree_sitter_cpp"),
    ".cxx":  ("cpp",        "tree_sitter_cpp"),
    ".cc":   ("cpp",        "tree_sitter_cpp"),
    ".hpp":  ("cpp",        "tree_sitter_cpp"),
}

# Node types that represent class definitions per language
_CLASS_TYPES: dict[str, set[str]] = {
    "python":     {"class_definition"},
    "typescript": {"class_declaration", "abstract_class_declaration"},
    "c":          {"struct_specifier", "union_specifier"},
    "cpp":        {"class_specifier", "struct_specifier", "union_specifier"},
}

# Node types that represent function/method definitions per language
_FUNCTION_TYPES: dict[str, set[str]] = {
    "python":     {"function_definition"},
    "typescript": {"function_declaration", "method_definition",
                   "arrow_function", "function_expression"},
    "c":          {"function_definition"},
    "cpp":        {"function_definition"},
}


def parse(path: Path) -> ParsedDocument:
    """
    Parse a source code file with Tree-sitter.

    Parameters
    ----------
    path:
        Path to the source file.

    Returns
    -------
    ParsedDocument
        Elements are code units (functions, classes, methods).

    Raises
    ------
    ImportError
        If tree-sitter or the language grammar is not installed.
    ValueError
        If the file extension is not a supported language.
    """
    suffix = path.suffix.lower()
    if suffix not in _LANG_MAP:
        raise ValueError(
            f"Unsupported code extension: {suffix}. "
            f"Supported: {sorted(_LANG_MAP.keys())}"
        )

    lang_name, module_name = _LANG_MAP[suffix]

    try:
        import tree_sitter  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "tree-sitter is not installed. Run: uv add tree-sitter "
            "tree-sitter-python tree-sitter-typescript tree-sitter-c tree-sitter-cpp"
        ) from exc

    try:
        lang_module = __import__(module_name)
        language = tree_sitter.Language(lang_module.language())
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            f"Tree-sitter grammar for {lang_name} not installed. "
            f"Run: uv add {module_name.replace('_', '-')}"
        ) from exc

    source = path.read_bytes()
    parser = tree_sitter.Parser(language)
    tree = parser.parse(source)
    source_str = source.decode("utf-8", errors="replace")

    elements: list[DocumentElement] = []
    class_types    = _CLASS_TYPES.get(lang_name, set())
    function_types = _FUNCTION_TYPES.get(lang_name, set())

    # Module-level docstring / file header
    elements.append(DocumentElement(
        element_type=ElementType.HEADING,
        content=path.name,
        level=1,
        heading=path.name,
    ))

    _walk(
        node=tree.root_node,
        source=source_str,
        elements=elements,
        class_types=class_types,
        function_types=function_types,
        current_class="",
        depth=0,
    )

    logger.info(
        "Code parser: %s → %d elements (%s)",
        path.name,
        len(elements),
        lang_name,
    )

    return ParsedDocument(
        filename=path.name,
        mime_type=mimetypes.guess_type(str(path))[0] or "text/plain",
        elements=elements,
        parser_used=f"tree-sitter-{lang_name}",
        metadata={"language": lang_name},
    )


def _walk(
    node,
    source: str,
    elements: list[DocumentElement],
    class_types: set[str],
    function_types: set[str],
    current_class: str,
    depth: int,
) -> None:
    """Recursively walk the parse tree, emitting elements for classes and functions."""
    if depth > 10:
        return

    ntype = node.type

    # ── Class / struct ────────────────────────────────────────────────
    if ntype in class_types:
        name = _get_name(node, source)
        if name:
            heading = name
            elements.append(DocumentElement(
                element_type=ElementType.HEADING,
                content=name,
                level=2,
                heading=name,
                metadata={"type": "class"},
            ))
            # Recurse into class body with updated context
            for child in node.children:
                _walk(child, source, elements, class_types, function_types,
                      current_class=name, depth=depth + 1)
            return

    # ── Function / method ─────────────────────────────────────────────
    if ntype in function_types:
        name = _get_name(node, source)
        if name:
            qualified = f"{current_class}.{name}" if current_class else name
            text = _node_text(node, source)
            elements.append(DocumentElement(
                element_type=ElementType.CODE,
                content=text,
                heading=current_class or "",
                metadata={
                    "type":     "method" if current_class else "function",
                    "name":     name,
                    "qualified": qualified,
                },
            ))
            return   # don't recurse into function bodies

    # ── Recurse into everything else ──────────────────────────────────
    for child in node.children:
        _walk(child, source, elements, class_types, function_types,
              current_class=current_class, depth=depth + 1)


def _get_name(node, source: str) -> str:
    """Extract the identifier name from a class/function node."""
    for child in node.children:
        if child.type in ("identifier", "type_identifier", "property_identifier"):
            return _node_text(child, source).strip()
    return ""


def _node_text(node, source: str) -> str:
    """Extract the source text of a node."""
    return source[node.start_byte:node.end_byte]
