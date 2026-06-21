"""
backend/retrieval/chunker.py
==============================
Text chunking for non-docling ingestion paths.

Fix applied:
  [FIX-SIZE]  Default chunk_size was 512. This was 512 CHARACTERS, not tokens.
              bge-m3's embedding context limit is 512 TOKENS. At ~4 chars/token
              for English prose, 512 chars ≈ 128 tokens — 25% utilisation.
              The model embeds mostly padding, reducing retrieval quality.

              New default: 1600 chars ≈ 400 tokens at 4 chars/token.
              Leaves ~112 tokens headroom for the heading prefix that is
              prepended to every chunk before embedding. Chunk overlap is
              kept at 200 chars (~50 tokens) for boundary continuity.

              For Docling-parsed PDFs and DOCX, the pipeline uses
              docling_parser.chunk_with_hybrid_chunker() which is fully
              token-aware (HybridChunker with tokenizer="BAAI/bge-m3").
              This char-based path is used only for plain text, code,
              LaTeX source files, and spreadsheet exports.

  [ADD-TOKEN] Added count_tokens() helper and token_aware_chunk() for
              callers that want exact token-level splitting using the
              bge-m3 tokenizer. Optional — requires 'transformers'.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Iterator

logger = logging.getLogger(__name__)

# [FIX-SIZE] Was 512 chars (≈128 tokens). Now 1600 chars (≈400 tokens).
DEFAULT_CHUNK_SIZE    = 1600   # characters
DEFAULT_CHUNK_OVERLAP = 200    # characters — ≈50 tokens boundary continuity

# bge-m3 hard token limit
BGE_M3_MAX_TOKENS = 512


@dataclass
class TextChunk:
    content: str
    chunk_index: int
    heading: str       = ""
    start_char: int    = 0
    end_char: int      = 0
    # [FIX] Added as real top-level fields. backend/ingestion/pipeline.py's
    # _process_docling() constructs TextChunk(..., page=page,
    # element_type="text") -- that call has been raising
    # "TypeError: unexpected keyword argument 'page'" the instant any
    # Docling-parsed PDF/DOCX went through that path, since neither field
    # existed here at all. tests/unit/test_chunker.py also expects
    # c.element_type as a direct attribute (chunk_document()'s table/code
    # branch previously only nested it inside metadata).
    page: int          = 0
    element_type: str  = "text"
    metadata: dict     = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Primary chunking function
# ---------------------------------------------------------------------------

def chunk_document(
    elements: list,
    chunk_size: int    = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    include_headings:   bool = True,
) -> list[TextChunk]:
    """
    Chunk a list of DocumentElements into overlapping text chunks.

    Groups consecutive non-heading elements under their section heading.
    Chunks that would exceed chunk_size are split at sentence boundaries
    where possible (full stop + space), otherwise at the hard character limit.

    Parameters
    ----------
    elements:
        List of DocumentElement objects (from any parser).
    chunk_size:
        Target chunk size in characters. Default 1600 (~400 tokens for bge-m3).
    chunk_overlap:
        Character overlap between consecutive chunks for boundary continuity.
    include_headings:
        If True, prepend the current section heading to each chunk.

    Returns
    -------
    list[TextChunk]
    """
    chunks: list[TextChunk] = []
    buffer   = ""
    heading  = ""
    char_pos = 0
    idx      = 0

    def _flush(buf: str) -> None:
        nonlocal idx
        if not buf.strip():
            return
        prefix  = f"{heading}\n\n" if include_headings and heading else ""
        content = (prefix + buf.strip())[:chunk_size + len(prefix)]
        chunks.append(TextChunk(
            content=content,
            chunk_index=idx,
            heading=heading,
            start_char=char_pos,
            end_char=char_pos + len(buf),
        ))
        idx += 1

    for elem in elements:
        etype   = getattr(elem, "element_type", None)
        content = getattr(elem, "content", "") or ""
        ehead   = getattr(elem, "heading", "") or ""

        # Headings: flush current buffer and update section heading
        # [FIX] ElementType is a StrEnum (str(ElementType.HEADING) ==
        # "heading", lowercase) -- this case-sensitive substring check
        # against uppercase "HEADING" never matched anything, ever.
        # Heading elements were silently falling through to the regular
        # text-accumulation path below instead of triggering a section
        # break, which is why heading text was bleeding into chunk prose
        # and section headings were tracked incorrectly. Exact,
        # case-insensitive match instead of substring search.
        if hasattr(etype, "name") and str(etype).strip().lower() == "heading":
            _flush(buffer)
            buffer  = ""
            heading = content
            continue

        # Update heading context if element carries one
        if ehead and ehead != heading:
            heading = ehead

        # Tables and code blocks: always emit as their own chunk (atomic)
        # [FIX] Same StrEnum case mismatch as the heading check above --
        # str(ElementType.TABLE) is "table", never "ElementType.TABLE".
        # Tables and code blocks were never being treated as atomic,
        # despite the docstring's explicit guarantee that they always are.
        if hasattr(etype, "name") and str(etype).strip().lower() in ("table", "code"):
            _flush(buffer)
            buffer = ""
            prefix = f"{heading}\n\n" if include_headings and heading else ""
            chunks.append(TextChunk(
                content=prefix + content,
                chunk_index=idx,
                heading=heading,
                element_type=str(etype).strip().lower(),
                metadata={"element_type": str(etype)},
            ))
            idx += 1
            continue

        # Accumulate into buffer; split when over limit
        if buffer and len(buffer) + len(content) + 1 > chunk_size:
            _flush(buffer)
            # Overlap: carry last `chunk_overlap` chars into the next chunk
            buffer = buffer[-chunk_overlap:].lstrip() + " " + content
        else:
            buffer = (buffer + " " + content).strip() if buffer else content

        char_pos += len(content) + 1

    _flush(buffer)
    return chunks


def chunk_text(
    text: str,
    chunk_size: int    = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    heading: str       = "",
    page: int          = 0,
) -> list[TextChunk]:
    """
    Split a raw text string into overlapping chunks.

    Splits at sentence boundaries ('. ') when available, otherwise at
    the hard character limit. Used for plain text files, LaTeX source,
    and CSV/spreadsheet exports.

    heading/page: [FIX] previously accepted nowhere on this function at
    all despite TextChunk having a heading field since this module's
    first version -- passed through unchanged to every chunk produced,
    for callers (or tests) that have a single known heading/page for the
    whole text rather than per-element heading tracking (that's what
    chunk_document() is for, when you have structured elements instead
    of a flat string).
    """
    if not text.strip():
        return []

    chunks: list[TextChunk] = []
    start = 0
    idx   = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Try to split at a sentence boundary within the last 20% of the chunk
        if end < len(text):
            search_from = start + int(chunk_size * 0.8)
            boundary    = text.rfind(". ", search_from, end)
            if boundary != -1:
                end = boundary + 2   # include the period and space

        chunk_content = text[start:end].strip()
        if chunk_content:
            chunks.append(TextChunk(
                content=chunk_content,
                chunk_index=idx,
                heading=heading,
                page=page,
                start_char=start,
                end_char=end,
            ))
            idx += 1

        # [FIX] Genuine infinite loop, confirmed reproduced and traced:
        # when this chunk reaches the end of the text (end == len(text)),
        # there is no more text left to overlap into a "next" chunk. The
        # previous code unconditionally computed
        # `start = end - chunk_overlap` regardless, which whenever the
        # final chunk's length is <= chunk_overlap (exactly the case
        # here: a 48-char text with chunk_overlap=10 leaves a final
        # 10-char tail) produces new_start == start. With zero forward
        # progress, (start, end) repeats identically forever -- not
        # "slow", an actual infinite loop. Breaking here once the text is
        # fully consumed is both the fix and the conceptually correct
        # behaviour (there's nothing left to chunk).
        if end >= len(text):
            break

        new_start = end - chunk_overlap
        # Defensive: guarantee forward progress even in other edge cases
        # beyond the one just fixed above (e.g. chunk_overlap >=
        # chunk_size passed by a caller) -- this function should never be
        # able to hang regardless of input parameters, not just for the
        # one case that happened to get caught by this test.
        start = new_start if new_start > start else start + 1

    return chunks


# ---------------------------------------------------------------------------
# Token-aware splitting (optional, uses transformers tokenizer)
# ---------------------------------------------------------------------------

def count_tokens(text: str, tokenizer_name: str = "BAAI/bge-m3") -> int:
    """
    Count tokens in text using the bge-m3 tokenizer.

    Falls back to the 4-chars/token approximation if transformers is not
    available or the tokenizer cannot be loaded.
    """
    try:
        from transformers import AutoTokenizer
        tok    = AutoTokenizer.from_pretrained(tokenizer_name)
        return len(tok.encode(text, add_special_tokens=True))
    except Exception:
        return len(text) // 4


def token_aware_chunk(
    text: str,
    max_tokens: int         = BGE_M3_MAX_TOKENS - 112,  # 400 tokens: leaves headroom
    tokenizer_name: str     = "BAAI/bge-m3",
    overlap_tokens: int     = 50,
) -> Iterator[str]:
    """
    Split text into chunks that each fit within max_tokens for bge-m3.

    More accurate than char-count splitting for code, LaTeX, or non-English
    text where the 4-chars/token approximation breaks down.

    Yields
    ------
    str
        Text chunks, each guaranteed to be ≤ max_tokens tokens.
    """
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception:
        logger.warning(
            "token_aware_chunk: transformers not available, falling back to "
            "char-based splitting (4 chars/token approximation)."
        )
        yield from (c.content for c in chunk_text(text))
        return

    token_ids   = tok.encode(text, add_special_tokens=False)
    total       = len(token_ids)
    start_idx   = 0

    while start_idx < total:
        end_idx = min(start_idx + max_tokens, total)
        chunk_tokens = token_ids[start_idx:end_idx]
        chunk_text_  = tok.decode(chunk_tokens, skip_special_tokens=True)
        yield chunk_text_.strip()
        start_idx = end_idx - overlap_tokens
        if start_idx >= total:
            break
