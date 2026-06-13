"""
backend/ingestion/parsers/spreadsheet_parser.py
================================================
Parses Excel (.xlsx) and CSV files into ``ParsedDocument``.

Strategy:
  - Each worksheet becomes a separate section (heading = sheet name).
  - Rows are serialised as Markdown tables (for retrieval) and also as
    row-level text records so individual data points are findable.
  - Non-data sheets (charts-only, very sparse) are skipped.
  - Column headers are treated as context for every row.
"""

from __future__ import annotations

import logging
from pathlib import Path

from backend.ingestion.parsers.base import (
    DocumentElement,
    ElementType,
    ParsedDocument,
)

logger = logging.getLogger(__name__)

_MIN_ROWS = 2          # skip sheets with fewer rows than this
_MAX_ROWS_PER_TABLE = 200  # very large sheets: chunk at this row count


def parse(path: Path) -> ParsedDocument:
    """
    Parse an Excel or CSV file.

    Parameters
    ----------
    path:
        Path to a ``.xlsx`` or ``.csv`` file.

    Returns
    -------
    ParsedDocument
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is not installed. Run: uv add pandas openpyxl") from exc

    suffix = path.suffix.lower()
    elements: list[DocumentElement] = []

    if suffix in {".xlsx", ".xls", ".xlsm"}:
        sheets = _read_excel(path, pd)
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        parser = "openpyxl+pandas"
    elif suffix == ".csv":
        df = pd.read_csv(path, dtype=str).fillna("")
        sheets = {path.stem: df}
        mime = "text/csv"
        parser = "pandas"
    else:
        raise ValueError(f"Unsupported spreadsheet format: {suffix}")

    for sheet_name, df in sheets.items():
        if len(df) < _MIN_ROWS:
            logger.debug("Skipping sparse sheet '%s' (%d rows)", sheet_name, len(df))
            continue

        # Sheet heading
        elements.append(DocumentElement(
            element_type=ElementType.HEADING,
            content=sheet_name,
            level=1,
            heading=sheet_name,
        ))

        # Column overview text
        col_summary = f"Sheet '{sheet_name}' columns: {', '.join(str(c) for c in df.columns)}"
        elements.append(DocumentElement(
            element_type=ElementType.TEXT,
            content=col_summary,
            heading=sheet_name,
        ))

        # Chunk into Markdown tables
        for start in range(0, len(df), _MAX_ROWS_PER_TABLE):
            chunk_df = df.iloc[start : start + _MAX_ROWS_PER_TABLE]
            md_table = _df_to_markdown(chunk_df)
            elements.append(DocumentElement(
                element_type=ElementType.TABLE,
                content=md_table,
                heading=sheet_name,
                metadata={
                    "sheet": sheet_name,
                    "row_start": start,
                    "row_end": start + len(chunk_df),
                    "has_table": True,
                },
            ))

    logger.info(
        "Spreadsheet parsed %s → %d elements across %d sheet(s)",
        path.name,
        len(elements),
        len(sheets),
    )

    return ParsedDocument(
        filename=path.name,
        mime_type=mime,
        elements=elements,
        parser_used=parser,
    )


def _read_excel(path: Path, pd: object) -> dict:
    """Read all sheets from an Excel file, return {sheet_name: DataFrame}."""
    import pandas as pd_ # type: ignore[import]
    xl = pd_.ExcelFile(path, engine="openpyxl")
    result = {}
    for sheet in xl.sheet_names:
        try:
            df = xl.parse(sheet, dtype=str).fillna("")
            result[sheet] = df
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not parse sheet '%s': %s", sheet, exc)
    return result


def _df_to_markdown(df) -> str:
    """Convert a pandas DataFrame to a Markdown table string."""
    headers = [str(c) for c in df.columns]
    sep = ["---"] * len(headers)
    rows = [headers, sep]
    for _, row in df.iterrows():
        rows.append([str(v).replace("|", "\\|") for v in row])
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)
