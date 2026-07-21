from __future__ import annotations

import json
import re
from typing import Any, Iterable

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.app.models.document import Document, DocumentChunk, SpreadsheetCell


def normalize_table_text(value: Any) -> str:
    """Normalize table labels without destroying Chinese business terms."""

    text = str(value or "").strip().lower()
    text = text.translate(str.maketrans({"（": "(", "）": ")", "：": ":", "／": "/"}))
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[/\\\-_—–]+", "", text)
    text = text.replace("第一季度", "1季度").replace("一季度", "1季度")
    text = text.replace("第二季度", "2季度").replace("二季度", "2季度")
    text = text.replace("第三季度", "3季度").replace("三季度", "3季度")
    text = text.replace("第四季度", "4季度").replace("四季度", "4季度")
    return text


def rebuild_spreadsheet_cell_index(
    db: Session,
    document: Document,
    chunks: Iterable[DocumentChunk] | None = None,
) -> int:
    """Replace one document's normalized cell index from persisted chunk metadata."""

    db.execute(delete(SpreadsheetCell).where(SpreadsheetCell.document_id == document.document_id))
    source_chunks = list(chunks) if chunks is not None else list(
        db.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document.document_id)
            .order_by(DocumentChunk.id.asc())
        )
    )
    created = 0
    for chunk in source_chunks:
        metadata = _metadata(chunk.chunk_metadata)
        if metadata.get("table_chunk_role") != "row":
            continue
        period = metadata.get("period") if isinstance(metadata.get("period"), dict) else {}
        sheet_name = str(metadata.get("sheet_name") or "")
        source_title = str(metadata.get("source_title") or metadata.get("table_title") or "")
        for cell in metadata.get("cells") or []:
            if not isinstance(cell, dict) or not str(cell.get("coordinate") or "").strip():
                continue
            numeric_value = cell.get("normalized_value")
            if isinstance(numeric_value, bool) or not isinstance(numeric_value, (int, float)):
                numeric_value = None
            db.add(
                SpreadsheetCell(
                    document_id=document.document_id,
                    chunk_id=chunk.chunk_id,
                    source_title=source_title or None,
                    source_title_norm=normalize_table_text(source_title),
                    sheet_name=sheet_name,
                    sheet_name_norm=normalize_table_text(sheet_name),
                    table_id=str(metadata.get("table_id") or "") or None,
                    table_title=str(metadata.get("table_title") or "") or None,
                    year=_int_or_none(period.get("year") or metadata.get("inferred_year")),
                    month=_int_or_none(period.get("month") or metadata.get("inferred_month")),
                    quarter=_int_or_none(period.get("quarter") or metadata.get("inferred_quarter")),
                    row_index=_int_or_none(cell.get("row")) or _int_or_none(metadata.get("row_index")) or 0,
                    column_index=_int_or_none(cell.get("column")) or 0,
                    coordinate=str(cell.get("coordinate")),
                    row_label=str(cell.get("row_label") or metadata.get("row_label") or ""),
                    row_label_norm=normalize_table_text(cell.get("row_label") or metadata.get("row_label")),
                    column_label=str(cell.get("column_label") or ""),
                    column_label_norm=normalize_table_text(cell.get("column_label")),
                    value=str(cell.get("value") or ""),
                    numeric_value=float(numeric_value) if numeric_value is not None else None,
                    unit=str(cell.get("unit") or metadata.get("unit") or "") or None,
                    is_formula=bool(cell.get("is_formula")),
                    formula=str(cell.get("formula") or "") or None,
                )
            )
            created += 1
    db.flush()
    return created


def _metadata(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None
