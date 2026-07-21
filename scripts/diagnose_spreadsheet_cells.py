from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from sqlalchemy import select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _bootstrap_data_dir() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir", type=Path)
    known, _ = parser.parse_known_args()
    if known.data_dir:
        os.environ["REGUMATE_DATA_DIR"] = str(known.data_dir.resolve())


_bootstrap_data_dir()

from backend.app.core.database import SessionLocal  # noqa: E402
from backend.app.models.document import Document, DocumentChunk, SpreadsheetCell  # noqa: E402
from backend.app.services.spreadsheet_cell_index_service import normalize_table_text  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only SpreadsheetCell diagnostic query.")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--source")
    parser.add_argument("--sheet")
    parser.add_argument("--year", type=int)
    parser.add_argument("--month", type=int)
    parser.add_argument("--quarter", type=int)
    parser.add_argument("--indicator")
    parser.add_argument("--row-label")
    parser.add_argument("--column-label")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    with SessionLocal() as db:
        statement = (
            select(SpreadsheetCell, DocumentChunk, Document)
            .join(DocumentChunk, SpreadsheetCell.chunk_id == DocumentChunk.chunk_id)
            .join(Document, SpreadsheetCell.document_id == Document.document_id)
            .where(Document.status.in_(["indexed", "table_indexed"]))
        )
        if args.source:
            statement = statement.where(SpreadsheetCell.source_title_norm.contains(normalize_table_text(args.source)))
        if args.sheet:
            statement = statement.where(SpreadsheetCell.sheet_name_norm.contains(normalize_table_text(args.sheet)))
        for key in ("year", "month", "quarter"):
            value = getattr(args, key)
            if value is not None:
                statement = statement.where(getattr(SpreadsheetCell, key) == value)

        rows = db.execute(statement.limit(5000)).all()
        terms = [
            normalize_table_text(value)
            for value in (args.indicator, args.row_label, args.column_label)
            if value
        ]
        filtered = []
        for cell, chunk, document in rows:
            label_text = normalize_table_text(
                " ".join([cell.row_label or "", cell.column_label or "", cell.table_title or "", cell.source_title or ""])
            )
            if terms and not all(term in label_text for term in terms):
                continue
            filtered.append((cell, chunk, document))

        payload = {
            "count": len(filtered),
            "base_count": len(rows),
            "query": {
                "source": args.source,
                "sheet": args.sheet,
                "year": args.year,
                "month": args.month,
                "quarter": args.quarter,
                "indicator": args.indicator,
                "row_label": args.row_label,
                "column_label": args.column_label,
            },
            "candidates": [
                {
                    "document_id": document.document_id,
                    "filename": document.filename,
                    "sheet": cell.sheet_name,
                    "coordinate": cell.coordinate,
                    "row_label": cell.row_label,
                    "column_label": cell.column_label,
                    "value": cell.value,
                    "unit": cell.unit,
                    "chunk_id": chunk.chunk_id,
                }
                for cell, chunk, document in filtered[: args.limit]
            ],
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
