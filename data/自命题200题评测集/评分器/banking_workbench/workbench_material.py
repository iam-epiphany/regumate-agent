"""Banking Workbench material extraction.

Evaluation-only.  Scans the indexed corpus (SQLite) for question-worthy
material: definition sentences, obligation/prohibition clauses, numeric
thresholds and spreadsheet cell coordinates, so the question author can build
the 100-case gold from real corpus evidence.

The output is a candidate-material list; authoring decisions stay manual.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB = PROJECT_ROOT / "data" / "evaluation" / "final_runtime" / "app.db"

DEFINITION_PATTERN = re.compile(r"(?:是指|定义为|称为|含义是|包括以下|属于)")
OBLIGATION_PATTERN = re.compile(r"(?:应当|应(?!用|当)|须|必须|需要|负责)")
PROHIBITION_PATTERN = re.compile(r"(?:不得|禁止|不应|不允许|不得以|严禁)")
THRESHOLD_PATTERN = re.compile(
    r"(?:[0-9]+(?:\.[0-9]+)?|[一二两三四五六七八九十百]+)\s*"
    r"(?:%|％|个百分点|个工作日|工作日|日|天|个月|月|年|倍|万元|亿元|元|项|个|笔|家|户|网点|次)"
)
ORDINAL_PATTERN = re.compile(r"^(?:第[一二三四五六七八九十百\d]+条|[一二三四五六七八九十]+、|\(\d+\)|\d+[.、])")


def extract_text_material(db_path: Path, limit_per_document: int = 8) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT d.filename, c.source_file, c.chunk_id, c.text, c.section_title "
            "FROM document_chunks c JOIN documents d ON d.document_id = c.document_id "
            "WHERE d.status = 'indexed' AND c.text IS NOT NULL AND length(c.text) > 20 "
            "ORDER BY d.filename, c.id"
        ).fetchall()
    finally:
        conn.close()

    material: list[dict[str, Any]] = []
    per_document: Counter = Counter()
    for filename, source_file, chunk_id, text, section_title in rows:
        if per_document[filename] >= limit_per_document:
            continue
        cleaned = re.sub(r"\s+", "", text)
        if len(cleaned) < 15 or len(cleaned) > 600:
            continue
        kind: list[str] = []
        if DEFINITION_PATTERN.search(cleaned):
            kind.append("definition")
        if OBLIGATION_PATTERN.search(cleaned):
            kind.append("obligation")
        if PROHIBITION_PATTERN.search(cleaned):
            kind.append("prohibition")
        if THRESHOLD_PATTERN.search(cleaned):
            kind.append("threshold")
        if not kind:
            continue
        # skip appendix/目录-like chunks and ordinal-only headings
        if ORDINAL_PATTERN.match(cleaned) and len(cleaned) < 40 and not kind:
            continue
        per_document[filename] += 1
        material.append(
            {
                "document": filename,
                "source_file": source_file,
                "chunk_id": chunk_id,
                "section_title": section_title,
                "text": text.strip(),
                "kinds": kind,
            }
        )
    return material


def extract_table_material(db_path: Path, limit_per_document: int = 12) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT d.filename, s.sheet_name, s.row_index, s.column_index, "
            "s.coordinate, s.row_label, s.column_label, s.value, s.numeric_value, s.unit "
            "FROM spreadsheet_cells s JOIN documents d ON d.document_id = s.document_id "
            "WHERE d.status = 'indexed' AND s.numeric_value IS NOT NULL "
            "ORDER BY d.filename, s.sheet_name, s.row_index, s.column_index"
        ).fetchall()
    finally:
        conn.close()

    material: list[dict[str, Any]] = []
    per_document: Counter = Counter()
    for filename, sheet, row, column, coordinate, row_label, column_label, value, numeric, unit in rows:
        if per_document[filename] >= limit_per_document:
            continue
        row_label = str(row_label or "").strip()
        column_label = str(column_label or "").strip()
        if not row_label or not column_label:
            continue
        if row_label in {"项目", "指标", "机构"} or column_label in {"项目", "指标", "机构"}:
            continue
        per_document[filename] += 1
        material.append(
            {
                "document": filename,
                "source_file": filename,
                "sheet": sheet,
                "row": row,
                "column": column,
                "coordinate": coordinate,
                "row_label": row_label,
                "column_label": column_label,
                "value": value,
                "numeric_value": numeric,
                "unit": unit,
            }
        )
    return material


def main() -> int:
    parser = argparse.ArgumentParser(description="Banking Workbench material extraction")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-limit", type=int, default=8)
    parser.add_argument("--table-limit", type=int, default=12)
    args = parser.parse_args()
    text_material = extract_text_material(args.db, args.text_limit)
    table_material = extract_table_material(args.db, args.table_limit)
    artifact = {
        "schema_version": 1,
        "text_material_count": len(text_material),
        "table_material_count": len(table_material),
        "document_count_text": len({item["document"] for item in text_material}),
        "document_count_table": len({item["document"] for item in table_material}),
        "text_material": text_material,
        "table_material": table_material,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in artifact.items() if k not in ("text_material", "table_material")},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
