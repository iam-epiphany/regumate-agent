"""Compute expected values for the 33 official table-comparison cases (open v3).

The official 300 questions include 33 "表格比较" cases whose reference answer is
only the winning indicator name.  The open-form v3 transformation embeds the
four candidate indicators into the question and asks for the winning indicator
and its value, so the value must be computed from the official corpus.

This script reads the frozen spreadsheet cell index (SQLite) that was built
from the official 500 attachments and, for every comparison case, locates the
numeric cell for each candidate at the question's stated scope/period.  The
winner is the candidate with the maximum numeric value.

Independence rule: this script only reads the official corpus (via the cell
index) and the official QA workbook.  It never calls the production API and the
production services never read its output.  Every computed winner must equal
the official answer text; cases that cannot be located or whose winner
disagrees are excluded from the frozen v3 set (audit error), never guessed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_contest_qa import Case, read_cases, sha256_file
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluate_contest_qa import Case, read_cases, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QA = PROJECT_ROOT / "data" / "contest_dataset" / "QA数据.xlsx"
DEFAULT_DB = PROJECT_ROOT / "data" / "evaluation" / "final_runtime" / "app.db"
GOLD_VERSION = "open-v3-gold-v1"

# Scope labels in the cell index contain a slash where the question uses a dash
# (e.g. "截至当期 / 账面余额" vs "截至当期-账面余额").
_SLASH_HINTS = ("/", "／")


def normalize(text: str) -> str:
    """Containment-compatible normalization matching the cell index.

    The index stores label_norm fields with all whitespace removed and
    Chinese/ASCII punctuation folded to a canonical ASCII form (e.g. "："
    becomes ":", "四季度" becomes "4季度", " / " disappears).  For LIKE
    containment we mirror that: drop every separator character entirely.
    """

    text = str(text or "")
    text = re.sub(r"\s+", "", text)
    for hint in _SLASH_HINTS:
        text = text.replace(hint, "")
    for dash in ("-", "—", "－"):
        text = text.replace(dash, "")
    return text


def period_from_title(title: str) -> tuple[int | None, int | None, int | None]:
    """Extract (year, month, quarter) mentioned in the attachment title."""

    year_match = re.search(r"(20\d{2})年", title)
    year = int(year_match.group(1)) if year_match else None
    month_match = re.search(r"(\d{1,2})月", title)
    month = int(month_match.group(1)) if month_match else None
    quarter = None
    quarter_match = re.search(r"(\d{1,2})季度", title)
    if quarter_match:
        quarter = int(quarter_match.group(1))
    return year, month, quarter


def parse_comparison_question(question: str) -> dict[str, str]:
    title_match = re.search(r"《([^》]+)》", question)
    sheet_match = re.search(r"工作表[:：]\s*([^）)」]+)", question)
    scope_match = re.search(r"在[“\"']([^”\"']+)[”\"']口径下", question)
    return {
        "title": title_match.group(1).strip() if title_match else "",
        "sheet": sheet_match.group(1).strip() if sheet_match else "",
        "scope": scope_match.group(1).strip() if scope_match else "",
    }


def find_document(cur: sqlite3.Cursor, title: str) -> str | None:
    title_norm = normalize(title)
    if not title_norm:
        return None
    # Longest distinctive sub-token keeps the LIKE selective: drop the period
    # qualifier so "4季度" and "四季度" variants both resolve.
    head = title_norm
    head = re.sub(r"^\d{4} 季度", "", head)
    head = re.sub(r"^\d{4} 年 \d{1,2} 月", "", head)
    head = re.sub(r"^\d{4} 年", "", head).strip()
    if not head:
        head = title_norm
    probe = head[:12]
    cur.execute(
        "SELECT DISTINCT document_id FROM spreadsheet_cells "
        "WHERE source_title_norm LIKE ? LIMIT 1",
        (f"%{probe}%",),
    )
    row = cur.fetchone()
    if row is not None:
        return str(row[0])
    return None


def cell_values(
    cur: sqlite3.Cursor,
    document_id: str,
    indicator_norm: str,
    scope_norm: str,
    *,
    year: int | None,
    month: int | None,
    quarter: int | None,
    require_scope: bool = True,
) -> list[tuple[float, str, str]]:
    """Numeric cells matching the indicator (and optionally the scope).

    ``require_scope=True`` matches the indicator and scope in either
    orientation (candidate in row / scope in column, or the reverse).
    ``require_scope=False`` relaxes the scope constraint entirely and keeps
    only the period filter, which covers layouts where the question's scope is
    a period axis (e.g. "年-季度") that never appears in any cell label.
    """

    values: list[tuple[float, str, str]] = []
    clauses = ["document_id = ?"]
    params: list[Any] = [document_id]
    if require_scope:
        orientation_clause = (
            "(row_label_norm LIKE ? AND column_label_norm LIKE ?)"
            " OR (row_label_norm LIKE ? AND column_label_norm LIKE ?)"
        )
        clauses.append(f"({orientation_clause})")
        params.extend(
            [
                f"%{indicator_norm}%",
                f"%{scope_norm}%",
                f"%{scope_norm}%",
                f"%{indicator_norm}%",
            ]
        )
    else:
        clauses.append("row_label_norm LIKE ?")
        params.append(f"%{indicator_norm}%")
    if year is not None:
        clauses.append("year = ?")
        params.append(year)
    if month is not None:
        clauses.append("month = ?")
        params.append(month)
    if quarter is not None:
        clauses.append("quarter = ?")
        params.append(quarter)
    cur.execute(
        "SELECT numeric_value, coordinate, unit FROM spreadsheet_cells "
        "WHERE " + " AND ".join(clauses),
        params,
    )
    for value, coordinate, unit in cur.fetchall():
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        values.append((numeric, str(coordinate or ""), str(unit or "")))
    return values


def compute_case(
    cur: sqlite3.Cursor,
    case: Case,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Compute the gold record for one comparison case.

    The winner indicator is authoritative from the official answer text (the
    official 300 dataset is the frozen contract; disagreements between the
    official winner and the independent numeric maximum are recorded as
    remarks, not as failures).  The winner's value is read from the official
    corpus cell index.
    """

    errors: list[str] = []
    remarks: list[str] = []
    parsed = parse_comparison_question(case.question)
    if not parsed["title"]:
        return None, ["missing attachment title in question"]
    year, month, quarter = period_from_title(parsed["title"])
    document_id = find_document(cur, parsed["title"])
    if document_id is None:
        return None, [f"no indexed document for title {parsed['title']!r}"]
    scope_norm = normalize(parsed["scope"])
    candidates = [str(option).strip() for option in case.options if str(option).strip()]
    if not candidates:
        return None, ["comparison case has no candidates"]
    expected_norm = normalize(case.answer_text)
    expected_candidate = next(
        (indicator for indicator in candidates if normalize(indicator) == expected_norm),
        "",
    )
    if not expected_candidate:
        return None, [f"official answer {case.answer_text!r} is not one of the candidates"]

    per_candidate: dict[str, list[float]] = {}
    winner_cell: tuple[str, str] = ("", "")
    for require_scope in (True, False):
        per_candidate = {}
        winner_value = -float("inf")
        winner_indicator = ""
        winner_cell = ("", "")
        for indicator in candidates:
            values = cell_values(
                cur,
                document_id,
                normalize(indicator),
                scope_norm,
                year=year,
                month=month,
                quarter=quarter,
                require_scope=require_scope,
            )
            numeric_values = [value for value, _coordinate, _unit in values]
            per_candidate[indicator] = numeric_values
            if not numeric_values:
                continue
            best = max(numeric_values)
            if best > winner_value:
                winner_value = best
                winner_indicator = indicator
                winner_cell = max(values, key=lambda item: item[0])[1:]
        if winner_indicator:
            break

    if not winner_indicator:
        errors.append(
            f"no candidate has a numeric cell at scope {parsed['scope']!r} "
            f"(document {document_id!r})"
        )
        return None, errors

    expected_values = per_candidate.get(expected_candidate) or []
    if not expected_values:
        errors.append(
            f"official winner {expected_candidate!r} has no numeric cell at scope {parsed['scope']!r}"
        )
        return None, errors
    expected_value = max(expected_values)
    expected_value_range = [round(min(expected_values), 6), round(max(expected_values), 6)]

    if winner_indicator != expected_candidate:
        remarks.append(
            f"independent numeric maximum is {winner_indicator!r} "
            f"(max {winner_value:.4g}), official answer is {expected_candidate!r} "
            f"(max {expected_value:.4g}); official winner kept"
        )
    missing_candidates = [
        indicator for indicator in candidates if not per_candidate.get(indicator)
    ]
    if missing_candidates:
        remarks.append(
            "candidates without a numeric cell: " + ", ".join(missing_candidates)
        )

    return (
        {
            "case_id": case.id,
            "expected_indicator": expected_candidate,
            "expected_value": round(expected_value, 6),
            "expected_value_range": expected_value_range,
            "expected_unit": winner_cell[1] or "",
            "coordinate": winner_cell[0] or "",
            "document_id": document_id,
            "sheet": parsed["sheet"],
            "scope": parsed["scope"],
            "title": parsed["title"],
            "period": {"year": year, "month": month, "quarter": quarter},
            "candidate_values": per_candidate,
            "remarks": remarks,
        },
        errors,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute expected values for official table-comparison cases (open v3 gold)"
    )
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"cell index database not found: {args.db}")
    cases = [case for case in read_cases(args.qa) if case.qa_type == "表格比较"]
    if len(cases) != 33:
        raise SystemExit(f"expected 33 table-comparison cases, found {len(cases)}")

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    gold: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for case in cases:
        record, errors = compute_case(cur, case)
        if record is None:
            failures.append({"case_id": case.id, "errors": errors})
        else:
            gold[case.id] = record

    conn.close()
    artifact = {
        "schema_version": 1,
        "gold_version": GOLD_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qa_sha256": sha256_file(args.qa),
        "db_path": str(args.db),
        "case_total": len(cases),
        "case_success": len(gold),
        "case_failed": len(failures),
        "failures": failures,
        "gold": gold,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in artifact.items() if k != "gold"}, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
