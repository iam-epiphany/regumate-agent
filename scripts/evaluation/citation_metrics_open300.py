"""Citation and latency metrics for an official-300 open-question results file.

Computed purely offline from the runner output (no service calls):

* ``citation_returned_rate``  - answered questions carrying >= 1 citation.
* ``structured_citation_rate`` - answered questions whose citations attach a
  clause number / table-cell coordinate / resolved or partial structural
  locator (the "附条款/坐标" citation rate).
* ``gold_source_hit_rate``    - the gold source file appears among citation
  filenames/titles (official-300 gold_evidence carries the numbered filename).
* ``gold_value_in_citation``  - the gold answer value/text appears in some
  cited excerpt.
* latency stats overall, by source_type and by answer_type.

Usage:
    python scripts/evaluation/citation_metrics_open300.py \
        --results outputs/official300_open/open_300_results_20261024.json \
        --questions outputs/official300_open/open_questions_all.jsonl \
        --out outputs/official300_open/citation_metrics_20261024.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


def _normalize(value: Any) -> str:
    text = str(value or "").lower()
    return re.sub(r"[\s·•、，,。.;；:：()（）\"'“”‘’\-_/\\]+", "", text)


def _numbers(text: str) -> list[float]:
    values: list[float] = []
    for match in re.finditer(r"-?\d+(?:,\d{3})*(?:\.\d+)?", str(text or "")):
        token = match.group(0).replace(",", "")
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _value_hit(gold: str, excerpts: str) -> bool:
    gold_numbers = _numbers(gold)
    if gold_numbers:
        excerpt_numbers = _numbers(excerpts)
        return any(
            any(
                abs(target - candidate) <= max(0.01, abs(target) * 0.001)
                for candidate in excerpt_numbers
            )
            for target in gold_numbers
        )
    gold_norm = _normalize(gold)
    excerpt_norm = _normalize(excerpts)
    return bool(gold_norm) and gold_norm[:30] in excerpt_norm


def _source_keys(source_title: str, gold_evidence: str) -> list[str]:
    keys: list[str] = []
    normalized_title = _normalize(source_title)
    if normalized_title:
        keys.append(normalized_title)
    basename = re.search(
        r"(\d{1,4})_[^/\\；;]+?\.(?:xlsx?|pdf|docx?)", str(gold_evidence or "")
    )
    if basename:
        keys.append(_normalize(basename.group(0).split("_", 1)[1]))
        keys.append(_normalize(basename.group(0)))
    return [key for key in dict.fromkeys(keys) if key]


def _citation_text(citation: dict[str, Any]) -> str:
    return " ".join(
        str(part)
        for part in (
            citation.get("excerpt"),
            citation.get("section_title"),
            citation.get("structure_label"),
        )
        if part
    )


def _has_clause_or_coordinate(citation: dict[str, Any]) -> bool:
    locator = citation.get("structure_locator") or {}
    if str(locator.get("resolution_status") or "") in {"resolved", "partial"}:
        return True
    metadata = citation.get("metadata") or {}
    if (
        metadata.get("chunk_type") == "table_cell"
        or citation.get("chunk_type") == "table_cell"
    ):
        return True
    if (
        citation.get("page_number") is not None
        or metadata.get("page_number") is not None
    ):
        return True
    if citation.get("section_number") or citation.get("parent_section_number"):
        return True
    for value in (citation.get("section_title"), citation.get("structure_label")):
        if re.search(r"第[一二三四五六七八九十百\d]+条", str(value or "")):
            return True
    path = locator.get("path") or []
    return bool(path)


def _has_broad_locator(citation: dict[str, Any]) -> bool:
    locator = citation.get("structure_locator") or {}
    return bool(str(citation.get("section_title") or "").strip()) or bool(
        str(locator.get("display") or "").strip()
    )


def _latency_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean_s": None, "p50_s": None, "p95_s": None, "max_s": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return {
        "count": len(values),
        "mean_s": round(sum(values) / len(values), 3),
        "p50_s": round(median(ordered), 3),
        "p95_s": round(ordered[p95_index], 3),
        "max_s": round(ordered[-1], 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8-sig"))
    rows = list(payload.get("results") or [])
    questions = {
        str(json.loads(line)["id"]): json.loads(line)
        for line in args.questions.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }

    answered = []
    citation_returned = 0
    structured = 0
    broad_structured = 0
    gold_source_hit = 0
    gold_value_hit = 0
    per_row: list[dict[str, Any]] = []
    latencies: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        response = row.get("response") or {}
        error = row.get("error")
        question = questions.get(str(row.get("id"))) or {}
        gold_answer = str(question.get("answer_text") or "")
        refused = bool(response.get("refused")) or response.get("answer_type") in {
            "clarification",
            "refusal",
        }
        citations = response.get("citations") or []
        latencies["all"].append(float(row.get("latency_ms") or 0) / 1000)
        latencies[f"source:{row.get('source_type')}"].append(
            float(row.get("latency_ms") or 0) / 1000
        )
        latencies[f"answer_type:{response.get('answer_type')}"].append(
            float(row.get("latency_ms") or 0) / 1000
        )
        if error or refused or not str(response.get("answer") or "").strip():
            per_row.append(
                {
                    "id": row.get("id"),
                    "answered": False,
                    "refused": refused,
                    "error": error,
                }
            )
            continue
        answered.append(row)
        excerpts = "\n".join(_citation_text(citation) for citation in citations)
        keys = _source_keys(
            str(question.get("source_title") or ""),
            str(question.get("gold_evidence") or ""),
        )
        citation_filenames = _normalize(
            " ".join(
                str(c.get("filename") or "") + str(c.get("source_title") or "")
                for c in citations
            )
        )
        source_hit = any(key and key in citation_filenames for key in keys)
        value_hit = _value_hit(gold_answer, excerpts) if gold_answer else False
        has_structure = any(
            _has_clause_or_coordinate(citation) for citation in citations
        )
        has_broad = any(_has_broad_locator(citation) for citation in citations)
        citation_returned += int(bool(citations))
        structured += int(has_structure)
        broad_structured += int(has_broad)
        gold_source_hit += int(source_hit)
        gold_value_hit += int(value_hit)
        per_row.append(
            {
                "id": row.get("id"),
                "answered": True,
                "citation_count": len(citations),
                "citation_returned": bool(citations),
                "structured_citation": has_structure,
                "gold_source_hit": source_hit,
                "gold_value_in_citation": value_hit,
            }
        )

    total_answered = len(answered)
    summary = {
        "total": len(rows),
        "answered": total_answered,
        "citation_returned_rate": round(citation_returned / total_answered, 4)
        if total_answered
        else None,
        "structured_citation_rate": round(structured / total_answered, 4)
        if total_answered
        else None,
        "structured_citation_rate_broad": round(broad_structured / total_answered, 4)
        if total_answered
        else None,
        "gold_source_hit_rate": round(gold_source_hit / total_answered, 4)
        if total_answered
        else None,
        "gold_value_in_citation_rate": round(gold_value_hit / total_answered, 4)
        if total_answered
        else None,
        "latency": {
            key: _latency_stats([value for value in values if value > 0])
            for key, values in sorted(latencies.items())
        },
        "metric_formula": {
            "citation_returned_rate": "实际作答题中 citations>=1 的占比",
            "structured_citation_rate": "实际作答题中至少一条引用附条款号/表格单元格坐标/页码/已解析结构定位（resolved|partial）的占比",
            "gold_source_hit_rate": "实际作答题中金标来源文件出现在引用 filename/source_title 的占比",
            "gold_value_in_citation_rate": "实际作答题中金标数值（±0.1%）或金标文本出现在引用摘录中的占比",
        },
    }
    artifact = {
        "schema_version": 1,
        "results_file": str(args.results),
        "summary": summary,
        "rows": per_row,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
