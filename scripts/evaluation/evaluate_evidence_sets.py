"""Offline evidence-set evaluation for paired QA outputs.

This script reads an existing batch output JSON file and the
matching ``gold.jsonl``.  It does not call the QA service, a model, or a
vector store, and it never feeds gold data into production code.

Reported signals are deliberately separated:

* ``document_hit``    - the cited file matches a required source file.
* ``evidence_hit``    - the required evidence text appears in a cited excerpt.
* ``article_locatable`` - the required evidence excerpt was found in a
  citation that also carries a resolved structural coordinate.  This is a
  locatability proxy, not a claim that the gold article number was supplied.
* ``structure_coverage`` - share of all citations with a resolved/partial
  document -> chapter/section -> article coordinate.
* ``sufficiency``     - the deterministic evidence-set audit attached to the
  response, if present.

Usage:
    python scripts/evaluation/evaluate_evidence_sets.py \
        --outputs evaluation/.../old_outputs.json \
        --gold data/.../gold.jsonl \
        --out evaluation/.../evidence_set_report.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[str(row.get("id"))] = row
    return rows


def _normalize(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("《", "").replace("》", "")
    text = re.sub(r"\.(?:pdf|docx?|xlsx?|xls|csv|txt|md)$", "", text)
    text = re.sub(r"[\s·•、，,。.;；:：()（）\"'“”‘’\-_]+", "", text)
    return text


def _file_key(value: Any) -> str:
    text = _normalize(value)
    return re.sub(r"^\d+[_-]+", "", text)


def _citation_source(citation: dict[str, Any]) -> str:
    metadata = citation.get("metadata") or {}
    return " ".join(
        str(part) for part in (
            citation.get("filename"),
            citation.get("source_title"),
            metadata.get("source_title"),
            metadata.get("filename"),
        ) if part
    )


def _citation_text(citation: dict[str, Any]) -> str:
    return " ".join(
        str(part) for part in (
            citation.get("excerpt"),
            citation.get("section_title"),
        ) if part
    )


def _locator_status(citation: dict[str, Any]) -> str:
    locator = citation.get("structure_locator") or {}
    return str(locator.get("resolution_status") or "unresolved")


def _evidence_present(text: str, evidence: str) -> tuple[bool, bool]:
    """Return (exact normalized containment, prefix containment)."""

    normalized_text = _normalize(text)
    normalized_evidence = _normalize(evidence)
    if not normalized_evidence:
        return False, False
    exact = normalized_evidence in normalized_text
    prefix = normalized_evidence[: max(24, min(60, len(normalized_evidence)))] in normalized_text
    return exact, prefix


def _evaluate_row(row: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    response = row.get("response") or {}
    citations = response.get("citations") or []
    required_sources = gold.get("required_sources") or []
    document_hits = 0
    evidence_exact = 0
    evidence_prefix = 0
    article_locatable = 0

    for required in required_sources:
        if not isinstance(required, dict):
            continue
        relative_path = str(required.get("relative_path") or "")
        evidence_text = str(required.get("evidence_text") or "")
        file_key = _file_key(relative_path)
        source_hit = bool(file_key) and any(
            file_key in _file_key(_citation_source(citation)) for citation in citations
        )
        document_hits += int(source_hit)

        matched_citations = [
            citation for citation in citations
            if _evidence_present(_citation_text(citation), evidence_text)[0]
        ]
        exact = bool(matched_citations)
        prefix = bool(matched_citations) or any(
            _evidence_present(_citation_text(citation), evidence_text)[1] for citation in citations
        )
        evidence_exact += int(exact)
        evidence_prefix += int(prefix)
        article_locatable += int(any(
            _locator_status(citation) == "resolved" for citation in matched_citations
        ))

    locator_counts = Counter(_locator_status(citation) for citation in citations)
    sufficiency = ((response.get("explanation") or {}).get("sufficiency")) or {}
    total = len(required_sources)
    return {
        "id": row.get("id"),
        "error": row.get("error"),
        "latency_ms": row.get("latency_ms"),
        "required_sources": total,
        "document_hit_rate": round(document_hits / total, 4) if total else None,
        "evidence_exact_rate": round(evidence_exact / total, 4) if total else None,
        "evidence_prefix_rate": round(evidence_prefix / total, 4) if total else None,
        "article_locatable_rate": round(article_locatable / total, 4) if total else None,
        "citation_count": len(citations),
        "structure_coverage": (
            round((locator_counts.get("resolved", 0) + locator_counts.get("partial", 0))
                  / len(citations), 4)
            if citations else None
        ),
        "sufficiency_status": sufficiency.get("status"),
        "missing_facets": len(sufficiency.get("missing_facets") or []),
        "expanded_chunks": int((sufficiency.get("expansion") or {}).get("expanded_chunks") or 0),
    }


def _mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return round(sum(present) / len(present), 4) if present else None


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if not row.get("error")]
    sufficiency = Counter(row.get("sufficiency_status") or "unrecorded" for row in completed)
    return {
        "rows": len(rows),
        "completed": len(completed),
        "errors": len(rows) - len(completed),
        "document_hit_rate": _mean(row.get("document_hit_rate") for row in completed),
        "evidence_exact_rate": _mean(row.get("evidence_exact_rate") for row in completed),
        "evidence_prefix_rate": _mean(row.get("evidence_prefix_rate") for row in completed),
        "article_locatable_rate": _mean(row.get("article_locatable_rate") for row in completed),
        "structure_coverage": _mean(row.get("structure_coverage") for row in completed),
        "mean_citations": _mean(float(row.get("citation_count") or 0) for row in completed),
        "mean_expanded_chunks": _mean(float(row.get("expanded_chunks") or 0) for row in completed),
        "sufficiency_counts": dict(sorted(sufficiency.items())),
        "mean_missing_facets": _mean(float(row.get("missing_facets") or 0) for row in completed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", required=True, help="Batch output JSON containing per-question responses")
    parser.add_argument("--gold", required=True, help="matching gold.jsonl")
    parser.add_argument("--out", required=True, help="report JSON path")
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N rows")
    args = parser.parse_args()

    outputs_path = Path(args.outputs)
    payload = json.loads(outputs_path.read_text(encoding="utf-8-sig"))
    results = list(payload.get("results") or [])
    if args.limit:
        results = results[: args.limit]
    gold_rows = _load_jsonl(Path(args.gold))

    rows = []
    missing_gold = 0
    for result in results:
        gold = gold_rows.get(str(result.get("id")))
        if gold is None:
            missing_gold += 1
            continue
        rows.append(_evaluate_row(result, gold))

    report = {
        "schema_version": "1.0",
        "outputs": str(outputs_path),
        "gold": str(Path(args.gold)),
        "evaluated": len(rows),
        "missing_gold": missing_gold,
        "aggregate": _aggregate(rows),
        "results": rows,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
