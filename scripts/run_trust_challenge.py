"""Execute the isolated trust challenge against the public QA API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any
import urllib.error
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--request-delay", type=float, default=0.05)
    args = parser.parse_args()
    cases = read_jsonl(args.gold)
    results: list[dict[str, Any]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        result = run_case(case, args.base_url, args.timeout)
        results.append(result)
        args.output.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results),
            encoding="utf-8",
        )
        if args.request_delay:
            time.sleep(args.request_delay)
    summary = {
        "case_count": len(cases),
        "completed_count": sum(not item.get("request_error") for item in results),
        "request_error_count": sum(bool(item.get("request_error")) for item in results),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["request_error_count"] == 0 else 2


def run_case(case: dict[str, Any], base_url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/qa/ask",
        data=json.dumps(
            {
                "question": case.get("question"),
                "options": case.get("options") or [],
                "include_debug": True,
            },
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return {"case_id": case.get("case_id"), "correct": False, "request_error": str(exc)}
    citations = payload.get("citations") or []
    grounding = payload.get("grounding_validation") or {}
    verdicts = grounding.get("claim_verdicts") or []
    evidence_coverage = payload.get("evidence_coverage") or {}
    cited_aspects = set(evidence_coverage.get("covered_aspect_ids") or [])
    for claim in payload.get("claims") or []:
        cited_aspects.update(str(value) for value in claim.get("aspect_ids") or [])
    accepted_documents = set(map(str, case.get("accepted_document_ids") or []))
    accepted_chunks = set(map(str, case.get("accepted_chunk_ids") or []))
    accepted_cells = set(map(str, case.get("accepted_cells") or []))
    actual_documents = {str(item.get("document_id")) for item in citations if item.get("document_id")}
    actual_chunks = {str(item.get("chunk_id")) for item in citations if item.get("chunk_id")}
    actual_cells: set[str] = set()
    for citation in citations:
        metadata = citation.get("metadata") or {}
        sheet = metadata.get("sheet_name")
        coordinate = metadata.get("coordinate")
        if coordinate:
            actual_cells.add(f"{sheet}!{coordinate}" if sheet else str(coordinate))
        for cell in metadata.get("calculation_cells") or []:
            if isinstance(cell, dict) and cell.get("cell"):
                actual_cells.add(
                    f"{cell.get('sheet_name')}!{cell.get('cell')}"
                    if cell.get("sheet_name")
                    else str(cell.get("cell"))
                )
    expected_refusal = bool(case.get("expected_refusal"))
    refused = bool(payload.get("refused"))
    reviewed = case.get("review_status") == "reviewed"
    evidence_ok = (
        (not accepted_documents or bool(actual_documents & accepted_documents))
        and (not accepted_chunks or bool(actual_chunks & accepted_chunks))
        and (not accepted_cells or accepted_cells.issubset(actual_cells))
    )
    answer = str(payload.get("answer") or "")
    forbidden = [value for value in case.get("forbidden_conclusions") or [] if value and value in answer]
    correct = reviewed and refused == expected_refusal and evidence_ok and not forbidden and bool(grounding.get("passed"))
    semantic_frame = verdicts[0].get("claim_frame") if verdicts and isinstance(verdicts[0], dict) else {}
    return {
        "case_id": case.get("case_id"),
        "correct": correct,
        "answer": answer,
        "refused": refused,
        "answer_type": payload.get("answer_type"),
        "citation_ids": sorted(actual_chunks),
        "source_document_ids": sorted(actual_documents),
        "cell_refs": sorted(actual_cells),
        "cited_aspect_ids": sorted(cited_aspects),
        "semantic_frame": semantic_frame or {},
        "grounding_validation": grounding,
        "forbidden_conclusions_found": forbidden,
        "request_error": None,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    raise SystemExit(main())
