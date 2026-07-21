"""Score the manually reviewed 60-question trust challenge set.

The gold file is never imported by the application.  This evaluator rejects
unreviewed gold records and scores answer correctness, required-Aspect citation
coverage, semantic verdicts, prohibited conclusions and refusal behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_CATEGORY_COUNTS = {
    "regulatory_semantics": 20,
    "mixed_text_excel": 15,
    "cross_document": 5,
    "metadata_version_filter": 10,
    "ambiguity_refusal": 10,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    gold = read_jsonl(args.gold)
    results = {str(item.get("case_id")): item for item in read_jsonl(args.results)}
    report = evaluate(gold, results)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0 if report["gate_passed"] else 2


def evaluate(gold: list[dict[str, Any]], results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    category_counts = {
        category: sum(item.get("category") == category for item in gold)
        for category in EXPECTED_CATEGORY_COUNTS
    }
    structural_errors: list[str] = []
    if len(gold) != 60:
        structural_errors.append(f"challenge set must contain 60 cases; received {len(gold)}")
    for category, expected in EXPECTED_CATEGORY_COUNTS.items():
        if category_counts[category] != expected:
            structural_errors.append(
                f"{category} must contain {expected} cases; received {category_counts[category]}"
            )
    unreviewed = [str(item.get("case_id")) for item in gold if item.get("review_status") != "reviewed"]
    if unreviewed:
        structural_errors.append(f"{len(unreviewed)} cases are not manually reviewed")

    scored: list[dict[str, Any]] = []
    for case in gold:
        case_id = str(case.get("case_id"))
        actual = results.get(case_id, {})
        required_aspects = {str(value) for value in case.get("required_aspects") or []}
        cited_aspects = {str(value) for value in actual.get("cited_aspect_ids") or []}
        expected_refusal = bool(case.get("expected_refusal"))
        refusal_correct = bool(actual) and bool(actual.get("refused")) == expected_refusal
        forbidden = [
            text
            for text in case.get("forbidden_conclusions") or []
            if text and text in str(actual.get("answer") or "")
        ]
        semantic_expected = case.get("expected_semantic_frame") or {}
        semantic_actual = actual.get("semantic_frame") or {}
        semantic_correct = _mapping_contains(semantic_actual, semantic_expected) and not forbidden
        aspect_complete = required_aspects.issubset(cited_aspects)
        answer_correct = bool(actual.get("correct")) and refusal_correct and semantic_correct
        scored.append(
            {
                "case_id": case_id,
                "category": case.get("category"),
                "answer_correct": answer_correct,
                "aspect_citation_complete": aspect_complete,
                "semantic_correct": semantic_correct,
                "refusal_correct": refusal_correct,
                "forbidden_conclusions_found": forbidden,
                "missing_aspects": sorted(required_aspects - cited_aspects),
                "result_present": bool(actual),
            }
        )
    total = len(scored) or 1
    accuracy = sum(item["answer_correct"] for item in scored) / total
    aspect_completeness = sum(item["aspect_citation_complete"] for item in scored) / total
    semantic_accuracy = sum(item["semantic_correct"] for item in scored) / total
    contradiction_cases = [item for item in scored if item["category"] == "regulatory_semantics"]
    contradiction_rejection = (
        sum(item["semantic_correct"] for item in contradiction_cases) / len(contradiction_cases)
        if contradiction_cases else 0.0
    )
    gate_passed = (
        not structural_errors
        and accuracy >= 0.95
        and aspect_completeness == 1.0
        and semantic_accuracy >= 0.95
        and contradiction_rejection == 1.0
    )
    return {
        "gate_passed": gate_passed,
        "summary": {
            "case_count": len(gold),
            "result_count": len(results),
            "accuracy": round(accuracy, 4),
            "aspect_citation_completeness": round(aspect_completeness, 4),
            "semantic_accuracy": round(semantic_accuracy, 4),
            "contradiction_rejection": round(contradiction_rejection, 4),
            "category_counts": category_counts,
            "structural_errors": structural_errors,
        },
        "cases": scored,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _mapping_contains(actual: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _mapping_contains(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and set(map(str, expected)).issubset(map(str, actual))
    return actual == expected


if __name__ == "__main__":
    raise SystemExit(main())
