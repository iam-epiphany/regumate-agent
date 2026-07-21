from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from evaluate_contest_qa import assign_splits, normalize_evidence, read_cases, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QA = PROJECT_ROOT / "data" / "contest_dataset" / "QA数据.xlsx"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a reviewable evidence manifest seed from a frozen baseline artifact."
    )
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-aspect-recall", type=float, default=0.55)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    baseline_by_id = {str(item["id"]): item for item in baseline.get("results") or []}
    cases = assign_splits(read_cases(args.qa))
    manifest_cases: list[dict[str, Any]] = []
    for case in cases:
        result = baseline_by_id.get(case.id) or {}
        if case.source_type == "excel":
            evidence = result.get("excel_evidence") or {}
            required_cells = evidence.get("expected_cells") or []
            manifest_cases.append(
                {
                    "case_id": case.id,
                    "source_type": case.source_type,
                    "required_aspects": [
                        {
                            "aspect_id": f"cell:{cell}",
                            "evidence": cell,
                            "acceptable_chunk_ids": _chunks_for_cell(result.get("citations") or [], cell),
                        }
                        for cell in required_cells
                    ],
                    "review_status": "reviewed" if required_cells else "needs_review",
                }
            )
            continue

        aspects = _evidence_aspects(case.evidence)
        manifest_aspects = []
        for index, aspect in enumerate(aspects, start=1):
            acceptable = []
            candidates = []
            for citation in result.get("citations") or []:
                recall = _bigram_recall(aspect, str(citation.get("excerpt") or ""))
                if recall > 0:
                    candidates.append(
                        {
                            "chunk_id": citation.get("chunk_id"),
                            "recall": round(recall, 4),
                        }
                    )
                if recall >= args.minimum_aspect_recall:
                    acceptable.append(str(citation.get("chunk_id")))
            manifest_aspects.append(
                {
                    "aspect_id": f"aspect:{index}",
                    "evidence": aspect,
                    "acceptable_chunk_ids": list(dict.fromkeys(acceptable)),
                    "baseline_candidates": sorted(candidates, key=lambda item: item["recall"], reverse=True),
                }
            )
        manifest_cases.append(
            {
                "case_id": case.id,
                "source_type": case.source_type,
                "required_aspects": manifest_aspects,
                "review_status": (
                    "seeded"
                    if manifest_aspects and all(item["acceptable_chunk_ids"] for item in manifest_aspects)
                    else "needs_review"
                ),
            }
        )

    payload = {
        "schema_version": "regumate-evidence-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "qa_sha256": sha256_file(args.qa),
        "baseline_sha256": sha256_file(args.baseline),
        "minimum_aspect_recall": args.minimum_aspect_recall,
        "review_policy": (
            "seeded entries are machine candidates, not final ground truth; every needs_review entry "
            "and any ambiguous seeded entry must be reviewed before strict quality gating"
        ),
        "cases": manifest_cases,
    }
    canonical = json.dumps(payload["cases"], ensure_ascii=False, sort_keys=True).encode("utf-8")
    payload["cases_sha256"] = hashlib.sha256(canonical).hexdigest()
    payload["summary"] = {
        "case_count": len(manifest_cases),
        "seeded_count": sum(item["review_status"] in {"seeded", "reviewed"} for item in manifest_cases),
        "needs_review_count": sum(item["review_status"] == "needs_review" for item in manifest_cases),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False))
    return 0


def _chunks_for_cell(citations: list[dict[str, Any]], cell: str) -> list[str]:
    target = cell.upper()
    result: list[str] = []
    for citation in citations:
        metadata = citation.get("metadata") or {}
        cells = [metadata.get("cell")]
        for key in ("calculation_cells", "comparison_cells"):
            cells.extend(item.get("cell") for item in metadata.get(key) or [] if isinstance(item, dict))
        if target in {str(item or "").upper() for item in cells}:
            result.append(str(citation.get("chunk_id")))
    return list(dict.fromkeys(result))


def _evidence_aspects(evidence: str) -> list[str]:
    return [item.strip() for item in re.split(r"[；;。\n]+", evidence) if item.strip()]


def _bigram_recall(expected: str, actual: str) -> float:
    expected_norm = normalize_evidence(expected)
    actual_norm = normalize_evidence(actual)
    if not expected_norm:
        return 0.0
    grams = (
        {expected_norm[index : index + 2] for index in range(len(expected_norm) - 1)}
        if len(expected_norm) >= 2
        else {expected_norm}
    )
    if not grams:
        return 0.0
    return sum(gram in actual_norm for gram in grams) / len(grams)


if __name__ == "__main__":
    raise SystemExit(main())
