"""Enforce contest-quality and latency gates on one frozen evaluation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--ood-result", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8-sig"))
    baseline = json.loads(args.baseline.read_text(encoding="utf-8-sig"))
    ood_result = json.loads(args.ood_result.read_text(encoding="utf-8-sig"))
    summary = result.get("summary") or {}
    baseline_summary = baseline.get("summary") or {}
    ood_summary = ood_result.get("summary") or {}
    p95 = float((summary.get("elapsed_ms") or {}).get("p95") or 0.0)
    baseline_p95 = float((baseline_summary.get("elapsed_ms") or {}).get("p95") or 0.0)
    gates = {
        "completed_without_errors": int(summary.get("error_count") or 0) == 0,
        "answer_accuracy_100": float(summary.get("answer_accuracy") or 0.0) == 1.0,
        "source_hit_100": float(summary.get("source_hit_rate") or 0.0) == 1.0,
        "excel_cell_recall_100": float(summary.get("excel_cell_recall") or 0.0) == 1.0,
        "citation_coverage_100": float(summary.get("citation_coverage_rate") or 0.0) == 1.0,
        "grounding_pass_100": float(summary.get("grounding_pass_rate") or 0.0) == 1.0,
        "key_entity_error_0": float(summary.get("key_entity_error_rate") or 0.0) == 0.0,
        "ood_refusal_100": (
            int(ood_summary.get("case_count") or 0) == 30
            and int(ood_summary.get("error_count") or 0) == 0
            and float(ood_summary.get("ood_refusal_rate") or 0.0) == 1.0
        ),
        "gpu_p95_at_most_8s": 0.0 < p95 <= 8000.0,
        "p95_regression_at_most_30pct": baseline_p95 > 0.0 and p95 <= baseline_p95 * 1.30,
    }
    report = {
        "passed": all(gates.values()),
        "result": str(args.result),
        "baseline": str(args.baseline),
        "ood_result": str(args.ood_result),
        "p95_ms": p95,
        "baseline_p95_ms": baseline_p95,
        "p95_regression_ratio": round(p95 / baseline_p95, 4) if baseline_p95 else None,
        "gates": gates,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
