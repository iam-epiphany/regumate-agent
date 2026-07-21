from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from evaluation_common import DEFAULT_OUTPUT_DIR, project_path, read_json, rel, summarize_qa_artifact, write_json


DEFAULT_RESULT = Path("data/evaluation/performance_experiments/phase1_gpu_full300/contest_qa_all_results.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize evidence, citation, and grounding metrics from QA results.")
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--label", default="phase1_gpu_full300")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    metrics = summarize_evidence(args.result, args.label)
    output = project_path(args.output_dir) / "evidence_metrics.json"
    write_json(output, metrics)
    print(f"wrote {rel(output)}")
    return 0


def summarize_evidence(result: Path, label: str) -> dict[str, Any]:
    p = project_path(result)
    if not p.exists():
        return {"runs": [], "missing": [{"name": "evidence result", "source_file": rel(p), "reason": "file not found"}]}
    qa = summarize_qa_artifact(p, label)
    artifact = read_json(p)
    rows = [row for row in artifact.get("results", []) if isinstance(row, dict)]
    invalid_citation_cases = []
    missing_inline_cases = []
    unsupported_entity_cases = []
    for row in rows:
        grounding = row.get("grounding_validation") or {}
        if not isinstance(grounding, dict):
            continue
        if grounding.get("invalid_citation_ids") or grounding.get("invalid_answer_citation_ids"):
            invalid_citation_cases.append(row.get("id"))
        if grounding.get("missing_inline_citation_ids") or grounding.get("missing_claim_citations"):
            missing_inline_cases.append(row.get("id"))
        if grounding.get("unsupported_entities") or grounding.get("unsupported_claim_entities"):
            unsupported_entity_cases.append(row.get("id"))
    return {
        "runs": [
            {
                "label": label,
                "source_file": qa["source_file"],
                "citation_coverage_rate": qa["citation_coverage_rate"],
                "grounding_pass_rate": qa["grounding_pass_rate"],
                "excel_cell_recall": qa["excel_cell_recall"],
                "text_evidence_coverage_rate": qa["text_evidence_coverage_rate"],
                "text_evidence_bigram_recall": qa["text_evidence_bigram_recall"],
                "evidence_manifest": qa["evidence_manifest"],
                "invalid_citation_case_count": len(invalid_citation_cases),
                "missing_inline_citation_case_count": len(missing_inline_cases),
                "unsupported_entity_case_count": len(unsupported_entity_cases),
                "invalid_citation_case_ids": invalid_citation_cases[:50],
                "missing_inline_citation_case_ids": missing_inline_cases[:50],
                "unsupported_entity_case_ids": unsupported_entity_cases[:50],
                "metric_limits": [
                    "Top-1/Top-3/Top-5 需要完整候选排名；现有最终 QA JSON 只有最终上下文/引用与 manifest final_recall。",
                    "幻觉率未做人工逐条标注；当前只能使用 key_entity_error_rate 与 grounding_pass_rate 作为代理指标。",
                ],
            }
        ]
    }


if __name__ == "__main__":
    raise SystemExit(main())
