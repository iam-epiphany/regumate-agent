from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from evaluation_common import (
    DEFAULT_OUTPUT_DIR,
    missing_item,
    project_path,
    qa_details_from_artifact,
    rel,
    summarize_qa_artifact,
    write_json,
    write_jsonl,
)


DEFAULT_GPU_QA = Path("data/evaluation/performance_experiments/phase1_gpu_full300/contest_qa_all_results.json")
DEFAULT_CPU_QA = Path("data/evaluation/performance_experiments/phase1_cpu_full300/contest_qa_all_results.json")
DEFAULT_GPU_OOD = Path("data/evaluation/performance_baseline/gpu_20260718_182304/ood/contest_qa_ood_results.json")
DEFAULT_CPU_OOD = Path("data/evaluation/performance_baseline/cpu_20260718_184347/ood/contest_qa_ood_results.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or summarize ReguMate QA evaluation results.")
    parser.add_argument("--dataset", type=Path, default=Path("data/contest_dataset/QA数据.xlsx"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--split", choices=["all", "dev", "holdout", "ood"], default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result", type=Path, help="Summarize a single existing contest_qa result JSON.")
    parser.add_argument("--from-existing", action="store_true", help="Use the known GPU/CPU result files instead of calling the backend.")
    parser.add_argument("--run-backend-eval", action="store_true", help="Call scripts/evaluate_contest_qa.py and then summarize its output.")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    output_dir = project_path(args.output_dir)
    if args.run_backend_eval:
        result = _run_backend_eval(args.dataset, args.base_url, args.split, output_dir / f"{args.device}_qa_run", args.limit)
        metrics = {"runs": [summarize_qa_artifact(result, f"{args.device}_{args.split}")], "missing": []}
        write_json(output_dir / "qa_metrics.json", metrics)
        write_jsonl(output_dir / "qa_details.jsonl", qa_details_from_artifact(result, f"{args.device}_{args.split}"))
        print(f"wrote {rel(output_dir / 'qa_metrics.json')}")
        return 0

    metrics = summarize_existing_qa(args.result)
    write_json(output_dir / "qa_metrics.json", metrics)
    detail_rows: list[dict[str, Any]] = []
    for run in metrics["runs"]:
        source = run.get("source_file")
        if source and project_path(source).exists():
            detail_rows.extend(qa_details_from_artifact(source, str(run.get("label"))))
    write_jsonl(output_dir / "qa_details.jsonl", detail_rows)
    error_rows = [row for row in detail_rows if row.get("answer_correct") is not True or row.get("error")]
    write_jsonl(output_dir / "error_cases.jsonl", error_rows)
    print(f"wrote {rel(output_dir / 'qa_metrics.json')}")
    print(f"wrote {rel(output_dir / 'qa_details.jsonl')}")
    return 0


def summarize_existing_qa(result: Path | None = None) -> dict[str, Any]:
    if result:
        p = project_path(result)
        if not p.exists():
            return {
                "runs": [],
                "missing": [
                    missing_item(
                        "QA 结果文件",
                        [rel(p)],
                        "指定结果文件不存在。",
                        "python scripts/evaluate_qa.py --run-backend-eval --dataset data/contest_dataset/QA数据.xlsx --base-url http://127.0.0.1:8000 --output-dir outputs/evaluation",
                    )
                ],
            }
        return {"runs": [summarize_qa_artifact(p, p.parent.name)], "missing": []}

    candidates = [
        ("phase1_gpu_full300", DEFAULT_GPU_QA),
        ("phase1_cpu_full300", DEFAULT_CPU_QA),
        ("gpu_ood30", DEFAULT_GPU_OOD),
        ("cpu_ood30", DEFAULT_CPU_OOD),
    ]
    runs = []
    missing = []
    for label, path in candidates:
        p = project_path(path)
        if p.exists():
            runs.append(summarize_qa_artifact(p, label))
        else:
            missing.append(
                missing_item(
                    f"{label} QA 结果",
                    [rel(p)],
                    "未找到默认结果文件。",
                    "python scripts/evaluate_qa.py --run-backend-eval --split all --output-dir outputs/evaluation/<new_run>",
                )
            )
    if runs:
        runs[0]["primary_for_report"] = True
    return {
        "runs": runs,
        "missing": missing,
        "metric_formulas": {
            "answer_accuracy": "answer_correct=True 的题数 / 有效评测题数",
            "source_hit_rate": "source_hit=True 的题数 / 有效评测题数",
            "citation_coverage_rate": "非拒答题中 citation_count>0 的题数 / 非拒答题数",
            "grounding_pass_rate": "grounding_validation.passed=True 的题数 / 进行 grounding 校验的题数",
            "refusal_accuracy": "应拒答/已拒答样本中 refused=True 且 answer_correct=True 的题数 / 拒答评测题数",
            "hallucination_rate_proxy": "现有结果未人工标注幻觉，报告使用 key_entity_error_rate 作为可核验代理指标。",
        },
    }


def _run_backend_eval(dataset: Path, base_url: str, split: str, output: Path, limit: int) -> Path:
    cmd = [
        sys.executable,
        str(project_path("scripts/evaluate_contest_qa.py")),
        "--qa",
        str(project_path(dataset)),
        "--output",
        str(output),
        "--base-url",
        base_url,
        "--split",
        split,
    ]
    if limit > 0:
        cmd.extend(["--limit", str(limit)])
    subprocess.check_call(cmd, cwd=project_path("."))
    return output / f"contest_qa_{split}_results.json"


if __name__ == "__main__":
    raise SystemExit(main())
