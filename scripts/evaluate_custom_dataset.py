from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation_common import DEFAULT_OUTPUT_DIR, pct, project_path, read_json, read_tabular, rel, write_json


DEFAULT_MANIFEST = Path("data/contest-data-self-made/manifest.json")
DEFAULT_QA = Path("data/contest-data-self-made/qa/self_made_qa.jsonl")
DEFAULT_QA_REPORT = Path("data/evaluation/self_made/self_made_qa_report.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize the self-made ReguMate evaluation dataset.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA)
    parser.add_argument("--qa-report", type=Path, default=DEFAULT_QA_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    metrics = summarize_custom_dataset(args.manifest, args.qa, args.qa_report)
    output = project_path(args.output_dir) / "custom_dataset_metrics.json"
    write_json(output, metrics)
    print(f"wrote {rel(output)}")
    accuracy = metrics.get("api_result_summary", {}).get("answer_accuracy")
    print(f"api_answer_accuracy={pct(accuracy) if accuracy is not None else '未实测'}")
    return 0


def summarize_custom_dataset(manifest_path: Path, qa_path: Path, qa_report_path: Path | None = DEFAULT_QA_REPORT) -> dict[str, Any]:
    manifest = read_json(manifest_path) if project_path(manifest_path).exists() else {}
    qa_rows = read_tabular(qa_path) if project_path(qa_path).exists() else []
    qa_type_counts = Counter(str(row.get("qa_type") or "未标注") for row in qa_rows)
    refusal_count = sum(1 for row in qa_rows if row.get("expected_refusal") is True)
    difficulty_counts = Counter()
    for row in qa_rows:
        tags = row.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        if row.get("expected_refusal") is True:
            difficulty_counts["越界/依据不足"] += 1
        elif "跨文件" in tags or "计算" in tags:
            difficulty_counts["中等"] += 1
        else:
            difficulty_counts["简单"] += 1
    api_summary: dict[str, Any] = {}
    run_mode = None
    if qa_report_path and project_path(qa_report_path).exists():
        report = read_json(qa_report_path)
        api_summary = report.get("summary") or report.get("qa_summary") or {}
        run_mode = report.get("run_mode")
    return {
        "dataset": {
            "name": manifest.get("name") or "contest-data-self-made",
            "manifest_file": rel(project_path(manifest_path)),
            "qa_file": rel(project_path(qa_path)),
            "document_count": len(manifest.get("documents") or []),
            "qa_case_count": len(qa_rows),
            "document_types": dict(Counter(str(doc.get("file_type") or "unknown") for doc in manifest.get("documents") or [])),
            "qa_type_counts": dict(sorted(qa_type_counts.items())),
            "difficulty_counts": dict(sorted(difficulty_counts.items())),
            "expected_refusal_count": refusal_count,
            "sensitive_customer_data": (manifest.get("data_compliance") or {}).get("sensitive_customer_data"),
            "real_bank_transaction_data": (manifest.get("data_compliance") or {}).get("real_bank_transaction_data"),
            "standard_answer_method": "由项目自制制度、填报说明和 CSV 报表中的显式事实/数值人工编写；不是从真实银行业务数据抽取。",
            "manual_review": "已在 manifest 与 QA 文件中人工确认题干、标准答案、来源文件和 expected_refusal 字段；未形成独立双人复核记录。",
        },
        "api_result_summary": api_summary,
        "run_mode": run_mode or "not_found",
        "sample_cases": qa_rows[:3],
        "limitations": [
            "自制 QA 已支持连接后端在线实测；未连接后端时仅生成数据集设计产物，不产生准确率、耗时和引用命中率。",
            "当前 8 题在线结果主要作为诊断样例，不能替代官方 300 题的统计口径；表格期间识别、跨文件表格计算和依据不足拒答需继续回归。",
            "数据规模为 8 题，适合作为功能覆盖与复现样例，不足以代表官方 300 题统计性能。",
        ],
        "rerun_command": "powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\run_self_made_evaluation.ps1",
    }


if __name__ == "__main__":
    raise SystemExit(main())
