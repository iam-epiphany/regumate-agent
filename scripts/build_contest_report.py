from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_contest_qa import summarize
except ModuleNotFoundError:  # Direct execution via ``python scripts/...``.
    from evaluate_contest_qa import summarize


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the frozen ReguMate contest evaluation report.")
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=ROOT / "data" / "evaluation" / "official_300_regression_merged_v2",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "evaluation" / "final_contest_report.md")
    parser.add_argument("--all-results", type=Path, help="Canonical 300-question result JSON.")
    parser.add_argument("--ood-results", type=Path, help="Canonical 30-question OOD result JSON.")
    parser.add_argument(
        "--directory-suffix",
        default="",
        help="Optional split-directory suffix, for example _frozen.",
    )
    args = parser.parse_args()

    ingest = _read_json(args.evaluation_dir / "ingest_manifest.json")
    legacy_doc = _read_json(args.evaluation_dir / "legacy_doc_report.json")
    excel_cells = _read_json(args.evaluation_dir / "official_excel_cells.json")
    if args.all_results or args.ood_results:
        if not args.all_results or not args.ood_results:
            raise SystemExit("--all-results and --ood-results must be provided together")
        all_evaluation = _read_json(args.all_results)
        ood_evaluation = _read_json(args.ood_results)
        all_rows = all_evaluation.get("results", [])
        evaluations = {
            "dev": _derived_evaluation(all_evaluation, all_rows, "dev"),
            "holdout": _derived_evaluation(all_evaluation, all_rows, "holdout"),
            "all": all_evaluation,
            "ood": ood_evaluation,
        }
        canonical_inputs = {
            "all": args.all_results.resolve(),
            "ood": args.ood_results.resolve(),
        }
    else:
        evaluations = {
            split: _read_json(
                args.evaluation_dir
                / f"{split}{args.directory_suffix}"
                / f"contest_qa_{split}_results.json"
            )
            for split in ("dev", "holdout", "all", "ood")
        }
        canonical_inputs = {
            split: (
                args.evaluation_dir
                / f"{split}{args.directory_suffix}"
                / f"contest_qa_{split}_results.json"
            ).resolve()
            for split in ("dev", "holdout", "all", "ood")
        }
    missing = [
        name
        for name, payload in {"ingest": ingest, "excel_cells": excel_cells, **evaluations}.items()
        if not payload
    ]
    if missing:
        raise SystemExit(f"missing evaluation artifacts: {', '.join(missing)}")

    all_summary = evaluations["all"]["summary"]
    ood_summary = evaluations["ood"]["summary"]
    ingest_summary = ingest["summary"]
    thresholds = {
        "500附件成功或明确降级": ingest_summary.get("success", 0) == 500 and ingest_summary.get("failed", 0) == 0,
        "300题无执行错误": all_summary.get("error_count", 1) == 0,
        "总体准确率>=99.67%": float(all_summary.get("answer_accuracy", 0)) >= 0.9967,
        "265标准单元格定位和值一致": (
            excel_cells.get("summary", {}).get("expected") == 265
            and excel_cells.get("summary", {}).get("located") == 265
            and excel_cells.get("summary", {}).get("values_matched") == 265
        ),
        "Excel准确率>=95%": _slice_accuracy(all_summary, "source_type", "excel") >= 0.95,
        "Word准确率>=90%": _slice_accuracy(all_summary, "source_type", "word") >= 0.90,
        "PDF准确率>=90%": _slice_accuracy(all_summary, "source_type", "pdf") >= 0.90,
        "来源命中率>=95%": float(all_summary.get("source_hit_rate", 0)) >= 0.95,
        "无答案拒答率>=90%": float(ood_summary.get("ood_refusal_rate", 0)) >= 0.90,
        "非拒答Grounding通过率=100%": float(all_summary.get("grounding_pass_rate", 0)) == 1.0,
        "关键实体错误率<=2%": float(all_summary.get("key_entity_error_rate", 1)) <= 0.02,
    }
    input_hashes = {
        name: _sha256_file(path) for name, path in canonical_inputs.items() if path.exists()
    }
    artifact = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "build_id": evaluations["all"].get("run", {}).get("backend", {}).get("build_id"),
            "all_run": evaluations["all"].get("run", {}),
            "ood_run": evaluations["ood"].get("run", {}),
            "inputs": {name: str(path) for name, path in canonical_inputs.items()},
            "input_sha256": input_hashes,
        },
        "ingest": ingest_summary,
        "dev": evaluations["dev"]["summary"],
        "holdout": evaluations["holdout"]["summary"],
        "all": all_summary,
        "ood": ood_summary,
        "legacy_doc": legacy_doc.get("summary", {}),
        "excel_cells": excel_cells.get("summary", {}),
        "thresholds": thresholds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_render(artifact, ingest, evaluations), encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(args.output)
    return 0 if all(thresholds.values()) else 2


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _derived_evaluation(
    source: dict[str, Any], rows: list[dict[str, Any]], split: str
) -> dict[str, Any]:
    selected = [row for row in rows if row.get("split") == split]
    return {
        "run": {**source.get("run", {}), "split": split, "case_count": len(selected), "derived": True},
        "summary": summarize(selected),
        "results": selected,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slice_accuracy(summary: dict[str, Any], field: str, value: str) -> float:
    return float(summary.get("by_slice", {}).get(field, {}).get(value, {}).get("accuracy", 0))


def _render(
    artifact: dict[str, Any], ingest: dict[str, Any], evaluations: dict[str, dict[str, Any]]
) -> str:
    all_summary = artifact["all"]
    ood_summary = artifact["ood"]
    failures = [item for item in evaluations["all"].get("results", []) if not item.get("answer_correct")]
    ood_failures = [
        item for item in evaluations["ood"].get("results", []) if not item.get("answer_correct")
    ]
    parse_failures = ingest.get("failures", [])
    lines = [
        "# ReguMate 官方数据冻结评测报告", "",
        f"生成时间：{artifact['created_at']}", "",
        "## 可复现身份", "",
        f"- Build ID：{artifact['provenance'].get('build_id') or '未报告'}",
        f"- QA 数据 SHA-256：{artifact['provenance'].get('all_run', {}).get('qa_sha256') or '未报告'}",
        f"- 评测器 SHA-256：{artifact['provenance'].get('all_run', {}).get('evaluator_sha256') or '未报告'}",
        *[
            f"- {name} 结果 SHA-256：{digest}"
            for name, digest in artifact["provenance"].get("input_sha256", {}).items()
        ], "",
        "## 数据入库", "",
        f"- 比赛附件：{ingest['run'].get('file_count', 0)}",
        f"- 成功：{artifact['ingest'].get('success', 0)}",
        f"- 降级：{artifact['ingest'].get('degraded', 0)}",
        f"- 失败：{artifact['ingest'].get('failed', 0)}",
        f"- chunks：{artifact['ingest'].get('chunks', 0)}",
        f"- 单元格：{artifact['ingest'].get('cells', 0)}",
        f"- Qdrant points：{artifact['ingest'].get('vectors', 0)}", "",
        "### 格式分项", "",
        "```json", json.dumps(artifact["ingest"].get("by_extension", {}), ensure_ascii=False, indent=2), "```", "",
        "### 旧版DOC双路径", "",
        f"- 官方.doc：{artifact['legacy_doc'].get('file_count', 0)}",
        f"- LibreOffice成功：{artifact['legacy_doc'].get('primary_success', 0)}",
        f"- antiword可降级：{artifact['legacy_doc'].get('fallback_success', 0)}", "",
        "### 官方 Excel 标准单元格", "",
        f"- 标准引用：{artifact['excel_cells'].get('expected', 0)}",
        f"- 成功定位：{artifact['excel_cells'].get('located', 0)}（{artifact['excel_cells'].get('location_rate', 0):.2%}）",
        f"- 值一致：{artifact['excel_cells'].get('values_matched', 0)}（{artifact['excel_cells'].get('value_match_rate', 0):.2%}）", "",
        "## 300题端到端结果", "",
        f"- 总体准确率：{all_summary.get('answer_accuracy', 0):.2%}",
        f"- 来源命中率：{all_summary.get('source_hit_rate', 0):.2%}",
        f"- Excel坐标召回率：{all_summary.get('excel_cell_recall', 0):.2%}",
        f"- Word/PDF标准证据覆盖率：{all_summary.get('text_evidence_coverage_rate', 0):.2%}",
        f"- Word/PDF标准证据二元组召回：{all_summary.get('text_evidence_bigram_recall', 0):.2%}",
        f"- Grounding通过率：{all_summary.get('grounding_pass_rate', 0):.2%}",
        f"- 关键实体错误率：{all_summary.get('key_entity_error_rate', 0):.2%}",
        f"- P95：{all_summary.get('elapsed_ms', {}).get('p95', 0):.2f} ms",
        f"- 30道派生题拒答率：{ood_summary.get('ood_refusal_rate', 0):.2%}", "",
        "### 固定划分（由同一轮 300 题结果派生）", "",
        f"- Dev：{artifact['dev'].get('case_count', 0)} 题，准确率 {artifact['dev'].get('answer_accuracy', 0):.2%}",
        f"- Holdout：{artifact['holdout'].get('case_count', 0)} 题，准确率 {artifact['holdout'].get('answer_accuracy', 0):.2%}", "",
        "## 分来源准确率", "",
        "```json", json.dumps(all_summary.get("by_slice", {}).get("source_type", {}), ensure_ascii=False, indent=2), "```", "",
        "## 验收阈值", "",
        *[f"- {'通过' if passed else '未通过'}：{name}" for name, passed in artifact["thresholds"].items()], "",
        "## 失败案例", "",
    ]
    lines.extend(
        f"- {item.get('id')}：expected={item.get('expected')} predicted={item.get('predicted')} error={item.get('error')}"
        for item in failures[:50]
    )
    if not failures:
        lines.append("- 无")
    lines.extend(["", "## 派生拒答题失败案例", ""])
    lines.extend(
        f"- {item.get('id')}：{item.get('question')}；answer={item.get('answer')}"
        for item in ood_failures[:50]
    )
    if not ood_failures:
        lines.append("- 无")
    lines.extend(["", "## 解析失败", ""])
    lines.extend(f"- {item.get('file')}：{item.get('error')}" for item in parse_failures)
    if not parse_failures:
        lines.append("- 无")
    lines.extend([
        "", "## 口径说明", "",
        "本报告使用固定种子20260713划分240道开发题和60道留出题；开发集与留出集指标由同一轮300题全量结果派生，并另行运行30道派生拒答题。文件命中不代替答案正确。",
    ])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
