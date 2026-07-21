from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def project_path(path: Path | str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return str(p)


def read_json(path: Path | str) -> Any:
    return json.loads(project_path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in project_path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def read_tabular(path: Path | str) -> list[dict[str, Any]]:
    p = project_path(path)
    if p.suffix.lower() == ".json":
        data = read_json(p)
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict):
            for key in ("results", "questions", "cases", "documents"):
                if isinstance(data.get(key), list):
                    return [row for row in data[key] if isinstance(row, dict)]
        return []
    if p.suffix.lower() == ".jsonl":
        return read_jsonl(p)
    if p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported input format: {p}")


def write_json(path: Path | str, data: Any) -> None:
    p = project_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path | str, rows: list[dict[str, Any]]) -> None:
    p = project_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def sha256_file(path: Path | str) -> str | None:
    p = project_path(path)
    if not p.exists() or not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_div(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if denominator in (None, 0):
        return None
    if numerator is None:
        return None
    return float(numerator) / float(denominator)


def pct(value: float | None) -> str:
    if value is None:
        return "未采集"
    return f"{value * 100:.2f}%"


def ms(value: float | int | None) -> str:
    if value is None:
        return "未采集"
    return f"{float(value) / 1000:.2f} s"


def seconds_from_ms(value: float | int | None) -> str:
    if value is None:
        return "未采集"
    return f"{float(value) / 1000:.2f} s"


def bytes_to_gib(value: float | int | None) -> str:
    if value is None:
        return "未采集"
    return f"{float(value) / (1024 ** 3):.2f} GiB"


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return round(ordered[int(position)], 2)
    return round(ordered[low] * (high - position) + ordered[high] * (position - low), 2)


def describe_numbers(values: list[float]) -> dict[str, float | int | None]:
    clean = [float(v) for v in values if isinstance(v, (int, float))]
    if not clean:
        return {"count": 0, "avg": None, "p50": None, "p90": None, "p95": None, "max": None}
    return {
        "count": len(clean),
        "avg": round(statistics.fmean(clean), 2),
        "p50": percentile(clean, 0.5),
        "p90": percentile(clean, 0.9),
        "p95": percentile(clean, 0.95),
        "max": round(max(clean), 2),
    }


def counter_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts = Counter(str(row.get(key) or "未标注") for row in rows)
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def accuracy_by(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key) or "未标注")].append(row)
    output: dict[str, dict[str, Any]] = {}
    for name, group in sorted(groups.items()):
        correct = sum(1 for row in group if row.get("answer_correct") is True)
        output[name] = {
            "count": len(group),
            "correct": correct,
            "accuracy": safe_div(correct, len(group)),
            "elapsed_ms": describe_numbers([row.get("elapsed_ms") for row in group]),
        }
    return output


def resource_peaks(artifact: dict[str, Any]) -> dict[str, Any]:
    perf = (((artifact.get("run") or {}).get("backend") or {}).get("performance") or {})
    resources = perf.get("resources") or {}
    peaks = resources.get("peaks") or {}
    return {
        "sampling": resources.get("sampling"),
        "sample_count": resources.get("samples"),
        "process_cpu_percent_peak": peaks.get("process_cpu_percent"),
        "rss_bytes_peak": peaks.get("rss_bytes"),
        "cuda_allocated_bytes_peak": peaks.get("cuda_peak_allocated_bytes")
        or peaks.get("cuda_allocated_bytes"),
        "cuda_reserved_bytes_peak": peaks.get("cuda_peak_reserved_bytes")
        or peaks.get("cuda_reserved_bytes"),
    }


def summarize_qa_artifact(path: Path | str, label: str | None = None) -> dict[str, Any]:
    p = project_path(path)
    artifact = read_json(p)
    results = [row for row in artifact.get("results", []) if isinstance(row, dict)]
    summary = artifact.get("summary") or {}
    elapsed = [row.get("elapsed_ms") for row in results]
    correct = sum(1 for row in results if row.get("answer_correct") is True)
    completed = sum(1 for row in results if row.get("error") in (None, ""))
    source_hits = sum(1 for row in results if row.get("source_hit") is True)
    citation_needed = [row for row in results if not row.get("refused")]
    citation_hits = sum(1 for row in citation_needed if int(row.get("citation_count") or 0) > 0)
    grounding_checked = [
        row for row in results if isinstance(row.get("grounding_validation"), dict)
    ]
    grounding_pass = sum(
        1 for row in grounding_checked if (row.get("grounding_validation") or {}).get("passed") is True
    )
    ood_rows = [row for row in results if str(row.get("expected")).upper() == "REFUSE" or row.get("refused")]
    refusal_correct = sum(1 for row in ood_rows if row.get("answer_correct") is True and row.get("refused") is True)
    text_evidence_rows = [row for row in results if isinstance(row.get("text_evidence"), dict)]
    excel_evidence_rows = [row for row in results if isinstance(row.get("excel_evidence"), dict)]
    manifest_summary = summary.get("evidence_manifest") if isinstance(summary.get("evidence_manifest"), dict) else {}
    topk = {"top1": None, "top3": None, "top5": None, "note": "原始 QA 结果未保存完整 pre-rerank/rerank 候选排名，不能从现有文件真实计算 Recall@K。"}
    elapsed_stats = describe_numbers(elapsed)
    summary_elapsed = summary.get("elapsed_ms") if isinstance(summary.get("elapsed_ms"), dict) else {}
    if summary_elapsed:
        for key in ("avg", "p50", "p95", "max"):
            if summary_elapsed.get(key) is not None:
                elapsed_stats[key] = summary_elapsed.get(key)
    return {
        "label": label or p.stem,
        "source_file": rel(p),
        "source_sha256": sha256_file(p),
        "run": artifact.get("run") or {},
        "case_count": len(results) or summary.get("case_count"),
        "completed_count": completed if results else summary.get("completed_count"),
        "error_count": (len(results) - completed) if results else summary.get("error_count"),
        "correct_count": correct if results else None,
        "answer_accuracy": safe_div(correct, len(results)) if results else summary.get("answer_accuracy"),
        "source_hit_count": source_hits if results else None,
        "source_hit_rate": safe_div(source_hits, len(results)) if results else summary.get("source_hit_rate"),
        "elapsed_ms": elapsed_stats,
        "summary_elapsed_ms": summary_elapsed,
        "citation_coverage_rate": safe_div(citation_hits, len(citation_needed)) if citation_needed else summary.get("citation_coverage_rate"),
        "grounding_pass_rate": safe_div(grounding_pass, len(grounding_checked)) if grounding_checked else summary.get("grounding_pass_rate"),
        "refusal_case_count": len(ood_rows),
        "refusal_correct_count": refusal_correct,
        "refusal_accuracy": safe_div(refusal_correct, len(ood_rows)) if ood_rows else summary.get("ood_refusal_rate"),
        "hallucination_rate_proxy": summary.get("key_entity_error_rate"),
        "hallucination_rate_note": "使用 key_entity_error_rate 作为可核验代理指标；现有结果未包含人工逐条幻觉标注。",
        "excel_cell_recall": summary.get("excel_cell_recall") if "excel_cell_recall" in summary else _avg_nested(excel_evidence_rows, "excel_evidence", "cell_recall"),
        "text_evidence_coverage_rate": summary.get("text_evidence_coverage_rate"),
        "text_evidence_bigram_recall": summary.get("text_evidence_bigram_recall"),
        "text_evidence_case_count": len(text_evidence_rows),
        "by_source_type": summary.get("by_slice", {}).get("source_type") or accuracy_by(results, "source_type"),
        "by_qa_type": summary.get("by_slice", {}).get("qa_type") or accuracy_by(results, "qa_type"),
        "by_difficulty": summary.get("by_slice", {}).get("difficulty") or accuracy_by(results, "difficulty"),
        "answer_type_counts": summary.get("answer_type_counts") or counter_by(results, "answer_type"),
        "generation_status_counts": summary.get("generation_status_counts") or counter_by(results, "generation_status"),
        "retrieval_topk": topk,
        "evidence_manifest": manifest_summary,
        "resource_peaks": resource_peaks(artifact),
    }


def _avg_nested(rows: list[dict[str, Any]], parent: str, key: str) -> float | None:
    values = []
    for row in rows:
        nested = row.get(parent)
        if isinstance(nested, dict) and isinstance(nested.get(key), (int, float)):
            values.append(float(nested[key]))
    if not values:
        return None
    return round(statistics.fmean(values), 4)


def qa_details_from_artifact(path: Path | str, run_label: str) -> list[dict[str, Any]]:
    artifact = read_json(path)
    rows = []
    for row in artifact.get("results", []):
        if not isinstance(row, dict):
            continue
        citations = row.get("citations") or []
        rows.append(
            {
                "run_label": run_label,
                "id": row.get("id"),
                "source_type": row.get("source_type"),
                "qa_type": row.get("qa_type"),
                "difficulty": row.get("difficulty"),
                "question": row.get("question"),
                "expected": row.get("expected"),
                "predicted": row.get("predicted"),
                "answer": row.get("answer"),
                "answer_correct": row.get("answer_correct"),
                "refused": row.get("refused"),
                "source_hit": row.get("source_hit"),
                "citation_count": row.get("citation_count"),
                "elapsed_ms": row.get("elapsed_ms"),
                "answer_type": row.get("answer_type"),
                "generation_status": row.get("generation_status"),
                "error": row.get("error"),
                "error_type": row.get("error_type"),
                "first_citation_file": citations[0].get("filename") if citations and isinstance(citations[0], dict) else None,
                "manifest_evidence": row.get("manifest_evidence"),
            }
        )
    return rows


def write_environment(path: Path | str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    env = {
        "created_at": now_iso(),
        "project_root": str(PROJECT_ROOT),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "git_commit": git_commit(),
        "env_vars_recorded": {
            "REGUMATE_MODEL_DEVICE": os.getenv("REGUMATE_MODEL_DEVICE"),
            "REGUMATE_PERFORMANCE_MODE": os.getenv("REGUMATE_PERFORMANCE_MODE"),
            "LLM_PROVIDER": os.getenv("LLM_PROVIDER"),
            "LLM_BASE_URL_CONFIGURED": bool(os.getenv("LLM_BASE_URL")),
            "LLM_MODEL": os.getenv("LLM_MODEL"),
            "LLM_API_KEY_CONFIGURED": bool(os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")),
        },
    }
    if extra:
        env.update(extra)
    write_json(path, env)
    return env


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def missing_item(name: str, checked: list[str], reason: str, command: str | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "status": "missing_or_unverifiable",
        "checked_locations": checked,
        "reason": reason,
        "recommended_rerun_command": command,
    }
