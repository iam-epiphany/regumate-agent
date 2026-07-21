from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from evaluation_common import (
    DEFAULT_OUTPUT_DIR,
    missing_item,
    pct,
    project_path,
    read_json,
    rel,
    safe_div,
    sha256_file,
    write_json,
)


DEFAULT_CPU_PARSE = Path("data/evaluation/final/ingest_manifest.json")
DEFAULT_FINAL_INGEST = Path("data/evaluation/final/ingest_manifest.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize ReguMate document ingestion results.")
    parser.add_argument("--result", type=Path, default=DEFAULT_CPU_PARSE, help="JSON result or manifest to summarize.")
    parser.add_argument("--final-ingest", type=Path, default=DEFAULT_FINAL_INGEST, help="Optional final ingest manifest for integrity checks.")
    parser.add_argument("--label", default="final_ingest_500")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    result = summarize_ingestion(args.result, args.label, args.device, args.final_ingest)
    output = project_path(args.output_dir) / "ingestion_metrics.json"
    write_json(output, result)
    print(f"wrote {rel(output)}")
    print(f"success_rate={pct(result['summary'].get('success_rate'))}")
    return 0


def summarize_ingestion(
    result_path: Path,
    label: str = "final_ingest_500",
    device: str = "cpu",
    final_ingest_path: Path | None = DEFAULT_FINAL_INGEST,
) -> dict[str, Any]:
    p = project_path(result_path)
    if not p.exists():
        return {
            "runs": [],
            "missing": [
                missing_item(
                    "文档入库结果",
                    [rel(p)],
                    "指定结果文件不存在。",
                    "python scripts/evaluate_ingestion.py --result <结果JSON> --output-dir outputs/evaluation",
                )
            ],
        }
    artifact = read_json(p)
    summary = artifact.get("summary") or {}
    docs = artifact.get("documents") or []
    failures = artifact.get("failures") or []
    performance = artifact.get("performance") or {}
    stages = performance.get("stages") or {}
    resources = performance.get("resources") or {}
    peaks = resources.get("peaks") or {}
    processed = int(summary.get("processed") or len(docs) or 0)
    success = int(summary.get("success") or sum(1 for doc in docs if str(doc.get("status", "")).lower() not in {"failed", "error"}))
    failed = int(summary.get("failed") or len(failures) or max(processed - success, 0))
    vectors = summary.get("vectors")
    chunks = summary.get("chunks")
    final_integrity = _integrity_checks(final_ingest_path) if final_ingest_path else None
    run = {
        "label": label,
        "device": device,
        "source_file": rel(p),
        "source_sha256": sha256_file(p),
        "run": artifact.get("run") or {},
        "summary": {
            "document_count": processed,
            "success_count": success,
            "failed_count": failed,
            "degraded_count": summary.get("degraded"),
            "success_rate": safe_div(success, processed),
            "chunk_count": chunks,
            "cell_count": summary.get("cells"),
            "vector_count": vectors,
            "total_elapsed_ms": performance.get("total_elapsed_ms"),
            "average_per_document_ms": (performance.get("per_document_ms") or {}).get("average"),
            "p50_per_document_ms": (performance.get("per_document_ms") or {}).get("p50"),
            "p95_per_document_ms": (performance.get("per_document_ms") or {}).get("p95"),
            "max_per_document_ms": (performance.get("per_document_ms") or {}).get("maximum"),
            "parse_avg_ms": (stages.get("index.parse") or {}).get("avg_ms"),
            "chunk_build_avg_ms": (stages.get("index.chunk_build") or {}).get("avg_ms"),
            "embedding_avg_ms": (stages.get("embedding.inference") or {}).get("avg_ms"),
            "sqlite_persist_avg_ms": (stages.get("index.sqlite_chunk_persist") or {}).get("avg_ms"),
            "index_write_avg_ms": None,
            "index_write_note": "现有结果记录 SQLite chunk persist；未单独记录 Qdrant 写入阶段耗时。",
        },
        "by_extension": summary.get("by_extension") or _group_docs(docs, "extension"),
        "by_parser": summary.get("by_parser") or _group_docs(docs, "parser_backend"),
        "failures": failures,
        "resource_peaks": {
            "process_cpu_percent_peak": peaks.get("process_cpu_percent"),
            "rss_bytes_peak": peaks.get("rss_bytes"),
            "cuda_allocated_bytes_peak": peaks.get("cuda_peak_allocated_bytes") or peaks.get("cuda_allocated_bytes"),
            "cuda_reserved_bytes_peak": peaks.get("cuda_peak_reserved_bytes") or peaks.get("cuda_reserved_bytes"),
            "sample_count": resources.get("samples"),
            "sampling": resources.get("sampling"),
        },
        "integrity_checks": final_integrity,
    }
    missing = []
    if vectors in (None, 0):
        missing.append(
            missing_item(
                "完整向量索引写入数量/耗时",
                [rel(p), rel(final_ingest_path) if final_ingest_path else ""],
                "当前指定结果 vectors 为 0 或未记录；请确认是否误传解析/分块性能实验，而不是 final/ingest_manifest.json。",
                "python scripts/ingest_contest_dataset.py --source data/contest_staging --output data/evaluation/<new_ingest_run>",
            )
        )
    missing.append(
        missing_item(
            "GPU 500 文档入库性能结果",
            [
                "data/evaluation/performance_experiments/",
                "data/evaluation/performance_baseline/",
                "docs/evaluation/performance_baseline_20260718.md",
            ],
            "已找到 CPU 解析/分块基线与 QA GPU/CPU 结果，但未找到同口径的 GPU 500 文档完整入库结果。",
            "使用 GPU 模式启动后执行 contest 数据入库脚本，并保留 JSON/日志输出。",
        )
    )
    return {"runs": [run], "summary": run["summary"], "missing": missing}


def _group_docs(docs: list[dict[str, Any]], key: str) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for doc in docs:
        name = str(doc.get(key) or "未标注")
        bucket = output.setdefault(name, {"processed": 0, "success": 0, "failed": 0, "chunks": 0, "cells": 0})
        bucket["processed"] += 1
        failed = str(doc.get("status") or "").lower() in {"failed", "error"}
        bucket["failed" if failed else "success"] += 1
        bucket["chunks"] += int(doc.get("chunk_count") or 0)
        bucket["cells"] += int(doc.get("cell_count") or 0)
    return dict(sorted(output.items()))


def _integrity_checks(path: Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = project_path(path)
    if not p.exists():
        return {"source_file": rel(p), "status": "missing"}
    data = read_json(p)
    summary = data.get("summary") or {}
    docs = data.get("documents") or []
    failures = data.get("failures") or []
    return {
        "source_file": rel(p),
        "source_sha256": sha256_file(p),
        "document_records": len(docs),
        "failure_records": len(failures),
        "summary_processed": summary.get("processed"),
        "summary_success": summary.get("success"),
        "summary_failed": summary.get("failed"),
        "summary_chunks": summary.get("chunks"),
        "summary_vectors": summary.get("vectors"),
        "document_count_matches_summary": len(docs) == summary.get("processed"),
        "failure_count_matches_summary": len(failures) == summary.get("failed"),
        "success_plus_failed_matches_processed": (
            (summary.get("success") or 0) + (summary.get("failed") or 0)
        )
        == summary.get("processed"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
