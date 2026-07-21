from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from evaluation_common import DEFAULT_OUTPUT_DIR, project_path, rel, summarize_qa_artifact, write_json


DEFAULT_GPU_QA = Path("data/evaluation/performance_experiments/phase1_gpu_full300/contest_qa_all_results.json")
DEFAULT_CPU_QA = Path("data/evaluation/performance_experiments/phase1_cpu_full300/contest_qa_all_results.json")
DEFAULT_INGESTION = Path("data/evaluation/performance_experiments/index_parse500.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare available CPU and GPU ReguMate evaluation outputs.")
    parser.add_argument("--gpu-qa", type=Path, default=DEFAULT_GPU_QA)
    parser.add_argument("--cpu-qa", type=Path, default=DEFAULT_CPU_QA)
    parser.add_argument("--ingestion", type=Path, default=DEFAULT_INGESTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    comparison = compare(args.gpu_qa, args.cpu_qa, args.ingestion)
    output = project_path(args.output_dir) / "cpu_gpu_comparison.json"
    write_json(output, comparison)
    print(f"wrote {rel(output)}")
    return 0


def compare(gpu_qa: Path, cpu_qa: Path, ingestion: Path | None = DEFAULT_INGESTION) -> dict[str, Any]:
    gpu = summarize_qa_artifact(gpu_qa, "gpu") if project_path(gpu_qa).exists() else None
    cpu = summarize_qa_artifact(cpu_qa, "cpu") if project_path(cpu_qa).exists() else None
    qa_comparison = None
    if gpu and cpu:
        gpu_avg = (gpu.get("elapsed_ms") or {}).get("avg")
        cpu_avg = (cpu.get("elapsed_ms") or {}).get("avg")
        qa_comparison = {
            "gpu_source_file": gpu["source_file"],
            "cpu_source_file": cpu["source_file"],
            "gpu_accuracy": gpu.get("answer_accuracy"),
            "cpu_accuracy": cpu.get("answer_accuracy"),
            "gpu_avg_ms": gpu_avg,
            "cpu_avg_ms": cpu_avg,
            "gpu_p95_ms": (gpu.get("elapsed_ms") or {}).get("p95"),
            "cpu_p95_ms": (cpu.get("elapsed_ms") or {}).get("p95"),
            "latency_speedup_cpu_avg_div_gpu_avg": round(float(cpu_avg) / float(gpu_avg), 2) if gpu_avg and cpu_avg else None,
            "gpu_resource_peaks": gpu.get("resource_peaks"),
            "cpu_resource_peaks": cpu.get("resource_peaks"),
            "by_source_type": _compare_slice(gpu, cpu, "by_source_type"),
            "by_qa_type": _compare_slice(gpu, cpu, "by_qa_type"),
            "by_difficulty": _compare_slice(gpu, cpu, "by_difficulty"),
        }
    return {
        "qa": qa_comparison,
        "ingestion": {
            "status": "partial",
            "available_source_file": rel(project_path(ingestion)) if ingestion and project_path(ingestion).exists() else None,
            "note": "当前仓库可核验 500 文档 CPU 解析/分块基线；未找到同口径 GPU/CPU 完整向量入库对比结果，故不计算入库加速比。",
        },
    }


def _compare_slice(gpu: dict[str, Any], cpu: dict[str, Any], key: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    gpu_slice = gpu.get(key) or {}
    cpu_slice = cpu.get(key) or {}
    for name in sorted(set(gpu_slice) | set(cpu_slice)):
        g = gpu_slice.get(name) or {}
        c = cpu_slice.get(name) or {}
        g_elapsed = g.get("elapsed_ms") or {}
        c_elapsed = c.get("elapsed_ms") or {}
        g_avg = g_elapsed.get("avg")
        c_avg = c_elapsed.get("avg")
        output[name] = {
            "gpu_count": g.get("count"),
            "cpu_count": c.get("count"),
            "gpu_accuracy": g.get("accuracy"),
            "cpu_accuracy": c.get("accuracy"),
            "gpu_avg_ms": g_avg,
            "cpu_avg_ms": c_avg,
            "speedup": round(float(c_avg) / float(g_avg), 2) if g_avg and c_avg else None,
        }
    return output


if __name__ == "__main__":
    raise SystemExit(main())
