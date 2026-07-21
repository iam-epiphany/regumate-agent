from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation_common import (
    DEFAULT_OUTPUT_DIR,
    describe_numbers,
    project_path,
    read_json,
    rel,
    safe_div,
    sha256_file,
    write_json,
    write_jsonl,
)


DEFAULT_QA_RESULT = Path("data/evaluation/performance_experiments/phase1_gpu_full300/contest_qa_all_results.json")
DEFAULT_EVIDENCE_MANIFEST = Path("data/evaluation/performance_experiments/evidence_manifest_seed.json")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute retrieval Recall@K from saved QA context_package results and evidence manifest."
    )
    parser.add_argument("--qa-result", type=Path, default=DEFAULT_QA_RESULT)
    parser.add_argument("--evidence-manifest", type=Path, default=DEFAULT_EVIDENCE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    metrics, details = summarize_retrieval_from_results(args.qa_result, args.evidence_manifest)
    output_dir = project_path(args.output_dir)
    write_json(output_dir / "retrieval_metrics.json", metrics)
    write_jsonl(output_dir / "retrieval_details.jsonl", details)
    print(f"wrote {rel(output_dir / 'retrieval_metrics.json')}")
    print(f"wrote {rel(output_dir / 'retrieval_details.jsonl')}")
    return 0


def summarize_retrieval_from_results(
    qa_result_path: Path = DEFAULT_QA_RESULT,
    evidence_manifest_path: Path = DEFAULT_EVIDENCE_MANIFEST,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    qa_path = project_path(qa_result_path)
    manifest_path = project_path(evidence_manifest_path)
    qa = read_json(qa_path)
    manifest = read_json(manifest_path)
    manifest_cases = {
        str(case.get("case_id")): case
        for case in manifest.get("cases", [])
        if isinstance(case, dict) and case.get("case_id")
    }
    details: list[dict[str, Any]] = []
    for result in qa.get("results", []):
        if not isinstance(result, dict):
            continue
        case = manifest_cases.get(str(result.get("id")))
        if not case:
            continue
        required_aspects = [
            aspect
            for aspect in case.get("required_aspects", [])
            if isinstance(aspect, dict) and aspect.get("acceptable_chunk_ids")
        ]
        if not required_aspects:
            continue
        retrieval_summary = ((result.get("context_package") or {}).get("retrieval_summary") or {})
        ranked_chunk_ids = _ranked_chunk_ids(retrieval_summary)
        selected_chunk_ids = _selected_chunk_ids(retrieval_summary)
        aspect_rows = []
        case_hit_at = {1: True, 3: True, 5: True}
        prompt_hit_all = True
        best_ranks = []
        for aspect in required_aspects:
            acceptable = {str(item) for item in aspect.get("acceptable_chunk_ids") or []}
            rank = _first_hit_rank(ranked_chunk_ids, acceptable)
            prompt_hit = bool(acceptable.intersection(selected_chunk_ids))
            if rank is not None:
                best_ranks.append(rank)
            for k in case_hit_at:
                if rank is None or rank > k:
                    case_hit_at[k] = False
            if not prompt_hit:
                prompt_hit_all = False
            aspect_rows.append(
                {
                    "aspect_id": aspect.get("aspect_id"),
                    "acceptable_chunk_ids": sorted(acceptable),
                    "best_hit_rank": rank,
                    "hit_at_1": rank is not None and rank <= 1,
                    "hit_at_3": rank is not None and rank <= 3,
                    "hit_at_5": rank is not None and rank <= 5,
                    "selected_for_prompt_hit": prompt_hit,
                }
            )
        details.append(
            {
                "id": result.get("id"),
                "source_type": result.get("source_type"),
                "qa_type": result.get("qa_type"),
                "difficulty": result.get("difficulty"),
                "review_status": case.get("review_status"),
                "required_aspect_count": len(required_aspects),
                "ranked_candidate_count": len(ranked_chunk_ids),
                "selected_chunk_count": len(selected_chunk_ids),
                "case_hit_at_1": case_hit_at[1],
                "case_hit_at_3": case_hit_at[3],
                "case_hit_at_5": case_hit_at[5],
                "case_prompt_hit_all": prompt_hit_all,
                "best_hit_rank": min(best_ranks) if best_ranks else None,
                "query_count": retrieval_summary.get("query_count"),
                "raw_candidate_count": retrieval_summary.get("raw_candidate_count"),
                "candidate_count": retrieval_summary.get("candidate_count"),
                "rerank_input_count": retrieval_summary.get("rerank_input_count"),
                "reranked_count": retrieval_summary.get("reranked_count"),
                "used_chunks": retrieval_summary.get("used_chunks"),
                "has_sufficient_context": retrieval_summary.get("has_sufficient_context"),
                "retrieval_covered_aspect_count": retrieval_summary.get("retrieval_covered_aspect_count"),
                "prompt_covered_aspect_count": retrieval_summary.get("prompt_covered_aspect_count"),
                "timings_ms": retrieval_summary.get("timings_ms") or {},
                "aspects": aspect_rows,
            }
        )
    metrics = _summarize(details, qa_path, manifest_path, qa, manifest)
    return metrics, details


def _ranked_chunk_ids(retrieval_summary: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for aspect in retrieval_summary.get("aspect_retrievals") or []:
        if not isinstance(aspect, dict):
            continue
        for chunk in aspect.get("retrieved_chunks") or []:
            if isinstance(chunk, dict) and chunk.get("chunk_id"):
                chunk_id = str(chunk["chunk_id"])
                if chunk_id not in ids:
                    ids.append(chunk_id)
    return ids


def _selected_chunk_ids(retrieval_summary: dict[str, Any]) -> set[str]:
    selected: set[str] = set()
    prompt_selection = retrieval_summary.get("prompt_selection") or {}
    for chunk_id in prompt_selection.get("final_prompt_chunk_ids") or []:
        selected.add(str(chunk_id))
    for chunk_id in retrieval_summary.get("final_prompt_chunk_ids") or []:
        selected.add(str(chunk_id))
    for aspect in retrieval_summary.get("aspect_retrievals") or []:
        if not isinstance(aspect, dict):
            continue
        for chunk_id in aspect.get("selected_chunk_ids") or []:
            selected.add(str(chunk_id))
        for chunk in aspect.get("retrieved_chunks") or []:
            if isinstance(chunk, dict) and chunk.get("selected_for_prompt") and chunk.get("chunk_id"):
                selected.add(str(chunk["chunk_id"]))
    return selected


def _first_hit_rank(ranked_chunk_ids: list[str], acceptable: set[str]) -> int | None:
    for index, chunk_id in enumerate(ranked_chunk_ids, start=1):
        if chunk_id in acceptable:
            return index
    return None


def _summarize(
    details: list[dict[str, Any]],
    qa_path: Path,
    manifest_path: Path,
    qa: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    case_count = len(details)
    aspect_rows = [aspect for row in details for aspect in row.get("aspects", [])]
    aspect_count = len(aspect_rows)
    by_status = Counter(str(row.get("review_status") or "unknown") for row in details)
    manifest_cases = [case for case in manifest.get("cases", []) if isinstance(case, dict)]
    manifest_status = Counter(str(case.get("review_status") or "unknown") for case in manifest_cases)
    no_acceptable_case_count = 0
    no_acceptable_aspect_count = 0
    for case in manifest_cases:
        aspects = [aspect for aspect in case.get("required_aspects", []) if isinstance(aspect, dict)]
        acceptable_aspects = [aspect for aspect in aspects if aspect.get("acceptable_chunk_ids")]
        no_acceptable_aspect_count += len(aspects) - len(acceptable_aspects)
        if not acceptable_aspects:
            no_acceptable_case_count += 1
    by_source_type = _slice(details, "source_type")
    by_qa_type = _slice(details, "qa_type")
    timings = _timing_summary(details)
    return {
        "source": "saved_qa_context_package_plus_evidence_manifest",
        "qa_result_file": rel(qa_path),
        "qa_result_sha256": sha256_file(qa_path),
        "evidence_manifest_file": rel(manifest_path),
        "evidence_manifest_sha256": sha256_file(manifest_path),
        "qa_created_at": (qa.get("run") or {}).get("created_at"),
        "manifest_created_at": manifest.get("created_at"),
        "manifest_review_policy": manifest.get("review_policy"),
        "source_hit_rate": (qa.get("summary") or {}).get("source_hit_rate"),
        "case_count": case_count,
        "aspect_count": aspect_count,
        "review_status_counts": dict(sorted(by_status.items())),
        "manifest_case_count": len(manifest_cases),
        "manifest_review_status_counts": dict(sorted(manifest_status.items())),
        "manifest_needs_review_case_count": manifest_status.get("needs_review", 0),
        "no_acceptable_case_count": no_acceptable_case_count,
        "no_acceptable_aspect_count": no_acceptable_aspect_count,
        "case_recall_at_1": _rate(details, "case_hit_at_1"),
        "case_recall_at_3": _rate(details, "case_hit_at_3"),
        "case_recall_at_5": _rate(details, "case_hit_at_5"),
        "aspect_recall_at_1": _aspect_rate(aspect_rows, "hit_at_1"),
        "aspect_recall_at_3": _aspect_rate(aspect_rows, "hit_at_3"),
        "aspect_recall_at_5": _aspect_rate(aspect_rows, "hit_at_5"),
        "prompt_context_case_recall": _rate(details, "case_prompt_hit_all"),
        "prompt_context_aspect_recall": _aspect_rate(aspect_rows, "selected_for_prompt_hit"),
        "best_hit_rank": describe_numbers(
            [float(row["best_hit_rank"]) for row in details if row.get("best_hit_rank") is not None]
        ),
        "candidate_count": describe_numbers(
            [float(row["candidate_count"]) for row in details if isinstance(row.get("candidate_count"), (int, float))]
        ),
        "reranked_count": describe_numbers(
            [float(row["reranked_count"]) for row in details if isinstance(row.get("reranked_count"), (int, float))]
        ),
        "used_chunks": describe_numbers(
            [float(row["used_chunks"]) for row in details if isinstance(row.get("used_chunks"), (int, float))]
        ),
        "retrieval_timings_ms": timings,
        "by_source_type": by_source_type,
        "by_qa_type": by_qa_type,
        "metric_formula": {
            "case_recall_at_k": "全部标准 aspect 均在该题已保存检索候选 Top-K 内命中 acceptable_chunk_ids 的题数 / 可评估题数",
            "aspect_recall_at_k": "在已保存检索候选 Top-K 内命中 acceptable_chunk_ids 的标准 aspect 数 / 可评估 aspect 数",
            "prompt_context_recall": "标准 aspect 的 acceptable_chunk_ids 是否进入最终 prompt 上下文",
        },
        "limitations": [
            "该脚本基于已保存 context_package 离线重算，不重新请求 Qdrant 或后端。",
            "evidence_manifest_seed 中 14 题为 needs_review 且无 acceptable_chunk_ids，未纳入 Recall@K 分母。",
            "多 aspect 题按题内去重后的候选 chunk 顺序计算 Top-K；这反映最终保存候选，不代表所有中间 query 的原始排名。",
        ],
    }


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    return safe_div(sum(1 for row in rows if row.get(key) is True), len(rows))


def _aspect_rate(rows: list[dict[str, Any]], key: str) -> float | None:
    return safe_div(sum(1 for row in rows if row.get(key) is True), len(rows))


def _slice(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key) or "未标注"), []).append(row)
    output: dict[str, dict[str, Any]] = {}
    for name, group in sorted(groups.items()):
        aspects = [aspect for row in group for aspect in row.get("aspects", [])]
        output[name] = {
            "case_count": len(group),
            "aspect_count": len(aspects),
            "case_recall_at_1": _rate(group, "case_hit_at_1"),
            "case_recall_at_3": _rate(group, "case_hit_at_3"),
            "case_recall_at_5": _rate(group, "case_hit_at_5"),
            "aspect_recall_at_1": _aspect_rate(aspects, "hit_at_1"),
            "aspect_recall_at_3": _aspect_rate(aspects, "hit_at_3"),
            "aspect_recall_at_5": _aspect_rate(aspects, "hit_at_5"),
            "prompt_context_case_recall": _rate(group, "case_prompt_hit_all"),
        }
    return output


def _timing_summary(details: list[dict[str, Any]]) -> dict[str, Any]:
    values_by_key: dict[str, list[float]] = {}
    for row in details:
        timings = row.get("timings_ms") or {}
        if not isinstance(timings, dict):
            continue
        for key, value in timings.items():
            if isinstance(value, (int, float)):
                values_by_key.setdefault(str(key), []).append(float(value))
    return {key: describe_numbers(values) for key, values in sorted(values_by_key.items())}


if __name__ == "__main__":
    raise SystemExit(main())
