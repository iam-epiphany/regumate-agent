"""Evaluate retrieval quality and latency against a small Chinese RAG case set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter
from typing import Any
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.database import SessionLocal, init_db  # noqa: E402
from backend.app.services.rag_service import retrieve_context_package  # noqa: E402
from backend.app.services.retrieval_service import RetrievalServiceUnavailable  # noqa: E402


DEFAULT_CASES_PATH = PROJECT_ROOT / "data" / "test_documents" / "retrieval_eval_cases.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ReguMate retrieval evaluation cases.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH, help="JSON case file path.")
    parser.add_argument(
        "--api-url",
        type=str,
        default="",
        help="Optional running backend base URL, for example http://127.0.0.1:8000/api.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for retrieval_metrics.json and retrieval_details.json.",
    )
    parser.add_argument("--output", type=Path, help="Optional explicit JSON output path.")
    args = parser.parse_args()

    cases = _load_cases(args.cases)
    results: list[dict[str, Any]] = []
    if args.api_url:
        for case in cases:
            results.append(_evaluate_case_via_api(args.api_url, case))
    else:
        init_db()
        with SessionLocal() as db:
            for case in cases:
                results.append(_evaluate_case(db, case))

    summary = _summarize(results)
    report = {"summary": summary, "cases": results}
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "retrieval_metrics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (args.output_dir / "retrieval_details.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    _print_report(report)


def _load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Evaluation case file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Evaluation case file must contain a JSON array.")
    return [case for case in data if isinstance(case, dict)]


def _evaluate_case(db: Any, case: dict[str, Any]) -> dict[str, Any]:
    question = str(case.get("question") or "").strip()
    expected_terms = [str(term) for term in case.get("expected_terms") or [] if str(term).strip()]
    expected_chunks = [str(chunk_id) for chunk_id in case.get("expected_chunks") or [] if str(chunk_id).strip()]
    should_refuse = bool(case.get("should_refuse", False))
    started_at = perf_counter()
    try:
        package = retrieve_context_package(db, question)
        elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
    except RetrievalServiceUnavailable as exc:
        elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
        return {
            "id": case.get("id"),
            "question": question,
            "error": str(exc),
            "latency_ms": elapsed_ms,
            "term_recall": 0.0,
            "precision_at_k": 0.0,
            "mrr": 0.0,
            "refusal_correct": should_refuse,
        }

    chunks = package.context_chunks
    top_k = [_chunk_result(chunk) for chunk in chunks]
    return _case_result_from_chunks(
        case=case,
        question=question,
        expected_terms=expected_terms,
        expected_chunks=expected_chunks,
        should_refuse=should_refuse,
        latency_ms=elapsed_ms,
        top_k=top_k,
        retrieval_summary=package.retrieval_summary,
    )


def _evaluate_case_via_api(api_url: str, case: dict[str, Any]) -> dict[str, Any]:
    question = str(case.get("question") or "").strip()
    expected_terms = [str(term) for term in case.get("expected_terms") or [] if str(term).strip()]
    expected_chunks = [str(chunk_id) for chunk_id in case.get("expected_chunks") or [] if str(chunk_id).strip()]
    should_refuse = bool(case.get("should_refuse", False))
    started_at = perf_counter()
    try:
        package = _retrieve_context_package_via_api(api_url, question)
        elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
    except Exception as exc:
        elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
        return {
            "id": case.get("id"),
            "question": question,
            "error": str(exc),
            "latency_ms": elapsed_ms,
            "term_recall": 0.0,
            "precision_at_k": 0.0,
            "mrr": 0.0,
            "refusal_correct": should_refuse,
        }

    top_k = [_api_chunk_result(chunk) for chunk in package.get("context_chunks", [])]
    return _case_result_from_chunks(
        case=case,
        question=question,
        expected_terms=expected_terms,
        expected_chunks=expected_chunks,
        should_refuse=should_refuse,
        latency_ms=elapsed_ms,
        top_k=top_k,
        retrieval_summary=package.get("retrieval_summary") or {},
    )


def _case_result_from_chunks(
    *,
    case: dict[str, Any],
    question: str,
    expected_terms: list[str],
    expected_chunks: list[str],
    should_refuse: bool,
    latency_ms: float,
    top_k: list[dict[str, Any]],
    retrieval_summary: dict[str, Any],
) -> dict[str, Any]:
    combined_text = "\n".join(str(chunk.get("text") or "") for chunk in top_k)
    matched_terms = [term for term in expected_terms if term in combined_text]
    relevant_ranks = [
        int(chunk["rank"])
        for chunk in top_k
        if chunk.get("rank") is not None and _is_relevant_chunk(chunk, expected_chunks, expected_terms)
    ]
    relevant_chunks = len(relevant_ranks)
    used_chunks = len(top_k)
    refused = not retrieval_summary.get("has_sufficient_context", False)
    hit_rank = min(relevant_ranks) if relevant_ranks else None
    refusal_correct = (refused or hit_rank is not None) if should_refuse else not refused
    return {
        "id": case.get("id"),
        "category": case.get("category"),
        "question": question,
        "latency_ms": latency_ms,
        "expected_source_file": case.get("expected_source_file"),
        "expected_section": case.get("expected_section"),
        "expected_chunks": expected_chunks,
        "expected_summary": case.get("expected_summary"),
        "success_criteria": case.get("success_criteria"),
        "retrieval_timings_ms": retrieval_summary.get("timings_ms", {}),
        "used_chunks": used_chunks,
        "query_count": retrieval_summary.get("query_count", 0),
        "candidate_count": retrieval_summary.get("candidate_count", 0),
        "reranked_count": retrieval_summary.get("reranked_count", 0),
        "filtered_count": retrieval_summary.get("filtered_count", 0),
        "term_recall": _safe_div(len(matched_terms), len(expected_terms)) if expected_terms else 1.0,
        "matched_terms": matched_terms,
        "precision_at_k": _safe_div(relevant_chunks, used_chunks),
        "mrr": 1 / hit_rank if hit_rank else (1.0 if should_refuse and refused else 0.0),
        "hit_rank": hit_rank,
        "hit_top1": hit_rank == 1,
        "hit_top3": hit_rank is not None and hit_rank <= 3,
        "hit_top5": hit_rank is not None and hit_rank <= 5,
        "refused": refused,
        "refusal_correct": refusal_correct,
        "has_sufficient_context": retrieval_summary.get("has_sufficient_context", False),
        "missing_aspects": retrieval_summary.get("missing_aspects", []),
        "query_plan": retrieval_summary.get("query_plan", {}),
        "top_k": top_k,
    }


def _retrieve_context_package_via_api(api_url: str, question: str) -> dict[str, Any]:
    base = api_url.rstrip("/")
    url = f"{base}/qa/retrieve"
    data = json.dumps({"question": question}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        return json.loads(response.read().decode("utf-8"))


def _chunk_result(chunk: Any) -> dict[str, Any]:
    metadata = chunk.metadata or {}
    return {
        "rank": chunk.rank,
        "chunk_id": chunk.chunk_id,
        "score": chunk.score,
        "rerank_score": metadata.get("rerank_score"),
        "fusion_score": metadata.get("aspect_query_fusion_score"),
        "source_doc": chunk.source_doc,
        "section_title": chunk.section_title,
        "section_path": chunk.section_path,
        "page_number": metadata.get("page_number"),
        "chunk_type": metadata.get("chunk_type"),
        "evidence_role": metadata.get("evidence_role"),
        "text": chunk.text,
    }


def _api_chunk_result(chunk: dict[str, Any]) -> dict[str, Any]:
    metadata = chunk.get("metadata") or {}
    return {
        "rank": chunk.get("rank"),
        "chunk_id": chunk.get("chunk_id"),
        "score": chunk.get("score"),
        "rerank_score": metadata.get("rerank_score"),
        "fusion_score": metadata.get("aspect_query_fusion_score"),
        "source_doc": chunk.get("source_doc"),
        "section_title": chunk.get("section_title"),
        "section_path": chunk.get("section_path") or [],
        "page_number": metadata.get("page_number"),
        "chunk_type": metadata.get("chunk_type"),
        "evidence_role": metadata.get("evidence_role"),
        "text": chunk.get("text") or "",
    }


def _is_relevant_chunk(chunk: dict[str, Any], expected_chunks: list[str], expected_terms: list[str]) -> bool:
    text = "\n".join(
        str(part)
        for part in [
            chunk.get("chunk_id"),
            chunk.get("section_title"),
            chunk.get("text"),
        ]
        if part
    )
    return bool(
        (expected_chunks and str(chunk.get("chunk_id")) in expected_chunks)
        or (expected_terms and any(term in text for term in expected_terms))
    )


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(result["latency_ms"]) for result in results]
    return {
        "case_count": len(results),
        "mean_term_recall": round(_mean(result["term_recall"] for result in results), 4),
        "mean_precision_at_k": round(_mean(result["precision_at_k"] for result in results), 4),
        "mean_mrr": round(_mean(result["mrr"] for result in results), 4),
        "refusal_accuracy": round(_mean(1.0 if result["refusal_correct"] else 0.0 for result in results), 4),
        "latency_p50_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "latency_p95_ms": round(_percentile(latencies, 0.95), 2) if latencies else 0.0,
    }


def _print_report(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("ReguMate retrieval evaluation")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print("Cases")
    for result in report["cases"]:
        print(
            f"- {result['id']}: recall={result['term_recall']:.2f}, "
            f"precision@k={result['precision_at_k']:.2f}, mrr={result['mrr']:.2f}, "
            f"hit_rank={result.get('hit_rank')}, latency={result['latency_ms']}ms, refused={result.get('refused')}"
        )
        for chunk in result.get("top_k", [])[:5]:
            print(
                "  #{rank} {chunk_id} score={score} rerank={rerank_score} page={page} section={section}".format(
                    rank=chunk.get("rank"),
                    chunk_id=chunk.get("chunk_id"),
                    score=_round_or_none(chunk.get("score")),
                    rerank_score=_round_or_none(chunk.get("rerank_score")),
                    page=chunk.get("page_number") or "-",
                    section=chunk.get("section_title") or "-",
                )
            )
        if result.get("missing_aspects"):
            print(f"  missing: {', '.join(result['missing_aspects'])}")
        if result.get("error"):
            print(f"  error: {result['error']}")


def _mean(values: Any) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round((len(ordered) - 1) * percentile)), len(ordered) - 1)
    return ordered[index]


def _safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _round_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


if __name__ == "__main__":
    main()
