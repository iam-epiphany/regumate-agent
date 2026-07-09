"""Evaluate retrieval quality and latency against a small Chinese RAG case set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.database import SessionLocal, init_db  # noqa: E402
from backend.app.services.rag_service import retrieve_context_package  # noqa: E402
from backend.app.services.retrieval_service import RetrievalServiceUnavailable  # noqa: E402


DEFAULT_CASES_PATH = PROJECT_ROOT / "data" / "test_documents" / "retrieval_eval_cases.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ReguMate retrieval evaluation cases.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH, help="JSON case file path.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    args = parser.parse_args()

    cases = _load_cases(args.cases)
    init_db()
    results: list[dict[str, Any]] = []
    with SessionLocal() as db:
        for case in cases:
            results.append(_evaluate_case(db, case))

    summary = _summarize(results)
    report = {"summary": summary, "cases": results}
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
    combined_text = "\n".join(chunk.text for chunk in chunks)
    matched_terms = [term for term in expected_terms if term in combined_text]
    relevant_ranks = [
        chunk.rank
        for chunk in chunks
        if expected_terms and any(term in chunk.text or term in (chunk.section_title or "") for term in expected_terms)
    ]
    relevant_chunks = len(relevant_ranks)
    used_chunks = len(chunks)
    refused = not package.retrieval_summary.get("has_sufficient_context", False)
    return {
        "id": case.get("id"),
        "question": question,
        "latency_ms": elapsed_ms,
        "retrieval_timings_ms": package.retrieval_summary.get("timings_ms", {}),
        "used_chunks": used_chunks,
        "query_count": package.retrieval_summary.get("query_count", 0),
        "candidate_count": package.retrieval_summary.get("candidate_count", 0),
        "reranked_count": package.retrieval_summary.get("reranked_count", 0),
        "filtered_count": package.retrieval_summary.get("filtered_count", 0),
        "term_recall": _safe_div(len(matched_terms), len(expected_terms)) if expected_terms else 1.0,
        "matched_terms": matched_terms,
        "precision_at_k": _safe_div(relevant_chunks, used_chunks),
        "mrr": 1 / min(relevant_ranks) if relevant_ranks else (1.0 if should_refuse and refused else 0.0),
        "refused": refused,
        "refusal_correct": refused == should_refuse,
        "missing_aspects": package.retrieval_summary.get("missing_aspects", []),
    }


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
            f"latency={result['latency_ms']}ms, refused={result.get('refused')}"
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


if __name__ == "__main__":
    main()
