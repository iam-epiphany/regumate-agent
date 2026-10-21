"""Official-300 open-question runner (no options, bounded parallel).

Every case is sent as a plain question.  Options and the answer key are not
sent to the QA service; they are only used later by the offline scorer.

Usage:
    python scripts/evaluation/run_official300_open.py --workers 6
    python scripts/evaluation/run_official300_open.py --workers 6 --limit 20
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.evaluate_contest_qa import read_cases

DEFAULT_OUT = Path("outputs/official300_open/open_300_results.json")
DEFAULT_OPEN_QUESTIONS = Path("outputs/official300_open/open_questions_all.jsonl")
KEEP_RESPONSE_KEYS = (
    "answer",
    "answer_type",
    "refused",
    "generation_status",
    "refusal_reason",
    "refusal_code",
    "confidence",
    "degraded",
    "citations",
    "claims",
    "grounding_validation",
    "evidence_coverage",
    "agent_trace",
    "explanation",
)


def _ask(base_url: str, question: str, timeout: float) -> dict:
    payload = {"question": question, "include_debug": True}
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/qa/ask",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _compact(body: dict) -> dict:
    result = {key: body.get(key) for key in KEEP_RESPONSE_KEYS}
    package = body.get("context_package") or {}
    summary = package.get("retrieval_summary") or {}
    result["retrieval"] = {
        "has_sufficient_context": summary.get("has_sufficient_context"),
        "used_chunks": summary.get("used_chunks"),
        "missing_requirements": summary.get("missing_requirements") or [],
        "evidence_sufficiency": summary.get("evidence_sufficiency"),
    }
    return result


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_existing(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(row.get("id")): row for row in payload.get("results") or []}


def _read_open_questions(path: Path) -> list[SimpleNamespace]:
    """Read the rewritten open-question file (options removed)."""

    cases = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cases.append(SimpleNamespace(
            id=str(row["id"]),
            question=str(row["open_question"]),
            source_type=str(row.get("source_type") or ""),
            qa_type=str(row.get("qa_type") or ""),
            answer=str(row.get("answer_key") or ""),
            answer_text=str(row.get("answer_text") or ""),
            options=(),
            evidence=str(row.get("gold_evidence") or ""),
            source_title=str(row.get("source_title") or ""),
        ))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--questions", type=Path, default=DEFAULT_OPEN_QUESTIONS,
                        help="rewritten open-question JSONL (preferred over the raw XLSX)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--question-delay", type=float, default=0.0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.questions.exists():
        cases = _read_open_questions(args.questions)
    else:
        # Compatibility fallback for checkouts that only have the raw workbook.
        cases = read_cases("data/contest_dataset/QA数据.xlsx")
    if args.start:
        cases = cases[args.start :]
    if args.limit:
        cases = cases[: args.limit]

    existing = {} if args.no_resume else _load_existing(args.out)
    pending = [case for case in cases if str(case.id) not in existing]
    results = dict(existing)
    lock = threading.Lock()
    completed = 0
    errors = 0
    started_at = time.perf_counter()

    print(
        f"[open300] total={len(cases)} done={len(existing)} pending={len(pending)} "
        f"workers={args.workers}",
        flush=True,
    )

    def run_case(case) -> dict:
        row = {
            "id": case.id,
            "question": case.question,
            "source_type": case.source_type,
            "qa_type": case.qa_type,
            "answer_key": case.answer,
            "answer_text": case.answer_text,
            "options": list(case.options),
            "gold_evidence": case.evidence,
            "gold_source_title": case.source_title,
            "response": None,
            "error": None,
            "attempts": 0,
            "latency_ms": None,
        }
        last_error = None
        for attempt in range(args.retries + 1):
            row["attempts"] = attempt + 1
            started = time.perf_counter()
            try:
                body = _ask(args.base_url, case.question, args.timeout)
                row["response"] = _compact(body)
                row["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                row["error"] = None
                return row
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"[:500]
                row["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                if attempt < args.retries:
                    time.sleep(2.0 + attempt * 2.0)
        row["error"] = last_error
        return row

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_case, case): case for case in pending}
        for future in as_completed(futures):
            row = future.result()
            with lock:
                results[str(row["id"])] = row
                completed += 1
                if row.get("error"):
                    errors += 1
                payload = {
                    "schema_version": 1,
                    "mode": "open_no_options",
                    "created_at": datetime.now(UTC).isoformat(),
                    "base_url": args.base_url,
                    "workers": args.workers,
                    "total": len(cases),
                    "completed": len(results),
                    "errors": errors,
                    "results": sorted(results.values(), key=lambda item: str(item.get("id"))),
                }
                _write_atomic(args.out, payload)
                elapsed = time.perf_counter() - started_at
                rate = completed / elapsed if elapsed else 0.0
                print(
                    f"[open300] {completed}/{len(pending)} errors={errors} "
                    f"elapsed={elapsed:.0f}s rate={rate:.2f}/s",
                    flush=True,
                )
            if args.question_delay:
                time.sleep(args.question_delay)

    payload = {
        "schema_version": 1,
        "mode": "open_no_options",
        "created_at": datetime.now(UTC).isoformat(),
        "base_url": args.base_url,
        "workers": args.workers,
        "total": len(cases),
        "completed": len(results),
        "errors": errors,
        "results": sorted(results.values(), key=lambda item: str(item.get("id"))),
    }
    _write_atomic(args.out, payload)
    print(
        json.dumps(
            {
                "completed": len(results),
                "errors": errors,
                "elapsed_seconds": round(time.perf_counter() - started_at, 1),
                "output": str(args.out),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
