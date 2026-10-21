"""Run a JSONL list of open questions through /api/qa/ask (no options).

Input rows must contain ``id`` and ``open_question``.  Gold fields are copied
through for the offline scorer but are never sent to the QA service.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from run_official300_open import _ask, _compact, _write_atomic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    questions = [
        json.loads(line)
        for line in args.questions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.start:
        questions = questions[args.start :]
    if args.limit:
        questions = questions[: args.limit]

    existing: dict[str, dict] = {}
    if not args.no_resume and args.out.exists():
        payload = json.loads(args.out.read_text(encoding="utf-8-sig"))
        existing = {str(row.get("id")): row for row in payload.get("results") or []}
    pending = [row for row in questions if str(row["id"]) not in existing]
    results = dict(existing)
    lock = threading.Lock()
    errors = 0
    started_at = time.perf_counter()
    print(f"[open-jsonl] total={len(questions)} done={len(existing)} pending={len(pending)}", flush=True)

    def run_one(row: dict) -> dict:
        result = {
            "id": row["id"],
            "question": row.get("open_question"),
            "original_question": row.get("original_question"),
            "rewrite_status": row.get("rewrite_status"),
            "leak_risk": row.get("leak_risk"),
            "source_type": row.get("source_type"),
            "qa_type": row.get("qa_type"),
            "answer_key": row.get("answer_key"),
            "answer_text": row.get("answer_text"),
            "options": row.get("options") or [],
            "gold_evidence": row.get("gold_evidence"),
            "source_title": row.get("source_title"),
            "response": None,
            "error": None,
            "attempts": 0,
            "latency_ms": None,
        }
        last_error = None
        for attempt in range(args.retries + 1):
            result["attempts"] = attempt + 1
            started = time.perf_counter()
            try:
                body = _ask(args.base_url, row["open_question"], args.timeout)
                result["response"] = _compact(body)
                result["error"] = None
                result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                validation = body.get("grounding_validation") or {}
                retrieval = result["response"].get("retrieval") or {}
                transient_refusal = (
                    body.get("answer_type") in {"refusal", "clarification"}
                    and not validation.get("missing_requirements")
                    and not retrieval.get("missing_requirements")
                )
                if (
                    (
                        body.get("answer_type") == "service_error"
                        or body.get("generation_status") == "error"
                        or transient_refusal
                    )
                    and attempt < args.retries
                ):
                    last_error = f"transient: {body.get('refusal_reason') or 'refusal'}"
                    time.sleep(2.0 + attempt * 2.0)
                    continue
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"[:500]
                result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                if attempt < args.retries:
                    time.sleep(2.0 + attempt * 2.0)
        result["error"] = last_error
        return result

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_one, row): row for row in pending}
        for future in as_completed(futures):
            result = future.result()
            with lock:
                results[str(result["id"])] = result
                if result.get("error"):
                    errors += 1
                payload = {
                    "schema_version": 1,
                    "mode": "open_no_options_jsonl",
                    "created_at": datetime.now(UTC).isoformat(),
                    "base_url": args.base_url,
                    "total": len(questions),
                    "completed": len(results),
                    "errors": errors,
                    "results": sorted(results.values(), key=lambda item: str(item.get("id"))),
                }
                _write_atomic(args.out, payload)
                elapsed = time.perf_counter() - started_at
                print(
                    f"[open-jsonl] {len(results)}/{len(questions)} errors={errors} "
                    f"elapsed={elapsed:.0f}s",
                    flush=True,
                )

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
