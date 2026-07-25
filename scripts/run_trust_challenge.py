"""Run public challenge questions without loading the offline gold file."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ReguMate trust challenge questions against /api/qa/ask.")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--split", choices=["dev", "holdout", "all"], default="all")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--request-delay", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--case-ids", help="Comma-separated case IDs for a bounded regression run.")
    args = parser.parse_args()

    all_questions = read_jsonl(args.questions)
    questions = [row for row in all_questions if args.split == "all" or row.get("split") == args.split]
    requested_case_ids = _case_id_filter(args.case_ids)
    if requested_case_ids:
        questions = [row for row in questions if str(row.get("id")) in requested_case_ids]
        missing = requested_case_ids - {str(row.get("id")) for row in questions}
        if missing:
            raise ValueError(f"Unknown or out-of-split case IDs: {sorted(missing)}")
    _validate_public_questions(questions)
    prior = _load_existing(args.output) if args.resume else {}
    records: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat()
    health = _get_json(f"{args.base_url.rstrip('/')}/api/health/ready", min(args.timeout, 30.0))

    for index, case in enumerate(questions, start=1):
        case_id = str(case["id"])
        if case_id in prior and not prior[case_id].get("request_error"):
            record = prior[case_id]
        else:
            record = run_case(case, args.base_url, args.timeout)
        records.append(record)
        _write_output(
            args.output,
            {
                "run": {
                    "started_at": started_at,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "base_url": args.base_url,
                    "split": args.split,
                    "question_count": len(questions),
                    "questions_sha256": _sha256_file(args.questions),
                    "runner_sha256": _sha256_file(Path(__file__)),
                    "health_snapshot": health,
                    "gold_loaded_by_runner": False,
                    "case_ids": sorted(requested_case_ids),
                },
                "results": records,
            },
        )
        print(f"[{index:03d}/{len(questions):03d}] {case_id} status={record.get('http_status')} error={bool(record.get('request_error'))}")
        if args.request_delay:
            time.sleep(args.request_delay)

    errors = sum(bool(item.get("request_error")) for item in records)
    summary = {"case_count": len(questions), "completed_count": len(records) - errors, "request_error_count": errors}
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if errors == 0 else 2


def run_case(case: dict[str, Any], base_url: str, timeout: float) -> dict[str, Any]:
    payload = {"question": case["question"], "include_debug": True}
    request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/qa/ask",
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    requested_at = datetime.now(timezone.utc).isoformat()
    status: int | None = None
    response_headers: dict[str, str] = {}
    response_body = b""
    error: str | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            response_headers = dict(response.headers.items())
            response_body = response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        response_headers = dict(exc.headers.items()) if exc.headers else {}
        response_body = exc.read()
        error = f"HTTPError: {exc}"
    except (OSError, urllib.error.URLError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    raw_text = response_body.decode("utf-8", errors="replace")
    parsed: dict[str, Any] | None = None
    if raw_text:
        try:
            value = json.loads(raw_text)
            parsed = value if isinstance(value, dict) else {"value": value}
        except json.JSONDecodeError as exc:
            error = error or f"JSONDecodeError: {exc}"
    if status is None or status < 200 or status >= 300:
        error = error or f"unexpected HTTP status: {status}"
    return {
        "case_id": case["id"],
        "scenario": case["scenario"],
        "question_type": case.get("question_type"),
        "answerable": case.get("answerable"),
        "difficulty": case["difficulty"],
        "split": case["split"],
        "question": case["question"],
        "requested_at": requested_at,
        "elapsed_ms": round(elapsed_ms, 3),
        "http_status": status,
        "response_headers": response_headers,
        "raw_response_text": raw_text,
        "response": parsed,
        "request_error": error,
    }


def _validate_public_questions(rows: list[dict[str, Any]]) -> None:
    allowed = {
        "id",
        "scenario",
        "difficulty",
        "split",
        "question",
        "question_type",
        "answerable",
        "scoring_type",
    }
    for row in rows:
        extra = set(row) - allowed
        if extra:
            raise ValueError(f"Public question record {row.get('id')} contains forbidden fields: {sorted(extra)}")
        if any(key in row for key in ("canonical_answer", "required_conclusions", "evidence", "forbidden_conclusions")):
            raise ValueError(f"Gold data leaked into public question record {row.get('id')}")


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return {"http_status": response.status, "body": payload}
    except Exception as exc:  # health capture must not hide per-case diagnostics
        return {"http_status": None, "error": f"{type(exc).__name__}: {exc}"}


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(item.get("case_id")): item for item in payload.get("results") or []}


def _write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _case_id_filter(value: str | None) -> set[str]:
    return {item.strip() for item in str(value or "").split(",") if item.strip()}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
