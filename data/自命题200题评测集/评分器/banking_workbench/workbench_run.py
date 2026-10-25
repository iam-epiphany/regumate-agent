"""Banking Workbench blind-run runner.

Evaluation-only.  Sends only the public question projection to the QA API.
The output file is created exclusively (never overwritten) so the first run
stays immutable; checkpoints are bound to the questions hash and base URL.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

try:
    from banking_workbench.workbench_common import sha256_file
except ModuleNotFoundError:
    from scripts.evaluation.banking_workbench.workbench_common import sha256_file


def request_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def run(questions_path: Path, output: Path, base_url: str, timeout: float, workers: int, resume: bool) -> int:
    questions = [json.loads(line) for line in questions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not questions:
        raise SystemExit("no questions")
    questions_sha256 = sha256_file(questions_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    exclusive = os.open(str(output), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(exclusive)

    rows_by_id: dict[str, dict[str, Any]] = {}
    if resume and output.exists():
        prior = json.loads(output.read_text(encoding="utf-8"))
        if prior.get("questions_sha256") != questions_sha256 or prior.get("base_url") != base_url:
            raise SystemExit("resume identity mismatch: questions hash or base URL changed")
        rows_by_id = {str(row["id"]): row for row in prior.get("results") or []}

    def perform(item: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        row: dict[str, Any] = {"id": item["id"], "question": item["question"]}
        try:
            body = request_json(
                f"{base_url.rstrip('/')}/api/qa/ask",
                {"question": item["question"]},
                timeout,
            )
            row.update({"response": body, "error": None})
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            row.update({"response": None, "error": str(exc)})
        row["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return row

    pending = [
        item
        for item in questions
        if str(item["id"]) not in rows_by_id or rows_by_id[str(item["id"])].get("error")
    ]
    completed = sum(1 for row in rows_by_id.values() if not row.get("error"))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(perform, item): item for item in pending}
        for future in as_completed(futures):
            item = futures[future]
            row = future.result()
            rows_by_id[str(row["id"])] = row
            completed += int(not row.get("error"))
            checkpoint = {
                "schema_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "base_url": base_url.rstrip("/"),
                "questions_sha256": questions_sha256,
                "complete": completed == len(questions),
                "results": [rows_by_id[str(item["id"])] for item in questions if str(item["id"]) in rows_by_id],
            }
            output.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[workbench {completed:03d}/{len(questions)}] {item['id']} error={bool(row['error'])}", flush=True)

    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url.rstrip("/"),
        "questions_sha256": questions_sha256,
        "complete": completed == len(questions),
        "runtime_error_count": sum(1 for row in rows_by_id.values() if row.get("error")),
        "results": [rows_by_id[str(item["id"])] for item in questions if str(item["id"]) in rows_by_id],
    }
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if artifact["runtime_error_count"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Banking Workbench blind-run runner")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    return run(args.questions, args.output, args.base_url, args.timeout, args.workers, args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
