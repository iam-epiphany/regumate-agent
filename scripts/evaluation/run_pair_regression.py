"""Serial pair-batch regression runner (old 100 + new 100 questions).

Usage:
    python tmp/run_pair_regression.py --questions <public/questions.jsonl> --out <outputs.json> [--base-url http://127.0.0.1:8000]

Resumes an interrupted run by re-loading the existing output file.  Serial by
design: the 4-worker container returned 503s under parallel load (stage-3
scope), so correctness regression runs one request at a time.
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone


def _sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ask(base_url: str, question: str, timeout: float) -> dict:
    payload = {"question": question, "include_debug": True}
    request = urllib.request.Request(
        base_url + "/api/qa/ask",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args()

    questions = [
        json.loads(line) for line in open(args.questions, encoding="utf-8") if line.strip()
    ]
    questions_sha256 = _sha256_of_file(args.questions)

    run: dict = {}
    try:
        run = json.load(open(args.out, encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    if not run:
        run = {
            "schema_version": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base_url": args.base_url,
            "questions_sha256": questions_sha256,
            "complete": False,
            "runtime_error_count": 0,
            "results": [],
        }
    else:
        # Never re-run against a different question set.
        assert run.get("questions_sha256") == questions_sha256, "question file changed"

    done_ids = {row.get("id") for row in run["results"]}
    pending = [q for q in questions if q["id"] not in done_ids]
    print(f"[regression] {len(done_ids)} done, {len(pending)} pending", flush=True)

    for index, question in enumerate(pending, start=1):
        started = time.perf_counter()
        row: dict = {"id": question["id"], "question": question["question"], "response": None, "error": None, "latency_ms": None}
        try:
            body = ask(args.base_url, question["question"], timeout=args.timeout)
            row["response"] = body
        except urllib.error.HTTPError as exc:
            row["error"] = f"HTTP {exc.code}: {exc.reason}"
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
                row["error"] += f" {detail}"
            except Exception:
                pass
        except Exception as exc:  # noqa: BLE001 - record and continue
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        run["results"].append(row)
        if row["error"]:
            run["runtime_error_count"] += 1
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(run, handle, ensure_ascii=False, indent=2)
        print(
            f"[regression {len(run['results'])}/{len(questions)}] {question['id']} "
            f"error={bool(row['error'])} latency={row['latency_ms']:.0f}ms",
            flush=True,
        )

    run["complete"] = True
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(run, handle, ensure_ascii=False, indent=2)
    print(f"[regression] complete: {len(run['results'])} rows, {run['runtime_error_count']} errors", flush=True)
    return 0 if run["runtime_error_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
