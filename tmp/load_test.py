"""Concurrent QA load test: N workers x M questions, report errors and latency.

Usage:
    python tmp/load_test.py --questions <jsonl> --workers 4 --limit 40 [--base-url http://127.0.0.1:8000]
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def ask(base_url: str, question: str, timeout: float) -> dict:
    payload = json.dumps({"question": question, "include_debug": False}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        base_url + "/api/qa/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args()

    questions = [
        json.loads(line) for line in open(args.questions, encoding="utf-8") if line.strip()
    ][: args.limit]
    errors: list[dict] = []
    latencies: list[float] = []
    refused = 0
    completed = 0

    def run_one(question: dict) -> dict:
        started = time.perf_counter()
        try:
            body = ask(args.base_url, question["question"], args.timeout)
            latency = (time.perf_counter() - started) * 1000
            return {"id": question["id"], "ok": True, "latency_ms": latency,
                    "refused": bool(body.get("refused")), "status": body.get("generation_status")}
        except urllib.error.HTTPError as exc:
            return {"id": question["id"], "ok": False, "error": f"HTTP {exc.code}: {exc.reason}",
                    "latency_ms": (time.perf_counter() - started) * 1000}
        except Exception as exc:  # noqa: BLE001
            return {"id": question["id"], "ok": False, "error": f"{type(exc).__name__}: {exc}",
                    "latency_ms": (time.perf_counter() - started) * 1000}

    print(f"[load] workers={args.workers} questions={len(questions)}", flush=True)
    started_at = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, q) for q in questions]
        for future in as_completed(futures):
            result = future.result()
            if result["ok"]:
                completed += 1
                if result["refused"]:
                    refused += 1
                latencies.append(result["latency_ms"])
            else:
                errors.append(result)
            print(f"  {result['id']}: {'OK' if result['ok'] else 'ERR'} "
                  f"{result.get('latency_ms', 0):.0f}ms {result.get('error', '')}", flush=True)
    total_elapsed = time.perf_counter() - started_at
    error_codes: dict[str, int] = {}
    for error in errors:
        code = str(error["error"]).split(":")[0]
        error_codes[code] = error_codes.get(code, 0) + 1
    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0
    p95 = latencies[int(len(latencies) * 0.95) - 1] if latencies else 0
    print(f"[load] completed={completed} refused={refused} errors={len(errors)} "
          f"error_codes={error_codes}", flush=True)
    print(f"[load] wall={total_elapsed:.1f}s p50={p50:.0f}ms p95={p95:.0f}ms "
          f"max={max(latencies, default=0):.0f}ms", flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
