# -*- coding: utf-8 -*-
"""Official-300 MCQ timing runner (serial, records latency + device snapshot).

Usage:
    python tmp/run_official300_timing.py [base_url] [out_dir]
"""

import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.evaluate_contest_qa import read_cases  # noqa: E402


def _is_correct(body: dict, case) -> bool:
    answer = str(body.get("answer") or "")
    label_match = re.search(r"(?:正确选项|答案为|选择)(?:是|为)?[:：]?\s*([A-D])", answer)
    if label_match and label_match.group(1) == str(case.answer).strip().upper():
        return True
    expected_index = ord(str(case.answer).strip().upper()) - ord("A")
    if 0 <= expected_index < len(case.options):
        expected_text = str(case.options[expected_index] or "")
        if expected_text and expected_text in answer:
            return True
    return False


def ask(base_url: str, question: str, options: list[str], timeout: float) -> dict:
    payload = {"question": question, "options": options, "include_debug": False}
    request = urllib.request.Request(
        base_url + "/api/qa/ask",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def health(base_url: str) -> dict:
    with urllib.request.urlopen(base_url + "/api/health/rag", timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8002"
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "data/evaluation/official300_timing")
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = read_cases("data/contest_dataset/QA数据.xlsx")

    # warmup: first MCQ request triggers model loading
    t0 = time.perf_counter()
    try:
        ask(base_url, cases[0].question, list(cases[0].options), 120.0)
        print(f"[warmup] done in {time.perf_counter() - t0:.1f}s", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[warmup] failed: {exc}", flush=True)

    results = []
    for i, case in enumerate(cases, start=1):
        started = time.perf_counter()
        row = {"id": case.id, "correct": False, "answer_type": None, "refused": None, "error": None, "latency_ms": None}
        try:
            body = ask(base_url, case.question, list(case.options), 120.0)
            row["correct"] = _is_correct(body, case)
            row["answer_type"] = body.get("answer_type")
            row["refused"] = body.get("refused")
        except Exception as exc:  # noqa: BLE001
            row["error"] = str(exc)[:300]
        row["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        results.append(row)
        if i % 50 == 0 or i == len(cases):
            print(f"[official300] {i}/{len(cases)}", flush=True)

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    errors = sum(1 for r in results if r.get("error"))
    lat = [r["latency_ms"] for r in results if r.get("latency_ms") is not None]
    lat_sorted = sorted(lat)
    stats = {
        "n": len(lat),
        "avg_ms": round(sum(lat) / len(lat), 2) if lat else None,
        "p50_ms": round(lat_sorted[len(lat_sorted) // 2], 2) if lat else None,
        "p95_ms": round(lat_sorted[min(len(lat_sorted) - 1, int(len(lat_sorted) * 0.95) - 1)], 2) if lat else None,
        "max_ms": round(max(lat), 2) if lat else None,
    }

    try:
        dev = health(base_url)
    except Exception:  # noqa: BLE001
        dev = {}

    artifact = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "device": {
            "selected_device": (dev.get("model_device") or {}).get("selected_device"),
            "cuda_available": (dev.get("model_device") or {}).get("cuda_available"),
            "cuda_device_name": (dev.get("model_device") or {}).get("cuda_device_name"),
            "performance_mode": (dev.get("performance") or {}).get("selected_mode"),
        },
        "summary": {
            "total": total,
            "correct": correct,
            "errors": errors,
            "accuracy": correct / total if total else None,
            "latency": stats,
        },
        "results": results,
    }
    (out_dir / "official_300_results.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(artifact["summary"], ensure_ascii=False, indent=1))
    print(json.dumps(artifact["device"], ensure_ascii=False))
    return 0 if correct == total and errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
