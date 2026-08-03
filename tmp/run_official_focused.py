"""Focused official-300 regression over the fix-affected surface:
all 100 table cases (planner table parsing changed) + 40 text cases
(boundary/refusal and generation fallback changed)."""

import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, ".")
from scripts.evaluate_contest_qa import read_cases  # noqa: E402

OUTPUT = Path("data/evaluation/official_300_focused_regression.json")
BASE_URL = "http://127.0.0.1:8000"

# Every 25th text case from the remaining 200, deterministic selection.
TEXT_SAMPLE = set(range(0, 200, 5))


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


def ask_once(question: str, options: list[str]) -> dict:
    payload = {"question": question, "options": options, "include_debug": False}
    request = urllib.request.Request(
        BASE_URL + "/api/qa/ask",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def perform(case) -> dict:
    for attempt in range(2):
        try:
            body = ask_once(case.question, list(case.options))
            return {"id": case.id, "correct": _is_correct(body, case), "error": None}
        except Exception as exc:  # noqa: BLE001
            if attempt == 1:
                return {"id": case.id, "correct": False, "error": str(exc)[:60]}
            time.sleep(1)


def main() -> int:
    cases = read_cases("data/contest_dataset/QA数据.xlsx")
    selected = [
        case
        for index, case in enumerate(cases)
        if case.source_type == "excel" or index in TEXT_SAMPLE
    ]
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(perform, case): case for case in selected}
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            done = len(results)
            if done % 25 == 0:
                print(f"[focused {done}/{len(selected)}]", flush=True)
    results.sort(key=lambda row: str(row["id"]))
    artifact = {
        "total": len(results),
        "correct": sum(1 for r in results if r["correct"]),
        "errors": sum(1 for r in results if r.get("error")),
        "results": results,
    }
    OUTPUT.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    wrong = [r["id"] for r in results if not r["correct"] and not r.get("error")]
    print(f"focused regression: {artifact['correct']}/{artifact['total']} correct, {artifact['errors']} errors")
    print("wrong:", wrong)
    return 0 if artifact["errors"] == 0 and not wrong else 1


if __name__ == "__main__":
    raise SystemExit(main())
