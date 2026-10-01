"""Temporary official-300 choice regression runner (options passed to the API)."""

import json
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, ".")
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


def main() -> int:
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    cases = read_cases("data/contest_dataset/QA数据.xlsx")
    results: list[dict] = []
    import os
    workers = int(os.environ.get("OFFICIAL300_WORKERS", "2"))
    out_path = os.environ.get("OFFICIAL300_OUT", "data/evaluation/official_300_regression_fix1.json")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(ask, base_url, case.question, list(case.options), 120.0): case
            for case in cases
        }
        for future in as_completed(futures):
            case = futures[future]
            try:
                body = future.result()
                correct = _is_correct(body, case)
                results.append({"id": case.id, "correct": correct, "answer_type": body.get("answer_type"), "refused": body.get("refused"), "error": None})
            except Exception as exc:  # noqa: BLE001
                results.append({"id": case.id, "correct": False, "error": str(exc)})
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    errors = sum(1 for r in results if r.get("error"))
    wrong = [r["id"] for r in results if not r["correct"] and not r.get("error")]
    print(f"official 300: {correct}/{total} correct, {errors} errors")
    print("wrong ids:", wrong)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"total": total, "correct": correct, "errors": errors, "results": results}, handle, ensure_ascii=False, indent=2)
    return 0 if correct == total and errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
