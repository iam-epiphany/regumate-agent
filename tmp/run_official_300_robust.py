"""Robust official-300 choice regression runner: bounded concurrency, retries,
checkpointing and incremental writes."""

import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, ".")
from scripts.evaluate_contest_qa import read_cases  # noqa: E402

OUTPUT = Path("data/evaluation/official_300_regression_fix1.json")
BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"


def _is_correct(body: dict, case) -> bool:
    sys.path.insert(0, "scripts/evaluation")
    sys.path.insert(0, ".")
    from banking_workbench.workbench_common import normalise

    answer = str(body.get("answer") or "")
    label_match = re.search(r"(?:正确选项|答案为|选择)(?:是|为)?[:：]?\s*([A-D])", answer)
    if label_match and label_match.group(1) == str(case.answer).strip().upper():
        return True
    expected_index = ord(str(case.answer).strip().upper()) - ord("A")
    if 0 <= expected_index < len(case.options):
        expected_text = str(case.options[expected_index] or "")
        if expected_text and normalise(expected_text) in normalise(answer):
            return True
    return False


def ask_once(question: str, options: list[str], timeout: float) -> dict:
    payload = {"question": question, "options": options, "include_debug": False}
    request = urllib.request.Request(
        BASE_URL + "/api/qa/ask",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    cases = read_cases("data/contest_dataset/QA数据.xlsx")
    results_by_id: dict[str, dict] = {}
    if OUTPUT.exists():
        prior = json.loads(OUTPUT.read_text(encoding="utf-8"))
        for row in prior.get("results") or []:
            if not row.get("error"):
                results_by_id[str(row["id"])] = row

    def perform(case) -> dict:
        for attempt in range(3):
            try:
                body = ask_once(case.question, list(case.options), 240.0)
                return {
                    "id": case.id,
                    "correct": _is_correct(body, case),
                    "answer_type": body.get("answer_type"),
                    "refused": body.get("refused"),
                    "error": None,
                }
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    return {"id": case.id, "correct": False, "error": str(exc)[:80]}
                time.sleep(2)

    pending = [case for case in cases if str(case.id) not in results_by_id]
    completed = len(results_by_id)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(perform, case): case for case in pending}
        for future in as_completed(futures):
            case = futures[future]
            row = future.result()
            results_by_id[str(case.id)] = row
            completed += 1
            done_rows = [results_by_id[str(case.id)] for case in cases if str(case.id) in results_by_id]
            artifact = {
                "total": len(cases),
                "correct": sum(1 for r in done_rows if r["correct"]),
                "errors": sum(1 for r in done_rows if r.get("error")),
                "results": done_rows,
            }
            OUTPUT.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
            if completed % 20 == 0:
                print(f"[official {completed}/300] correct={artifact['correct']} errors={artifact['errors']}", flush=True)

    artifact = {
        "total": len(cases),
        "correct": sum(1 for r in results_by_id.values() if r["correct"]),
        "errors": sum(1 for r in results_by_id.values() if r.get("error")),
        "results": [results_by_id[str(case.id)] for case in cases],
    }
    OUTPUT.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    wrong = [r["id"] for r in artifact["results"] if not r["correct"] and not r.get("error")]
    print(f"official 300: {artifact['correct']}/{artifact['total']} correct, {artifact['errors']} errors")
    print("wrong:", wrong)
    return 0 if artifact["correct"] == artifact["total"] and artifact["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
