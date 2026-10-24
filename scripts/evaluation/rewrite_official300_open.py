"""Rewrite official-300 multiple-choice items into neutral open questions.

The official questions are multiple-choice; 233/300 stems say "以下哪一项正确"
and carry the actual topic only in the options.  Removing options verbatim
therefore produces an ill-posed question (the system asks for options, which
we verified).  This script rewrites each item into an equivalent open question
that keeps the topic but not the option text or the answer's specific values.

Numeric extraction questions keep their original wording: "……的取值是多少？"
is already an open question.

Output: JSONL with one row per case, resumable by id.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.evaluate_contest_qa import read_cases
from backend.app.core.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

PROMPT = """你是银行监管知识库评测的出题助手。下面是一道选择题。请把它改写成一道等价的开卷问答题。
要求：
1. 只问正确选项所表达的那个知识点，不要问“以下哪一项正确”。
2. 题干不得出现 A/B/C/D、选项、以上都对/错。
3. 题干不得直接复述正确选项的结论、数字或完整措辞；只保留必要的主题、制度名、对象、指标名。
4. 问题必须能仅凭监管原文回答，不要要求比较四个选项。
5. 输出 JSON：{{"open_question":"...","topic":"...","leak_risk":"low/medium/high"}}

制度名：{title}
原题：{question}
正确选项：{answer}
"""


def _api_key() -> str:
    if LLM_API_KEY:
        return LLM_API_KEY
    raise RuntimeError("MODEL_API_KEY is not configured in .env or the environment")


def _call(prompt: str, key: str, timeout: float) -> dict:
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        LLM_BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return json.loads(body["choices"][0]["message"]["content"])


def _valid(question: str, answer: str) -> bool:
    if not question or len(question) < 10 or len(question) > 240:
        return False
    banned = ("以下哪", "哪一项", "哪项", "A/B/C/D", "选项", "以上都")
    if any(word in question for word in banned):
        return False
    numbers = re.findall(r"\d+(?:\.\d+)?", answer)
    return not (numbers and any(number in question for number in numbers))


def _fallback(case) -> str:
    topic = case.source_title or "该制度"
    return f"请根据《{topic}》回答：{case.question.rstrip('。？?')}。请直接给出规定内容，不要要求提供选项。"


def _rewrite(case, key: str, timeout: float) -> dict:
    answer = case.answer_text.strip()
    if re.fullmatch(r"[-+]?\d[\d,\.]*\s*[%％]?", answer):
        return {
            "id": case.id,
            "open_question": case.question,
            "topic": "",
            "leak_risk": "low",
            "rewrite_status": "numeric_original",
        }
    prompt = PROMPT.format(title=case.source_title, question=case.question, answer=answer)
    last_error = None
    for attempt in range(3):
        try:
            payload = _call(prompt, key, timeout)
            question = str(payload.get("open_question") or "").strip()
            leak = str(payload.get("leak_risk") or "medium").lower()
            if _valid(question, answer) and leak != "high":
                return {
                    "id": case.id,
                    "open_question": question,
                    "topic": str(payload.get("topic") or ""),
                    "leak_risk": leak,
                    "rewrite_status": "llm",
                }
            last_error = f"invalid rewrite: {question[:80]} leak={leak}"
            break
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"[:300]
            time.sleep(1.5 + attempt)
    return {
        "id": case.id,
        "open_question": _fallback(case),
        "topic": "",
        "leak_risk": "unknown",
        "rewrite_status": "fallback",
        "rewrite_error": last_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/official300_open/open_questions.jsonl"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    cases = read_cases("data/contest_dataset/QA数据.xlsx")
    if args.start:
        cases = cases[args.start :]
    if args.limit:
        cases = cases[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[str(row["id"])] = row
    pending = [case for case in cases if str(case.id) not in done]
    key = _api_key()
    print(f"[rewrite] total={len(cases)} done={len(done)} pending={len(pending)}", flush=True)

    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_rewrite, case, key, args.timeout): case for case in pending}
        for future in as_completed(futures):
            case = futures[future]
            row = future.result()
            row.update(
                source_title=case.source_title,
                original_question=case.question,
                answer_key=case.answer,
                answer_text=case.answer_text,
                gold_evidence=case.evidence,
                source_type=case.source_type,
                qa_type=case.qa_type,
            )
            with lock:
                done[str(case.id)] = row
                with args.out.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                print(f"[rewrite] {len(done)}/{len(cases)} {case.id} {row['rewrite_status']}", flush=True)

    rows = [done[str(case.id)] for case in cases if str(case.id) in done]
    args.out.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    status = {}
    for row in rows:
        status[row["rewrite_status"]] = status.get(row["rewrite_status"], 0) + 1
    print(json.dumps({"rows": len(rows), "status": status, "output": str(args.out)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
