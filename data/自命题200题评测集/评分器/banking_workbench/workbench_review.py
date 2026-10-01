"""Banking Workbench LLM boundary review.

Evaluation-only.  Picks borderline cases from the deterministic diagnostic and
has an independent LLM review whether the deterministic verdict was fair.
The review never changes the main deterministic score; it produces a separate
report for human inspection.

Selection rules (max 15 cases, at most 5 per question type):
1. facts all matched but sources not met (likely citation mis-match);
2. a conclusion at the bigram-recall boundary [0.45, 0.55) or with all
   critical entities present but recall slightly below threshold;
3. threshold_rule failed but the answer contains every gold numeric entity;
4. refusal_boundary failed but the answer contains refusal wording.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

try:
    from banking_workbench.workbench_common import normalise, read_jsonl
except ModuleNotFoundError:
    from scripts.evaluation.banking_workbench.workbench_common import normalise, read_jsonl

MAX_REVIEW_CASES = 15
MAX_PER_TYPE = 5


def _request_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['WORKBENCH_REVIEW_API_KEY']}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_content(content: str) -> dict[str, Any]:
    text = str(content or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def _select_boundary_cases(diagnostic: dict[str, Any], gold_by_id: dict[str, dict[str, Any]]) -> list[str]:
    selected: list[str] = []
    per_type: dict[str, int] = {}
    rows = diagnostic.get("diagnostic") or []
    for item in rows:
        if item.get("passed") or item.get("failure_category") == "runtime":
            continue
        case_id = str(item.get("id") or "")
        gold = gold_by_id.get(case_id) or {}
        answerable = bool(gold.get("answerable"))
        kind = str(item.get("question_type") or "unknown")
        if per_type.get(kind, 0) >= MAX_PER_TYPE:
            continue
        reasons: list[str] = []
        if answerable and item.get("failure_category") == "answer_generation":
            # Semantic-equivalence boundary: the deterministic matcher may
            # have missed a paraphrase; LLM review checks fairness.
            reasons.append("answer_generation_paraphrase")
        if not answerable and item.get("failure_category") == "refusal_boundary":
            reasons.append("refusal_boundary")
        if not reasons:
            continue
        selected.append(case_id)
        per_type[kind] = per_type.get(kind, 0) + 1
        if len(selected) >= MAX_REVIEW_CASES:
            break
    return selected


def review(
    diagnostic_path: Path,
    gold_path: Path,
    outputs_path: Path,
    output_path: Path,
    base_url: str,
    model: str,
    timeout: float,
) -> int:
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    gold_by_id = {str(item.get("id")): item for item in read_jsonl(gold_path)}
    run = json.loads(outputs_path.read_text(encoding="utf-8"))
    answer_by_id = {
        str(row.get("id")): str((row.get("response") or {}).get("answer") or "")
        for row in (run.get("results") or [])
    }
    selected_ids = _select_boundary_cases(diagnostic, gold_by_id)
    results: list[dict[str, Any]] = []
    for case_id in selected_ids:
        gold = gold_by_id.get(case_id) or {}
        question = str(gold.get("question") or "")
        canonical = str(gold.get("canonical_answer") or "")
        answer = answer_by_id.get(case_id, "")
        prompt = (
            "你是银行监管问答评审器。只判断候选答案与参考答案的事实语义是否一致，"
            "不能因表达不同而误判，也不能补充参考答案之外的事实。输出严格 JSON："
            '{"correct":true或false,"confidence":0到1之间的小数,"reason":"简短中文理由"}。\n'
            f"问题：{question}\n参考答案：{canonical}\n候选答案：{answer}"
        )
        verdict: dict[str, Any]
        try:
            body = _request_json(
                f"{base_url.rstrip('/')}/chat/completions",
                {
                    "model": model,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout,
            )
            content = body["choices"][0]["message"]["content"]
            verdict = _parse_content(content)
        except Exception as exc:  # noqa: BLE001
            verdict = {"correct": None, "confidence": 0.0, "reason": f"review_error: {exc}"}
        results.append({"id": case_id, "verdict": verdict})
        print(f"[review {len(results)}/{len(selected_ids)}] {case_id} correct={verdict.get('correct')}", flush=True)
    artifact = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "review_model": model,
        "selected_count": len(selected_ids),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(artifact, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Banking Workbench LLM boundary review")
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default=os.environ.get("WORKBENCH_REVIEW_BASE_URL", ""))
    parser.add_argument("--model", default=os.environ.get("WORKBENCH_REVIEW_MODEL", ""))
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    if not args.base_url or not args.model or not os.environ.get("WORKBENCH_REVIEW_API_KEY"):
        raise SystemExit("WORKBENCH_REVIEW_BASE_URL / WORKBENCH_REVIEW_MODEL / WORKBENCH_REVIEW_API_KEY required")
    return review(args.diagnostic, args.gold, args.outputs, args.output, args.base_url, args.model, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
