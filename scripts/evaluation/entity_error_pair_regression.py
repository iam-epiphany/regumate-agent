"""Deterministic entity-error rate over pair-regression (200-question) outputs.

Definition (reported with the metric, mirrors the historical "关键数字/日期/
机构/文号错误率（按事实计）"): over every fact of every answered (non-refusal)
case, count a fact as an entity error when the case failed AND the answer
asserts a conflicting key entity against that fact's gold conclusion:

* a number that matches no gold number of that fact (0.5% tolerance);
* a dated period (年/月/季度) absent from that fact's gold periods;
* a 《document title》 absent from that fact's gold titles.

entity_error_rate = conflicting facts / total facts over answered cases.

Usage:
    python scripts/evaluation/entity_error_pair_regression.py \
        --outputs <dir>/old_outputs.json --diagnostic <dir>/old_outputs_diagnostic.json \
        --gold data/自命题200题评测集/去锚100题/gold.jsonl [...]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

NUMBER_TOLERANCE = 0.005


def _numbers(text: str) -> list[float]:
    values: list[float] = []
    for match in re.finditer(r"-?\d+(?:,\d{3})*(?:\.\d+)?", str(text or "")):
        token = match.group(0).replace(",", "")
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _periods(text: str) -> set[str]:
    found = set()
    for match in re.finditer(
        r"\d{4}\s*年(?:\s*\d{1,2}\s*月)?(?:\s*\d{1,2}\s*[日号])?|\d{1,2}\s*月(?:末|度|份)?|[一二三四1234]\s*季度(?:末)?",
        str(text or ""),
    ):
        found.add(re.sub(r"\s+", "", match.group(0)))
    return found


def _titles(text: str) -> set[str]:
    return set(re.findall(r"《([^》]{2,60})》", str(text or "")))


def _conflicts(gold_conclusion: str, answer: str) -> list[str]:
    conflicts: list[str] = []
    gold_numbers = _numbers(gold_conclusion)
    if gold_numbers:
        answer_numbers = _numbers(answer)
        matched = any(
            any(
                abs(target - candidate) <= max(0.01, abs(target) * NUMBER_TOLERANCE)
                for candidate in answer_numbers
            )
            for target in gold_numbers
        )
        if not matched:
            conflicts.append("number")
    gold_periods = _periods(gold_conclusion)
    if gold_periods:
        answer_periods = _periods(answer)
        if answer_periods and not (answer_periods & gold_periods):
            conflicts.append("period")
    gold_titles = _titles(gold_conclusion)
    if gold_titles:
        answer_titles = _titles(answer)
        if answer_titles and not (answer_titles & gold_titles):
            conflicts.append("title")
    return conflicts


def evaluate(
    outputs_path: Path, diagnostic_path: Path, gold_path: Path
) -> dict[str, Any]:
    outputs = json.loads(outputs_path.read_text(encoding="utf-8-sig"))
    rows = {str(row.get("id")): row for row in outputs.get("results") or []}
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8-sig"))
    diag_rows = {str(row.get("id")): row for row in diagnostic.get("diagnostic") or []}
    gold_rows = {
        str(json.loads(line)["id"]): json.loads(line)
        for line in gold_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }
    total_facts = 0
    error_facts = 0
    details: list[dict[str, Any]] = []
    for case_id, gold in gold_rows.items():
        if not gold.get("answerable", True):
            continue
        row = rows.get(case_id) or {}
        diag = diag_rows.get(case_id) or {}
        response = row.get("response") or {}
        refused = bool(response.get("refused")) or response.get("answer_type") in {
            "clarification",
            "refusal",
        }
        if row.get("error") or refused or not str(response.get("answer") or "").strip():
            continue
        facts_total = int(diag.get("facts_total") or 0)
        total_facts += facts_total
        if diag.get("passed"):
            continue
        answer = str(response.get("answer") or "")
        conclusions = list(
            gold.get("required_conclusions") or [gold.get("canonical_answer") or ""]
        )
        for conclusion in conclusions:
            conflicts = _conflicts(str(conclusion), answer)
            if conflicts:
                error_facts += 1
                details.append(
                    {
                        "id": case_id,
                        "conflicts": conflicts,
                        "gold_conclusion": str(conclusion)[:160],
                        "answer_excerpt": answer[:160],
                    }
                )
    return {
        "outputs": str(outputs_path),
        "answered_facts_total": total_facts,
        "entity_error_facts": error_facts,
        "entity_error_rate": round(error_facts / total_facts, 4)
        if total_facts
        else None,
        "details": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch",
        action="append",
        nargs=3,
        metavar=("OUTPUTS", "DIAGNOSTIC", "GOLD"),
        required=True,
        help="outputs.json diagnostic.json gold.jsonl triplets, repeatable",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    batches = [
        evaluate(Path(outputs), Path(diag), Path(gold))
        for outputs, diag, gold in args.batch
    ]
    total_facts = sum(batch["answered_facts_total"] for batch in batches)
    error_facts = sum(batch["entity_error_facts"] for batch in batches)
    report = {
        "schema_version": 1,
        "metric_formula": (
            "entity_error_rate = 与金标结论冲突（数字±0.5%不匹配/期间不含金标期间/《文号》不含金标文号）的事实数 / 实际作答题事实总数；"
            "仅在评分未通过的案例上计数，拒答题不计"
        ),
        "batches": batches,
        "total": {
            "answered_facts_total": total_facts,
            "entity_error_facts": error_facts,
            "entity_error_rate": round(error_facts / total_facts, 4)
            if total_facts
            else None,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(report["total"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
