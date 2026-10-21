"""Score the open-question official-300 run against the answer key.

This is a diagnostic scorer, not a replacement for the official MCQ grader.
It separates:

* answered vs clarification/refusal;
* exact/numeric answer match;
* cases where the answer key and the cited source disagree (fact-review list).

Usage:
    python scripts/evaluation/score_official300_open.py \
        --results outputs/official300_open/literal_no_options.json \
        --out-dir outputs/official300_open/literal_report
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.evaluate_contest_qa import read_cases


def _normalize(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\[\d+\]", "", text)
    text = text.replace("《", "").replace("》", "")
    text = re.sub(r"[\s·•、，,。.;；:：()（）\"'“”‘’\-_/]+", "", text)
    return text


def _numbers(value: Any) -> list[float]:
    values: list[float] = []
    for match in re.finditer(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", str(value or "")):
        token = match.group(0).replace(",", "")
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _number_match(gold: str, answer: str) -> tuple[bool, str]:
    gold_numbers = _numbers(gold)
    answer_numbers = _numbers(answer)
    if not gold_numbers:
        return False, "no_gold_number"
    for target in gold_numbers:
        for candidate in answer_numbers:
            tolerance = max(0.01, abs(target) * 0.001)
            if abs(target - candidate) <= tolerance:
                return True, f"number:{target}"
    return False, "number_missing"


def _bigrams(text: str) -> set[str]:
    normalized = _normalize(text)
    return {normalized[index : index + 2] for index in range(max(0, len(normalized) - 1))}


def _text_match(gold: str, answer: str) -> tuple[bool, str, float]:
    gold_norm = _normalize(gold)
    answer_norm = _normalize(answer)
    if not gold_norm or not answer_norm:
        return False, "empty", 0.0
    if gold_norm in answer_norm:
        return True, "containment", 1.0
    if answer_norm in gold_norm and len(answer_norm) >= max(8, len(gold_norm) // 3):
        return True, "answer_contained", 1.0
    gold_grams = _bigrams(gold)
    answer_grams = _bigrams(answer)
    if not gold_grams:
        return False, "no_grams", 0.0
    recall = len(gold_grams & answer_grams) / len(gold_grams)
    precision = len(gold_grams & answer_grams) / max(len(answer_grams), 1)
    score = 0.7 * recall + 0.3 * precision
    if score >= 0.62:
        return True, f"bigram:{score:.2f}", score
    return False, f"bigram_low:{score:.2f}", score


def _citation_text(response: dict[str, Any]) -> str:
    parts = []
    for citation in (response.get("citations") or [])[:5]:
        parts.append(str(citation.get("excerpt") or ""))
        parts.append(str(citation.get("section_title") or ""))
    return "\n".join(part for part in parts if part)


def _wrong_option_hit(response: dict[str, Any], gold: dict[str, Any]) -> bool:
    answer = _normalize(response.get("answer"))
    if not answer:
        return False
    correct_index = "abcd".find(str(gold.get("answer_key") or "").lower())
    for index, option in enumerate(gold.get("options") or []):
        if index == correct_index:
            continue
        normalized = _normalize(option)
        if len(normalized) >= 20 and normalized in answer:
            return True
    return False


def _score_row(row: dict[str, Any]) -> dict[str, Any]:
    response = row.get("response") or {}
    answer = str(response.get("answer") or "")
    gold_text = str(row.get("answer_text") or "")
    result: dict[str, Any] = {
        "id": row.get("id"),
        "source_type": row.get("source_type"),
        "qa_type": row.get("qa_type"),
        "question": row.get("question"),
        "gold_answer_key": row.get("answer_key"),
        "gold_answer_text": gold_text,
        "system_answer": answer,
        "answer_type": response.get("answer_type"),
        "refused": response.get("refused"),
        "error": row.get("error"),
        "match": False,
        "basis": "not_answered",
        "score": 0.0,
        "citation_supports_gold": False,
        "wrong_option": False,
        "needs_fact_review": False,
        "citations": [
            {
                "filename": citation.get("filename"),
                "section_title": citation.get("section_title"),
                "structure_label": citation.get("structure_label"),
                "excerpt": str(citation.get("excerpt") or "")[:400],
            }
            for citation in (response.get("citations") or [])[:3]
        ],
    }
    if row.get("error"):
        result["basis"] = "error"
        return result
    if response.get("refused") or response.get("answer_type") in {"clarification", "refusal"}:
        result["basis"] = response.get("answer_type") or "refused"
        if answer.strip() and response.get("answer_type") != "clarification":
            # A refusal can still contain the correct fact (over-refusal).  Score
            # it, but keep the refusal flag visible in the report.
            if gold_text and re.fullmatch(r"[-+]?\d[\d,\.]*\s*[%％]?", gold_text.strip()):
                matched, basis = _number_match(gold_text, answer)
                score = 1.0 if matched else 0.0
            else:
                matched, basis, score = _text_match(gold_text, answer)
                matched = bool(matched and (basis in {"containment", "answer_contained"} or score >= 0.75))
            if matched:
                result["match"] = True
                result["basis"] = "refusal_with_fact"
                result["score"] = round(score, 4)
        return result
    if not answer.strip():
        result["basis"] = "empty_answer"
        return result

    if gold_text and re.fullmatch(r"[-+]?\d[\d,\.]*\s*[%％]?", gold_text.strip()):
        matched, basis = _number_match(gold_text, answer)
    else:
        matched, basis, score = _text_match(gold_text, answer)
        result["score"] = round(score, 4)
    result["match"] = bool(matched)
    result["basis"] = basis
    result["wrong_option"] = _wrong_option_hit(response, row)

    citation_text = _citation_text(response)
    citation_numbers = _numbers(citation_text)
    gold_numbers = _numbers(gold_text)
    if gold_numbers and citation_numbers:
        result["citation_supports_gold"] = any(
            abs(target - candidate) <= max(0.01, abs(target) * 0.001)
            for target in gold_numbers
            for candidate in citation_numbers
        )
    elif gold_text:
        result["citation_supports_gold"] = _normalize(gold_text)[:30] in _normalize(citation_text)
    result["needs_fact_review"] = bool(
        not result["match"] and (result["citation_supports_gold"] or result["wrong_option"])
    )
    return result


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    answer_types = Counter(str(row.get("answer_type")) for row in rows)
    bases = Counter(str(row.get("basis")) for row in rows)
    answered = sum(
        1 for row in rows
        if row.get("basis") not in {
            "clarification", "refusal", "refused", "service_error",
            "error", "empty_answer", "refusal_with_fact",
        }
    )
    correct = sum(1 for row in rows if row.get("match"))
    correct_real = sum(
        1 for row in rows if row.get("match") and row.get("basis") != "refusal_with_fact"
    )
    return {
        "total": total,
        "answered": answered,
        "answered_rate": round(answered / total, 4) if total else None,
        "correct": correct,
        "correct_real_answers": correct_real,
        "accuracy_over_total": round(correct / total, 4) if total else None,
        "accuracy_over_answered": round(correct_real / answered, 4) if answered else None,
        "clarification": answer_types.get("clarification", 0),
        "refusal": answer_types.get("refusal", 0),
        "service_error": answer_types.get("service_error", 0),
        "errors": sum(1 for row in rows if row.get("error")),
        "wrong_option": sum(1 for row in rows if row.get("wrong_option")),
        "refusal_with_fact": sum(1 for row in rows if row.get("basis") == "refusal_with_fact"),
        "citation_supports_gold": sum(1 for row in rows if row.get("citation_supports_gold")),
        "needs_fact_review": sum(1 for row in rows if row.get("needs_fact_review")),
        "answer_types": dict(sorted(answer_types.items())),
        "match_basis": dict(sorted(bases.items())),
    }


def _by(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key)), []).append(row)
    return {name: _summary(group) for name, group in sorted(groups.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8-sig"))
    cases = {case.id: case for case in read_cases("data/contest_dataset/QA数据.xlsx")}
    raw_rows = payload.get("results") or []
    for row in raw_rows:
        case = cases.get(str(row.get("id")))
        if case is not None:
            row.setdefault("options", list(case.options))
            row.setdefault("answer_key", case.answer)
            row.setdefault("answer_text", case.answer_text)
            row.setdefault("source_type", case.source_type)
            row.setdefault("qa_type", case.qa_type)
    rows = [_score_row(row) for row in raw_rows]
    summary = _summary(rows)
    report = {
        "schema_version": 1,
        "results": str(args.results),
        "summary": summary,
        "by_source_type": _by(rows, "source_type"),
        "by_qa_type": _by(rows, "qa_type"),
        "disagreements": [
            row for row in rows
            if not row.get("match") and row.get("basis") not in {"clarification", "refusal", "refused", "error", "empty_answer"}
        ],
        "rows": rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "score.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in (
        "total", "answered", "correct", "correct_real_answers", "accuracy_over_total", "accuracy_over_answered",
        "clarification", "refusal", "service_error", "errors", "wrong_option", "refusal_with_fact", "needs_fact_review",
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
