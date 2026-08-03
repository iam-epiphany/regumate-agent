"""Banking Workbench deterministic scorer.

Evaluation-only.  Production code never imports this package.

Scoring is fully deterministic and never calls an LLM:
- every required conclusion must be matched by the answer (exact substring,
  or bigram recall >= 0.55 with all critical entities present and no
  polarity conflict);
- every required source document must appear in the answer citations;
- runtime errors and refusal-boundary violations are classified separately.

The diagnostic output contains only safe fields (no gold content).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

try:
    from banking_workbench.workbench_common import (
    bigram_recall,
    normalise,
    normalise_title,
    polarity_conflict,
    read_jsonl,
    sha256_file,
    )
except ModuleNotFoundError:
    from scripts.evaluation.banking_workbench.workbench_common import (
    bigram_recall,
    normalise,
    normalise_title,
    polarity_conflict,
    read_jsonl,
    sha256_file,
)

SCORER_VERSION = "rigorous-banking-workbench-scorer-v1"

BANKING_UNITS = (
    "%",
    "％",
    "个百分点",
    "个工作日",
    "工作日",
    "日",
    "天",
    "个月",
    "月",
    "年",
    "倍",
    "万元",
    "亿元",
    "元",
    "项",
    "个",
    "笔",
    "家",
    "户",
    "网点",
    "次",
    "户均",
)
_UNIT_ALTERNATION = "|".join(re.escape(unit) for unit in sorted(BANKING_UNITS, key=len, reverse=True))
ENTITY_PATTERN = re.compile(
    rf"(?:20\d{{2}}年(?:\d{{1,2}}月(?:\d{{1,2}}日)?)?)|"
    rf"(?:[一二两三四五六七八九十百]+|[0-9]+(?:\.[0-9]+)?)(?:{_UNIT_ALTERNATION})|"
    rf"[0-9]+(?:\.[0-9]+)?%?"
)
DOCUMENT_PATTERN = re.compile(r"《([^》]+)》")


def _critical_entities(text: str) -> set[str]:
    return set(ENTITY_PATTERN.findall(text)) | set(DOCUMENT_PATTERN.findall(text))


def _fact_match(answer: str, gold: str) -> dict[str, Any]:
    """Match one required conclusion against the answer."""

    normalised_answer = normalise(answer)
    normalised_gold = normalise(gold)
    if not normalised_gold:
        return {"matched": False, "bigram_recall": 0.0, "critical_entities_ok": False}
    if normalised_gold in normalised_answer:
        return {"matched": True, "bigram_recall": 1.0, "critical_entities_ok": True, "exact": True}
    recall = bigram_recall(normalised_answer, normalised_gold)
    gold_entities = _critical_entities(gold)
    answer_entities = _critical_entities(answer)
    # No extractable entities (pure definition sentences) means nothing to
    # verify beyond the bigram match; an empty expected set must not fail.
    entities_ok = (not gold_entities) or gold_entities <= answer_entities
    if recall >= 0.55 and entities_ok and not polarity_conflict(answer, gold):
        return {"matched": True, "bigram_recall": round(recall, 4), "critical_entities_ok": True}
    return {
        "matched": False,
        "bigram_recall": round(recall, 4),
        "critical_entities_ok": bool(entities_ok),
        "missing_entities": sorted(gold_entities - answer_entities) if gold_entities else [],
        "polarity_conflict": polarity_conflict(answer, gold),
    }


def _source_key(relative_path: str) -> str:
    return normalise_title(Path(str(relative_path or "")).name)


def _answer_source_keys(citations: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for citation in citations or []:
        for field in ("filename", "source_title"):
            value = citation.get(field)
            if value:
                keys.add(normalise_title(value))
        metadata = citation.get("metadata") or {}
        for field in ("source_title", "source_filename", "filename"):
            value = metadata.get(field)
            if value:
                keys.add(normalise_title(value))
    return keys


def _sources_met(gold: dict[str, Any], citations: list[dict[str, Any]]) -> dict[str, Any]:
    expected = [
        _source_key(str(source.get("relative_path") or ""))
        for source in (gold.get("required_sources") or [])
        if source.get("relative_path")
    ]
    actual = _answer_source_keys(citations)
    met = [key for key in expected if any(key in item or item in key for item in actual)]
    return {
        "expected_count": len(expected),
        "matched_count": len(met),
        "met": len(expected) > 0 and len(met) == len(expected),
        "missing": [key for key in expected if key not in met],
    }


def _refusal_keywords(answer: str) -> bool:
    return any(
        marker in answer
        for marker in ("无法", "不能", "不掌握", "不在资料库", "依据不足", "未检索到", "无法给出", "无法确定", "不予")
    )


def _failure_category(row: dict[str, Any], gold: dict[str, Any], facts: dict[str, Any], sources: dict[str, Any]) -> str:
    if row.get("error") or row.get("response") is None:
        return "runtime"
    answer = str((row.get("response") or {}).get("answer") or "")
    citations = (row.get("response") or {}).get("citations") or []
    if gold.get("answerable") is False:
        refused = bool((row.get("response") or {}).get("refused"))
        if refused and (row.get("response") or {}).get("refusal_reason"):
            return "ok"
        if _refusal_keywords(answer):
            return "refusal_boundary"
        return "refusal_boundary"
    if not facts.get("met"):
        return "answer_generation"
    if not citations:
        return "citation_missing"
    return "ok"


def score(questions_path: Path, gold_path: Path, outputs_path: Path, diagnostic_path: Path) -> int:
    questions = read_jsonl(questions_path)
    gold_by_id = {str(item.get("id")): item for item in read_jsonl(gold_path)}
    run = json.loads(outputs_path.read_text(encoding="utf-8"))
    rows = run.get("results") or []
    diagnostics: list[dict[str, Any]] = []
    for row in rows:
        gold = gold_by_id.get(str(row.get("id"))) or {}
        answer = str((row.get("response") or {}).get("answer") or "")
        facts: dict[str, Any] = {"met": True, "facts_met": 0, "facts_total": 0, "details": []}
        if gold.get("answerable"):
            conclusions = gold.get("required_conclusions") or []
            details = [_fact_match(answer, str(conclusion)) for conclusion in conclusions]
            facts = {
                "met": all(item["matched"] for item in details),
                "facts_met": sum(1 for item in details if item["matched"]),
                "facts_total": len(details),
                "details": details,
            }
        sources = _sources_met(gold, (row.get("response") or {}).get("citations") or [])
        category = _failure_category(row, gold, facts, sources)
        passed = category == "ok"
        diagnostics.append(
            {
                "id": row.get("id"),
                "question_type": gold.get("question_type"),
                "difficulty": gold.get("difficulty"),
                "answerable": gold.get("answerable"),
                "passed": passed,
                "failure_category": category,
                "facts_met": facts.get("facts_met"),
                "facts_total": facts.get("facts_total"),
                "source_evidence_met": sources.get("met"),
                "matched_source_count": sources.get("matched_count"),
                "citation_returned": bool((row.get("response") or {}).get("citations")),
                "latency_ms": row.get("latency_ms"),
            }
        )
    total = len(diagnostics)
    passed_count = sum(1 for item in diagnostics if item["passed"])
    by_type: dict[str, dict[str, Any]] = {}
    by_category: dict[str, int] = {}
    for item in diagnostics:
        kind = str(item["question_type"] or "unknown")
        bucket = by_type.setdefault(kind, {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += int(item["passed"])
        by_category[str(item["failure_category"])] = by_category.get(str(item["failure_category"]), 0) + 1
    artifact = {
        "schema_version": 1,
        "scorer_version": SCORER_VERSION,
        "scorer_sha256": sha256_file(Path(__file__).resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "outputs_sha256": sha256_file(outputs_path),
        "summary": {
            "total": total,
            "passed": passed_count,
            "pass_rate": passed_count / total if total else 0.0,
            "by_question_type": {
                kind: {"passed": bucket["passed"], "total": bucket["total"]}
                for kind, bucket in sorted(by_type.items())
            },
            "failure_categories": by_category,
        },
        "diagnostic": diagnostics,
    }
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(artifact["summary"], ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Banking Workbench deterministic scorer")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score")
    s.add_argument("--questions", type=Path, required=True)
    s.add_argument("--gold", type=Path, required=True)
    s.add_argument("--outputs", type=Path, required=True)
    s.add_argument("--diagnostic", type=Path, required=True)
    s.set_defaults(func=score)
    args = parser.parse_args()
    return args.func(args.questions, args.gold, args.outputs, args.diagnostic)


if __name__ == "__main__":
    raise SystemExit(main())
