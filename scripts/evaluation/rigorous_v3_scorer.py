"""Isolated scorer and freezer for the rigorous v3 100-case evaluation.

Production code never imports this module.  It evaluates answer facts with
bounded semantic overlap plus critical-entity checks instead of requiring a
verbatim copy of an entire evidence sentence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCORER_VERSION = "rigorous-generalization-scorer-v3"
ANSWERABLE_TYPE_QUOTAS = {
    "fact_definition": 15,
    "rule_scope": 12,
    "single_document_synthesis": 8,
    "cross_document_evidence": 10,
    "table_lookup": 15,
    "table_calculation": 10,
    "formula_calculation": 5,
    "policy_reporting_joint": 5,
}
DIFFICULTY_QUOTAS = {"easy": 20, "medium": 55, "hard": 25}
PRIVATE_FIELDS = {
    "canonical_answer",
    "acceptable_answers",
    "required_conclusions",
    "required_sources",
    "required_evidence_aspects",
    "calculation",
    "expected_refusal_code",
    "refusal_rationale",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corpus_hash(corpus: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in corpus.rglob("*") if item.is_file()):
        digest.update(
            f"{path.relative_to(corpus).as_posix()}\t{sha256(path)}\n".encode("utf-8")
        )
    return digest.hexdigest()


def normalise(text: str) -> str:
    return re.sub(r"\W+", "", str(text or "").casefold())


def _bigrams(text: str) -> set[str]:
    value = normalise(text)
    if len(value) < 2:
        return {value} if value else set()
    return {value[index : index + 2] for index in range(len(value) - 1)}


def _bigram_recall(answer: str, conclusion: str) -> float:
    expected = _bigrams(conclusion)
    if not expected:
        return 0.0
    return len(expected & _bigrams(answer)) / len(expected)


def _critical_entities(text: str) -> list[str]:
    entities = re.findall(
        r"(?:\d+(?:\.\d+)?|[一二两三四五六七八九十百]+)\s*"
        r"(?:%|％|个工作日|工作日|日|个月|月|年|倍|万元|亿元|元|项|个)",
        str(text or ""),
    )
    document_terms = re.findall(r"《([^》]{2,80})》", str(text or ""))
    return list(dict.fromkeys([*entities, *document_terms]))


def fact_match(answer: str, conclusion: str) -> dict[str, Any]:
    answer_norm = normalise(answer)
    conclusion_norm = normalise(conclusion)
    exact = bool(conclusion_norm) and conclusion_norm in answer_norm
    recall = _bigram_recall(answer, conclusion)
    entities = _critical_entities(conclusion)
    missing_entities = [
        entity for entity in entities if normalise(entity) not in answer_norm
    ]
    polarity_ok = True
    if any(term in conclusion for term in ("不得", "禁止", "不应")):
        polarity_ok = any(term in answer for term in ("不得", "禁止", "不应", "不能"))
    if "至少" in conclusion:
        polarity_ok = polarity_ok and any(term in answer for term in ("至少", "不低于"))
    matched = exact or (
        recall >= 0.55
        and not missing_entities
        and polarity_ok
    )
    return {
        "matched": matched,
        "bigram_recall": round(recall, 4),
        "critical_entities_ok": not missing_entities,
        "polarity_ok": polarity_ok,
    }


def _source_key(value: str) -> str:
    stem = Path(str(value or "")).stem
    stem = re.sub(r"^\d+_", "", stem)
    return normalise(stem)


def _citation_sources(citations: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for citation in citations:
        for key in ("filename", "source_title"):
            value = str(citation.get(key) or "").strip()
            if value:
                values.append(_source_key(value))
        metadata = citation.get("metadata")
        if isinstance(metadata, dict):
            for cell in metadata.get("calculation_cells") or []:
                if isinstance(cell, dict):
                    value = str(cell.get("filename") or cell.get("source_title") or "")
                    if value:
                        values.append(_source_key(value))
    return list(dict.fromkeys(value for value in values if value))


def _sources_met(
    required_sources: list[dict[str, Any]],
    citations: list[dict[str, Any]],
) -> tuple[bool, int]:
    expected = list(
        dict.fromkeys(
            _source_key(str(source.get("relative_path") or ""))
            for source in required_sources
            if source.get("relative_path")
        )
    )
    actual = _citation_sources(citations)
    matched = 0
    for source in expected:
        if any(source in candidate or candidate in source for candidate in actual):
            matched += 1
    return matched == len(expected), matched


def validate_batch(
    public_questions: list[dict[str, Any]],
    gold: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    if len(public_questions) != 100 or len(gold) != 100:
        errors.append(
            f"batch_size_mismatch: public={len(public_questions)} gold={len(gold)}"
        )
    public_ids = [str(row.get("id")) for row in public_questions]
    gold_ids = [str(row.get("id")) for row in gold]
    if public_ids != gold_ids:
        errors.append("public_private_id_order_mismatch")
    if len(set(public_ids)) != len(public_ids):
        errors.append("duplicate_ids")
    for row in public_questions:
        leaked = PRIVATE_FIELDS & set(row)
        if leaked:
            errors.append(f"{row.get('id')}:public_private_field_leak:{sorted(leaked)}")
        question = str(row.get("question") or "")
        if "\ufffd" in question or question.count("?") >= 3:
            errors.append(f"{row.get('id')}:question_encoding_error")
    answerable = [row for row in public_questions if row.get("answerable")]
    refusal = [row for row in public_questions if not row.get("answerable")]
    if len(answerable) != 80 or len(refusal) != 20:
        errors.append(
            f"answerable_refusal_quota_mismatch:{len(answerable)}/{len(refusal)}"
        )
    type_counts = Counter(str(row.get("question_type")) for row in answerable)
    if dict(type_counts) != ANSWERABLE_TYPE_QUOTAS:
        errors.append(f"type_quota_mismatch:{dict(type_counts)}")
    difficulty_counts = Counter(str(row.get("difficulty")) for row in public_questions)
    if dict(difficulty_counts) != DIFFICULTY_QUOTAS:
        errors.append(f"difficulty_quota_mismatch:{dict(difficulty_counts)}")
    questions_normalised = [normalise(str(row.get("question") or "")) for row in public_questions]
    if len(set(questions_normalised)) != len(questions_normalised):
        errors.append("duplicate_normalised_questions")
    for row in gold:
        if row.get("answerable"):
            conclusions = [
                str(value)
                for value in row.get("required_conclusions") or []
                if str(value).strip()
            ]
            if not conclusions:
                errors.append(f"{row.get('id')}:missing_required_conclusions")
            if not row.get("required_sources"):
                errors.append(f"{row.get('id')}:missing_required_sources")
        elif not row.get("expected_refusal_code"):
            errors.append(f"{row.get('id')}:missing_expected_refusal_code")
    return errors


def score(
    questions_path: Path,
    gold_path: Path,
    outputs_path: Path,
    diagnostic_path: Path,
) -> dict[str, Any]:
    questions = load_jsonl(questions_path)
    gold = {str(row["id"]): row for row in load_jsonl(gold_path)}
    outputs = {str(row["id"]): row for row in load_jsonl(outputs_path)}
    diagnostic: list[dict[str, Any]] = []
    by_type: dict[str, dict[str, int]] = {}
    passed = 0
    source_expected = 0
    source_matched = 0
    for question in questions:
        case_id = str(question["id"])
        key = gold.get(case_id, {})
        response = outputs.get(case_id, {})
        answer = str(response.get("answer") or "")
        citations = response.get("citations") or []
        if question.get("answerable"):
            conclusions = [
                str(value)
                for value in key.get("required_conclusions") or []
                if str(value).strip()
            ]
            matches = [fact_match(answer, conclusion) for conclusion in conclusions]
            facts_ok = bool(matches) and all(item["matched"] for item in matches)
            sources_ok, matched_sources = _sources_met(
                list(key.get("required_sources") or []),
                citations,
            )
            source_expected += len(
                {
                    _source_key(str(source.get("relative_path") or ""))
                    for source in key.get("required_sources") or []
                    if source.get("relative_path")
                }
            )
            source_matched += matched_sources
            ok = facts_ok and sources_ok and not response.get("error")
            category = (
                None
                if ok
                else "runtime"
                if response.get("error")
                else "retrieval_or_citation"
                if not sources_ok
                else "answer_generation"
            )
        else:
            facts_ok = True
            sources_ok = True
            matched_sources = 0
            ok = (
                str(response.get("refusal_code") or "")
                == str(key.get("expected_refusal_code") or "")
                and not response.get("error")
            )
            category = None if ok else "refusal_boundary"
            matches = []
        passed += int(ok)
        question_type = str(question.get("question_type") or "unknown")
        bucket = by_type.setdefault(question_type, {"total": 0, "passed": 0})
        bucket["total"] += 1
        bucket["passed"] += int(ok)
        diagnostic.append(
            {
                "id": case_id,
                "question_type": question_type,
                "difficulty": question.get("difficulty"),
                "answerable": bool(question.get("answerable")),
                "passed": ok,
                "failure_category": category,
                "facts_met": facts_ok,
                "source_evidence_met": sources_ok,
                "matched_source_count": matched_sources,
                "fact_match_count": sum(item["matched"] for item in matches),
                "required_fact_count": len(matches),
                "citation_returned": bool(citations),
                "latency_ms": response.get("latency_ms"),
            }
        )
    report = {
        "schema_version": 1,
        "scorer_version": SCORER_VERSION,
        "scorer_sha256": sha256(Path(__file__)),
        "total": len(questions),
        "passed": passed,
        "pass_rate": passed / len(questions) if questions else 0.0,
        "source_evidence_hit_rate": (
            source_matched / source_expected if source_expected else 1.0
        ),
        "by_question_type": by_type,
        "diagnostic": diagnostic,
    }
    write_json(diagnostic_path, report)
    return report


def freeze(
    public_root: Path,
    private_root: Path,
    corpus: Path,
) -> dict[str, Any]:
    lock_path = public_root / "lock.json"
    if lock_path.exists():
        raise FileExistsError("v3 round is already frozen")
    questions_path = public_root / "public" / "questions.jsonl"
    gold_path = private_root / "gold.jsonl"
    errors = validate_batch(load_jsonl(questions_path), load_jsonl(gold_path))
    if errors:
        raise ValueError(f"batch validation failed: {errors}")
    lock = {
        "schema_version": 1,
        "round": public_root.name,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "scorer_version": SCORER_VERSION,
        "hashes": {
            "questions_sha256": sha256(questions_path),
            "gold_sha256": sha256(gold_path),
            "scorer_sha256": sha256(Path(__file__)),
            "corpus_snapshot_sha256": corpus_hash(corpus),
        },
        "first_run_immutable": True,
    }
    write_json(lock_path, lock)
    return lock


def verify(public_root: Path, private_root: Path, corpus: Path) -> dict[str, Any]:
    lock = json.loads((public_root / "lock.json").read_text(encoding="utf-8"))
    expected = lock["hashes"]
    checks = {
        "questions": sha256(public_root / "public" / "questions.jsonl")
        == expected["questions_sha256"],
        "gold": sha256(private_root / "gold.jsonl") == expected["gold_sha256"],
        "scorer": sha256(Path(__file__)) == expected["scorer_sha256"],
        "corpus": corpus_hash(corpus) == expected["corpus_snapshot_sha256"],
    }
    return {"passed": all(checks.values()), "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("freeze", "verify"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--public-root", type=Path, required=True)
        sub.add_argument("--private-root", type=Path, required=True)
        sub.add_argument("--corpus", type=Path, required=True)
    sub = subparsers.add_parser("score")
    sub.add_argument("--questions", type=Path, required=True)
    sub.add_argument("--gold", type=Path, required=True)
    sub.add_argument("--outputs", type=Path, required=True)
    sub.add_argument("--diagnostic", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        print(json.dumps(freeze(args.public_root, args.private_root, args.corpus)))
    elif args.command == "verify":
        report = verify(args.public_root, args.private_root, args.corpus)
        print(json.dumps(report))
        return 0 if report["passed"] else 2
    else:
        print(
            json.dumps(
                score(args.questions, args.gold, args.outputs, args.diagnostic),
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
