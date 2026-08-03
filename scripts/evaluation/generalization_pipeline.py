"""Isolated tooling for ReguMate's generated-question evaluations.

This module deliberately lives outside the production application.  It moves
only public question text through the online runner; answer keys and evidence
remain below a sibling private root.  It is also usable by a separate question
authoring process, which is the intended way to keep optimisation blind.
"""
from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import hashlib
import json
import re
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PRIVATE_ROOT = ROOT.parent / "ReguMate-Eval-Private" / "generalization_100"
SCORER_VERSION = "generalization-scorer-v2"
PRIVATE_FIELDS = {"canonical_answer", "acceptable_answers", "required_conclusions", "forbidden_conclusions", "required_sources", "required_evidence_aspects", "calculation", "refusal_rationale"}

# These quotas define the capability mix of every independent 100-case batch.
# They deliberately describe capabilities, rather than documents, facts, or
# answers, so that they cannot become an evaluation-targeted answer map.
ANSWERABLE_TYPE_QUOTAS = {
    "fact_definition": 20,
    "rule_scope": 12,
    "single_document_synthesis": 10,
    "cross_document_evidence": 8,
    "table_lookup": 12,
    "table_calculation": 10,
    "formula_calculation": 4,
    "policy_reporting_joint": 4,
}
DIFFICULTY_QUOTAS = {"easy": 20, "medium": 55, "hard": 25}
REFUSAL_COUNT = 20
PUBLIC_FORBIDDEN_FIELDS = PRIVATE_FIELDS | {
    "evidence", "answer", "source", "sources", "sheet", "cell",
    "row", "column", "formula", "reasoning_steps", "expected_refusal_code",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows), encoding="utf-8")


def normalise(text: str) -> str:
    return re.sub(r"\W+", "", text.casefold())


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[\u4e00-\u9fff]{1,4}|[a-z0-9]+", text.casefold()))


def question_text_errors(text: str) -> list[str]:
    """Return encoding/integrity errors that make a question unauditable.

    A single ASCII question mark is valid in English text.  Repeated literal
    question marks, however, are the characteristic irreversible replacement
    produced when Unicode text is sent through a mismatched Windows code page.
    Chinese full-width punctuation (``？``) is unaffected by this check.
    """

    value = str(text)
    errors: list[str] = []
    if "\ufffd" in value:
        errors.append("question_contains_unicode_replacement_character")
    if any(ord(char) < 32 and char not in "\t\r\n" for char in value):
        errors.append("question_contains_control_character")
    ascii_question_marks = value.count("?")
    visible_length = len(re.sub(r"\s+", "", value))
    if ascii_question_marks >= 3 or (
        ascii_question_marks >= 2
        and visible_length
        and Decimal(ascii_question_marks) / Decimal(visible_length) >= Decimal("0.08")
    ):
        errors.append("question_suspected_codepage_replacement")
    if not normalise(value):
        errors.append("question_has_no_searchable_content")
    return errors


def duplicate_reason(item: dict[str, Any], history: list[dict[str, Any]]) -> str | None:
    """Conservative string/semantic/evidence/conclusion/reasoning dedupe."""
    question = str(item.get("question", "")); tokens = token_set(question)
    sources = {str(x.get("relative_path")) for x in item.get("required_sources", []) if isinstance(x, dict)}
    conclusion = normalise(str(item.get("canonical_answer", "")))
    structure = (item.get("question_type"), bool(item.get("answerable")), tuple(sorted(item.get("reasoning_steps", []))))
    question_core = normalise(question)
    for phrase in ("请说明", "请给出", "请问", "是什么", "是如何", "如何", "什么", "的"):
        question_core = question_core.replace(normalise(phrase), "")
    for old in history:
        old_question = str(old.get("question", ""))
        if normalise(question) == normalise(old_question): return "string_duplicate"
        old_sources = {str(x.get("relative_path")) for x in old.get("required_sources", []) if isinstance(x, dict)}
        old_tokens = token_set(old_question); union = tokens | old_tokens
        clean, old_clean = normalise(question), normalise(old_question)
        token_overlap = len(tokens & old_tokens) / len(union) if union else 0
        sequence_similarity = SequenceMatcher(None, clean, old_clean).ratio()
        old_core = old_clean
        for phrase in ("请说明", "请给出", "请问", "是什么", "是如何", "如何", "什么", "的"):
            old_core = old_core.replace(normalise(phrase), "")
        same_evidence_domain = bool(sources & old_sources) or (not sources and not old_sources)
        if (
            same_evidence_domain
            and (
                (question_core and question_core == old_core)
                or sequence_similarity >= 0.98
                or (len(tokens & old_tokens) >= 5 and token_overlap >= 0.90)
            )
        ):
            return "semantic_near_duplicate"
        if sources and sources == old_sources and conclusion and conclusion == normalise(str(old.get("canonical_answer", ""))): return "evidence_conclusion_duplicate"
        if structure == (old.get("question_type"), bool(old.get("answerable")), tuple(sorted(old.get("reasoning_steps", [])))) and old.get("reasoning_steps"): return "reasoning_structure_duplicate"
    return None


def public_projection(item: dict[str, Any]) -> dict[str, Any]:
    forbidden = PUBLIC_FORBIDDEN_FIELDS | {"question"}
    # question is intentionally put back after stripping potential answer fields.
    public = {k: v for k, v in item.items() if k not in forbidden}
    public["id"] = item["id"]
    public["question"] = item["question"]
    public["answerable"] = bool(item["answerable"])
    public["question_type"] = item["question_type"]
    public["difficulty"] = item.get("difficulty", "medium")
    return public


def validate_batch(rows: list[dict[str, Any]], public_rows: list[dict[str, Any]] | None = None) -> list[str]:
    """Validate a completed authoring batch without exposing its answer key.

    This is intentionally stricter than per-row evidence auditing: a valid
    individual question is not sufficient if the batch has missing capability
    coverage, a skewed difficulty mix, or public answer-key leakage.
    """

    errors: list[str] = []
    if len(rows) != 100:
        errors.append("batch_must_contain_100_cases")
    ids = [str(row.get("id", "")) for row in rows]
    if len(ids) != len(set(ids)) or any(not item for item in ids):
        errors.append("duplicate_or_missing_case_id")
    if any(question_text_errors(str(row.get("question", ""))) for row in rows):
        errors.append("question_text_integrity_failed")
    accepted_for_dedupe: list[dict[str, Any]] = []
    for row in rows:
        if duplicate_reason(row, accepted_for_dedupe):
            errors.append("semantic_or_evidence_duplicate")
            break
        accepted_for_dedupe.append(row)

    answerable = [row for row in rows if bool(row.get("answerable"))]
    refusals = [row for row in rows if not bool(row.get("answerable"))]
    if len(answerable) != sum(ANSWERABLE_TYPE_QUOTAS.values()):
        errors.append("answerable_quota_mismatch")
    if len(refusals) != REFUSAL_COUNT:
        errors.append("refusal_quota_mismatch")
    type_counts = {kind: sum(row.get("question_type") == kind for row in answerable) for kind in ANSWERABLE_TYPE_QUOTAS}
    if type_counts != ANSWERABLE_TYPE_QUOTAS:
        errors.append("answerable_type_quota_mismatch")
    difficulty_counts = {level: sum(row.get("difficulty") == level for row in rows) for level in DIFFICULTY_QUOTAS}
    if difficulty_counts != DIFFICULTY_QUOTAS:
        errors.append("difficulty_quota_mismatch")

    source_documents = {
        str(source.get("relative_path"))
        for row in answerable for source in (row.get("required_sources") or [])
        if isinstance(source, dict) and source.get("relative_path")
    }
    if len(source_documents) < 40:
        errors.append("insufficient_source_document_coverage")
    if public_rows is not None:
        if len(public_rows) != len(rows) or [row.get("id") for row in public_rows] != ids:
            errors.append("public_private_identity_mismatch")
        elif any(
            str(public.get("question", "")) != str(private.get("question", ""))
            for private, public in zip(rows, public_rows)
        ):
            errors.append("public_private_question_text_mismatch")
        for row in public_rows:
            leaked = PUBLIC_FORBIDDEN_FIELDS & set(row)
            if leaked:
                errors.append("public_projection_contains_private_fields")
                break
            if set(row) - {"id", "question", "answerable", "question_type", "difficulty", "business_relevance"}:
                errors.append("public_projection_contains_unapproved_fields")
                break
        if any(question_text_errors(str(row.get("question", ""))) for row in public_rows):
            errors.append("public_question_text_integrity_failed")
    return errors


def audit_gold(rows: list[dict[str, Any]], corpus: Path) -> list[dict[str, Any]]:
    """Revalidate candidates without trusting generator claims.

    The audit deliberately checks structural/evidence references and exact
    Decimal calculations.  Text-evidence semantic verification is performed
    by the separate authoring/audit process before candidates enter this tool.
    """
    results = []
    seen_q: set[str] = set()
    source_text_cache: dict[Path, str] = {}
    for item in rows:
        errors: list[str] = []
        for key in ("id", "question", "answerable", "question_type"):
            if key not in item or item[key] in (None, ""):
                errors.append(f"missing_{key}")
        errors.extend(question_text_errors(str(item.get("question", ""))))
        q = normalise(str(item.get("question", "")))
        if not q or q in seen_q:
            errors.append("duplicate_question")
        seen_q.add(q)
        if item.get("answerable"):
            if not item.get("canonical_answer"):
                errors.append("missing_canonical_answer")
            sources = item.get("required_sources")
            if not isinstance(sources, list) or not sources:
                errors.append("missing_required_sources")
            for source in sources or []:
                rel = source.get("relative_path") if isinstance(source, dict) else None
                if not rel or not (corpus / rel).is_file():
                    errors.append("source_not_in_corpus")
                # A table key must be independently reproducible from the
                # original workbook, not merely point at an existing file.
                elif all(key in source for key in ("sheet", "row", "column")):
                    try:
                        actual = pd.read_excel(corpus / rel, sheet_name=source.get("sheet_raw", source["sheet"]), header=None, dtype=object).iat[int(source["row"]) - 1, int(source["column"]) - 1]
                        expected_value = source.get("expected_value", item.get("canonical_answer", ""))
                        try:
                            equal = Decimal(str(actual)) == Decimal(str(expected_value))
                        except InvalidOperation:
                            equal = normalise(str(actual)) == normalise(str(expected_value))
                        if not equal:
                            errors.append("table_cell_value_mismatch")
                    except Exception:
                        errors.append("table_cell_unreadable")
                elif source.get("evidence_text"):
                    try:
                        source_path = corpus / rel
                        if source_path not in source_text_cache:
                            from backend.app.services.document_parser import parse_document_text

                            source_text_cache[source_path] = parse_document_text(source_path)
                        if normalise(str(source["evidence_text"])) not in normalise(source_text_cache[source_path]):
                            errors.append("text_evidence_not_in_source")
                    except Exception:
                        errors.append("text_evidence_unreadable")
        elif not item.get("expected_refusal_code") or not item.get("refusal_rationale"):
            errors.append("invalid_refusal_gold")
        calc = item.get("calculation")
        if isinstance(calc, dict):
            try:
                left, right = Decimal(str(calc["left"])), Decimal(str(calc["right"]))
                op = calc["operator"]
                actual = {"+": left + right, "-": left - right, "*": left * right, "/": left / right}[op]
                if actual != Decimal(str(calc["result"])):
                    errors.append("calculation_mismatch")
            except (KeyError, InvalidOperation, ZeroDivisionError):
                errors.append("invalid_calculation")
        results.append({"id": item.get("id"), "passed": not errors, "errors": errors})
    return results


def freeze(round_name: str, public_root: Path, private_root: Path, corpus: Path) -> Path:
    qpath, gpath = public_root / "public" / "questions.jsonl", private_root / "gold.jsonl"
    rows = load_jsonl(gpath)
    audit = audit_gold(rows, corpus)
    if not all(x["passed"] for x in audit):
        raise ValueError("audit failed; replace invalid cases before freezing")
    public_rows = load_jsonl(qpath)
    if len(rows) != len(public_rows) or [x["id"] for x in rows] != [x["id"] for x in public_rows]:
        raise ValueError("public/private question identity mismatch")
    batch_errors = validate_batch(rows, public_rows)
    if batch_errors:
        raise ValueError(f"batch audit failed: {', '.join(batch_errors)}")
    source = Path(__file__)
    lock = {"schema_version": 1, "round": round_name, "frozen_at": datetime.now(timezone.utc).isoformat(), "scorer_version": SCORER_VERSION,
            "hashes": {"questions_sha256": sha(qpath), "gold_sha256": sha(gpath), "scorer_sha256": sha(source), "corpus_snapshot_sha256": corpus_manifest_hash(corpus)},
            "artifacts": [{"path": "public/questions.jsonl", "sha256": sha(qpath)}], "private_artifacts": ["gold.jsonl"], "first_run_immutable": True}
    path = public_root / "lock.json"; path.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def verify_round(public_root: Path, private_root: Path, corpus: Path) -> dict[str, Any]:
    lock_path = public_root / "lock.json"; lock = json.loads(lock_path.read_text(encoding="utf-8"))
    qpath, gpath = public_root / "public" / "questions.jsonl", private_root / "gold.jsonl"
    expected = lock.get("hashes", {})
    checks = {"questions": qpath.is_file() and sha(qpath) == expected.get("questions_sha256"), "gold": gpath.is_file() and sha(gpath) == expected.get("gold_sha256"), "scorer": sha(Path(__file__)) == expected.get("scorer_sha256"), "corpus": corpus_manifest_hash(corpus) == expected.get("corpus_snapshot_sha256")}
    return {"passed": all(checks.values()), "checks": checks, "scorer_version": lock.get("scorer_version")}


def corpus_manifest_hash(corpus: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(x for x in corpus.rglob("*") if x.is_file()):
        h.update(f"{p.relative_to(corpus).as_posix()}\t{sha(p)}\n".encode())
    return h.hexdigest()


def score(public_questions: Path, gold: Path, outputs: Path, diagnostic: Path) -> dict[str, Any]:
    questions, gold_rows = load_jsonl(public_questions), {x["id"]: x for x in load_jsonl(gold)}
    responses = {x["id"]: x for x in load_jsonl(outputs)}
    safe: list[dict[str, Any]] = []; passed = 0
    by_type: dict[str, dict[str, int]] = {}
    for q in questions:
        response, key = responses.get(q["id"], {}), gold_rows.get(q["id"], {})
        answer = str(response.get("answer", ""))
        if q["answerable"]:
            accepted = [str(key.get("canonical_answer", "")), *map(str, key.get("acceptable_answers", []))]
            required_conclusions = [
                str(value)
                for value in key.get("required_conclusions", [])
                if str(value).strip()
            ]
            exact_or_accepted = any(x and normalise(x) in normalise(answer) for x in accepted)
            conclusions_met = bool(required_conclusions) and all(
                normalise(conclusion) in normalise(answer)
                for conclusion in required_conclusions
            )
            ok = exact_or_accepted or conclusions_met
        else:
            ok = str(response.get("refusal_code", "")) == str(key.get("expected_refusal_code", ""))
        passed += ok
        question_type = str(q.get("question_type", "unknown"))
        bucket = by_type.setdefault(question_type, {"total": 0, "passed": 0})
        bucket["total"] += 1
        bucket["passed"] += int(ok)
        if ok:
            category = None
        elif not q["answerable"]:
            category = "refusal_boundary"
        elif not response.get("citations"):
            category = "retrieval_or_citation"
        else:
            category = "answer_generation"
        # This package must never reveal the key, evidence, source, or numbers.
        safe.append({"id": q["id"], "question": q["question"], "answerable": bool(q["answerable"]), "question_type": question_type, "difficulty": q.get("difficulty"), "passed": ok, "failure_category": category, "latency_ms": response.get("latency_ms"), "citation_returned": bool(response.get("citations")), "expected_source_count": len(key.get("required_sources", [])), "matched_aspect_count": 0})
    report = {"schema_version": 1, "scorer_version": SCORER_VERSION, "scorer_sha256": sha(Path(__file__)), "total": len(questions), "passed": passed, "pass_rate": passed / len(questions) if questions else 0, "by_question_type": by_type, "diagnostic": safe}
    diagnostic.parent.mkdir(parents=True, exist_ok=True); diagnostic.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def runner(public_questions: Path, out: Path, base_url: str) -> None:
    rows = []
    for q in load_jsonl(public_questions):
        request = urllib.request.Request(base_url.rstrip("/") + "/api/qa/ask", data=json.dumps({"question": q["question"]}).encode(), headers={"Content-Type": "application/json"})
        started = datetime.now().timestamp()
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode())
        rows.append({"id": q["id"], "answer": payload.get("answer", ""), "citations": payload.get("citations", []),
                     "refused": bool(payload.get("refused", False)), "refusal_code": payload.get("refusal_code"),
                     "latency_ms": round((datetime.now().timestamp()-started)*1000, 2)})
    if out.exists(): raise FileExistsError("runner output already exists; first-run outputs are immutable")
    write_jsonl(out, rows)


def main() -> int:
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="command", required=True)
    for name in ("project", "audit"):
        x=sub.add_parser(name); x.add_argument("--input", type=Path, required=True); x.add_argument("--output", type=Path, required=True)
    x=sub.add_parser("freeze"); x.add_argument("--round", required=True); x.add_argument("--public-root", type=Path, required=True); x.add_argument("--private-root", type=Path, required=True); x.add_argument("--corpus", type=Path, required=True)
    x=sub.add_parser("validate-batch"); x.add_argument("--gold", type=Path, required=True); x.add_argument("--public-questions", type=Path, required=True); x.add_argument("--report", type=Path, required=True)
    x=sub.add_parser("generate"); x.add_argument("--proposals", type=Path, required=True); x.add_argument("--history", type=Path); x.add_argument("--accepted", type=Path, required=True); x.add_argument("--rejected", type=Path, required=True)
    x=sub.add_parser("verify"); x.add_argument("--public-root", type=Path, required=True); x.add_argument("--private-root", type=Path, required=True); x.add_argument("--corpus", type=Path, required=True); x.add_argument("--report", type=Path, required=True)
    x=sub.add_parser("score"); x.add_argument("--questions", type=Path, required=True); x.add_argument("--gold", type=Path, required=True); x.add_argument("--outputs", type=Path, required=True); x.add_argument("--diagnostic", type=Path, required=True)
    x=sub.add_parser("run"); x.add_argument("--questions", type=Path, required=True); x.add_argument("--output", type=Path, required=True); x.add_argument("--base-url", required=True)
    a=p.parse_args()
    if a.command == "project": write_jsonl(a.output, [public_projection(x) for x in load_jsonl(a.input)])
    elif a.command == "audit": a.output.write_text(json.dumps(audit_gold(load_jsonl(a.input), ROOT / "data" / "contest_dataset" / "dataset" / "nfra_page_attachments_500"), ensure_ascii=False, indent=2), encoding="utf-8")
    elif a.command == "freeze": print(freeze(a.round,a.public_root,a.private_root,a.corpus))
    elif a.command == "validate-batch":
        errors = validate_batch(load_jsonl(a.gold), load_jsonl(a.public_questions))
        payload = {"passed": not errors, "errors": errors}
        a.report.parent.mkdir(parents=True, exist_ok=True); a.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False)); return 0 if not errors else 2
    elif a.command == "generate":
        history = load_jsonl(a.history) if a.history and a.history.is_file() else []
        accepted=[]; rejected=[]
        for item in load_jsonl(a.proposals):
            reason=duplicate_reason(item, history + accepted)
            (rejected if reason else accepted).append({"id":item.get("id"),"reason":reason} if reason else item)
        write_jsonl(a.accepted, accepted); a.rejected.parent.mkdir(parents=True, exist_ok=True); a.rejected.write_text(json.dumps(rejected,ensure_ascii=False,indent=2),encoding="utf-8")
    elif a.command == "verify":
        report=verify_round(a.public_root,a.private_root,a.corpus); a.report.parent.mkdir(parents=True,exist_ok=True); a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(report)); return 0 if report["passed"] else 2
    elif a.command == "score": print(json.dumps(score(a.questions,a.gold,a.outputs,a.diagnostic),ensure_ascii=False))
    else: runner(a.questions,a.output,a.base_url)
    return 0

if __name__ == "__main__": raise SystemExit(main())
