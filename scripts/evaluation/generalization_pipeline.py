"""Isolated tooling for ReguMate's generated-question evaluations.

This module deliberately lives outside the production application.  It moves
only public question text through the online runner; answer keys and evidence
remain below a sibling private root.  It is also usable by a separate question
authoring process, which is the intended way to keep optimisation blind.
"""
from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROOT = ROOT.parent / "ReguMate-Eval-Private" / "generalization_100"
SCORER_VERSION = "generalization-scorer-v1"
PRIVATE_FIELDS = {"canonical_answer", "acceptable_answers", "required_conclusions", "forbidden_conclusions", "required_sources", "required_evidence_aspects", "calculation", "refusal_rationale"}


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


def duplicate_reason(item: dict[str, Any], history: list[dict[str, Any]]) -> str | None:
    """Conservative string/semantic/evidence/conclusion/reasoning dedupe."""
    question = str(item.get("question", "")); tokens = token_set(question)
    sources = {str(x.get("relative_path")) for x in item.get("required_sources", []) if isinstance(x, dict)}
    conclusion = normalise(str(item.get("canonical_answer", "")))
    structure = (item.get("question_type"), bool(item.get("answerable")), tuple(sorted(item.get("reasoning_steps", []))))
    for old in history:
        old_question = str(old.get("question", ""))
        if normalise(question) == normalise(old_question): return "string_duplicate"
        old_tokens = token_set(old_question); union = tokens | old_tokens
        characters = set(normalise(question)); old_characters = set(normalise(old_question)); char_union = characters | old_characters
        character_overlap = len(characters & old_characters) / min(len(characters), len(old_characters)) if characters and old_characters else 0
        clean, old_clean = normalise(question), normalise(old_question)
        shared_ngrams = {clean[i:i+4] for i in range(max(0, len(clean)-3))} & {old_clean[i:i+4] for i in range(max(0, len(old_clean)-3))}
        if (union and len(tokens & old_tokens) / len(union) >= 0.82) or (char_union and len(characters & old_characters) / len(char_union) >= 0.72) or character_overlap >= 0.78 or len(shared_ngrams) >= 2: return "semantic_near_duplicate"
        old_sources = {str(x.get("relative_path")) for x in old.get("required_sources", []) if isinstance(x, dict)}
        if sources and sources == old_sources and conclusion and conclusion == normalise(str(old.get("canonical_answer", ""))): return "evidence_conclusion_duplicate"
        if structure == (old.get("question_type"), bool(old.get("answerable")), tuple(sorted(old.get("reasoning_steps", [])))) and old.get("reasoning_steps"): return "reasoning_structure_duplicate"
    return None


def public_projection(item: dict[str, Any]) -> dict[str, Any]:
    forbidden = PRIVATE_FIELDS | {"question", "evidence", "answer"}
    # question is intentionally put back after stripping potential answer fields.
    public = {k: v for k, v in item.items() if k not in forbidden}
    public["id"] = item["id"]
    public["question"] = item["question"]
    public["answerable"] = bool(item["answerable"])
    public["question_type"] = item["question_type"]
    public["difficulty"] = item.get("difficulty", "medium")
    return public


def audit_gold(rows: list[dict[str, Any]], corpus: Path) -> list[dict[str, Any]]:
    """Revalidate candidates without trusting generator claims.

    The audit deliberately checks structural/evidence references and exact
    Decimal calculations.  Text-evidence semantic verification is performed
    by the separate authoring/audit process before candidates enter this tool.
    """
    results = []
    seen_q: set[str] = set()
    for item in rows:
        errors: list[str] = []
        for key in ("id", "question", "answerable", "question_type"):
            if key not in item or item[key] in (None, ""):
                errors.append(f"missing_{key}")
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
    for q in questions:
        response, key = responses.get(q["id"], {}), gold_rows.get(q["id"], {})
        answer = str(response.get("answer", ""))
        if q["answerable"]:
            accepted = [str(key.get("canonical_answer", "")), *map(str, key.get("acceptable_answers", []))]
            ok = any(x and normalise(x) in normalise(answer) for x in accepted)
        else:
            ok = str(response.get("refusal_code", "")) == str(key.get("expected_refusal_code", ""))
        passed += ok
        # This package must never reveal the key, evidence, source, or numbers.
        safe.append({"id": q["id"], "question": q["question"], "passed": ok, "failure_category": None if ok else ("refusal_boundary" if not q["answerable"] else "answer_generation"), "latency_ms": response.get("latency_ms"), "golden_evidence_retrieved": bool(response.get("citations")), "expected_source_count": len(key.get("required_sources", [])), "matched_aspect_count": 0})
    report = {"schema_version": 1, "scorer_version": SCORER_VERSION, "scorer_sha256": sha(Path(__file__)), "total": len(questions), "passed": passed, "pass_rate": passed / len(questions) if questions else 0, "diagnostic": safe}
    diagnostic.parent.mkdir(parents=True, exist_ok=True); diagnostic.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def runner(public_questions: Path, out: Path, base_url: str) -> None:
    rows = []
    for q in load_jsonl(public_questions):
        request = urllib.request.Request(base_url.rstrip("/") + "/api/qa/ask", data=json.dumps({"question": q["question"]}).encode(), headers={"Content-Type": "application/json"})
        started = datetime.now().timestamp()
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode())
        rows.append({"id": q["id"], "answer": payload.get("answer", ""), "citations": payload.get("citations", []), "latency_ms": round((datetime.now().timestamp()-started)*1000, 2)})
    if out.exists(): raise FileExistsError("runner output already exists; first-run outputs are immutable")
    write_jsonl(out, rows)


def main() -> int:
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="command", required=True)
    for name in ("project", "audit"):
        x=sub.add_parser(name); x.add_argument("--input", type=Path, required=True); x.add_argument("--output", type=Path, required=True)
    x=sub.add_parser("freeze"); x.add_argument("--round", required=True); x.add_argument("--public-root", type=Path, required=True); x.add_argument("--private-root", type=Path, required=True); x.add_argument("--corpus", type=Path, required=True)
    x=sub.add_parser("generate"); x.add_argument("--proposals", type=Path, required=True); x.add_argument("--history", type=Path); x.add_argument("--accepted", type=Path, required=True); x.add_argument("--rejected", type=Path, required=True)
    x=sub.add_parser("verify"); x.add_argument("--public-root", type=Path, required=True); x.add_argument("--private-root", type=Path, required=True); x.add_argument("--corpus", type=Path, required=True); x.add_argument("--report", type=Path, required=True)
    x=sub.add_parser("score"); x.add_argument("--questions", type=Path, required=True); x.add_argument("--gold", type=Path, required=True); x.add_argument("--outputs", type=Path, required=True); x.add_argument("--diagnostic", type=Path, required=True)
    x=sub.add_parser("run"); x.add_argument("--questions", type=Path, required=True); x.add_argument("--output", type=Path, required=True); x.add_argument("--base-url", required=True)
    a=p.parse_args()
    if a.command == "project": write_jsonl(a.output, [public_projection(x) for x in load_jsonl(a.input)])
    elif a.command == "audit": a.output.write_text(json.dumps(audit_gold(load_jsonl(a.input), ROOT / "data" / "contest_dataset" / "dataset" / "nfra_page_attachments_500"), ensure_ascii=False, indent=2), encoding="utf-8")
    elif a.command == "freeze": print(freeze(a.round,a.public_root,a.private_root,a.corpus))
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
