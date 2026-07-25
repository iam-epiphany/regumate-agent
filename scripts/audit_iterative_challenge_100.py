"""Audit an Iterative-100 challenge round without importing its builder."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROUND_DIR = ROOT / "data/evaluation/iterative_challenge_100/round_1"
DATABASE = ROOT / "data/evaluation/final_runtime/app.db"
DEFAULT_PRIOR_QUESTIONS = (
    ROOT / "data/evaluation/trust_challenge_100/questions.jsonl",
    ROOT / "data/evaluation/hard_challenge_50/round_1/questions.jsonl",
    ROOT / "data/evaluation/hard_challenge_50/round_2/questions.jsonl",
    ROOT / "data/evaluation/hard_challenge_70/round_1/questions.jsonl",
)
PUBLIC_KEYS = {
    "id",
    "scenario",
    "question_type",
    "difficulty",
    "split",
    "question",
    "answerable",
    "scoring_type",
}
GOLD_ONLY_KEYS = {
    "canonical_answer",
    "equivalent_answers",
    "allowed_expressions",
    "required_conclusions",
    "forbidden_conclusions",
    "required_documents",
    "evidence",
    "calculation",
    "expected_refusal_code",
}
EXPECTED_TYPE_COUNTS = {
    "cross_document": 10,
    "fact": 10,
    "formula": 5,
    "formula_refusal": 4,
    "judgment": 10,
    "mixed_regulation_table": 5,
    "multiple_choice": 10,
    "summary": 15,
    "table_comparison": 10,
    "table_cross_period_calculation": 5,
    "table_refusal": 8,
    "version_refusal": 8,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit an Iterative-100 challenge package.")
    parser.add_argument("--round-dir", type=Path, default=DEFAULT_ROUND_DIR)
    parser.add_argument("--db", type=Path, default=DATABASE)
    parser.add_argument("--prior-question", type=Path, action="append", default=[])
    args = parser.parse_args()

    report = audit(args.round_dir, args.db, tuple(args.prior_question or DEFAULT_PRIOR_QUESTIONS))
    report_path = args.round_dir / "independent_audit.json"
    markdown_path = args.round_dir / "independent_audit.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "errors": len(report["errors"])}, ensure_ascii=False))
    return 0 if report["passed"] else 2


def audit(round_dir: Path, db_path: Path = DATABASE, prior_question_files: tuple[Path, ...] = DEFAULT_PRIOR_QUESTIONS) -> dict[str, Any]:
    questions_path = round_dir / "questions.jsonl"
    gold_path = round_dir / "gold.jsonl"
    questions = _read_jsonl(questions_path)
    gold = _read_jsonl(gold_path)
    errors: list[str] = []
    warnings: list[str] = []

    if len(questions) != 100 or len(gold) != 100:
        errors.append(f"expected 100 public/gold rows, got {len(questions)}/{len(gold)}")
    q_ids = [str(row.get("id")) for row in questions]
    g_ids = [str(row.get("id")) for row in gold]
    if q_ids != g_ids:
        errors.append("IDs/order are not identical in public and gold files")
    if len(set(q_ids)) != len(q_ids):
        errors.append("duplicate case IDs in public questions")
    if q_ids and not all(re.fullmatch(r"I100-R\d{2}-\d{3}", case_id) for case_id in q_ids):
        errors.append("case IDs must follow I100-RNN-NNN")
    if any(set(row) != PUBLIC_KEYS for row in questions):
        errors.append("public question file contains missing or extra fields")
    if any(set(row) & GOLD_ONLY_KEYS for row in questions):
        errors.append("public question file leaks gold-only fields")

    prior_norms = _prior_question_norms(prior_question_files)
    question_norms = [_normalize_question(str(row.get("question") or "")) for row in questions]
    if len(set(question_norms)) != len(question_norms):
        errors.append("duplicate public question text inside iterative round")
    if any(norm in prior_norms for norm in question_norms):
        errors.append("one or more public questions exactly duplicate a prior challenge")

    type_counts = Counter(str(row.get("question_type")) for row in gold)
    if dict(type_counts) != EXPECTED_TYPE_COUNTS:
        errors.append(f"question type distribution mismatch: {dict(type_counts)}")
    answerable = sum(row.get("answerable") is True for row in gold)
    refusal = len(gold) - answerable
    if answerable != 80 or refusal != 20:
        errors.append(f"expected 80 answerable and 20 refusal cases, got {answerable}/{refusal}")
    if any(row.get("difficulty") != "hard" or row.get("split") != "all" for row in gold):
        errors.append("all gold cases must be difficulty=hard and split=all")
    if any(row.get("review_status") != "codex_verified" or row.get("expert_reviewed") is not False for row in gold):
        errors.append("review status must be codex_verified and expert_reviewed=false")

    documents, chunks = _load_corpus(db_path, errors)
    referenced_documents: set[str] = set()
    evidence_chunks: set[str] = set()
    format_counts: Counter[str] = Counter()
    case_formats: defaultdict[str, set[str]] = defaultdict(set)
    refusal_search_checks: dict[str, int] = {}
    for public, case in zip(questions, gold, strict=False):
        case_id = str(case.get("id"))
        for key in PUBLIC_KEYS:
            if public.get(key) != case.get(key):
                errors.append(f"{case_id}: public/gold mismatch for {key}")
        if not case.get("canonical_answer") or not case.get("required_conclusions"):
            errors.append(f"{case_id}: missing canonical answer or required conclusions")
        if case.get("answerable") and not case.get("evidence"):
            errors.append(f"{case_id}: answerable case has no evidence")
        if not case.get("answerable") and str(case.get("question_type")) not in {"version_refusal", "table_refusal", "formula_refusal"}:
            errors.append(f"{case_id}: refusal case has unexpected question_type={case.get('question_type')}")

        for item in case.get("required_documents") or []:
            document_id = str(item.get("document_id") or "")
            document = documents.get(document_id)
            if document is None:
                errors.append(f"{case_id}: missing required document {document_id}")
                continue
            referenced_documents.add(document_id)
            if item.get("filename") and str(item.get("filename")) != str(document["filename"]):
                errors.append(f"{case_id}: filename mismatch for {document_id}")
            if item.get("file_sha256") and str(item.get("file_sha256")) != str(document["file_sha256"]):
                errors.append(f"{case_id}: file SHA mismatch for {document_id}")
            group = _format_group(str(document.get("file_type") or ""), str(document["filename"]))
            format_counts[group] += 1
            case_formats[case_id].add(group)

        for evidence in case.get("evidence") or []:
            chunk_id = str(evidence.get("chunk_id") or "")
            if "-FORMULA-BLOCK-" not in chunk_id:
                chunk = chunks.get(chunk_id)
                if chunk is None:
                    errors.append(f"{case_id}: missing evidence chunk {chunk_id}")
                    continue
                if str(chunk["document_id"]) != str(evidence.get("document_id")):
                    errors.append(f"{case_id}: evidence chunk/document mismatch for {chunk_id}")
            evidence_chunks.add(chunk_id)
            text = str(evidence.get("evidence_text") or "")
            if len(_normalize_question(text)) < 12:
                errors.append(f"{case_id}: evidence text too short for {chunk_id}")

        search = case.get("corpus_search") or {}
        if not case.get("answerable") and search:
            query = str(search.get("query") or "")
            if search.get("match_count") != 0:
                errors.append(f"{case_id}: refusal corpus_search.match_count must be 0")
            refusal_search_checks[case_id] = int(search.get("match_count") or 0)
            if not query:
                warnings.append(f"{case_id}: refusal case has an empty corpus_search query")

    report = {
        "schema_version": "iterative100-independent-audit-v1",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "round_name": round_dir.name,
        "passed": not errors,
        "review_status": "codex_verified",
        "expert_reviewed": False,
        "third_party_blind_test": False,
        "counts": {
            "cases": len(gold),
            "answerable": answerable,
            "refusal": refusal,
            "frozen_documents": len(documents),
            "referenced_documents": len(referenced_documents),
            "evidence_chunks": len(evidence_chunks),
        },
        "coverage": {
            "question_types": dict(sorted(type_counts.items())),
            "formats": dict(sorted(format_counts.items())),
            "cross_document_cases": sum(len(case_formats[str(row.get("id"))]) > 1 or len(row.get("required_documents") or []) > 1 for row in gold),
            "table_cases": sum(str(row.get("question_type")) in {"table_comparison", "table_cross_period_calculation", "mixed_regulation_table", "table_refusal"} for row in gold),
            "formula_cases": sum("formula" in str(row.get("question_type")) for row in gold),
            "refusal_search_checks": refusal_search_checks,
        },
        "policy": {
            "first_run_is_blind_evidence": True,
            "failed_cases_are_regression_only": True,
            "release_requires_next_fresh_round": True,
        },
        "hashes": {
            "questions_sha256": _sha256_file(questions_path),
            "gold_sha256": _sha256_file(gold_path),
            "auditor_sha256": _sha256_file(Path(__file__)),
            "db_sha256": _sha256_file(db_path) if db_path.exists() else None,
        },
        "errors": errors,
        "warnings": warnings,
    }
    return report


def _load_corpus(db_path: Path, errors: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if not db_path.exists():
        errors.append(f"SQLite corpus is missing: {db_path}")
        return {}, {}
    connection = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    documents = {
        str(row["document_id"]): dict(row)
        for row in connection.execute(
            "SELECT document_id, filename, file_type, file_sha256 FROM documents"
        )
    }
    chunks = {
        str(row["chunk_id"]): dict(row)
        for row in connection.execute(
            "SELECT chunk_id, document_id, text FROM document_chunks"
        )
    }
    if len(documents) != 500:
        errors.append(f"frozen corpus must contain 500 documents, got {len(documents)}")
    return documents, chunks


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Iterative-100 {report['round_name']} 独立证据审计",
        "",
        f"- 审计结论：**{'通过' if report['passed'] else '未通过'}**",
        f"- 题目数：{report['counts']['cases']}（可答 {report['counts']['answerable']}，拒答 {report['counts']['refusal']}）",
        f"- 冻结知识库文档数：{report['counts']['frozen_documents']}",
        f"- 引用源文档数：{report['counts']['referenced_documents']}",
        f"- 证据 chunk 数：{report['counts']['evidence_chunks']}",
        f"- 题目文件 SHA-256：`{report['hashes']['questions_sha256']}`",
        f"- 金标文件 SHA-256：`{report['hashes']['gold_sha256']}`",
        "",
        "## 迭代约束",
        "",
        "- 首跑结果作为盲测证据保存。",
        "- 失败题只能用于回归定位，不得作为最终达标证明。",
        "- 达标后仍需下一批全新 100 题验收。",
        "",
    ]
    if report["errors"]:
        lines.extend(["## 阻断问题", ""])
        lines.extend(f"- {item}" for item in report["errors"])
        lines.append("")
    if report["warnings"]:
        lines.extend(["## 提示", ""])
        lines.extend(f"- {item}" for item in report["warnings"])
        lines.append("")
    return "\n".join(lines)


def _format_group(file_type: str, filename: str) -> str:
    suffix = (file_type or Path(filename).suffix.lstrip(".")).lower()
    if suffix in {"doc", "docx"}:
        return "Word"
    if suffix == "pdf":
        return "PDF"
    if suffix in {"xls", "xlsx", "csv"}:
        return "Excel"
    return suffix.upper() or "UNKNOWN"


def _prior_question_norms(paths: tuple[Path, ...]) -> set[str]:
    norms: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for row in _read_jsonl(path):
            question = str(row.get("question") or "")
            if question:
                norms.add(_normalize_question(question))
    return norms


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _normalize_question(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
