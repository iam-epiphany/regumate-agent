"""Offline, gold-isolated deterministic scorer for the ReguMate trust challenge."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sqlite3
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import median
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_SCENARIOS = {"表格精确取数", "表格比较与计算"}
REFUSAL_TYPES = {"refusal", "formula_refusal", "version_uncertainty"}
HARD_PROFILES = {"hard50", "hard70", "iterative100"}
HARD_PROFILE_EXPECTED_CASES = {"hard50": 50, "hard70": 70, "iterative100": 100}
HARD_PROFILE_EXPECTED_REFUSALS = {"hard50": None, "hard70": 20, "iterative100": 20}


def main() -> int:
    parser = argparse.ArgumentParser(description="Score raw trust-challenge HTTP results against offline gold.")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data/evaluation/final_runtime/app.db")
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--split", choices=["dev", "holdout", "all"], default="all")
    parser.add_argument("--profile", choices=["trust100", "hard50", "hard70", "iterative100"], default="trust100")
    parser.add_argument("--case-ids", help="Comma-separated case IDs for a bounded regression report.")
    args = parser.parse_args()

    gold_all = read_jsonl(args.gold)
    questions_all = read_jsonl(args.questions)
    gold = [row for row in gold_all if args.split == "all" or row.get("split") == args.split]
    questions = [row for row in questions_all if args.split == "all" or row.get("split") == args.split]
    requested_case_ids = _case_id_filter(args.case_ids)
    if requested_case_ids:
        gold = [row for row in gold if str(row.get("id")) in requested_case_ids]
        questions = [row for row in questions if str(row.get("id")) in requested_case_ids]
        missing = requested_case_ids - {str(row.get("id")) for row in gold}
        if missing:
            raise ValueError(f"Unknown or out-of-split case IDs: {sorted(missing)}")
    result_payload = json.loads(args.results.read_text(encoding="utf-8"))
    results = {str(row.get("case_id")): row for row in result_payload.get("results") or []}
    lock = json.loads(args.lock.read_text(encoding="utf-8")) if args.lock and args.lock.exists() else None
    report = evaluate(gold, questions, results, args.db, args.split, lock, args.gold, args.questions, args.results, args.profile)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0 if report["gate_passed"] else 2


def evaluate(
    gold: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    db_path: Path,
    split: str,
    lock: dict[str, Any] | None,
    gold_path: Path,
    questions_path: Path,
    results_path: Path,
    profile: str = "trust100",
) -> dict[str, Any]:
    structural_errors = _structural_errors(gold, questions, results, split, lock, gold_path, questions_path, profile)
    valid_chunks, chunk_documents = _chunk_index(db_path)
    scored = [_score_case(case, results.get(str(case["id"])), valid_chunks, chunk_documents, profile) for case in gold]
    summary = _summarize(scored, structural_errors, split, lock, profile)
    return {
        "gate_passed": summary["gate_passed"],
        "release_eligible": split == "all" and summary["gate_passed"],
        "split": split,
        "profile": profile,
        "limitations": [
            "未经过银行监管专家人工复核。",
            _profile_limitation(profile),
            "评分器不读取或采用系统返回的 grounding.passed 作为答案正确性结论。",
        ],
        "artifacts": {
            "questions_sha256": _sha256_file(questions_path),
            "gold_sha256": _sha256_file(gold_path),
            "results_sha256": _sha256_file(results_path),
            "evaluator_sha256": _sha256_file(Path(__file__)),
        },
        "summary": summary,
        "cases": scored,
    }


def _profile_limitation(profile: str) -> str:
    if profile == "hard50":
        return "50题为 Codex 独立构建并证据复核的困难回归集，不是第三方独立盲测。"
    if profile == "hard70":
        return "70题为 Codex 独立构建并证据复核的困难迭代集，其中20题为拒答题；不是第三方独立盲测。"
    if profile == "iterative100":
        return "100题为 Codex 独立构建并证据复核的迭代盲测集，其中20题为拒答题；失败题只能作为回归子集，达标后仍需下一批全新100题验收。"
    return "60题为封存回归集，不是第三方独立盲测。"


def _case_id_filter(value: str | None) -> set[str]:
    return {item.strip() for item in str(value or "").split(",") if item.strip()}


def _score_case(
    case: dict[str, Any],
    result_record: dict[str, Any] | None,
    valid_chunks: set[str],
    chunk_documents: dict[str, str],
    profile: str = "trust100",
) -> dict[str, Any]:
    response = (result_record or {}).get("response") or {}
    answer = str(response.get("answer") or "")
    refused = bool(response.get("refused"))
    refusal_code = str(
        response.get("refusal_code")
        or response.get("refusal_reason")
        or (response.get("grounding_validation") or {}).get("reason")
        or ""
    )
    citations = response.get("citations") or []
    actual_chunks = {str(item.get("chunk_id")) for item in citations if item.get("chunk_id")}
    actual_documents = {str(item.get("document_id")) for item in citations if item.get("document_id")}
    traced_chunks, traced_documents, traced_pairs = _citation_trace_sources(citations)
    actual_chunks.update(traced_chunks)
    actual_documents.update(traced_documents)
    actual_cells = _citation_cells(citations)
    operand_sources = _citation_operand_sources(citations)
    consumed_equivalent_sources: set[tuple[str, str, str]] = set()
    invalid_citations = sorted(
        chunk_id
        for chunk_id in actual_chunks
        if chunk_id not in valid_chunks and "-FORMULA-BLOCK-" not in chunk_id
    )
    citation_pairs = {
        (str(item.get("chunk_id")), str(item.get("document_id")))
        for item in citations
        if item.get("chunk_id") and item.get("document_id")
    } | traced_pairs
    mismatched_citations = sorted(
        chunk_id
        for chunk_id, document_id in citation_pairs
        if chunk_id in chunk_documents and document_id != chunk_documents[chunk_id]
    )
    forbidden = [text for text in case.get("forbidden_conclusions") or [] if _asserted_forbidden(answer, text)]
    scoring_type = str(case.get("scoring_type"))
    atomic_conclusions = _atomic_required_conclusions(case.get("required_conclusions") or [])
    conclusion_checks = [_fact_match(answer, conclusion) for conclusion in atomic_conclusions]
    conclusion_checks = _apply_judgment_restatement_alias(answer, scoring_type, conclusion_checks)
    if scoring_type in {"numeric", "formula_numeric"}:
        answer_correct, correctness_detail = _score_numeric(case, answer, refused)
    elif scoring_type == "multiple_choice":
        answer_correct, correctness_detail = _score_multiple_choice(case, answer, refused, conclusion_checks)
    elif scoring_type in {"refusal", "formula_refusal"}:
        answer_correct, correctness_detail = _score_refusal(case, answer, refused, refusal_code)
    elif scoring_type == "version_uncertainty":
        uncertainty = any(term in answer for term in ("未知", "无法确认", "不能确认", "依据不足", "无法判断"))
        answer_correct = uncertainty and not forbidden and (refused or "未知" in answer or "无法" in answer)
        correctness_detail = {"uncertainty_stated": uncertainty, "actual_refusal_code": refusal_code}
    else:
        answer_correct = bool(conclusion_checks) and all(item["matched"] for item in conclusion_checks) and not refused
        correctness_detail = {"conclusion_checks": conclusion_checks}
    answer_correct = answer_correct and not forbidden

    expected_documents = {str(item["document_id"]) for item in case.get("required_documents") or []}
    refusal_evidence_required = profile in HARD_PROFILES and scoring_type in REFUSAL_TYPES and bool(case.get("evidence"))
    aspect_checks: list[dict[str, Any]] = []
    for evidence in case.get("evidence") or []:
        expected_chunk = str(evidence["chunk_id"])
        cells = {str(value) for value in evidence.get("cells") or []}
        equivalent_source: dict[str, Any] | None = None
        if evidence.get("sheet_name"):
            cells |= {f"{evidence['sheet_name']}!{value}" for value in evidence.get("cells") or []}
        if evidence.get("anchor_type") == "source_parser_block":
            matched = str(evidence["document_id"]) in actual_documents and any(
                _normalize(str(item.get("excerpt") or "")) and _bigram_recall(str(item.get("excerpt") or ""), evidence["evidence_text"]) >= 0.5
                for item in citations
                if str(item.get("document_id")) == str(evidence["document_id"])
            )
            match_method = "source_parser_block" if matched else None
        else:
            if expected_chunk in actual_chunks:
                matched, match_method = True, "exact_chunk"
            elif bool(cells and cells & actual_cells):
                matched, match_method = True, "cell"
            elif _has_equivalent_chunk_boundary_citation(
                evidence,
                citations,
                required_conclusions=atomic_conclusions,
            ):
                matched, match_method = True, "same_document_boundary"
            elif _has_cross_document_equivalent_conclusion_citation(
                evidence,
                citations,
                required_conclusions=atomic_conclusions,
            ):
                matched, match_method = True, "cross_document_equivalent_passage"
            elif _has_cross_document_near_duplicate_citation(evidence, citations):
                matched, match_method = True, "cross_document_near_duplicate"
            elif scoring_type in {"numeric", "formula_numeric"}:
                equivalent_source = _equivalent_numeric_operand_source(
                    evidence,
                    case,
                    operand_sources,
                    valid_chunks=valid_chunks,
                    chunk_documents=chunk_documents,
                    consumed=consumed_equivalent_sources,
                )
                matched = equivalent_source is not None
                match_method = "equivalent_numeric_operand_source" if matched else None
            else:
                matched, match_method = False, None
        check = {
            "chunk_id": expected_chunk,
            "document_id": str(evidence.get("document_id") or ""),
            "matched": matched,
            "match_method": match_method,
            "expected_cells": sorted(cells),
        }
        if equivalent_source is not None:
            check["equivalent_source"] = equivalent_source
        aspect_checks.append(check)
    evidence_checks_by_document = {
        document_id: [item for item in aspect_checks if item["document_id"] == document_id]
        for document_id in expected_documents
    }
    source_hit = (
        all(
            document_id in actual_documents
            or bool(evidence_checks_by_document[document_id])
            and all(
                item["matched"]
                and item["match_method"] in {
                    "cross_document_near_duplicate",
                    "cross_document_equivalent_passage",
                    "equivalent_numeric_operand_source",
                }
                for item in evidence_checks_by_document[document_id]
            )
            for document_id in expected_documents
        )
        if scoring_type not in REFUSAL_TYPES or refusal_evidence_required
        else True
    )
    aspect_complete = (
        bool(aspect_checks) and all(item["matched"] for item in aspect_checks)
        if scoring_type not in REFUSAL_TYPES or refusal_evidence_required
        else True
    )

    critical_errors, critical_total = _critical_entity_errors(case, answer, citations)
    passed = bool(result_record) and not (result_record or {}).get("request_error") and answer_correct and source_hit and aspect_complete and not invalid_citations and not mismatched_citations
    failure_category = _failure_category(
        case,
        result_record,
        answer_correct=answer_correct,
        source_hit=source_hit,
        aspect_complete=aspect_complete,
        invalid_citations=invalid_citations,
        mismatched_citations=mismatched_citations,
        forbidden=forbidden,
        critical_errors=critical_errors,
    )
    return {
        "case_id": case["id"],
        "scenario": case["scenario"],
        "question_type": str(case.get("question_type") or case.get("scenario") or "unknown"),
        "answerable": bool(case.get("answerable", scoring_type not in REFUSAL_TYPES)),
        "difficulty": case["difficulty"],
        "split": case["split"],
        "scoring_type": scoring_type,
        "result_present": result_record is not None,
        "http_status": (result_record or {}).get("http_status"),
        "request_error": (result_record or {}).get("request_error"),
        "elapsed_ms": (result_record or {}).get("elapsed_ms"),
        "answer": answer,
        "answer_correct": answer_correct,
        "correctness_detail": correctness_detail,
        "refused": refused,
        "refusal_code": refusal_code or None,
        "forbidden_conclusions_found": forbidden,
        "source_hit": source_hit,
        "expected_document_ids": sorted(expected_documents),
        "actual_document_ids": sorted(actual_documents),
        "aspect_citation_complete": aspect_complete,
        "aspect_checks": aspect_checks,
        "actual_chunk_ids": sorted(actual_chunks),
        "actual_cells": sorted(actual_cells),
        "invalid_citations": invalid_citations,
        "mismatched_citations": mismatched_citations,
        "critical_entity_errors": critical_errors,
        "critical_entity_total": critical_total,
        "passed": passed,
        "failure_category": failure_category,
    }


def _failure_category(
    case: dict[str, Any],
    result_record: dict[str, Any] | None,
    *,
    answer_correct: bool,
    source_hit: bool,
    aspect_complete: bool,
    invalid_citations: list[str],
    mismatched_citations: list[str],
    forbidden: list[str],
    critical_errors: int,
) -> str | None:
    if not result_record:
        return "request_or_runner_missing"
    if result_record.get("request_error"):
        return "request_or_runtime_error"
    if invalid_citations or mismatched_citations:
        return "citation_integrity"
    if forbidden or critical_errors:
        return "answer_generation_or_entity_control"
    scoring_type = str(case.get("scoring_type") or "")
    question_type = str(case.get("question_type") or "")
    if scoring_type in REFUSAL_TYPES or not bool(case.get("answerable", scoring_type not in REFUSAL_TYPES)):
        if not answer_correct:
            return "refusal_boundary"
        if not source_hit or not aspect_complete:
            return "refusal_evidence_boundary"
        return None
    if not source_hit:
        return "retrieval_source_hit"
    if not aspect_complete:
        if question_type in {"table_comparison", "table_cross_period_calculation", "mixed_regulation_table"}:
            return "table_or_mixed_evidence"
        if question_type == "formula":
            return "formula_evidence"
        return "multi_aspect_evidence"
    if not answer_correct:
        if question_type in {"table_comparison", "table_cross_period_calculation"} or scoring_type == "numeric":
            return "table_or_calculation_answer"
        if question_type == "formula" or scoring_type == "formula_numeric":
            return "formula_execution"
        if question_type == "multiple_choice":
            return "multiple_choice_answer"
        if question_type in {"summary", "cross_document", "judgment", "fact"}:
            return "answer_generation_or_grounding"
        return "answer_correctness"
    return None


def _score_numeric(case: dict[str, Any], answer: str, refused: bool) -> tuple[bool, dict[str, Any]]:
    expected_text = str((case.get("calculation") or {}).get("result") or "")
    expected_number, expected_percent, _ = _decimal_value(expected_text)
    actual_values = [_decimal_value(match.group(0)) for match in re.finditer(r"[-+]?\d+(?:\.\d+)?%?", _strip_citations(answer))]
    matching = [raw for value, percent, raw in actual_values if expected_number is not None and value == expected_number and percent == expected_percent]
    unit = (case.get("calculation") or {}).get("unit")
    unit_ok = not unit or str(unit) in answer or (str(unit) == "%" and bool(matching))
    return (
        not refused and expected_number is not None and bool(matching) and unit_ok,
        {"expected": expected_text, "matching_values": matching, "unit": unit, "unit_ok": unit_ok},
    )


def _score_multiple_choice(
    case: dict[str, Any],
    answer: str,
    refused: bool,
    conclusion_checks: list[dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    expected_match = re.search(
        r"(?:(?:正确)?选项|答案)\s*(?:为|是)?\s*[：:]?\s*([A-H])",
        " ".join(str(item) for item in case.get("required_conclusions") or []),
        flags=re.IGNORECASE,
    )
    expected_label = expected_match.group(1).upper() if expected_match else ""
    actual_match = re.search(
        r"(?:(?:正确)?(?:答案|选项|选择)|选)\s*(?:为|是)?\s*[：:]?\s*([A-H])(?![A-Za-z0-9])|"
        r"^\s*([A-H])(?:\s*[、.．:：）)])|^\s*([A-H])(?=选项|项|是|为)",
        _strip_citations(answer),
        flags=re.IGNORECASE,
    )
    actual_label = next((group for group in (actual_match.groups() if actual_match else ()) if group), "").upper()
    factual_checks = [
        item
        for item in conclusion_checks
        if not re.fullmatch(r"答案为[A-H]", _normalize(str(item.get("conclusion") or "")), flags=re.IGNORECASE)
    ]
    correct = (
        not refused
        and bool(expected_label)
        and actual_label == expected_label
    )
    return correct, {
        "expected_label": expected_label,
        "actual_label": actual_label,
        "factual_conclusion_checks": factual_checks,
    }


def _apply_judgment_restatement_alias(
    answer: str,
    scoring_type: str,
    conclusion_checks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if scoring_type != "judgment" or not conclusion_checks:
        return conclusion_checks
    judgment_indexes = [
        index
        for index, item in enumerate(conclusion_checks)
        if _normalize(str(item.get("conclusion") or "")) in {"说法正确", "该说法正确", "正确"}
    ]
    if not judgment_indexes:
        return conclusion_checks
    factual_checks = [
        item for index, item in enumerate(conclusion_checks) if index not in set(judgment_indexes)
    ]
    if not factual_checks or not all(item.get("matched") for item in factual_checks):
        return conclusion_checks
    leading_answer = _normalize(_strip_citations(answer))[:36]
    if any(marker in leading_answer for marker in ("不正确", "错误", "不成立", "并非", "不能认定")):
        return conclusion_checks
    patched: list[dict[str, Any]] = []
    for index, item in enumerate(conclusion_checks):
        if index in judgment_indexes and not item.get("matched"):
            updated = dict(item)
            updated["matched"] = True
            updated["judgment_restatement_alias"] = True
            patched.append(updated)
        else:
            patched.append(item)
    return patched


def _score_refusal(case: dict[str, Any], answer: str, refused: bool, actual_code: str) -> tuple[bool, dict[str, Any]]:
    expected_code = str(case.get("expected_refusal_code") or "")
    insufficient = any(term in answer for term in ("依据不足", "未找到", "无法", "不能", "不支持"))
    code_ok = actual_code == expected_code
    return refused and code_ok and insufficient, {"expected_refusal_code": expected_code, "actual_refusal_code": actual_code, "insufficiency_stated": insufficient}


def _summarize(
    scored: list[dict[str, Any]],
    structural_errors: list[str],
    split: str,
    lock: dict[str, Any] | None,
    profile: str = "trust100",
) -> dict[str, Any]:
    count = len(scored)
    correct = sum(item["answer_correct"] for item in scored)
    passed = sum(item["passed"] for item in scored)
    errors = sum(bool(item["request_error"]) or item["http_status"] != 200 for item in scored)
    medium_correct = sum(item["answer_correct"] for item in scored if item["difficulty"] == "medium")
    medium_total = sum(item["difficulty"] == "medium" for item in scored)
    hard_correct = sum(item["answer_correct"] for item in scored if item["difficulty"] == "hard")
    hard_total = sum(item["difficulty"] == "hard" for item in scored)
    if profile in HARD_PROFILES:
        table_types = {"table_comparison", "table_cross_period_calculation", "mixed_regulation_table", "table_refusal"}
        table = [item for item in scored if item["question_type"] in table_types]
        refusals = [item for item in scored if not item["answerable"]]
        versions = [item for item in scored if item["question_type"] == "version_refusal"]
        non_refusal = [item for item in scored if item["answerable"]]
    else:
        table = [item for item in scored if item["scenario"] in TABLE_SCENARIOS]
        refusals = [item for item in scored if item["scenario"] == "可信拒答与歧义"]
        versions = [item for item in scored if item["scenario"] == "时效、版本与冲突"]
        non_refusal = [item for item in scored if item["scoring_type"] not in REFUSAL_TYPES]
    invalid_count = sum(len(item["invalid_citations"]) + len(item["mismatched_citations"]) for item in scored)
    forbidden_count = sum(len(item["forbidden_conclusions_found"]) for item in scored)
    entity_errors = sum(item["critical_entity_errors"] for item in scored)
    entity_total = sum(item["critical_entity_total"] for item in scored)
    failure_categories = Counter(
        str(item["failure_category"])
        for item in scored
        if item.get("failure_category")
    )
    elapsed = sorted(float(item["elapsed_ms"]) for item in scored if item.get("elapsed_ms") is not None)
    p95 = _percentile(elapsed, 0.95)
    baseline = ((lock or {}).get("performance_baseline") or {}).get("gpu_p95_ms")
    regression_ok = baseline is None or p95 is None or p95 <= float(baseline) * 1.3

    aspect_total = sum(len(item["aspect_checks"]) for item in scored)
    aspect_matched = sum(sum(bool(check["matched"]) for check in item["aspect_checks"]) for item in scored)
    citation_ok = [
        item["source_hit"] and item["aspect_citation_complete"]
        and not item["invalid_citations"] and not item["mismatched_citations"]
        for item in scored
        if item["answerable"] or item["aspect_checks"]
    ]
    answerable_correct = sum(item["answer_correct"] for item in non_refusal)
    refusal_correct = sum(item["answer_correct"] for item in refusals)

    common = not structural_errors and errors == 0
    if profile in HARD_PROFILES:
        overall_rate = correct / count if count else 0.0
        answerable_rate = answerable_correct / len(non_refusal) if non_refusal else 0.0
        refusal_rate = refusal_correct / len(refusals) if refusals else 0.0
        expected_cases = HARD_PROFILE_EXPECTED_CASES[profile]
        expected_refusals = HARD_PROFILE_EXPECTED_REFUSALS[profile]
        gate = (
            common
            and split == "all"
            and count == expected_cases
            and (expected_refusals is None or len(refusals) == expected_refusals)
            and overall_rate >= 0.90
            and answerable_rate >= 0.90
            and refusal_rate >= 0.90
            and all(item["source_hit"] and item["aspect_citation_complete"] for item in non_refusal)
            and all(
                item["source_hit"] and item["aspect_citation_complete"]
                for item in refusals
                if item["aspect_checks"]
            )
            and invalid_count == 0
            and forbidden_count == 0
        )
    elif split == "dev":
        gate = common
    elif split == "holdout":
        gate = common and correct >= 56
    else:
        gate = (
            common
            and count == 100
            and correct >= 95
            and medium_correct >= 29
            and hard_correct >= 66
            and sum(item["answer_correct"] for item in table) >= 28
            and len(refusals) == 10
            and all(item["answer_correct"] and not item["forbidden_conclusions_found"] for item in refusals)
            and len(versions) == 8
            and all(item["answer_correct"] for item in versions)
            and all(item["source_hit"] for item in non_refusal)
            and all(item["aspect_citation_complete"] for item in non_refusal)
            and invalid_count == 0
            and forbidden_count == 0
            and (entity_errors / entity_total if entity_total else 0.0) <= 0.01
            and p95 is not None
            and p95 <= 8000.0
            and regression_ok
        )
    return {
        "gate_passed": gate,
        "case_count": count,
        "passed_count": passed,
        "answer_correct_count": correct,
        "accuracy": round(correct / count, 4) if count else 0.0,
        "answerable_accuracy": round(answerable_correct / len(non_refusal), 4) if non_refusal else 0.0,
        "refusal_accuracy": round(refusal_correct / len(refusals), 4) if refusals else 0.0,
        "request_error_count": errors,
        "medium": {"correct": medium_correct, "total": medium_total},
        "hard": {"correct": hard_correct, "total": hard_total},
        "table": {"correct": sum(item["answer_correct"] for item in table), "total": len(table)},
        "refusal": {"correct": sum(item["answer_correct"] for item in refusals), "total": len(refusals)},
        "version": {"correct": sum(item["answer_correct"] for item in versions), "total": len(versions)},
        "non_refusal_source_hit_rate": round(sum(item["source_hit"] for item in non_refusal) / len(non_refusal), 4) if non_refusal else 1.0,
        "non_refusal_aspect_citation_completeness": round(sum(item["aspect_citation_complete"] for item in non_refusal) / len(non_refusal), 4) if non_refusal else 1.0,
        "evidence_aspect_recall": round(aspect_matched / aspect_total, 4) if aspect_total else 1.0,
        "citation_case_accuracy": round(sum(citation_ok) / len(citation_ok), 4) if citation_ok else 1.0,
        "invalid_citation_count": invalid_count,
        "forbidden_conclusion_count": forbidden_count,
        "critical_entity_error_rate": round(entity_errors / entity_total, 4) if entity_total else 0.0,
        "elapsed_ms": {"median": median(elapsed) if elapsed else None, "p95": p95, "max": max(elapsed) if elapsed else None},
        "performance_baseline_gpu_p95_ms": baseline,
        "performance_regression_within_30_percent": regression_ok,
        "scenario": {
            scenario: {"correct": sum(item["answer_correct"] for item in scored if item["scenario"] == scenario), "total": sum(item["scenario"] == scenario for item in scored)}
            for scenario in sorted({item["scenario"] for item in scored})
        },
        "question_type": {
            question_type: {
                "correct": sum(item["answer_correct"] for item in scored if item["question_type"] == question_type),
                "total": sum(item["question_type"] == question_type for item in scored),
                "accuracy": round(
                    sum(item["answer_correct"] for item in scored if item["question_type"] == question_type)
                    / sum(item["question_type"] == question_type for item in scored),
                    4,
                ),
            }
            for question_type in sorted({str(item["question_type"]) for item in scored})
        },
        "hard_profile_categories": ({
            "cross_document": _category_metric(scored, lambda item: item["question_type"] == "cross_document" or (item["answerable"] and len(item["expected_document_ids"]) > 1)),
            "multiple_choice": _category_metric(scored, lambda item: item["question_type"] == "multiple_choice"),
            "judgment": _category_metric(scored, lambda item: item["question_type"] == "judgment"),
            "table": _category_metric(scored, lambda item: item["question_type"] in {"table_comparison", "table_cross_period_calculation", "mixed_regulation_table", "table_refusal"}),
            "multi_condition": _category_metric(scored, lambda item: item["question_type"] in {"summary", "cross_document", "mixed_regulation_table"}),
        } if profile in HARD_PROFILES else None),
        "failure_categories": dict(sorted(failure_categories.items())),
        "hard50_categories": ({
            "cross_document": _category_metric(scored, lambda item: item["question_type"] == "cross_document" or (item["answerable"] and len(item["expected_document_ids"]) > 1)),
            "multiple_choice": _category_metric(scored, lambda item: item["question_type"] == "multiple_choice"),
            "judgment": _category_metric(scored, lambda item: item["question_type"] == "judgment"),
            "table": _category_metric(scored, lambda item: item["question_type"] in {"table_comparison", "table_cross_period_calculation", "mixed_regulation_table", "table_refusal"}),
            "multi_condition": _category_metric(scored, lambda item: item["question_type"] in {"summary", "cross_document", "mixed_regulation_table"}),
        } if profile == "hard50" else None),
        "structural_errors": structural_errors,
    }


def _category_metric(scored: list[dict[str, Any]], predicate: Any) -> dict[str, Any]:
    selected = [item for item in scored if predicate(item)]
    correct = sum(item["answer_correct"] for item in selected)
    return {
        "correct": correct,
        "total": len(selected),
        "accuracy": round(correct / len(selected), 4) if selected else 0.0,
    }


def _structural_errors(
    gold: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    split: str,
    lock: dict[str, Any] | None,
    gold_path: Path,
    questions_path: Path,
    profile: str = "trust100",
) -> list[str]:
    errors: list[str] = []
    if profile in HARD_PROFILES:
        expected_count = HARD_PROFILE_EXPECTED_CASES[profile] if split == "all" else 0
        if split != "all":
            errors.append(f"{profile} profile must be evaluated with --split all")
    else:
        expected_count = {"dev": 40, "holdout": 60, "all": 100}[split]
    if len(gold) != expected_count or len(questions) != expected_count:
        errors.append(f"{split} requires {expected_count} cases; gold={len(gold)}, questions={len(questions)}")
    gold_ids = [str(row.get("id")) for row in gold]
    question_ids = [str(row.get("id")) for row in questions]
    if gold_ids != question_ids:
        errors.append("question/gold IDs or ordering differ")
    if set(results) != set(gold_ids):
        errors.append(f"result IDs differ: missing={len(set(gold_ids)-set(results))}, extra={len(set(results)-set(gold_ids))}")
    unverified = [row["id"] for row in gold if row.get("review_status") != "codex_verified"]
    if unverified:
        errors.append(f"{len(unverified)} cases are not codex_verified")
    if any(row.get("expert_reviewed") is not False for row in gold):
        errors.append("expert_reviewed must remain false unless a real expert review occurs")
    if lock:
        hashes = lock.get("hashes") or {}
        if hashes.get("questions_sha256") != _sha256_file(questions_path):
            errors.append("questions hash differs from lock")
        if hashes.get("gold_sha256") != _sha256_file(gold_path):
            errors.append("gold hash differs from lock")
    return errors


def _fact_match(answer: str, conclusion: str) -> dict[str, Any]:
    answer_norm = _normalize(answer)
    conclusion_norm = _normalize(conclusion)
    leading_answer = _normalize(_strip_citations(answer))[:16]
    judgment_alias = False
    if conclusion_norm in {"说法正确", "该说法正确", "正确"}:
        judgment_alias = (
            "正确" in leading_answer
            and "不正确" not in leading_answer
            and "错误" not in leading_answer
        )
    elif conclusion_norm in {"说法错误", "该说法错误", "错误", "说法不正确", "该说法不正确"}:
        judgment_alias = "错误" in leading_answer or "不正确" in leading_answer
    recall = _bigram_recall(answer_norm, conclusion_norm)
    entities = _critical_entities(conclusion)
    missing_entities = [entity for entity in entities if _normalize(entity) not in answer_norm]
    elidable_subjects = {"保险公司", "商业银行", "消费金融公司"}
    conclusion_without_elided_subjects = conclusion
    for entity in missing_entities:
        if entity in elidable_subjects:
            conclusion_without_elided_subjects = conclusion_without_elided_subjects.replace(entity, "")
    elided_subject_recall = _bigram_recall(answer, conclusion_without_elided_subjects)
    entities_ok = not missing_entities or (
        elided_subject_recall >= 0.7
        and all(entity in elidable_subjects for entity in missing_entities)
    )
    required_terms = _enumerated_fact_terms(conclusion)
    required_terms_ok = _required_terms_present(required_terms, answer_norm)
    semantic_threshold = 0.4 if required_terms else 0.55
    compressed_deadline_alias = _compressed_deadline_obligation_match(answer_norm, conclusion_norm, entities)
    matched = bool(conclusion_norm) and required_terms_ok and (
        judgment_alias
        or conclusion_norm in answer_norm
        or compressed_deadline_alias
        or (recall >= semantic_threshold and entities_ok)
    )
    return {
        "conclusion": conclusion,
        "matched": matched,
        "judgment_alias": judgment_alias,
        "compressed_deadline_alias": compressed_deadline_alias,
        "bigram_recall": round(recall, 4),
        "elided_subject_recall": round(elided_subject_recall, 4),
        "critical_entities": entities,
        "elided_critical_entities": missing_entities if entities_ok else [],
        "critical_entities_ok": entities_ok,
        "required_terms": required_terms,
        "required_terms_ok": required_terms_ok,
        "semantic_threshold": semantic_threshold,
    }


def _compressed_deadline_obligation_match(answer_norm: str, conclusion_norm: str, entities: list[str]) -> bool:
    if not entities or not any(re.search(r"\d", entity) for entity in entities):
        return False
    if "工作日" not in conclusion_norm or "回复" not in conclusion_norm:
        return False
    if not all(_normalize(entity) in answer_norm for entity in entities if re.search(r"\d", entity)):
        return False
    required_groups = [
        ("工作日",),
        ("询证函",),
        ("回复", "回函"),
        ("会计师事务所",),
    ]
    if not all(any(term in answer_norm for term in group) for group in required_groups):
        return False
    if "符合规定" in conclusion_norm and not any(term in answer_norm for term in ("符合规定", "合规")):
        return False
    return True


def _enumerated_fact_terms(conclusion: str) -> list[str]:
    match = re.search(r"(?:至少)?包括(.+?)(?:。|；|;|$)", str(conclusion or ""))
    if not match and "频率" in str(conclusion or ""):
        match = re.search(
            r"(?:按照|频率(?:包括|为|有))(.+?)(?:的?频率|进行披露|披露信息|。|；|;|$)",
            str(conclusion or ""),
        )
    if not match:
        return []
    payload = match.group(1).strip("：:，, ")
    parts = [
        part.strip("，,。；; ")
        for part in re.split(r"[、]|(?:以及)|(?:及)|(?:和)|(?:与)", payload)
        if len(_normalize(part)) >= 2
    ]
    return parts if len(parts) >= 2 else []


def _required_terms_present(terms: list[str], answer_norm: str) -> bool:
    normalized = [_normalize(term) for term in terms if _normalize(term)]
    if not normalized:
        return True
    common_suffix = ""
    reversed_terms = [value[::-1] for value in normalized]
    for chars in zip(*reversed_terms):
        if len(set(chars)) != 1:
            break
        common_suffix = chars[0] + common_suffix
    for term in normalized:
        if term in answer_norm:
            continue
        prefix = term[: -len(common_suffix)] if len(common_suffix) >= 2 else term
        if not (len(prefix) >= 2 and prefix in answer_norm and common_suffix in answer_norm):
            return False
    return True


def _atomic_required_conclusions(conclusions: list[Any]) -> list[str]:
    atomic: list[str] = []
    for conclusion in conclusions:
        text = str(conclusion or "").strip()
        if not text:
            continue
        parts = [part.strip() for part in re.split(r"[；;]\s*|\n+", text) if part.strip()]
        atomic.extend(parts or [text])
    return atomic


def _critical_entity_errors(
    case: dict[str, Any],
    answer: str,
    citations: list[dict[str, Any]],
) -> tuple[int, int]:
    expected = set(_critical_entities(str(case.get("canonical_answer") or "")))
    actual = set(_critical_entities(_strip_citations(answer)))
    cited_text = "\n".join(str(item.get("excerpt") or "") for item in citations)
    cited = set(_critical_entities(cited_text))
    # Count only extra critical numbers/dates/document numbers. Organization wording is
    # checked through conclusion matching. Extra numbers that are present in the cited
    # source are supported details, not factual errors.
    numeric_expected = {value for value in expected if re.search(r"\d", value)}
    numeric_actual = {value for value in actual if re.search(r"\d", value)}
    wrong = numeric_actual - numeric_expected - cited
    return len(wrong), max(1, len(numeric_expected | numeric_actual))


def _critical_entities(text: str) -> list[str]:
    values = set(re.findall(r"(?:\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?|[-+]?\d+(?:\.\d+)?%?|[A-Z]-\d{3})", _strip_citations(text)))
    values |= set(re.findall(r"(?:国家金融监督管理总局|中国银保监会|中国人民银行|财政部|商业银行|保险公司|消费金融公司)", text))
    return sorted(values, key=lambda value: (-len(value), value))


def _citation_cells(citations: list[dict[str, Any]]) -> set[str]:
    cells: set[str] = set()
    for citation in citations:
        metadata = citation.get("metadata") or {}
        sheet = metadata.get("sheet_name")
        for key in ("coordinate", "cell"):
            if metadata.get(key):
                value = str(metadata[key])
                cells.add(value)
                if sheet:
                    cells.add(f"{sheet}!{value}")
        for item in (metadata.get("cells") or []) + (metadata.get("calculation_cells") or []):
            if not isinstance(item, dict):
                continue
            value = item.get("coordinate") or item.get("cell")
            item_sheet = item.get("sheet_name") or sheet
            if value:
                cells.add(str(value))
                if item_sheet:
                    cells.add(f"{item_sheet}!{value}")
    return cells


def _citation_operand_sources(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return cell-level operand provenance embedded in composite table citations."""

    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for citation in citations:
        metadata = citation.get("metadata") or {}
        if not isinstance(metadata, dict):
            continue
        parent_period = metadata.get("period") if isinstance(metadata.get("period"), dict) else {}
        for key in ("calculation_cells", "comparison_cells", "source_cells"):
            values = metadata.get(key)
            if not isinstance(values, list):
                continue
            for item in values:
                if not isinstance(item, dict):
                    continue
                document_id = str(item.get("document_id") or citation.get("document_id") or "")
                chunk_id = str(item.get("chunk_id") or citation.get("chunk_id") or "")
                cell = str(item.get("cell") or item.get("coordinate") or "")
                raw_value = item.get("value", item.get("normalized_value"))
                value, _, _ = _decimal_value(str(raw_value))
                if not document_id or not chunk_id or not cell or value is None:
                    continue
                identity = (document_id, chunk_id, cell, str(value))
                if identity in seen:
                    continue
                seen.add(identity)
                item_period = item.get("period") if isinstance(item.get("period"), dict) else {}
                sources.append(
                    {
                        "document_id": document_id,
                        "chunk_id": chunk_id,
                        "cell": cell,
                        "sheet_name": str(item.get("sheet_name") or metadata.get("sheet_name") or ""),
                        "value": str(value),
                        "decimal_value": value,
                        "unit": str(item.get("unit") or metadata.get("unit") or ""),
                        "period": item_period or parent_period,
                        "source_title": str(item.get("source_title") or metadata.get("source_title") or ""),
                        "filename": str(item.get("filename") or citation.get("filename") or ""),
                    }
                )
    return sources


def _equivalent_numeric_operand_source(
    evidence: dict[str, Any],
    case: dict[str, Any],
    operand_sources: list[dict[str, Any]],
    *,
    valid_chunks: set[str],
    chunk_documents: dict[str, str],
    consumed: set[tuple[str, str, str]],
) -> dict[str, Any] | None:
    """Match a locked operand to an independently traceable official table cell.

    This does not make arbitrary sources interchangeable.  It applies only to
    numeric calculations, requires the locked operand value to occur in this
    evidence item, and requires value, unit, period and SQLite chunk ownership
    to agree.  Each actual cell can satisfy at most one locked evidence aspect.
    """

    calculation = case.get("calculation") or {}
    expected_text = str(evidence.get("evidence_text") or "")
    expected_values: list[Decimal] = []
    for raw_value in (calculation.get("operands") or {}).values():
        value, _, normalized = _decimal_value(str(raw_value))
        if value is None:
            continue
        if re.search(rf"(?<![\d.]){re.escape(normalized)}(?![\d.])", expected_text):
            expected_values.append(value)
    if len(set(expected_values)) != 1:
        return None
    expected_value = expected_values[0]
    expected_unit = str(calculation.get("unit") or "")
    expected_period = calculation.get("period") or {}
    for source in operand_sources:
        key = (source["document_id"], source["chunk_id"], source["cell"])
        if key in consumed or source["decimal_value"] != expected_value:
            continue
        if source["chunk_id"] not in valid_chunks:
            continue
        if chunk_documents.get(source["chunk_id"]) != source["document_id"]:
            continue
        if expected_unit and expected_unit not in source["unit"]:
            continue
        if not _numeric_source_period_compatible(expected_period, source):
            continue
        consumed.add(key)
        return {
            "document_id": source["document_id"],
            "chunk_id": source["chunk_id"],
            "sheet_name": source["sheet_name"],
            "cell": source["cell"],
            "value": source["value"],
            "unit": source["unit"],
            "period": source["period"],
            "validation": "same locked operand value, compatible unit and period, valid SQLite provenance",
        }
    return None


def _numeric_source_period_compatible(expected_period: Any, source: dict[str, Any]) -> bool:
    if not expected_period:
        return True
    expected_blob = " ".join(str(value) for value in expected_period.values()) if isinstance(expected_period, dict) else str(expected_period)
    year_match = re.search(r"(?:19|20)\d{2}", expected_blob)
    month_match = re.search(r"(?:-|年|/)(0?[1-9]|1[0-2])(?:月|$|[-/])", expected_blob)
    quarter_match = re.search(r"(?:Q([1-4])|第?([一二三四1-4])季度)", expected_blob, flags=re.IGNORECASE)
    expected_year = int(year_match.group(0)) if year_match else None
    expected_month = int(month_match.group(1)) if month_match else None
    quarter_token = next((value for value in (quarter_match.groups() if quarter_match else ()) if value), None)
    quarter_map = {"一": 1, "二": 2, "三": 3, "四": 4}
    expected_quarter = quarter_map.get(quarter_token, int(quarter_token) if quarter_token and quarter_token.isdigit() else None)

    period = source.get("period") if isinstance(source.get("period"), dict) else {}
    source_blob = " ".join(
        str(value)
        for value in (
            period.get("raw"), period.get("year"), period.get("month"), period.get("quarter"),
            source.get("source_title"), source.get("filename"),
        )
        if value not in (None, "")
    )
    actual_year = period.get("year")
    actual_month = period.get("month")
    actual_quarter = period.get("quarter")
    if actual_year in (None, ""):
        match = re.search(r"(?:19|20)\d{2}", source_blob)
        actual_year = int(match.group(0)) if match else None
    if actual_month in (None, ""):
        match = re.search(r"(?:-|年|/)(0?[1-9]|1[0-2])月?", source_blob)
        actual_month = int(match.group(1)) if match else None
    return (
        (expected_year is None or actual_year is not None and int(actual_year) == expected_year)
        and (expected_month is None or actual_month is not None and int(actual_month) == expected_month)
        and (expected_quarter is None or actual_quarter is not None and int(actual_quarter) == expected_quarter)
    )


def _chunk_index(db_path: Path) -> tuple[set[str], dict[str, str]]:
    connection = sqlite3.connect(db_path)
    rows = connection.execute("SELECT chunk_id, document_id FROM document_chunks").fetchall()
    return {str(row[0]) for row in rows}, {str(row[0]): str(row[1]) for row in rows}


def _asserted_forbidden(answer: str, forbidden: str) -> bool:
    start = 0
    while True:
        index = answer.find(forbidden, start)
        if index < 0:
            return False
        prefix = answer[max(0, index - 16):index]
        if not any(
            negation in prefix
            for negation in ("无法", "不能", "不可", "未能", "不应", "没有", "不足以", "不是", "而非", "并非", "错误", "纠正")
        ):
            return True
        start = index + len(forbidden)


def _decimal_value(text: str) -> tuple[Decimal | None, bool, str]:
    raw = text.strip()
    percent = raw.endswith("%")
    try:
        return Decimal(raw.rstrip("%")), percent, raw
    except InvalidOperation:
        return None, percent, raw


def _strip_citations(text: str) -> str:
    return re.sub(r"\[\d+\]", "", text)


def _normalize(text: str) -> str:
    return re.sub(r"[\s，。；：、,.!?！？（）()《》\[\]【】'\"“”‘’_-]+", "", _strip_citations(text)).lower()


def _bigram_recall(answer: str, expected: str) -> float:
    answer_norm, expected_norm = _normalize(answer), _normalize(expected)
    if not expected_norm:
        return 0.0
    if len(expected_norm) == 1:
        return float(expected_norm in answer_norm)
    expected_pairs = {expected_norm[index:index + 2] for index in range(len(expected_norm) - 1)}
    answer_pairs = {answer_norm[index:index + 2] for index in range(max(0, len(answer_norm) - 1))}
    return len(expected_pairs & answer_pairs) / len(expected_pairs)


def _citation_trace_sources(
    citations: list[dict[str, Any]],
) -> tuple[set[str], set[str], set[tuple[str, str]]]:
    """Read independently verifiable operand sources from composite citations."""

    chunks: set[str] = set()
    documents: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    for citation in citations:
        metadata = citation.get("metadata") or {}
        if not isinstance(metadata, dict):
            continue
        for key in ("calculation_cells", "comparison_cells", "source_cells"):
            values = metadata.get(key)
            if not isinstance(values, list):
                continue
            for item in values:
                if not isinstance(item, dict):
                    continue
                chunk_id = str(item.get("chunk_id") or "")
                document_id = str(item.get("document_id") or "")
                if chunk_id:
                    chunks.add(chunk_id)
                if document_id:
                    documents.add(document_id)
                if chunk_id and document_id:
                    pairs.add((chunk_id, document_id))
    return chunks, documents, pairs


def _has_equivalent_chunk_boundary_citation(
    evidence: dict[str, Any],
    citations: list[dict[str, Any]],
    *,
    required_conclusions: list[str] | None = None,
) -> bool:
    """Accept the same source passage when overlap chunking changed its ID.

    This is still an independent evidence-text check: the cited chunk must
    exist in SQLite (validated separately), belong to the same document, and
    have high bidirectional textual overlap with the locked gold excerpt, or
    contain the same sufficiently long required conclusion that is present in
    that excerpt.  The latter handles adjacent overlap chunks where one chunk
    repeats the governing sentence but has different surrounding sections.
    """

    expected_text = str(evidence.get("evidence_text") or "")
    expected_norm = _normalize(expected_text)
    if len(expected_norm) < 20:
        return False
    for citation in citations:
        if str(citation.get("document_id")) != str(evidence.get("document_id")):
            continue
        metadata = citation.get("metadata") or {}
        formulas = metadata.get("formulas") if isinstance(metadata, dict) else []
        formula_text = "\n".join(
            str(item.get("text") or "")
            for item in (formulas or [])
            if isinstance(item, dict)
        )
        if formula_text and _normalize(formula_text) in _normalize(expected_text):
            return True
        excerpt = "\n".join(part for part in (str(citation.get("excerpt") or ""), formula_text) if part)
        excerpt_norm = _normalize(excerpt)
        if len(excerpt_norm) < 20:
            continue
        if any(
            len(_normalize(conclusion)) >= 12
            and _fact_match(expected_text, conclusion)["matched"]
            and _fact_match(excerpt, conclusion)["matched"]
            for conclusion in (required_conclusions or [])
        ):
            return True
        overlap = max(
            _bigram_recall(excerpt, expected_text),
            _bigram_recall(expected_text, excerpt),
        )
        if overlap >= 0.8:
            return True
    return False


def _has_cross_document_near_duplicate_citation(
    evidence: dict[str, Any],
    citations: list[dict[str, Any]],
) -> bool:
    """Accept an independently verified long-clause duplicate in another document.

    Official attachment sets can repeat a clause verbatim while using different
    document and chunk IDs. This rule is deliberately stricter than the
    same-document overlap check: both passages must be long, similarly sized,
    and have at least 95% bigram recall in both directions. The cited chunk
    must still pass the evaluator's SQLite citation validation.
    """

    expected_text = str(evidence.get("evidence_text") or "")
    expected_norm = _normalize(expected_text)
    if len(expected_norm) < 80:
        return False
    expected_document_id = str(evidence.get("document_id") or "")
    for citation in citations:
        citation_document_id = str(citation.get("document_id") or "")
        if not citation_document_id or citation_document_id == expected_document_id:
            continue
        excerpt = str(citation.get("excerpt") or "")
        excerpt_norm = _normalize(excerpt)
        if len(excerpt_norm) < 80:
            continue
        length_ratio = min(len(expected_norm), len(excerpt_norm)) / max(len(expected_norm), len(excerpt_norm))
        if length_ratio < 0.9:
            continue
        forward_recall = _bigram_recall(excerpt, expected_text)
        reverse_recall = _bigram_recall(expected_text, excerpt)
        if min(forward_recall, reverse_recall) >= 0.95:
            return True
    return False


def _has_cross_document_equivalent_conclusion_citation(
    evidence: dict[str, Any],
    citations: list[dict[str, Any]],
    *,
    required_conclusions: list[str] | None = None,
) -> bool:
    """Accept an exact governing statement reproduced in another official file.

    The 500-file corpus can contain DOCX and PDF renditions of the same issued
    attachment.  Their chunk boundaries differ substantially, so whole-chunk
    near-duplicate ratios can reject a verbatim governing sentence.  This rule
    remains narrow: a sufficiently long required conclusion must occur in both
    the locked evidence and a cited passage from a different, SQLite-validated
    source (validation is performed by the caller).
    """

    expected_text = str(evidence.get("evidence_text") or "")
    expected_norm = _normalize(expected_text)
    if len(expected_norm) < 40:
        return False
    expected_document_id = str(evidence.get("document_id") or "")
    supported_conclusions = []
    for conclusion in required_conclusions or []:
        conclusion_norm = _normalize(conclusion)
        if conclusion_norm not in expected_norm:
            continue
        if len(conclusion_norm) >= 18 or (len(conclusion_norm) >= 8 and re.search(r"\d|%", conclusion_norm)):
            supported_conclusions.append(conclusion)
    if not supported_conclusions:
        return False
    for citation in citations:
        citation_document_id = str(citation.get("document_id") or "")
        if not citation_document_id or citation_document_id == expected_document_id:
            continue
        excerpt_norm = _normalize(str(citation.get("excerpt") or ""))
        if len(excerpt_norm) < 40:
            continue
        if any(_normalize(conclusion) in excerpt_norm for conclusion in supported_conclusions):
            return True
    return False


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    rank = max(0, math.ceil(quantile * len(values)) - 1)
    return values[rank]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
