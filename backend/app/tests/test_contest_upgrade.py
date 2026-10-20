"""Mechanism tests for the contest-version answer and display upgrade."""
from __future__ import annotations

from types import SimpleNamespace

from backend.app.core.config import (
    QA_COMPLEX_REQUEST_TOTAL_BUDGET_SECONDS,
    QA_REQUEST_TOTAL_BUDGET_SECONDS,
)
from backend.app.schemas.qa import AnswerClaim, EvidenceRequirement, RetrievalResult
from backend.app.services.answer_explanation_service import build_answer_explanation
from backend.app.services.answer_validation_service import validation_passed
from backend.app.services.performance_metrics import (
    remaining_request_budget_seconds,
    request_budget_context,
    request_budget_snapshot,
    upgrade_request_budget,
)
from backend.app.services.query_planner_service import (
    QueryAspect,
    QueryPlan,
)


def _chunk(chunk_id: str, text: str, *, source: str, label: str, aspect: str,
           metadata: dict | None = None) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        rank=int(label.strip("[]")),
        source_doc=source,
        text=text,
        citation_label=label,
        section_title="第二章 定义",
        metadata={
            "document_id": f"doc-{chunk_id}",
            "source_title": source,
            "aspect_id": aspect,
            "prompt_matched_aspects": [aspect],
            **(metadata or {}),
        },
    )


def _plan(*aspects: QueryAspect, budget: dict | None = None) -> QueryPlan:
    return QueryPlan(
        "比较商业银行版恢复计划示例与商业银行版处置计划建议示例中“关键功能”的定义是否一致。",
        aspects,
        "test",
        budget=budget or {"max_aspects": 2, "detected_items": []},
    )


def test_complex_plan_budget_upgrades_once_from_original_start() -> None:
    from backend.app.services.request_budget_service import (
        apply_plan_budget,
        classify_plan_budget,
    )

    aspects = (
        QueryAspect("source_a", "说明商业银行版恢复计划示例的定义。", (), "定义", (),
                    table_filters={"source_title": "商业银行版恢复计划示例"}),
        QueryAspect("source_b", "说明商业银行版处置计划建议示例的定义。", (), "定义", (),
                    table_filters={"source_title": "商业银行版处置计划建议示例"}),
    )
    plan = _plan(*aspects)
    decision = classify_plan_budget(plan)
    assert decision.complex_task is True
    assert "multiple_required_sources" in decision.reasons

    with request_budget_context(QA_REQUEST_TOTAL_BUDGET_SECONDS):
        before = remaining_request_budget_seconds()
        applied = apply_plan_budget(plan)
        assert applied.upgraded is True
        after = remaining_request_budget_seconds()
        assert after > before
        assert after <= QA_COMPLEX_REQUEST_TOTAL_BUDGET_SECONDS
        snapshot = request_budget_snapshot()
        assert snapshot["category"] == "complex"
        assert snapshot["total_seconds"] == QA_COMPLEX_REQUEST_TOTAL_BUDGET_SECONDS
        # A second plan or a nested repair cannot open a fresh 240 seconds.
        assert upgrade_request_budget(QA_COMPLEX_REQUEST_TOTAL_BUDGET_SECONDS + 30) is False
        assert remaining_request_budget_seconds() <= after


def test_recovered_budget_deducts_elapsed_time_and_keeps_category() -> None:
    with request_budget_context(
        240, elapsed_seconds=239.5, category="complex"
    ):
        remaining = remaining_request_budget_seconds()
        assert remaining is not None and remaining <= 0.6
        snapshot = request_budget_snapshot()
        assert snapshot["category"] == "complex"
        assert snapshot["total_seconds"] == 240


def test_standard_and_mixed_plans_are_classified_structurally() -> None:
    from backend.app.services.request_budget_service import classify_plan_budget

    standard = _plan(QueryAspect("a", "资本充足率是什么？", (), "定义", ()))
    assert classify_plan_budget(standard).complex_task is False

    mixed = _plan(
        QueryAspect("text", "说明制度要求。", (), "定义", (), modality="text"),
        QueryAspect("table", "读取报表数值。", (), "取值", (), modality="table"),
    )
    assert "text_and_table_evidence" in classify_plan_budget(mixed).reasons

    three_items = _plan(
        QueryAspect("a", "事项一。", (), "取证", ()),
        QueryAspect("b", "事项二。", (), "取证", ()),
        QueryAspect("c", "事项三。", (), "取证", ()),
        budget={"max_aspects": 3, "detected_items": ["一", "二", "三"]},
    )
    assert "three_or_more_user_items" in classify_plan_budget(three_items).reasons


def test_validation_success_requires_every_report_condition() -> None:
    complete = {
        "status": "passed", "passed": True, "task_complete": True,
        "citation_integrity": True, "facts_supported": True,
    }
    assert validation_passed(complete)
    for missing in ("task_complete", "citation_integrity", "facts_supported"):
        report = dict(complete)
        report[missing] = False
        assert not validation_passed(report)
    assert not validation_passed({**complete, "status": "not_run"})
    assert not validation_passed({**complete, "passed": False})


def test_explanation_links_requirements_claims_sources_and_calculation() -> None:
    chunk = _chunk(
        "chunk-1", "关键功能是指向第三方提供的金融服务。", source="商业银行版恢复计划示例",
        label="[1]", aspect="source_a",
        metadata={
            "dynamic_table_evidence": True,
            "operation": "difference",
            "calculation_formula": "Sheet1!B2 - Sheet1!B3",
            "calculation_display_formula": "3500 - 1800 = 1700",
            "calculation_result": 1700,
            "calculation_cells": [
                {"sheet_name": "Sheet1", "cell": "B2", "normalized_value": 3500, "unit": "万元"},
                {"sheet_name": "Sheet1", "cell": "B3", "normalized_value": 1800, "unit": "万元"},
            ],
        },
    )
    package = SimpleNamespace(
        context_chunks=[chunk],
        retrieval_summary={
            "query_plan": {"aspects": [{"aspect_id": "source_a", "question": "定义"}]},
            "evidence_bundle": {
                "requirements": [{
                    "requirement_id": "source_a", "question": "定义", "mandatory": True,
                    "source_titles": ["商业银行版恢复计划示例"], "attribute_terms": [],
                }],
                "coverage": {"source_a": ["chunk-1"]},
                "missing_requirements": [],
                "missing_optional_requirements": [],
                "decisions": [{
                    "chunk_id": "chunk-1", "requirement_id": "source_a",
                    "stage": "selected", "reason": "source_and_requirement_match",
                }],
            },
        },
    )
    claim = AnswerClaim(text="答案为 A。关键功能是指向第三方提供的金融服务。[1]",
                        citation_ids=["[1]"], aspect_ids=["source_a"])
    generated = SimpleNamespace(
        answer="答案为 A。关键功能是指向第三方提供的金融服务。[1]",
        answer_type="llm_grounded",
        refused=False,
        refusal_reason=None,
        claims=[claim],
        grounding_validation={
            "status": "passed", "passed": True, "task_complete": True,
            "citation_integrity": True, "facts_supported": True,
            "method": "model", "reasons": [], "missing_requirements": [],
        },
    )
    explanation = build_answer_explanation(
        question="商业银行版恢复计划示例如何定义关键功能？",
        generated=generated,
        package=package,
        cited_results=[chunk],
        started_at=None,
    )
    assert explanation.validation.passed is True
    assert explanation.requirements[0].status == "covered"
    assert explanation.requirements[0].evidence_ids == ["[1]"]
    assert explanation.claims[0].evidence_ids == ["[1]"]
    assert explanation.claims[0].requirement_ids == ["source_a"]
    assert explanation.sources[0].cell is None
    assert explanation.calculations[0].display_formula == "3500 - 1800 = 1700"
    assert len(explanation.calculations[0].operands) == 2
    assert explanation.calculations[0].operands[0].cell == "B2"
    # Explanation is a local projection; no prompt/debug package is required.
    assert explanation.budget.model_call_limit >= 1


def test_explanation_marks_missing_requirement_and_unverified_outcome() -> None:
    package = SimpleNamespace(
        context_chunks=[],
        retrieval_summary={
            "query_plan": {"aspects": []},
            "evidence_bundle": {
                "requirements": [{
                    "requirement_id": "source_b", "question": "处置计划建议示例的定义",
                    "mandatory": True, "source_titles": ["商业银行版处置计划建议示例"],
                }],
                "coverage": {"source_b": []},
                "missing_requirements": ["source_b"],
                "missing_optional_requirements": [],
                "decisions": [{
                    "chunk_id": "", "requirement_id": "source_b",
                    "stage": "missing", "reason": "not_recalled",
                }],
            },
        },
    )
    generated = SimpleNamespace(
        answer="当前证据不足以完整回答。缺少：《商业银行版处置计划建议示例》的直接依据",
        answer_type="refusal",
        refused=True,
        refusal_reason="insufficient_context",
        claims=[],
        grounding_validation={
            "status": "not_run", "passed": False, "task_complete": False,
            "citation_integrity": False, "facts_supported": False,
            "failure_stage": "evidence", "missing_requirements": ["source_b"],
        },
    )
    explanation = build_answer_explanation(
        question="比较两份材料。", generated=generated, package=package, cited_results=[]
    )
    assert explanation.outcome == "evidence_missing"
    assert explanation.requirements[0].status == "missing"
    assert explanation.missing_requirements
    assert explanation.validation.passed is False


def test_definition_requirement_reserves_the_defining_clause(monkeypatch) -> None:
    from backend.app.services import rag_service, retrieval_support
    from backend.app.services.retrieval_support import AspectRetrieval

    aspect = QueryAspect(
        "source_a", "《恢复计划示例（商业银行版）》中，关键功能的定义是什么？",
        (), "定义原文", ("关键功能",),
        table_filters={"source_title": "恢复计划示例（商业银行版）"},
    )
    requirement = EvidenceRequirement(
        requirement_id="source_a", question=aspect.question, intent="define",
        target_terms=["关键功能"], attribute_terms=["定义"],
        source_titles=["恢复计划示例（商业银行版）"],
    )
    plan = QueryPlan("比较定义", (aspect,), "test", requirements=(requirement,))
    candidate = _chunk(
        "candidate", "相邻段落", source="恢复计划示例（商业银行版）",
        label="[1]", aspect="source_a",
    )
    document_chunks = [
        SimpleNamespace(
            chunk_id="neighbour", text="关键功能在恢复阶段应优先保护。",
            source_file="500_银行保险机构恢复和处置计划实施暂行办法_附件1：恢复计划示例（商业银行版）.docx",
            section_title=None, chunk_metadata="{}", document_id="doc", index_status="indexed",
        ),
        SimpleNamespace(
            chunk_id="definition", text="关键功能是提供给第三方的关键业务或产品等金融服务。",
            source_file="500_银行保险机构恢复和处置计划实施暂行办法_附件1：恢复计划示例（商业银行版）.docx",
            section_title=None, chunk_metadata="{}", document_id="doc", index_status="indexed",
        ),
    ]
    monkeypatch.setattr(retrieval_support, "_chunks_by_document",
                        lambda *args: {"doc": document_chunks})
    item = AspectRetrieval(aspect, [candidate], [], {}, [])
    selected = rag_service._reserve_definition_clauses(
        SimpleNamespace(scalars=True), [], plan, [item]
    )
    assert selected[0].chunk_id == "definition"
    assert selected[0].metadata["prompt_selection_reason"] == "definition_clause_reserved"


def test_table_plus_regulation_question_is_split_by_named_source() -> None:
    from backend.app.services import query_planner_service

    question = (
        "《恢复计划示例（商业银行版）》中，关键功能的定义是什么？同时，"
        "《2023年3季度保险业资金运用情况表》中，资金运用余额的账面余额是多少？"
    )
    token = query_planner_service._TABLE_SOURCES.set((
        "148_2023年3季度保险业资金运用情况表_2023年三季度保险业资金运用情况表.xlsx",
    ))
    try:
        aspects = query_planner_service._multi_source_table_aspects(question, [])
    finally:
        query_planner_service._TABLE_SOURCES.reset(token)
    assert len(aspects) == 2
    by_modality = {aspect.modality: aspect for aspect in aspects}
    assert by_modality["table"].table_filters["source_title"] == "2023年3季度保险业资金运用情况表"
    assert by_modality["text"].table_filters["source_title"] == "恢复计划示例（商业银行版）"
    assert "关键功能的定义" in by_modality["text"].question
    assert "资金运用余额" in by_modality["table"].question


def test_cross_document_source_aspects_are_required_and_umbrella_is_optional(monkeypatch) -> None:
    from backend.app.services import query_planner_service

    question = "商业银行版恢复计划示例与商业银行版处置计划建议示例中“关键功能”的定义是否一致？"
    aspects = (
        QueryAspect("source_a", "商业银行版恢复计划示例如何定义关键功能？", (),
                    "定义", (), subject_scope=("商业银行版恢复计划示例",)),
        QueryAspect("source_b", "商业银行版处置计划建议示例如何定义关键功能？", (),
                    "定义", (), subject_scope=("商业银行版处置计划建议示例",)),
        QueryAspect("compare", "比较两个来源的定义是否一致。", (), "对比", (),
                    subject_scope=("商业银行版恢复计划示例", "商业银行版处置计划建议示例")),
    )
    plan = _plan(*aspects)
    monkeypatch.setattr(query_planner_service, "_plan_query_impl",
                        lambda query, options=None: plan)
    result = query_planner_service.plan_query(question)
    mandatory = {aspect.aspect_id: aspect.mandatory for aspect in result.aspects}
    assert mandatory["source_a"] is True
    assert mandatory["source_b"] is True
    assert mandatory["compare"] is False
