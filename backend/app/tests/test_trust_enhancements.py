from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.database import Base
from backend.app.models.document import Document
from backend.app.schemas.qa import AnswerClaim, RetrievalResult
from backend.app.services.answer_generation_service import GeneratedAnswer, generate_answer
from backend.app.services.regulatory_semantic_grounding_service import (
    deterministic_semantic_conflicts,
    extract_regulatory_semantic_frame,
    validate_regulatory_semantics,
)
from backend.app.services.retrieval_service import filter_candidates_by_metadata
from backend.app.services.retrieval_metadata_filter_service import build_retrieval_metadata_filter
from backend.app.services.query_planner_service import QueryAspect, QuerySearchQuery
from backend.app.services.rag_service import _vector_metadata_filter_inputs
from backend.app.services.vector_store_service import VectorSearchResult, _retrieval_filter
from scripts.evaluate_trust_challenge import EXPECTED_CATEGORY_COUNTS, evaluate, read_jsonl


def _document(document_id: str, title: str, authority: str) -> Document:
    return Document(
        document_id=document_id,
        filename=f"{title}.pdf",
        filename_norm=f"{title}.pdf".casefold(),
        file_type="pdf",
        size=10,
        storage_path=f"/tmp/{document_id}.pdf",
        title=title,
        issuing_authority=authority,
        version_status="current",
        status="indexed",
        index_version="bge-m3-qdrant-v3-grounded-cells",
    )


def test_explicit_filter_resolves_document_scope_and_keeps_inferred_soft() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add_all(
            [
                _document("DOC-1", "资本管理办法", "国家金融监督管理总局"),
                _document("DOC-2", "统计制度", "中国人民银行"),
            ]
        )
        db.commit()
        compiled = build_retrieval_metadata_filter(
            db,
            {"source_title": "资本管理办法", "regulatory_topic": "资本监管"},
            ("source_title",),
        )
    assert compiled.status == "applied"
    assert compiled.document_ids == ("DOC-1",)
    assert compiled.explicit == {"source_title": "资本管理办法"}
    assert compiled.inferred == {"regulatory_topic": "资本监管"}


def test_explicit_filter_zero_match_never_becomes_unscoped() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add(_document("DOC-1", "统计制度", "中国人民银行"))
        db.commit()
        compiled = build_retrieval_metadata_filter(
            db,
            {"source_title": "不存在的制度"},
            ("source_title",),
        )
    assert compiled.no_match
    assert compiled.status == "metadata_filter_no_match"
    assert compiled.qdrant_filter()["document_ids"] == []


def test_mixed_table_locator_does_not_filter_regulation_vectors() -> None:
    aspect = QueryAspect(
        aspect_id="mixed",
        question="结合表格和制度判断",
        search_queries=(QuerySearchQuery("监管依据", "semantic_question"),),
        evidence_need="制度与表格",
        keywords=("监管",),
        modality="mixed",
        table_task="lookup",
        table_filters={
            "source_title": "2024年统计表",
            "year": 2024,
            "issuing_authority": "国家金融监督管理总局",
        },
        explicit_filter_keys=("source_title", "year", "issuing_authority"),
    )
    values, explicit_keys, table_scoped = _vector_metadata_filter_inputs(aspect)
    assert values == {"issuing_authority": "国家金融监督管理总局"}
    assert explicit_keys == {"issuing_authority"}
    assert set(table_scoped) == {"source_title", "year"}


@dataclass
class _MatchValue:
    value: object


@dataclass
class _MatchAny:
    any: list[str]


@dataclass
class _FieldCondition:
    key: str
    match: object


@dataclass
class _Filter:
    must: list[object]


class _Models:
    MatchValue = _MatchValue
    MatchAny = _MatchAny
    FieldCondition = _FieldCondition
    Filter = _Filter


def test_qdrant_filter_applies_same_scope_to_prefetch() -> None:
    compiled = _retrieval_filter(
        _Models,
        {"document_ids": ["DOC-1", "DOC-2"], "article_number": "第十条"},
    )
    conditions = {item.key: item.match for item in compiled.must}
    assert conditions["document_id"].any == ["DOC-1", "DOC-2"]
    assert conditions["article_number"].value == "第十条"
    assert "index_version" in conditions


def test_semantic_frames_reject_negation_and_modality_changes() -> None:
    negated_claim = extract_regulatory_semantic_frame("商业银行不得报送该项目。")
    positive_evidence = extract_regulatory_semantic_frame("商业银行应当报送该项目。")
    assert "polarity_conflict" in deterministic_semantic_conflicts(negated_claim, positive_evidence)

    mandatory_claim = extract_regulatory_semantic_frame("商业银行必须提交说明。")
    permissive_evidence = extract_regulatory_semantic_frame("商业银行可以提交说明。")
    assert "modality_strengthened" in deterministic_semantic_conflicts(mandatory_claim, permissive_evidence)


def test_semantic_validation_focuses_on_relevant_cited_sentence() -> None:
    claim = AnswerClaim(
        text="消费金融公司可以在全国范围内开展业务。[1]",
        citation_ids=["[1]"],
        role="conclusion",
        aspect_ids=["multiple_choice_evidence"],
    )
    evidence = RetrievalResult(
        chunk_id="RULE-FOCUS",
        rank=1,
        source_doc="rule.pdf",
        text=(
            "消费金融公司应当建立风险管理制度。"
            "第十四条 消费金融公司可以在全国范围内开展业务。"
            "符合条件的公司可以申请经营其他人民币业务。"
        ),
        citation_label="[1]",
    )

    result = validate_regulatory_semantics([claim], [evidence], mode="risk_based")

    assert result["semantic_passed"] is True
    verdict = result["claim_verdicts"][0]
    assert verdict["verdict"] == "supported"
    assert "第十四条" in verdict["evidence_focus"]
    assert "建立风险管理制度" not in verdict["evidence_focus"]


def test_multiple_choice_evidence_scope_commentary_is_not_a_regulatory_claim(monkeypatch) -> None:
    from backend.app.services import regulatory_semantic_grounding_service as service

    monkeypatch.setattr(
        service,
        "_verify_risky_claims",
        lambda _items: (_ for _ in ()).throw(AssertionError("verifier should not be called")),
    )
    claim = AnswerClaim(
        text="其他选项在知识库中无相关证据，无法判断其正确性。[1]",
        citation_ids=["[1]"],
        role="explanation",
        aspect_ids=["multiple_choice_evidence"],
    )
    evidence = RetrievalResult(
        chunk_id="RULE-SCOPE",
        rank=1,
        source_doc="rule.pdf",
        text="银行业金融机构应当在十个工作日内回函。",
        citation_label="[1]",
    )

    result = validate_regulatory_semantics([claim], [evidence], mode="risk_based")

    assert result["semantic_passed"] is True
    assert result["claim_verdicts"][0]["validation_mode"] == "deterministic_scope_commentary"


def test_active_document_metadata_repairs_legacy_vector_payload(monkeypatch) -> None:
    from backend.app.services import retrieval_service as service

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as db:
        db.add(_document("DOC-PDF", "银行函证工作操作指引", "国家金融监督管理总局"))
        db.commit()
    monkeypatch.setattr(service, "SessionLocal", session_factory)
    candidate = VectorSearchResult(
        chunk_id="CHUNK-1",
        document_id="DOC-PDF",
        filename="银行函证工作操作指引.pdf",
        section_title=None,
        page_number=1,
        text="回函时限",
        embedding_text="回函时限",
        token_count=4,
        score=0.9,
        metadata={},
    )

    active = service.filter_active_candidates([candidate])
    filtered = filter_candidates_by_metadata(active, {"file_type": "pdf"})

    assert filtered == [candidate]
    assert candidate.metadata["source_format"] == "pdf"


def test_calculation_trace_normalizes_sheet_whitespace_and_formula_order() -> None:
    from backend.app.services.answer_generation_service import _calculation_trace_valid

    metadata = {
        "operation": "difference",
        "calculation_formula": " 保障房贷款!E6 -  保障房贷款!B6",
        "calculation_result": -251.14228369600096,
        "calculation_cells": [
            {"sheet_name": " 保障房贷款", "cell": "B6", "normalized_value": 23486.373080154},
            {"sheet_name": " 保障房贷款", "cell": "E6", "normalized_value": 23235.230796458},
        ],
    }

    assert _calculation_trace_valid(metadata) is True
    metadata["calculation_formula"] = "保障房贷款!E6 - 保障房贷款!C6"
    assert _calculation_trace_valid(metadata) is False


def test_document_snapshot_cache_reuses_and_invalidates_chunk_snapshot() -> None:
    from backend.app.services import rag_service
    from backend.app.models.document import DocumentChunk

    class _Result:
        def __init__(self, rows):
            self.rows = rows

        def all(self):
            return self.rows

    class _DB:
        def __init__(self, rows):
            self.rows = rows
            self.calls = 0

        def scalars(self, _statement):
            self.calls += 1
            return _Result(self.rows)

    chunk = DocumentChunk(
        id=901,
        chunk_id="CACHE-1",
        document_id="DOC-CACHE",
        text="缓存测试",
        embedding_text="缓存测试",
        index_status="indexed",
        source_file="cache.pdf",
    )
    rag_service.clear_document_snapshot_cache()
    db = _DB([chunk])

    first = rag_service._chunks_by_document(db, {"DOC-CACHE"}, document_chunk_cache={})
    second = rag_service._chunks_by_document(db, {"DOC-CACHE"}, document_chunk_cache={})
    assert db.calls == 1
    assert first["DOC-CACHE"][0].chunk_id == second["DOC-CACHE"][0].chunk_id

    rag_service.clear_document_snapshot_cache({"DOC-CACHE"})
    rag_service._chunks_by_document(db, {"DOC-CACHE"}, document_chunk_cache={})
    assert db.calls == 2


def test_risk_claim_uses_one_structured_verifier_call(monkeypatch) -> None:
    from backend.app.services import regulatory_semantic_grounding_service as service

    calls = []

    def fake_verify(items):
        calls.append(items)
        return (
            [
                {
                    "claim_index": 1,
                    "verdict": "supported",
                    "reason_codes": [],
                    "evidence_snippet": "仅当完成审批时方可办理",
                    "reason": "claim保留了条件",
                }
            ],
            "completed",
        )

    monkeypatch.setattr(service, "_verify_risky_claims", fake_verify)
    claim = AnswerClaim(
        text="仅当完成审批时方可办理。[1]",
        citation_ids=["[1]"],
        role="conclusion",
    )
    evidence = RetrievalResult(
        chunk_id="RISK-1",
        rank=1,
        source_doc="rule.pdf",
        text="仅当完成审批时方可办理，但特定情形除外。",
        citation_label="[1]",
    )
    result = validate_regulatory_semantics([claim], [evidence], mode="risk_based")
    assert result["semantic_passed"] is True
    assert result["verifier_status"] == "completed"
    assert len(calls) == 1


def test_mixed_answer_does_not_take_table_only_early_exit(monkeypatch) -> None:
    from backend.app.services import answer_generation_service as service

    chunks = [
        RetrievalResult(
            chunk_id="TABLE-1",
            rank=1,
            source_doc="report.xlsx",
            text="A1=10",
            citation_label="[1]",
            metadata={
                "dynamic_table_evidence": True,
                "value": 10,
                "coordinate": "A1",
                "aspect_id": "table",
                "evidence_role": "direct_evidence",
            },
        ),
        RetrievalResult(
            chunk_id="RULE-1",
            rank=2,
            source_doc="rule.pdf",
            text="该指标应当如实填报。",
            citation_label="[2]",
            metadata={"aspect_id": "rule", "evidence_role": "direct_evidence"},
        ),
    ]

    def fake_call(*_args, **_kwargs):
        return GeneratedAnswer(
            answer="表格值为10。[1]\n\n该指标应当如实填报。[2]",
            answer_type="mixed_grounded",
            generation_status="completed",
            claims=[
                AnswerClaim(text="表格值为10。[1]", citation_ids=["[1]"], role="table_fact", aspect_ids=["table"]),
                AnswerClaim(text="该指标应当如实填报。[2]", citation_ids=["[2]"], role="regulatory_basis", aspect_ids=["rule"]),
            ],
        )

    monkeypatch.setattr(service, "ANSWER_GENERATION_ENABLED", True)
    monkeypatch.setattr(service, "ANSWER_GENERATION_API_KEY", "test")
    monkeypatch.setattr(service, "_call_llm", fake_call)
    result = generate_answer(
        "结合制度解释表格值",
        chunks,
        has_sufficient_context=True,
        answer_mode="mixed",
        required_aspect_ids=["table", "rule"],
    )
    assert result.answer_type == "mixed_grounded"
    assert not result.refused
    assert result.grounding_validation["mixed_evidence_complete"] is True


def test_answer_claim_remains_backward_compatible() -> None:
    claim = AnswerClaim.model_validate({"text": "依据摘录", "citation_ids": ["[1]"]})
    assert claim.role == "other"
    assert claim.aspect_ids == []


def test_challenge_template_has_fixed_distribution_and_cannot_pass_unreviewed() -> None:
    rows = read_jsonl(Path("data/evaluation/trust_challenge_60.jsonl"))
    assert len(rows) == 60
    assert {
        category: sum(item["category"] == category for item in rows)
        for category in EXPECTED_CATEGORY_COUNTS
    } == EXPECTED_CATEGORY_COUNTS
    report = evaluate(rows, {})
    assert report["gate_passed"] is False
    assert "cases are not manually reviewed" in " ".join(report["summary"]["structural_errors"])
