from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.database import Base
from backend.app.models.document import Document, DocumentChunk
from backend.app.schemas.qa import AnswerClaim, Citation, RetrievalResult
from backend.app.services.answer_generation_service import (
    GeneratedAnswer,
    _deterministic_table_output_unit,
    generate_answer,
)
from backend.app.services.document_identity_query_service import answer_document_identity_question
from backend.app.services.regulatory_semantic_grounding_service import (
    deterministic_semantic_conflicts,
    extract_regulatory_semantic_frame,
    validate_regulatory_semantics,
)
from backend.app.services.retrieval_service import RetrievalDiagnostics, filter_candidates_by_metadata
from backend.app.services.rerank_service import RerankedChunk
from backend.app.services.retrieval_metadata_filter_service import build_retrieval_metadata_filter
from backend.app.services.query_planner_service import QueryAspect, QueryPlan, QuerySearchQuery
from backend.app.services.rag_service import (
    AspectRetrieval,
    _audit_citation_payload,
    _best_non_duplicate_candidate,
    _best_shared_candidate,
    _bounded_document_lexical_support_matches,
    _bounded_lexical_phrases,
    _chunk_matches_query_aspect,
    _exact_anchor_support_matches,
    _mcq_exact_support_matches,
    _mcq_matches_from_reranked,
    _mcq_reliability_question,
    _mcq_required_exact_terms,
    _mcq_seed_document_ids,
    _normalize_exact_support_text,
    _recover_missing_aspects_from_sibling_documents,
    _select_prompt_chunks,
    _vector_metadata_filter_inputs,
)
from backend.app.services.formula_parser_service import (
    _extract_variable_assignments,
    _format_formula_result,
)
from backend.app.services.vector_store_service import VectorSearchResult, _retrieval_filter
from scripts.evaluate_trust_challenge import (
    _asserted_forbidden,
    _atomic_required_conclusions,
    _critical_entity_errors,
    _decimal_value,
    _fact_match,
    _citation_trace_sources,
    _has_equivalent_chunk_boundary_citation,
    _has_cross_document_near_duplicate_citation,
    _score_multiple_choice,
    _summarize,
    read_jsonl,
)


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


def test_exact_anchor_support_recovers_heading_inside_resolved_document() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    document_id = "DOC-EXACT-ANCHOR"
    aspect = QueryAspect(
        aspect_id="approval_heading",
        question="根据《申请材料目录》，请说明与“中资商业银行分行筹建审批”相关的规定。",
        search_queries=(QuerySearchQuery("分行筹建审批", "keyword_anchor"),),
        evidence_need="机构设立事项",
        keywords=("分行筹建审批",),
        table_filters={"source_title": "申请材料目录"},
        explicit_filter_keys=("source_title",),
    )
    with Session(engine) as db:
        db.add(_document(document_id, "申请材料目录", "国家金融监督管理总局"))
        db.add_all(
            [
                DocumentChunk(
                    chunk_id=f"{document_id}-CHUNK-0001",
                    document_id=document_id,
                    text="一、机构设立。1.3 中资商业银行分行筹建审批。",
                    embedding_text="机构设立 分行筹建审批",
                    source_file="申请材料目录.pdf",
                    index_status="indexed",
                ),
                DocumentChunk(
                    chunk_id=f"{document_id}-CHUNK-0002",
                    document_id=document_id,
                    text="其他行政许可事项。",
                    source_file="申请材料目录.pdf",
                    index_status="indexed",
                ),
            ]
        )
        db.commit()
        matches = _exact_anchor_support_matches(
            db,
            aspect,
            [],
            document_ids={document_id},
            document_chunk_cache={},
        )

    assert [match.citation.chunk_id for match in matches] == [f"{document_id}-CHUNK-0001"]
    assert matches[0].metadata["fusion_method"] == "bounded_document_exact_anchor"


def test_bounded_lexical_support_recovers_each_clause_only_inside_resolved_document() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    document_id = "DOC-BOUNDED-LEXICAL"
    aspect = QueryAspect(
        aspect_id="compound_policy",
        question="根据《消费者保护办法》，概括绩效薪酬延期支付要求以及催收不得涉及无关第三人的边界。",
        search_queries=(
            QuerySearchQuery("绩效薪酬延期支付不得少于三年", "document_style_statement"),
            QuerySearchQuery("催收不得涉及与债务无关的第三人", "document_style_statement"),
        ),
        evidence_need="两项制度约束",
        keywords=("绩效薪酬", "催收"),
        table_filters={"source_title": "消费者保护办法"},
        explicit_filter_keys=("source_title",),
    )
    with Session(engine) as db:
        db.add(_document(document_id, "消费者保护办法", "监管机关"))
        db.add_all(
            [
                DocumentChunk(
                    chunk_id=f"{document_id}-CHUNK-0001",
                    document_id=document_id,
                    text="关键岗位绩效薪酬延期支付期限不得少于三年。",
                    embedding_text="绩效薪酬 延期支付 不得少于三年",
                    source_file="消费者保护办法.pdf",
                    index_status="indexed",
                ),
                DocumentChunk(
                    chunk_id=f"{document_id}-CHUNK-0002",
                    document_id=document_id,
                    text="催收不得涉及与债务无关的第三人。",
                    embedding_text="催收 无关第三人",
                    source_file="消费者保护办法.pdf",
                    index_status="indexed",
                ),
            ]
        )
        db.commit()
        matches = _bounded_document_lexical_support_matches(
            db,
            aspect,
            [],
            document_ids={document_id},
            document_chunk_cache={},
        )

    assert {match.citation.chunk_id for match in matches} == {
        f"{document_id}-CHUNK-0001",
        f"{document_id}-CHUNK-0002",
    }
    assert all(match.metadata["fusion_method"] == "bounded_document_lexical_support" for match in matches)


def test_bounded_lexical_support_prefers_definition_clause_for_definition_question() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    document_id = "DOC-BOUNDED-DEFINITION"
    aspect = QueryAspect(
        aspect_id="definition",
        question="意外伤害保险如何定义？",
        search_queries=(QuerySearchQuery("意外伤害保险 定义", "semantic_question"),),
        evidence_need="概念定义",
        keywords=("意外伤害保险", "定义"),
    )
    with Session(engine) as db:
        db.add(_document(document_id, "意外伤害保险办法", "监管机关"))
        db.add_all(
            [
                DocumentChunk(
                    chunk_id=f"{document_id}-CHUNK-0001",
                    document_id=document_id,
                    text="本办法用于规范意外伤害保险经营活动。",
                    source_file="意外伤害保险办法.pdf",
                    index_status="indexed",
                ),
                DocumentChunk(
                    chunk_id=f"{document_id}-CHUNK-0002",
                    document_id=document_id,
                    text="本办法所称意外伤害保险，是以约定事故为给付保险金条件的人身保险。",
                    source_file="意外伤害保险办法.pdf",
                    index_status="indexed",
                ),
            ]
        )
        db.commit()
        matches = _bounded_document_lexical_support_matches(
            db,
            aspect,
            [],
            document_ids={document_id},
            document_chunk_cache={},
        )

    assert matches
    assert matches[0].citation.chunk_id.endswith("0002")


def test_bounded_lexical_support_ignores_repeated_source_metadata_in_embedding_text() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    document_id = "DOC-VERSION-TITLE"
    title = "中资商业银行行政许可事项申请材料目录及格式要求（2023年版）"
    aspect = QueryAspect(
        aspect_id="version",
        question=f"《{title}》的版本是什么？",
        search_queries=(QuerySearchQuery(f"{title} 版本", "semantic_question"),),
        evidence_need="文件版本",
        keywords=("2023年版", "版本"),
    )
    with Session(engine) as db:
        db.add(_document(document_id, title, "监管机关"))
        db.add_all(
            [
                DocumentChunk(
                    chunk_id=f"{document_id}-CHUNK-0001",
                    document_id=document_id,
                    text=title,
                    embedding_text=f"来源文件：{title}.pdf\n{title}",
                    source_file=f"{title}.pdf",
                    index_status="indexed",
                ),
                DocumentChunk(
                    chunk_id=f"{document_id}-CHUNK-0002",
                    document_id=document_id,
                    text="电子材料应符合纸质材料相关要求。",
                    embedding_text=f"来源文件：{title}.pdf\n电子材料应符合纸质材料相关要求。",
                    source_file=f"{title}.pdf",
                    index_status="indexed",
                ),
            ]
        )
        db.commit()
        matches = _bounded_document_lexical_support_matches(
            db,
            aspect,
            [],
            document_ids={document_id},
            document_chunk_cache={},
        )

    assert matches
    assert matches[0].citation.chunk_id == f"{document_id}-CHUNK-0001"
    assert all(match.citation.chunk_id != f"{document_id}-CHUNK-0002" for match in matches)


def test_bounded_lexical_phrases_keep_predicate_after_interrogative_object() -> None:
    aspect = QueryAspect(
        aspect_id="communication_subjects",
        question="说明沟通策略对象。",
        search_queries=(
            QuerySearchQuery("说明沟通策略 与哪些主体开展有效沟通。", "document_style_statement"),
        ),
        evidence_need="沟通对象",
        keywords=("沟通策略", "有效沟通"),
    )

    assert "开展有效沟通" in _bounded_lexical_phrases(aspect)


def test_bounded_lexical_phrases_extract_unknown_threshold_relation_anchors() -> None:
    question = "商业银行大额风险暴露相对一级资本净额的门槛是多少？"
    aspect = QueryAspect(
        aspect_id="threshold",
        question=question,
        search_queries=(QuerySearchQuery(question, "semantic_question"),),
        evidence_need="量化门槛",
        keywords=("大额风险暴露", "一级资本净额"),
    )

    phrases = _bounded_lexical_phrases(aspect)

    assert "大额风险暴露" in phrases
    assert "一级资本净额" in phrases


def test_sibling_document_recovery_supplements_weak_nonempty_aspect() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    wrong_document_id = "DOC-WEAK-EXPOSURE"
    shared_document_id = "DOC-SIBLING-DIRECT"
    target_chunk_id = f"{shared_document_id}-CHUNK-0001"
    sibling_chunk_id = f"{shared_document_id}-CHUNK-0002"
    weak_chunk_id = f"{wrong_document_id}-CHUNK-0001"
    target_statement = (
        "大额风险暴露是指商业银行对单一客户或一组关联客户超过其一级资本净额2.5%的风险暴露"
    )
    with Session(engine) as db:
        db.add_all(
            [
                _document(wrong_document_id, "风险暴露分类材料", "监管机关"),
                _document(shared_document_id, "恢复和处置计划示例", "监管机关"),
                DocumentChunk(
                    chunk_id=weak_chunk_id,
                    document_id=wrong_document_id,
                    text="小微企业风险暴露不得超过规定限额。",
                    source_file="风险暴露分类材料.pdf",
                    index_status="indexed",
                ),
                DocumentChunk(
                    chunk_id=target_chunk_id,
                    document_id=shared_document_id,
                    text=target_statement,
                    source_file="恢复和处置计划示例.pdf",
                    index_status="indexed",
                ),
                DocumentChunk(
                    chunk_id=sibling_chunk_id,
                    document_id=shared_document_id,
                    text="处置策略建议应坚持自救为本的基本原则。",
                    source_file="恢复和处置计划示例.pdf",
                    index_status="indexed",
                ),
            ]
        )
        db.commit()
        threshold_aspect = QueryAspect(
            aspect_id="threshold",
            question="商业银行大额风险暴露的门槛是多少？",
            search_queries=(
                QuerySearchQuery(
                    "商业银行大额风险暴露相对一级资本净额的门槛是多少",
                    "semantic_question",
                ),
            ),
            evidence_need="大额风险暴露门槛",
            keywords=("大额风险暴露", "一级资本净额"),
        )
        strategy_aspect = QueryAspect(
            aspect_id="strategy",
            question="处置策略基本原则是什么？",
            search_queries=(QuerySearchQuery("处置策略建议应坚持自救为本", "semantic_question"),),
            evidence_need="处置策略原则",
            keywords=("处置策略", "自救为本"),
        )
        retrievals = [
            AspectRetrieval(
                aspect=threshold_aspect,
                candidates=[
                    RetrievalResult(
                        chunk_id=weak_chunk_id,
                        rank=1,
                        source_doc="风险暴露分类材料.pdf",
                        text="小微企业风险暴露不得超过规定限额。",
                        citation_label="[1]",
                        score=0.8,
                        metadata={"document_id": wrong_document_id},
                    )
                ],
                diagnostics=[],
                citation_validation={"checked_chunks": 1, "valid_chunks": 1, "invalid_chunks": 0, "invalid_chunk_ids": []},
                selected_chunk_ids=[],
                retrieval_covered=True,
            ),
            AspectRetrieval(
                aspect=strategy_aspect,
                candidates=[
                    RetrievalResult(
                        chunk_id=sibling_chunk_id,
                        rank=1,
                        source_doc="恢复和处置计划示例.pdf",
                        text="处置策略建议应坚持自救为本的基本原则。",
                        citation_label="[1]",
                        score=0.9,
                        metadata={"document_id": shared_document_id},
                    )
                ],
                diagnostics=[],
                citation_validation={"checked_chunks": 1, "valid_chunks": 1, "invalid_chunks": 0, "invalid_chunk_ids": []},
                selected_chunk_ids=[],
                retrieval_covered=True,
            ),
        ]

        _recover_missing_aspects_from_sibling_documents(
            db,
            retrievals,
            document_chunk_cache={},
        )

    assert retrievals[0].candidates[0].chunk_id == target_chunk_id
    assert retrievals[0].diagnostics[-1]["recovery_mode"] == "supplemented"


def test_mcq_lexical_seed_recovers_literal_option_when_vector_matches_are_empty() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    document_id = "DOC-MCQ-SEED"
    statement = "保险期限一年及以下的个人意外险平均附加费用率上限为35%"
    aspect = QueryAspect(
        aspect_id="multiple_choice_evidence",
        question="下列哪项关于意外伤害保险的表述正确？",
        search_queries=(QuerySearchQuery(statement, "document_style_statement"),),
        evidence_need="选项事实核验",
        keywords=("意外伤害保险", "35%"),
    )
    with Session(engine) as db:
        db.add(_document(document_id, "意外伤害保险业务监管办法", "监管机关"))
        db.add(
            DocumentChunk(
                chunk_id=f"{document_id}-CHUNK-0001",
                document_id=document_id,
                text=f"本办法规定，{statement}。",
                embedding_text=statement,
                source_file="意外伤害保险业务监管办法.pdf",
                index_status="indexed",
            )
        )
        db.commit()
        matches = _mcq_exact_support_matches(db, aspect, [], document_chunk_cache={})

    assert matches
    assert matches[0].citation.document_id == document_id
    assert matches[0].metadata["exact_support_statement"] == statement


def test_lexical_seed_accepts_distinctive_four_to_seven_character_policy_term() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    document_id = "DOC-SHORT-SEED"
    with Session(engine) as db:
        db.add(_document(document_id, "处置计划建议示例", "监管机关"))
        db.add(
            DocumentChunk(
                chunk_id=f"{document_id}-CHUNK-0001",
                document_id=document_id,
                text="大额风险暴露是指超过一级资本净额规定比例的风险暴露。",
                source_file="处置计划建议示例.docx",
                index_status="indexed",
            )
        )
        db.commit()

        assert _mcq_seed_document_ids(db, ["大额风险暴露"]) == {document_id}


def test_mcq_lexical_seed_does_not_lose_rare_threshold_behind_common_term_limit() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    target_document_id = "DOC-MCQ-RARE-THRESHOLD"
    statement = "核心一级资本充足率降至5.125%或以下构成持续经营触发事件"
    aspect = QueryAspect(
        aspect_id="multiple_choice_evidence",
        question="下列哪项关于资本工具触发事件的表述正确？",
        search_queries=(QuerySearchQuery(statement, "document_style_statement"),),
        evidence_need="选项事实核验",
        keywords=("资本工具", "5.125%"),
    )
    with Session(engine) as db:
        for index in range(170):
            document_id = f"DOC-MCQ-COMMON-{index:03d}"
            db.add(_document(document_id, f"资本管理材料{index}", "监管机关"))
            db.add(
                DocumentChunk(
                    chunk_id=f"{document_id}-CHUNK-0001",
                    document_id=document_id,
                    text="资本工具应当按照规定吸收损失。",
                    embedding_text="资本工具 吸收损失 持续经营",
                    source_file=f"材料{index}.pdf",
                    index_status="indexed",
                )
            )
        db.add(_document(target_document_id, "资本工具触发事件规定", "监管机关"))
        db.add(
            DocumentChunk(
                chunk_id=f"{target_document_id}-CHUNK-0001",
                document_id=target_document_id,
                text=statement,
                embedding_text=statement,
                source_file="资本工具触发事件规定.pdf",
                index_status="indexed",
            )
        )
        db.commit()
        matches = _mcq_exact_support_matches(db, aspect, [], document_chunk_cache={})

    assert matches
    assert matches[0].citation.document_id == target_document_id


def test_mcq_exact_support_preserves_governing_modal_list_preamble() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    document_id = "DOC-MCQ-LIST-SCOPE"
    statement = "不得向特定团体成员以外的个人销售团体意外险"
    aspect = QueryAspect(
        aspect_id="multiple_choice_evidence",
        question="下列哪项销售行为符合规定？",
        search_queries=(QuerySearchQuery(statement, "document_style_statement"),),
        evidence_need="选项事实核验",
        keywords=("团体意外险",),
    )
    with Session(engine) as db:
        db.add(_document(document_id, "意外险销售规定", "监管机关"))
        for index, text in enumerate(
            [
                "保险公司开展业务活动不得存在以下行为：\n（一）强迫消费者订立合同；",
                "（二）虚构业务套取资金；",
                "（九）向特定团体成员以外的个人销售团体意外险；",
            ],
            start=1,
        ):
            db.add(
                DocumentChunk(
                    chunk_id=f"{document_id}-CHUNK-{index:04d}",
                    document_id=document_id,
                    text=text,
                    embedding_text=text,
                    source_file="意外险销售规定.pdf",
                    index_status="indexed",
                )
            )
        db.commit()
        matches = _mcq_exact_support_matches(db, aspect, [], document_chunk_cache={})

    exact = next(item for item in matches if item.metadata["evidence_role"] == "mcq_exact_support")
    preamble = next(item for item in matches if item.metadata["evidence_role"] == "mcq_rule_preamble")
    assert exact.citation.chunk_id.endswith("0003")
    assert preamble.citation.chunk_id.endswith("0001")
    assert preamble.metadata["governs_chunk_ids"] == [exact.citation.chunk_id]


def test_multiple_choice_evaluator_accepts_label_option_format_without_forcing_definition_restatement() -> None:
    correct, detail = _score_multiple_choice(
        {
            "required_conclusions": [
                "答案为C",
                "大额风险暴露是指商业银行对单一客户超过一级资本净额2.5%的风险暴露",
            ]
        },
        "C选项‘门槛为一级资本净额2.5%’正确。[1]",
        False,
        [
            {"conclusion": "答案为C", "matched": True},
            {"conclusion": "大额风险暴露是指商业银行对单一客户超过一级资本净额2.5%的风险暴露", "matched": False},
        ],
    )

    assert correct is True
    assert detail["actual_label"] == "C"


@pytest.mark.parametrize(
    ("answer", "label"),
    [
        ("正确选项为：A、核心一级资本充足率降至5.125%。[1]", "A"),
        ("选项B“真实性、准确性、完整性、一致性和可比性”正确。[1]", "B"),
        ("因此选项D正确。[1]", "D"),
    ],
)
def test_multiple_choice_evaluator_accepts_common_explicit_label_phrasings(answer: str, label: str) -> None:
    correct, detail = _score_multiple_choice(
        {"required_conclusions": [f"答案为{label}"]},
        answer,
        False,
        [],
    )

    assert correct is True
    assert detail["actual_label"] == label


def test_mcq_reliability_gate_uses_public_option_facts_instead_of_generic_stem() -> None:
    aspect = QueryAspect(
        aspect_id="multiple_choice_evidence",
        question="关于第三支柱补充披露，下列哪项完整列出了材料要求？",
        search_queries=(
            QuerySearchQuery("关于第三支柱补充披露", "keyword_anchor"),
            QuerySearchQuery(
                "要求真实性、准确性、完整性、一致性和可比性",
                "document_style_statement",
            ),
            QuerySearchQuery("要求及时性、盈利性和保密性", "document_style_statement"),
        ),
        evidence_need="选项事实核验",
        keywords=("第三支柱",),
    )

    reliability_question = _mcq_reliability_question(aspect)

    assert "真实性、准确性、完整性、一致性和可比性" in reliability_question
    assert "及时性、盈利性和保密性" in reliability_question
    assert "哪项完整列出" not in reliability_question


def test_mcq_reliability_gate_evaluates_cross_document_option_facts_separately() -> None:
    aspect = QueryAspect(
        aspect_id="multiple_choice_evidence",
        question="下列哪组同时符合两份材料？",
        search_queries=(
            QuerySearchQuery("太平洋保险集团属于保险控股型集团", "document_style_statement"),
            QuerySearchQuery("查询服务至少保留三个月", "document_style_statement"),
        ),
        evidence_need="选项事实核验",
        keywords=("两份材料",),
    )
    reranked = [
        RerankedChunk(
            candidate=VectorSearchResult(
                chunk_id="DOC-MCQ-A-CHUNK-0001",
                document_id="DOC-MCQ-A",
                filename="group.pdf",
                section_title="保险控股型集团",
                page_number=1,
                text="太平洋保险集团属于保险控股型集团。",
                embedding_text="太平洋保险集团属于保险控股型集团。",
                token_count=12,
                score=0.8,
                chunk_type="paragraph",
            ),
            rerank_score=0.44,
        ),
        RerankedChunk(
            candidate=VectorSearchResult(
                chunk_id="DOC-MCQ-B-CHUNK-0001",
                document_id="DOC-MCQ-B",
                filename="accident.pdf",
                section_title="信息查询服务",
                page_number=2,
                text="保单信息查询服务应至少保留三个月。",
                embedding_text="保单信息查询服务应至少保留三个月。",
                token_count=14,
                score=0.8,
                chunk_type="paragraph",
            ),
            rerank_score=0.44,
        ),
    ]

    matches = _mcq_matches_from_reranked(aspect, reranked, RetrievalDiagnostics())

    assert {match.citation.document_id for match in matches} == {"DOC-MCQ-A", "DOC-MCQ-B"}


def test_mcq_prompt_selection_preserves_bounded_exact_support_below_generic_threshold() -> None:
    aspect = QueryAspect(
        aspect_id="multiple_choice_evidence",
        question="下列哪组同时符合两份材料？",
        search_queries=(QuerySearchQuery("跨文件选项事实", "document_style_statement"),),
        evidence_need="选项事实核验",
        keywords=("两份材料",),
    )
    exact_chunks = [
        RetrievalResult(
            chunk_id=f"DOC-MCQ-{index}-CHUNK-0001",
            rank=index,
            source_doc=f"source-{index}.pdf",
            text=f"跨文件选项事实 {index}",
            citation_label=f"[{index}]",
            score=0.38,
            metadata={"evidence_role": "mcq_exact_support", "rerank_score": 0.38},
        )
        for index in (1, 2)
    ]
    unrelated = RetrievalResult(
        chunk_id="DOC-HIGH-CHUNK-0001",
        rank=3,
        source_doc="unrelated.pdf",
        text="两份材料背景信息",
        citation_label="[3]",
        score=0.9,
        metadata={"evidence_role": "direct_evidence", "rerank_score": 0.9},
    )
    retrieval = AspectRetrieval(
        aspect=aspect,
        candidates=[*exact_chunks, unrelated],
        diagnostics=[],
        citation_validation={"valid_chunks": 3},
        selected_chunk_ids=[],
        retrieval_covered=True,
    )
    plan = QueryPlan(original_question=aspect.question, aspects=(aspect,), planner="test")

    selected, _summary = _select_prompt_chunks(aspect.question, plan, [retrieval])

    assert {chunk.chunk_id for chunk in exact_chunks}.issubset({chunk.chunk_id for chunk in selected})


def test_mcq_prompt_selection_preserves_cross_document_option_evidence() -> None:
    aspect = QueryAspect(
        aspect_id="multiple_choice_evidence",
        question="下列哪一组同时正确描述两份材料？",
        search_queries=(QuerySearchQuery("选项事实", "document_style_statement"),),
        evidence_need="选项事实核验",
        keywords=("两份材料",),
    )
    crowded_document_chunks = [
        RetrievalResult(
            chunk_id=f"DOC-A-CHUNK-{index:04d}",
            rank=index,
            source_doc="a.pdf",
            text=f"材料 A 选项事实 {index}",
            citation_label=f"[{index}]",
            score=0.95 - index * 0.01,
            metadata={
                "document_id": "DOC-A",
                "evidence_role": "mcq_exact_support",
                "rerank_score": 0.95 - index * 0.01,
                "query_hits": [{"query_type": "document_style_statement"}],
            },
        )
        for index in range(1, 6)
    ]
    second_document = RetrievalResult(
        chunk_id="DOC-B-CHUNK-0001",
        rank=6,
        source_doc="b.pdf",
        text="材料 B 的直接选项事实。",
        citation_label="[6]",
        score=0.82,
        metadata={
            "document_id": "DOC-B",
            "evidence_role": "direct_evidence",
            "rerank_score": 0.82,
            "query_hits": [{"query_type": "document_style_statement"}],
        },
    )
    retrieval = AspectRetrieval(
        aspect=aspect,
        candidates=[*crowded_document_chunks, second_document],
        diagnostics=[],
        citation_validation={"valid_chunks": 6},
        selected_chunk_ids=[],
        retrieval_covered=True,
    )
    plan = QueryPlan(original_question=aspect.question, aspects=(aspect,), planner="test")

    selected, _summary = _select_prompt_chunks(aspect.question, plan, [retrieval])

    assert "DOC-B-CHUNK-0001" in {chunk.chunk_id for chunk in selected}


def test_mcq_required_exact_terms_reject_cross_family_application_material() -> None:
    statement = "申请书 内容包括但不限于 拟设立中资商业银行 名称 拟设地 注册资本 股权结构 业务范围 基本信息"
    nonbank_text = "拟设立金融租赁公司名称、拟设地、注册资本、股权结构、业务范围等。"
    bank_text = "拟设立中资商业银行的名称、拟设地、注册资本、股权结构、业务范围等基本信息。"

    required = _mcq_required_exact_terms(statement)

    assert "中资商业银行" in required
    assert not all(
        _normalize_exact_support_text(term) in _normalize_exact_support_text(nonbank_text)
        for term in required
    )
    assert all(
        _normalize_exact_support_text(term) in _normalize_exact_support_text(bank_text)
        for term in required
    )


def test_mcq_required_exact_terms_inherit_family_from_sibling_option_alias() -> None:
    aspect = QueryAspect(
        aspect_id="multiple_choice_evidence",
        question="行政许可申请材料选择题",
        search_queries=(
            QuerySearchQuery("申请书列名称、拟设地、注册资本、股权结构和业务范围，目录为2023年版", "document_style_statement"),
            QuerySearchQuery(
                "申请书 内容包括但不限于 拟设立中资商业银行 名称 拟设地 注册资本 股权结构 业务范围 基本信息",
                "document_style_statement",
            ),
        ),
        evidence_need="选项事实核验",
        keywords=("行政许可申请材料",),
    )

    required = _mcq_required_exact_terms("申请书列名称、拟设地、注册资本、股权结构和业务范围", aspect)

    assert "中资商业银行" in required


def test_source_title_filter_normalizes_insurance_alias_and_optional_situation_token() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add(_document("DOC-TITLE-ALIAS", "2023年10月人身保险公司经营情况表", "监管机关"))
        db.commit()
        concise = build_retrieval_metadata_filter(
            db,
            {"source_title": "2023年10月人身险公司经营情况表"},
            ("source_title",),
        )
        omitted = build_retrieval_metadata_filter(
            db,
            {"source_title": "2023年10月人身保险公司经营表"},
            ("source_title",),
        )

    assert concise.document_ids == ("DOC-TITLE-ALIAS",)
    assert omitted.document_ids == ("DOC-TITLE-ALIAS",)


def test_audit_citation_payload_is_bounded_and_keeps_reproducibility_fields() -> None:
    citation = Citation(
        document_id="DOC-AUDIT",
        chunk_id="DOC-AUDIT-CHUNK-0001",
        filename="公式附件.docx",
        excerpt="证据" * 10000,
        metadata={
            "aspect_id": "formula",
            "calculation_formula": "RWA = E * R * 12.5",
            "formulas": [
                {"text": "RWA = E * R * 12.5", "source_type": "word_omml", "order_index": index, "raw": "x" * 10000}
                for index in range(100)
            ],
            "unbounded_debug_blob": "y" * 5_000_000,
        },
    )

    payload = _audit_citation_payload(citation)
    encoded = __import__("json").dumps(payload, ensure_ascii=False)

    assert len(encoded) < 10_000
    assert payload["metadata"]["calculation_formula"] == "RWA = E * R * 12.5"
    assert len(payload["metadata"]["formulas"]) == 12
    assert "unbounded_debug_blob" not in payload["metadata"]


def test_compact_formula_assignments_and_percentage_output() -> None:
    assignments = _extract_variable_assignments(
        "核心一级资本900、扣除项0、风险加权资产10000，核心一级资本充足率是多少百分比？",
        expected_variables=["核心一级资本", "扣除项", "风险加权资产"],
    )

    assert next(iter(assignments["核心一级资本"]))[0] == 900
    assert next(iter(assignments["扣除项"]))[0] == 0
    assert next(iter(assignments["风险加权资产"]))[0] == 10000
    assert _format_formula_result(0.09, "(核心一级资本-扣除项)/风险加权资产", "结果是多少百分比") == "9%"


def test_evaluator_accepts_elided_regulated_subject_and_shared_suffix_enumeration() -> None:
    subject = _fact_match(
        "团体意外险不得向特定团体成员以外的个人销售。",
        "保险公司不得向特定团体成员以外的个人销售团体意外险",
    )
    enumeration = _fact_match(
        "压力测试至少覆盖系统性、自身和混合压力情景。",
        "压力测试至少包括系统性压力情景、自身压力情景和混合压力情景。",
    )

    assert subject["matched"] is True
    assert enumeration["matched"] is True


def test_evaluator_reads_cross_document_sources_from_calculation_trace() -> None:
    chunks, documents, pairs = _citation_trace_sources(
        [
            {
                "metadata": {
                    "calculation_cells": [
                        {"document_id": "DOC-A", "chunk_id": "CHUNK-A", "cell": "C5"},
                        {"document_id": "DOC-B", "chunk_id": "CHUNK-B", "cell": "C5"},
                    ]
                }
            }
        ]
    )

    assert chunks == {"CHUNK-A", "CHUNK-B"}
    assert documents == {"DOC-A", "DOC-B"}
    assert pairs == {("CHUNK-A", "DOC-A"), ("CHUNK-B", "DOC-B")}


def test_evaluator_requires_each_semicolon_delimited_conclusion() -> None:
    conclusions = _atomic_required_conclusions(["应使用自有资金；应与监管部门和客户有效沟通。"])

    assert conclusions == ["应使用自有资金", "应与监管部门和客户有效沟通。"]
    assert all(_fact_match("应使用自有资金。", item)["matched"] for item in conclusions) is False


def test_evaluator_accepts_explicit_judgment_alias_and_contextual_subject_elision() -> None:
    assert _fact_match("正确。该比例与期限均符合要求。", "说法正确")["matched"]
    assert _fact_match(
        "账簿划分政策和程序的内部审计频率为每年一次，内部审计结果需留档备查。",
        "商业银行应每年对划分政策和程序开展内部审计，内部审计结果需留档备查",
    )["matched"]
    assert not _fact_match("商业银行每年披露一次信息。", "商业银行应每年开展内部审计并留档备查")["matched"]


def test_fullwidth_ordinal_number_is_not_a_factual_entity() -> None:
    from backend.app.services.grounding_validation_service import (
        _extract_entities,
        _strip_presentation_markers,
    )

    claim_text = "表述（1）说法正确。材料规定授信额度最高不得超过人民币20万元。"

    stripped = _strip_presentation_markers(claim_text)
    entities = _extract_entities(stripped)

    assert "1" not in entities
    assert "20" in entities


def test_evaluator_does_not_count_cited_extra_number_as_entity_error() -> None:
    case = {"canonical_answer": "应在10个工作日内完成。"}
    citations = [{"excerpt": "2025年底前，特殊情形可延长至14个工作日。"}]

    errors, total = _critical_entity_errors(
        case,
        "应在10个工作日内完成；2025年底前特殊情形可延长至14个工作日。[1]",
        citations,
    )

    assert errors == 0
    assert total == 3


def test_evaluator_accepts_same_document_overlap_chunk_with_required_clause() -> None:
    conclusion = "补充披露信息应符合真实性、准确性、完整性、一致性和可比性要求"
    evidence = {
        "document_id": "DOC-DISCLOSURE",
        "evidence_text": f"四、补充披露要求。其他前置规定。{conclusion}。",
    }
    citation = {
        "document_id": "DOC-DISCLOSURE",
        "excerpt": f"{conclusion}。五、追溯披露和过渡期披露要求。其他后续规定。",
    }

    assert _has_equivalent_chunk_boundary_citation(
        evidence,
        [citation],
        required_conclusions=[conclusion],
    )
    assert not _has_equivalent_chunk_boundary_citation(
        evidence,
        [{**citation, "document_id": "DOC-OTHER"}],
        required_conclusions=[conclusion],
    )


def test_evaluator_accepts_only_strict_cross_document_clause_duplicates() -> None:
    repeated_clause = (
        "在处置计划实施过程中，应当与监管部门、地方政府、股东、客户、员工和社会公众等开展有效沟通，"
        "维持正常经营秩序，防范区域性与系统性风险和其他不利于处置的事件，降低处置影响，提高处置可行性。"
    )
    evidence = {"document_id": "DOC-A", "evidence_text": repeated_clause}
    duplicate = repeated_clause.replace("降低处置影响", "降低处置的影响")

    assert _has_cross_document_near_duplicate_citation(
        evidence,
        [{"document_id": "DOC-B", "excerpt": duplicate}],
    )
    assert not _has_cross_document_near_duplicate_citation(
        evidence,
        [{"document_id": "DOC-A", "excerpt": duplicate}],
    )
    assert not _has_cross_document_near_duplicate_citation(
        evidence,
        [{"document_id": "DOC-B", "excerpt": "本段只共享处置计划和有效沟通等少量通用词语，并不包含同一监管要求。" * 5}],
    )


def test_evaluator_requires_each_item_in_an_includes_conclusion() -> None:
    conclusion = "银行业金融机构公示信息包括函证范围和回函用章。"

    incomplete = _fact_match("银行业金融机构公示信息包括办理机构和联系方式。", conclusion)
    complete = _fact_match("公示信息包括回函用章以及询证函的函证范围。", conclusion)

    assert incomplete["matched"] is False
    assert incomplete["required_terms"] == ["函证范围", "回函用章"]
    assert complete["matched"] is True


def test_prompt_selection_reuses_one_chunk_for_two_explicit_aspects() -> None:
    shared = RetrievalResult(
        chunk_id="C1",
        rank=1,
        source_doc="制度.pdf",
        text="不得以市场事件为由转换账簿；除非产品性质变化，否则账簿转换不可撤销。",
        citation_label="[1]",
        score=0.8,
        metadata={"rerank_score": 0.9},
    )
    aspect = QueryAspect(
        aspect_id="condition_2",
        question="请说明与“除非产品性质变化，否则账簿转换不可撤销”相关的规定。",
        search_queries=(QuerySearchQuery("账簿转换不可撤销", "keyword_anchor"),),
        evidence_need="转换例外",
        keywords=("账簿转换不可撤销",),
    )

    reused = _best_shared_candidate([shared], [shared], aspect)

    assert reused is shared


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


def test_exact_support_normalization_unifies_common_insurance_aliases() -> None:
    left = _normalize_exact_support_text(
        "万能险、投连险、变额年金及中短存续期产品"
    )
    right = _normalize_exact_support_text(
        "万能保险、投资连结保险、变额年金及中短存续期保险产品"
    )

    assert left == right


def test_prompt_aspect_gate_preserves_bounded_lexical_support() -> None:
    aspect = QueryAspect(
        aspect_id="regulatory_basis",
        question="说明产品综合溢价",
        search_queries=(QuerySearchQuery("产品综合溢价", "semantic_question"),),
        evidence_need="产品综合溢价依据",
        keywords=("并不逐字出现的长问题",),
    )
    chunk = RetrievalResult(
        chunk_id="DOC-TEST-CHUNK-0001",
        rank=1,
        score=0.9,
        source_doc="test.docx",
        section_title=None,
        section_path=[],
        text="原文直接支持该产品的综合溢价。",
        citation_label="[1]",
        metadata={"evidence_role": "bounded_lexical_support"},
    )

    assert _chunk_matches_query_aspect(chunk, aspect)


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


def test_challenge_100_is_gold_isolated_and_has_fixed_distribution() -> None:
    root = Path("data/evaluation/trust_challenge_100")
    questions = read_jsonl(root / "questions.jsonl")
    gold = read_jsonl(root / "gold.jsonl")
    assert len(questions) == len(gold) == 100
    assert [row["id"] for row in questions] == [row["id"] for row in gold]
    assert {row["difficulty"] for row in questions} == {"medium", "hard"}
    assert sum(row["difficulty"] == "medium" for row in questions) == 30
    assert sum(row["difficulty"] == "hard" for row in questions) == 70
    assert sum(row["split"] == "dev" for row in questions) == 40
    assert sum(row["split"] == "holdout" for row in questions) == 60
    assert all("canonical_answer" not in row and "evidence" not in row for row in questions)
    assert all(row["review_status"] == "codex_verified" and row["expert_reviewed"] is False for row in gold)


def test_hard_50_is_gold_isolated_and_independently_audited() -> None:
    root = Path("data/evaluation/hard_challenge_50/round_1")
    if not (root / "questions.jsonl").is_file():
        pytest.skip("optional internal hard-50 dataset is not included in the submission package")
    questions = read_jsonl(root / "questions.jsonl")
    gold = read_jsonl(root / "gold.jsonl")
    audit = __import__("json").loads((root / "independent_audit.json").read_text(encoding="utf-8"))

    assert len(questions) == len(gold) == 50
    assert [row["id"] for row in questions] == [row["id"] for row in gold]
    assert all(row["difficulty"] == "hard" for row in questions)
    assert sum(row["answerable"] is True for row in questions) == 44
    assert all("canonical_answer" not in row and "evidence" not in row for row in questions)
    assert audit["passed"] is True
    assert audit["expert_reviewed"] is False


def test_hard_50_gate_requires_accuracy_refusal_and_direct_citations() -> None:
    records = []
    for index in range(50):
        answerable = index < 44
        records.append({
            "case_id": f"HC{index + 1:03d}",
            "scenario": "test",
            "question_type": "fact" if answerable else "table_refusal",
            "answerable": answerable,
            "difficulty": "hard",
            "scoring_type": "text_fact" if answerable else "refusal",
            "answer_correct": True,
            "passed": True,
            "request_error": None,
            "http_status": 200,
            "elapsed_ms": 100.0,
            "source_hit": True,
            "aspect_citation_complete": True,
            "aspect_checks": ([{"matched": True}] if answerable else []),
            "invalid_citations": [],
            "mismatched_citations": [],
            "forbidden_conclusions_found": [],
            "critical_entity_errors": 0,
            "critical_entity_total": 1,
            "expected_document_ids": (["DOC-1"] if answerable else []),
        })

    passed = _summarize(records, [], "all", None, "hard50")
    assert passed["gate_passed"] is True
    assert passed["answerable_accuracy"] == 1.0
    assert passed["refusal_accuracy"] == 1.0
    assert passed["evidence_aspect_recall"] == 1.0

    records[0]["aspect_citation_complete"] = False
    failed = _summarize(records, [], "all", None, "hard50")
    assert failed["gate_passed"] is False


def test_challenge_scorer_handles_facts_decimals_and_negated_forbidden_phrases() -> None:
    fact = _fact_match(
        "消费金融公司是不吸收公众存款、服务境内居民个人的非银行金融机构。",
        "消费金融公司是不吸收公众存款、为中国境内居民个人提供消费贷款的非银行金融机构。",
    )
    assert fact["matched"] is True
    assert _decimal_value("9%")[:2] == (Decimal("9"), True)
    assert _asserted_forbidden("现有依据无法确认现行有效。", "确认现行有效") is False
    assert _asserted_forbidden("该文件确认现行有效。", "确认现行有效") is True


def test_document_identity_question_returns_explicit_unknown_without_legal_inference() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as db:
        document = _document("DOC-IDENTITY-1", "集中度风险指标统计表", "")
        document.filename = "387_保险集团集中度风险监管指引_附件1：集中度风险指标统计表.doc"
        document.filename_norm = document.filename.casefold()
        document.file_sha256 = "a" * 64
        document.version_status = "unknown"
        document.identity_review_status = "unreviewed"
        db.add(document)
        db.commit()
        response = answer_document_identity_question(
            db,
            "根据知识库中的文档身份信息，能否确认《附件1：集中度风险指标统计表》现行有效？",
        )
    assert response is not None
    assert response.refused is True
    assert response.refusal_code == "version_status_unknown"
    assert "未知" in str(response.answer)


def test_deterministic_table_unit_uses_metric_semantics_for_mixed_unit_tables() -> None:
    assert _deterministic_table_output_unit({"unit": "亿元、万件", "row_label": "原保险保费收入"}) == "亿元"
    assert _deterministic_table_output_unit({"unit": "亿元、万件", "row_label": "保单件数"}) == "万件"
    assert _deterministic_table_output_unit({"unit": "亿元、%", "row_label": "不良贷款率"}) == "%"


def test_source_title_filter_ignores_human_readable_file_format_suffix() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add(_document("DOC-1", "银行函证工作操作指引", "财政部"))
        db.commit()
        compiled = build_retrieval_metadata_filter(
            db,
            {"source_title": "银行函证工作操作指引（PDF）"},
            ("source_title",),
        )

    assert compiled.document_ids == ("DOC-1",)


def test_prompt_core_prefers_explicit_condition_match_over_generic_higher_score() -> None:
    aspect = QueryAspect(
        aspect_id="audit_requirement",
        question="请说明与“商业银行应每年对划分政策和程序开展内部审计”相关的规定。",
        search_queries=(QuerySearchQuery("内部审计", "keyword_anchor", "test"),),
        evidence_need="内部审计频率和留档要求",
        keywords=("商业银行", "内部审计", "留档备查"),
    )
    generic = RetrievalResult(
        chunk_id="GENERIC",
        rank=1,
        score=0.99,
        source_doc="rules.pdf",
        text="商业银行应建立内部审计制度并完善治理机制。",
        citation_label="[1]",
        metadata={},
    )
    exact = RetrievalResult(
        chunk_id="EXACT",
        rank=2,
        score=0.8,
        source_doc="rules.pdf",
        text="商业银行应每年对划分政策和程序开展内部审计，内部审计结果需留档备查。",
        citation_label="[2]",
        metadata={},
    )

    selected = _best_non_duplicate_candidate([generic, exact], [], aspect)

    assert selected is exact
