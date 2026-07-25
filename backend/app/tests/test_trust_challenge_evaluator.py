from scripts import evaluate_trust_challenge


def test_multiple_choice_expected_label_accepts_correct_option_wording() -> None:
    correct, detail = evaluate_trust_challenge._score_multiple_choice(
        {"required_conclusions": ["正确选项为：B"]},
        "答案为 B、交易头寸包括自营、做市、对客及相关对冲。",
        False,
        [],
    )

    assert correct is True
    assert detail["expected_label"] == "B"
    assert detail["actual_label"] == "B"


def test_frequency_enumeration_allows_filler_omission_but_requires_every_item() -> None:
    conclusion = "商业银行应根据表格要求，分别按照季度、半年和年度的频率披露信息"

    complete = evaluate_trust_challenge._fact_match(
        "商业银行第三支柱信息披露频率为季度、半年和年度。",
        conclusion,
    )
    incomplete = evaluate_trust_challenge._fact_match(
        "商业银行第三支柱信息披露频率为季度和年度。",
        conclusion,
    )

    assert complete["matched"] is True
    assert complete["required_terms"] == ["季度", "半年", "年度"]
    assert incomplete["matched"] is False


def test_deadline_obligation_accepts_compressed_equivalent_wording() -> None:
    conclusion = "银行业金融机构应当自收到符合规定的询证函之日起10个工作日内，按照要求将回函直接回复会计师事务所"

    complete = evaluate_trust_challenge._fact_match(
        "银行收到合规询证函后一般应在10个工作日内直接回复会计师事务所。",
        conclusion,
    )
    incomplete = evaluate_trust_challenge._fact_match(
        "银行收到询证函后应尽快处理并回复客户。",
        conclusion,
    )

    assert complete["matched"] is True
    assert complete["compressed_deadline_alias"] is True
    assert incomplete["matched"] is False


def _numeric_alternate_source_case() -> dict:
    return {
        "id": "HC-ALT",
        "scenario": "跨表勾稽计算",
        "difficulty": "hard",
        "split": "test",
        "scoring_type": "numeric",
        "answerable": True,
        "canonical_answer": "差额为0.01亿元。",
        "required_conclusions": ["差额为0.01亿元。"],
        "forbidden_conclusions": [],
        "required_documents": [{"document_id": "DOC-LOCKED"}],
        "evidence": [
            {
                "document_id": "DOC-LOCKED",
                "chunk_id": "DOC-LOCKED-CHUNK-1",
                "evidence_text": "全国合计单元格 C4 的原始值为 45167.98，单位：亿元。",
                "cells": ["C4"],
            }
        ],
        "calculation": {
            "operands": {"保险业合计": "45167.98", "人身险": "31739.18", "财产险": "13428.79"},
            "unit": "亿元",
            "period": {"raw": "2023-10"},
            "result": "0.01",
        },
    }


def _numeric_alternate_source_result(*, year: int = 2023, month: int = 10) -> dict:
    return {
        "response": {
            "answer": "计算结果为0.01亿元。",
            "refused": False,
            "citations": [
                {
                    "document_id": "DOC-ACTUAL",
                    "chunk_id": "DOC-ACTUAL-CHUNK-1",
                    "filename": f"{year}年{month}月保险业经营情况表.xls",
                    "metadata": {
                        "period": {"year": year, "month": month, "raw": f"{year}年{month}月"},
                        "unit": "亿元、万件",
                        "calculation_cells": [
                            {
                                "document_id": "DOC-ACTUAL",
                                "chunk_id": "DOC-ACTUAL-CHUNK-1",
                                "sheet_name": "保险业经营数据（月度）",
                                "cell": "C5",
                                "value": "45167.98",
                                "unit": "亿元、万件",
                            }
                        ],
                    },
                }
            ],
        }
    }


def test_numeric_evaluator_accepts_traceable_same_period_equivalent_operand_source() -> None:
    scored = evaluate_trust_challenge._score_case(
        _numeric_alternate_source_case(),
        _numeric_alternate_source_result(),
        {"DOC-ACTUAL-CHUNK-1"},
        {"DOC-ACTUAL-CHUNK-1": "DOC-ACTUAL"},
        profile="hard50",
    )

    assert scored["answer_correct"] is True
    assert scored["source_hit"] is True
    assert scored["aspect_citation_complete"] is True
    assert scored["aspect_checks"][0]["match_method"] == "equivalent_numeric_operand_source"
    assert scored["aspect_checks"][0]["equivalent_source"]["cell"] == "C5"


def test_numeric_evaluator_rejects_equivalent_value_from_wrong_period() -> None:
    scored = evaluate_trust_challenge._score_case(
        _numeric_alternate_source_case(),
        _numeric_alternate_source_result(year=2024, month=10),
        {"DOC-ACTUAL-CHUNK-1"},
        {"DOC-ACTUAL-CHUNK-1": "DOC-ACTUAL"},
        profile="hard50",
    )

    assert scored["answer_correct"] is True
    assert scored["source_hit"] is False
    assert scored["aspect_citation_complete"] is False
    assert scored["aspect_checks"][0]["match_method"] is None


def test_cross_document_equivalent_passage_requires_same_long_locked_conclusion() -> None:
    conclusion = "数字化回函与纸质回函具有同等法律效力和证明力"
    evidence = {
        "document_id": "DOC-PDF",
        "evidence_text": f"（四）回函效力说明。{conclusion}。无论采取何种方式均应加强校验并对内容负责。",
    }
    matching = [
        {
            "document_id": "DOC-DOCX",
            "excerpt": f"操作指引明确：{conclusion}。银行应当加强内部稽核、校验并对真实性负责。",
        }
    ]
    unrelated = [
        {
            "document_id": "DOC-DOCX",
            "excerpt": "本材料仅讨论纸质询证函寄送流程、联系人填写和签章校验要求，不涉及回函法律效力。",
        }
    ]

    assert evaluate_trust_challenge._has_cross_document_equivalent_conclusion_citation(
        evidence,
        matching,
        required_conclusions=[conclusion],
    ) is True
    assert evaluate_trust_challenge._has_cross_document_equivalent_conclusion_citation(
        evidence,
        unrelated,
        required_conclusions=[conclusion],
    ) is False


def test_judgment_scoring_accepts_positive_restatement_after_required_facts() -> None:
    case = {
        "id": "HC-JUDGMENT",
        "scenario": "判断题",
        "difficulty": "hard",
        "split": "test",
        "scoring_type": "judgment",
        "required_conclusions": ["说法正确", "数字化回函与纸质回函具有同等法律效力和证明力"],
        "forbidden_conclusions": [],
        "required_documents": [],
        "evidence": [],
    }
    result = {
        "response": {
            "answer": "数字化回函与纸质回函具有同等法律效力和证明力。",
            "refused": False,
            "citations": [],
        }
    }

    scored = evaluate_trust_challenge._score_case(case, result, set(), {}, profile="hard50")

    assert scored["answer_correct"] is True
    checks = scored["correctness_detail"]["conclusion_checks"]
    assert checks[0]["judgment_restatement_alias"] is True


def test_judgment_scoring_rejects_restatement_when_required_fact_missing() -> None:
    case = {
        "id": "HC-JUDGMENT",
        "scenario": "判断题",
        "difficulty": "hard",
        "split": "test",
        "scoring_type": "judgment",
        "required_conclusions": ["说法正确", "银行业金融机构应当在10个工作日内直接回复会计师事务所"],
        "forbidden_conclusions": [],
        "required_documents": [],
        "evidence": [],
    }
    result = {
        "response": {
            "answer": "该材料讨论银行函证回函流程。",
            "refused": False,
            "citations": [],
        }
    }

    scored = evaluate_trust_challenge._score_case(case, result, set(), {}, profile="hard50")

    assert scored["answer_correct"] is False


def test_cross_document_equivalent_passage_accepts_short_exact_numeric_conclusion() -> None:
    conclusion = "终极利率暂定为4.5%"
    evidence = {
        "document_id": "DOC-PDF",
        "evidence_text": f"本附件规定基础利率曲线和综合溢价，其中{conclusion}，并说明过渡曲线采用二次插值方法计算得到。",
    }
    matching = [
        {
            "document_id": "DOC-DOCX",
            "excerpt": f"计算现金流现值所采用的折现率曲线由基础利率曲线加综合溢价形成；{conclusion}，相关过渡曲线按规则计算。",
        }
    ]

    assert evaluate_trust_challenge._has_cross_document_equivalent_conclusion_citation(
        evidence,
        matching,
        required_conclusions=[conclusion],
    ) is True


def _hard_profile_scored_case(case_id: str, *, answerable: bool = True, correct: bool = True) -> dict:
    return {
        "case_id": case_id,
        "scenario": "hard70",
        "question_type": "fact" if answerable else "refusal",
        "answerable": answerable,
        "difficulty": "hard",
        "split": "all",
        "scoring_type": "fact" if answerable else "refusal",
        "request_error": False,
        "http_status": 200,
        "elapsed_ms": 100.0,
        "answer_correct": correct,
        "source_hit": True,
        "aspect_citation_complete": True,
        "invalid_citations": [],
        "mismatched_citations": [],
        "forbidden_conclusions_found": [],
        "critical_entity_errors": 0,
        "critical_entity_total": 1,
        "aspect_checks": [{"matched": True}] if answerable else [],
        "expected_document_ids": ["DOC-1"] if answerable else [],
        "passed": correct,
    }


def test_hard70_profile_gate_requires_70_cases_and_20_refusals() -> None:
    scored = [
        _hard_profile_scored_case(f"HC70-A{i:03d}", answerable=True)
        for i in range(50)
    ] + [
        _hard_profile_scored_case(f"HC70-R{i:03d}", answerable=False)
        for i in range(20)
    ]

    summary = evaluate_trust_challenge._summarize(scored, [], "all", None, profile="hard70")

    assert summary["gate_passed"] is True
    assert summary["case_count"] == 70
    assert summary["answerable_accuracy"] == 1.0
    assert summary["refusal_accuracy"] == 1.0
    assert summary["refusal"]["total"] == 20
    assert summary["hard_profile_categories"] is not None
    assert summary["hard50_categories"] is None


def test_hard70_profile_rejects_wrong_refusal_count_even_when_accuracy_passes() -> None:
    scored = [
        _hard_profile_scored_case(f"HC70-A{i:03d}", answerable=True)
        for i in range(51)
    ] + [
        _hard_profile_scored_case(f"HC70-R{i:03d}", answerable=False)
        for i in range(19)
    ]

    summary = evaluate_trust_challenge._summarize(scored, [], "all", None, profile="hard70")

    assert summary["case_count"] == 70
    assert summary["accuracy"] == 1.0
    assert summary["refusal"]["total"] == 19
    assert summary["gate_passed"] is False


def test_hard70_structural_gate_requires_all_split(tmp_path) -> None:
    questions = [{"id": f"HC70-{i:03d}", "review_status": "codex_verified", "expert_reviewed": False} for i in range(70)]
    results = {row["id"]: {} for row in questions}

    assert evaluate_trust_challenge._structural_errors(
        questions,
        list(questions),
        results,
        "all",
        None,
        tmp_path / "gold.jsonl",
        tmp_path / "questions.jsonl",
        profile="hard70",
    ) == []

    errors = evaluate_trust_challenge._structural_errors(
        questions[:40],
        questions[:40],
        {row["id"]: {} for row in questions[:40]},
        "dev",
        None,
        tmp_path / "gold.jsonl",
        tmp_path / "questions.jsonl",
        profile="hard70",
    )

    assert "hard70 profile must be evaluated with --split all" in errors


def test_iterative100_profile_gate_requires_100_cases_and_20_refusals() -> None:
    scored = [
        _hard_profile_scored_case(f"I100-A{i:03d}", answerable=True)
        for i in range(80)
    ] + [
        _hard_profile_scored_case(f"I100-R{i:03d}", answerable=False)
        for i in range(20)
    ]

    summary = evaluate_trust_challenge._summarize(scored, [], "all", None, profile="iterative100")

    assert summary["gate_passed"] is True
    assert summary["case_count"] == 100
    assert summary["answerable_accuracy"] == 1.0
    assert summary["refusal_accuracy"] == 1.0
    assert summary["refusal"]["total"] == 20
    assert summary["hard_profile_categories"] is not None


def test_iterative100_profile_rejects_reusing_regression_sized_batches() -> None:
    scored = [
        _hard_profile_scored_case(f"I100-A{i:03d}", answerable=True)
        for i in range(44)
    ] + [
        _hard_profile_scored_case(f"I100-R{i:03d}", answerable=False)
        for i in range(6)
    ]

    summary = evaluate_trust_challenge._summarize(scored, [], "all", None, profile="iterative100")

    assert summary["case_count"] == 50
    assert summary["accuracy"] == 1.0
    assert summary["gate_passed"] is False


def test_failure_category_reports_common_root_cause() -> None:
    case = {
        "id": "I100-R01-001",
        "scenario": "迭代100题/跨制度文件推理",
        "question_type": "cross_document",
        "difficulty": "hard",
        "split": "all",
        "scoring_type": "multi_assertion",
        "answerable": True,
        "required_conclusions": ["应完整覆盖两个制度事实"],
        "forbidden_conclusions": [],
        "required_documents": [{"document_id": "DOC-A"}, {"document_id": "DOC-B"}],
        "evidence": [
            {"document_id": "DOC-A", "chunk_id": "CHUNK-A", "evidence_text": "制度事实A"},
            {"document_id": "DOC-B", "chunk_id": "CHUNK-B", "evidence_text": "制度事实B"},
        ],
    }
    result = {
        "response": {
            "answer": "只回答了制度事实A。",
            "refused": False,
            "citations": [{"document_id": "DOC-A", "chunk_id": "CHUNK-A"}],
        }
    }

    scored = evaluate_trust_challenge._score_case(
        case,
        result,
        {"CHUNK-A", "CHUNK-B"},
        {"CHUNK-A": "DOC-A", "CHUNK-B": "DOC-B"},
        profile="iterative100",
    )
    summary = evaluate_trust_challenge._summarize([scored], [], "all", None, profile="iterative100")

    assert scored["passed"] is False
    assert scored["failure_category"] == "retrieval_source_hit"
    assert summary["failure_categories"] == {"retrieval_source_hit": 1}
