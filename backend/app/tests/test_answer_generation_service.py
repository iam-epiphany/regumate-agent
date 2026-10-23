import json
from dataclasses import replace

import pytest

from backend.app.core.rag_features import RagFeatures
from backend.app.schemas.qa import AnswerClaim, RetrievalResult
from backend.app.services import answer_generation_service
from backend.app.services.answer_generation_service import (
    GeneratedAnswer,
    generate_answer,
)
from backend.app.services.grounding_validation_service import validate_grounded_answer


class _SSEStream:
    def __init__(self, lines: list[bytes | str]) -> None:
        self.lines = lines

    def __iter__(self):
        return iter(self.lines)


def _chunk(
    *,
    metadata=None,
    text="监管报送期限为2025年12月31日。",
    citation_label="[1]",
    chunk_id="C1",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id, rank=1, score=0.9, source_doc="rules.xlsx",
        section_title="测试", section_path=["测试"], text=text,
        citation_label=citation_label, metadata=metadata or {},
    )










def test_table_answer_is_deterministic_and_matches_option() -> None:
    result = generate_answer(
        "数值是多少？",
        [_chunk(metadata={"dynamic_table_evidence": True, "value": "123.45", "cell": "C5"}, text="C5=123.45")],
        options=["12", "123.45", "456", "789"],
        has_sufficient_context=True,
    )

    assert result.refused is False
    assert result.answer_type == "table_deterministic"
    assert "答案为 123.45" in result.answer
    assert "答案为 B" not in result.answer
    assert result.claims[0].citation_ids == ["[1]"]


def test_table_answer_keeps_user_option_label_when_provided() -> None:
    result = generate_answer(
        "数值是多少？",
        [_chunk(metadata={"dynamic_table_evidence": True, "value": "123.45", "cell": "C5"}, text="C5=123.45")],
        options=["12", "123.45", "456", "789"],
        option_labels=["A", "B", "C", "D"],
        has_sufficient_context=True,
    )

    assert result.refused is False
    assert "答案为 B。123.45" in result.answer


def test_choice_conclusion_normalization_uses_extractor_delimiters() -> None:
    normalize = answer_generation_service._normalize_choice_conclusion
    assert normalize("正确选项为：D。D项说明[1]") == "答案为 D。D项说明[1]"
    assert normalize("答案为 D、新增保险金额。") == "答案为 D。新增保险金额。"
    assert normalize("答案为 D：新增保险金额。") == "答案为 D。新增保险金额。"
    assert normalize("答案为 D。新增保险金额。") == "答案为 D。新增保险金额。"
    assert normalize("无法确定正确选项：证据不足。") == "无法确定正确选项：证据不足。"


def test_generation_loop_retries_refusal_with_exact_support(monkeypatch) -> None:
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_ENABLED", True)
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", "test")
    statement = "鼓励商业银行制定专门的知识产权金融业务办法和管理制度"
    ctx = answer_generation_service.GenerationContext(
        question="下列哪项表述正确？",
        context_chunks=[_chunk(metadata={
            "evidence_role": "mcq_exact_support",
            "exact_support_statement": statement,
        })],
        options=["选项一", "选项二", "选项三", statement + "。"],
        option_labels=["A", "B", "C", "D"],
        llm_prompt=None,
        answer_mode="choice",
        required=[],
        requirements=None,
        table_findings=[],
        deterministic=None,
        cancellation_checker=None,
    )
    attempts: list[int] = []

    def draft(context, *, attempt, correction):
        attempts.append(attempt)
        if attempt == 0:
            return GeneratedAnswer(answer="无法确定", answer_type="refusal", generation_status="completed",
                                   refused=True, refusal_reason="model_refusal", claims=[])
        return GeneratedAnswer(
            answer=f"答案为 D。{statement}。[1]", answer_type="llm_grounded", generation_status="completed",
            claims=[AnswerClaim(text=f"答案为 D。{statement}。[1]", citation_ids=["[1]"])],
        )

    def verify(context, generated):
        return {"passed": True, "status": "passed", "task_complete": True,
                "citation_integrity": True, "facts_supported": True,
                "missing_requirements": [], "reasons": []}

    result = answer_generation_service.run_generation_loop(ctx, draft=draft, verify=verify)
    assert attempts == [0, 1]
    assert result.answer.startswith("答案为 D。")


def test_parse_generated_answer_normalizes_choice_in_production_text_mode() -> None:
    content = json.dumps({
        "refused": False,
        "refusal_reason": None,
        "claims": [{
            "text": "正确选项为：D。D项说明[1]",
            "citation_ids": ["[1]"],
            "role": "conclusion",
            "aspect_ids": [],
        }],
    }, ensure_ascii=False)
    generated = answer_generation_service._parse_generated_answer_content(
        content,
        question="下列哪项正确？",
        context_chunks=[_chunk()],
        options=["甲", "乙", "丙", "丁"],
        option_labels=["A", "B", "C", "D"],
        answer_mode="text",
    )
    assert generated.answer.startswith("答案为 D。")


def _support_context(statement: str, options: list[str]) -> answer_generation_service.GenerationContext:
    return answer_generation_service.GenerationContext(
        question="下列哪项表述正确？",
        context_chunks=[_chunk(metadata={
            "evidence_role": "mcq_exact_support",
            "exact_support_statement": statement,
        }, text=statement)],
        options=options,
        option_labels=["A", "B", "C", "D"],
        llm_prompt=None,
        answer_mode="text",
        required=[],
        requirements=None,
        table_findings=[],
        deterministic=None,
        cancellation_checker=None,
    )


def test_persistent_option_refusal_stays_unverified_refusal(monkeypatch) -> None:
    """A literal option match cannot certify itself after the model refuses."""

    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_ENABLED", True)
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", "test")
    statement = "鼓励商业银行制定专门的知识产权金融业务办法和管理制度"
    ctx = _support_context(statement, ["选项一", "选项二", "选项三", statement + "。"])
    attempts: list[int] = []

    def draft(context, *, attempt, correction):
        attempts.append(attempt)
        return GeneratedAnswer(answer="无法确定", answer_type="refusal", generation_status="completed",
                               refused=True, refusal_reason="model_refusal", claims=[])

    def verify(context, generated):
        raise AssertionError("a refused draft must not be certified by exact-support similarity")

    result = answer_generation_service.run_generation_loop(ctx, draft=draft, verify=verify)
    assert attempts == [0, 1]
    assert result.refused is True
    assert result.refusal_reason == "model_refusal"
    assert result.grounding_validation["status"] == "not_run"
    assert not result.grounding_validation["passed"]


def test_model_refusal_reason_is_rendered_as_clean_clarification(monkeypatch) -> None:
    """Internal refusal markers must never become the user-visible answer."""

    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_ENABLED", True)
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", "test")
    ctx = answer_generation_service.GenerationContext(
        question="2026年城商行备案额度上限是多少？",
        context_chunks=[_chunk()],
        options=[],
        option_labels=[],
        llm_prompt=None,
        answer_mode="text",
        required=[],
        requirements=None,
        table_findings=[],
        deterministic=None,
        cancellation_checker=None,
    )

    for reason in (
        "needed: 具体城商行名称",
        {"needed": True, "question": "请补充具体城商行名称。"},
    ):
        def draft(context, *, attempt, correction, reason=reason):
            return GeneratedAnswer(
                answer=None,
                answer_type="refusal",
                generation_status="completed",
                refused=True,
                refusal_reason=reason,
                claims=[],
            )

        result = answer_generation_service.run_generation_loop(
            ctx,
            draft=draft,
            verify=lambda *args, **kwargs: pytest.fail("a refusal must not be verified"),
        )

        assert "needed" not in result.answer
        assert "具体城商行名称" in result.answer
        assert result.answer_type == "clarification"


def test_table_calculation_result_not_in_options_refuses() -> None:
    result = answer_generation_service._deterministic_table_answer(
        [_chunk(metadata={"dynamic_table_evidence": True, "calculation_result": 42, "value": 42,
                          "selection_assumptions": ["期间标签未指定具体日期，按表内顺序取首项。"]})],
        ["1", "2", "3", "4"],
        ["A", "B", "C", "D"],
    )
    assert result is not None
    assert result.refused is True
    assert result.refusal_reason == "calculation_result_not_in_options"


def test_calculation_answer_shows_numeric_formula() -> None:
    result = generate_answer(
        "贷款余额比证券投资多多少万元？",
        [
            _chunk(
                metadata={
                    "dynamic_table_evidence": True,
                    "operation": "difference",
                    "calculation_formula": "CSV!B3 - CSV!B4",
                    "calculation_display_formula": "3500 - 1800 = 1700",
                    "calculation_result": 1700,
                    "calculation_cells": [
                        {"sheet_name": "CSV", "cell": "B3", "normalized_value": 3500},
                        {"sheet_name": "CSV", "cell": "B4", "normalized_value": 1800},
                    ],
                },
                text="表格计算证据：CSV!B3=3500；CSV!B4=1800；3500 - 1800 = 1700。",
            )
        ],
        has_sufficient_context=True,
    )

    assert result.refused is False
    assert result.answer_type == "table_deterministic"
    assert "3500 - 1800 = 1700" in result.answer


def test_calculation_trace_accepts_three_operand_reconciliation() -> None:
    metadata = {
        "operation": "difference",
        "calculation_formula": "总表!C5 - 人身险!C5 - 财产险!C6",
        "calculation_result": -0.18,
        "calculation_cells": [
            {"sheet_name": "总表", "cell": "C5", "normalized_value": 21745},
            {"sheet_name": "人身险", "cell": "C5", "normalized_value": 16590.18},
            {"sheet_name": "财产险", "cell": "C6", "normalized_value": 5155},
        ],
    }

    assert answer_generation_service._calculation_trace_valid(metadata) is True


def test_calculation_trace_accepts_scaled_percentage_ratio() -> None:
    metadata = {
        "operation": "ratio",
        "calculation_formula": "Sheet1!C5 / Sheet1!D5 * 100",
        "calculation_result": 25.0,
        "result_scale": 100,
        "calculation_cells": [
            {"sheet_name": "Sheet1", "cell": "C5", "normalized_value": 25.0},
            {"sheet_name": "Sheet1", "cell": "D5", "normalized_value": 100.0},
        ],
    }

    assert answer_generation_service._calculation_trace_valid(metadata) is True


def test_calculation_trace_recomputes_from_displayed_values() -> None:
    """trace 校验与计算同源：显示值字符串优先，避免 float 噪声导致误拒。"""
    metadata = {
        "operation": "difference",
        "calculation_formula": "表!G5 - 表!G38",
        "calculation_result": 169.578795,
        "calculation_cells": [
            {"sheet_name": "表", "cell": "G5", "value": "8234.537443", "normalized_value": 8234.537442863},
            {"sheet_name": "表", "cell": "G38", "value": "8064.958648", "normalized_value": 8064.958647601},
        ],
    }

    assert answer_generation_service._calculation_trace_valid(metadata) is True


def test_table_value_formatter_honors_requested_decimal_places() -> None:
    assert answer_generation_service._format_value(17.8159210005, decimal_places=2) == "17.82"


def test_word_pdf_formula_answer_calculates_when_variables_are_complete() -> None:
    result = generate_answer(
        "calculate K, A=2, B=3, C=5",
        [
            _chunk(
                metadata={
                    "contains_formula": True,
                    "formulas": [{"text": "K=(A+B)/C", "source_type": "omml", "order_index": 1}],
                },
                text="Formula context: [公式] K=(A+B)/C. A is exposure, B is buffer, C is coefficient.",
            )
        ],
        has_sufficient_context=True,
    )

    assert result.refused is False
    assert result.answer_type == "formula_deterministic"
    assert "计算结果为 1" in result.answer
    assert "代入计算式为 (2+3)/5 = 1" in result.answer
    assert result.claims[0].role == "calculation"


def test_word_formula_prefers_complete_visible_equation_over_fragmented_omml_metadata() -> None:
    result = generate_answer(
        "请计算 LGD*，LGD_s=0.4，E_s=60，E=100，H_e=0.2，LGD_u=0.6，E_u=40。",
        [
            _chunk(
                metadata={
                    "contains_formula": True,
                    "formulas": [
                        {"text": "LGD_s", "source_type": "omml", "order_index": 1},
                        {"text": "H_e", "source_type": "omml", "order_index": 2},
                    ],
                },
                text=(
                    "采用抵质押品时，调整后的违约损失率为：\n"
                    "[公式] LGD^*=LGD_s*(E_s)/(E*(1+H_e))+LGD_u*(E_u)/(E*(1+H_e))\n"
                    "其中各符号为监管公式变量。"
                ),
            )
        ],
        has_sufficient_context=True,
    )

    assert result.refused is False
    assert result.answer_type == "formula_deterministic"
    assert "计算结果为 0.4" in result.answer
    assert "LGD_s*(E_s)/(E*(1+H_e))" in result.answer


def test_word_formula_ranks_requested_variables_before_unrelated_earlier_formula() -> None:
    result = generate_answer(
        "请计算 LGD*，LGD_s=0.4，E_s=60，E=100，H_e=0.2，LGD_u=0.6，E_u=40。",
        [
            _chunk(
                chunk_id="WRONG",
                citation_label="[1]",
                metadata={"contains_formula": True},
                text="多种抵质押品公式：[公式] LGD^*=∑(LGD_i*E_i/E)。其中变量按抵质押品逐项计算。",
            ),
            _chunk(
                chunk_id="RIGHT",
                citation_label="[2]",
                metadata={"contains_formula": True},
                text=(
                    "单种抵质押品调整公式：[公式] "
                    "LGD^*=LGD_s*(E_s)/(E*(1+H_e))+LGD_u*(E_u)/(E*(1+H_e))。"
                    "其中各符号为监管公式变量。"
                ),
            ),
        ],
        has_sufficient_context=True,
    )

    assert result.refused is False
    assert "计算结果为 0.4" in result.answer
    assert result.claims[0].citation_ids == ["[2]"]
    assert result.answer.endswith("[2]")


def test_word_pdf_formula_answer_refuses_missing_variables() -> None:
    result = generate_answer(
        "calculate K, A=2, B=3",
        [
            _chunk(
                metadata={
                    "contains_formula": True,
                    "formulas": [{"text": "K=(A+B)/C", "source_type": "omml", "order_index": 1}],
                },
                text="Formula context: [公式] K=(A+B)/C. A is exposure, B is buffer, C is coefficient.",
            )
        ],
        has_sufficient_context=True,
    )

    assert result.refused is True
    assert result.answer_type == "formula_refusal"
    assert result.refusal_code == "missing_variables"
    assert result.missing_variables == ["C"]
    assert "缺少必要变量取值：C" in result.answer


def test_word_pdf_formula_answer_refuses_ambiguous_variables() -> None:
    result = generate_answer(
        "calculate K, A=2, A=4, B=3, C=5",
        [
            _chunk(
                metadata={
                    "contains_formula": True,
                    "formulas": [{"text": "K=(A+B)/C", "source_type": "pdf_text", "order_index": 1}],
                },
                text="Formula context: [公式] K=(A+B)/C. A is exposure, B is buffer, C is coefficient.",
            )
        ],
        has_sufficient_context=True,
    )

    assert result.refused is True
    assert result.refusal_code == "ambiguous_variables"
    assert result.ambiguous_variables == ["A"]


def test_word_pdf_formula_answer_refuses_unit_conflict() -> None:
    result = generate_answer(
        "calculate K, A=2元, A=2万元, B=3, C=5",
        [
            _chunk(
                metadata={
                    "contains_formula": True,
                    "formulas": [{"text": "K=(A+B)/C", "source_type": "omml", "order_index": 1}],
                },
                text="Formula context: [公式] K=(A+B)/C. A is exposure, B is buffer, C is coefficient.",
            )
        ],
        has_sufficient_context=True,
    )

    assert result.refused is True
    assert result.refusal_code == "unit_conflict"


def test_word_pdf_formula_answer_refuses_unsupported_operation() -> None:
    result = generate_answer(
        "calculate K, A=2",
        [
            _chunk(
                metadata={
                    "contains_formula": True,
                    "formulas": [{"text": "K=log(A)", "source_type": "omml", "order_index": 1}],
                },
                text="Formula context: [公式] K=log(A). A is exposure.",
            )
        ],
        has_sufficient_context=True,
    )

    assert result.refused is True
    assert result.refusal_code == "unsupported_operation"


def test_word_pdf_formula_answer_calculates_chinese_percent_variables() -> None:
    result = generate_answer(
        "请计算杠杆率，核心一级资本=900，核心一级资本扣除项=0，调整后表内外资产余额=10000。",
        [
            _chunk(
                metadata={
                    "contains_formula": True,
                    "formulas": [
                        {
                            "text": "杠杆率=(核心一级资本-核心一级资本扣除项)/(调整后表内外资产余额)*100%",
                            "source_type": "omml",
                            "order_index": 1,
                        }
                    ],
                },
                text=(
                    "资本监管要求中给出杠杆率计算公式：[公式] "
                    "杠杆率=(核心一级资本-核心一级资本扣除项)/(调整后表内外资产余额)*100%。"
                    "核心一级资本、核心一级资本扣除项和调整后表内外资产余额均为监管口径变量。"
                ),
            )
        ],
        has_sufficient_context=True,
    )

    assert result.refused is False
    assert result.answer_type == "formula_deterministic"
    assert "计算结果为 9%" in result.answer


def test_word_pdf_formula_answer_refuses_unsupported_max_before_missing_variables() -> None:
    result = generate_answer(
        "请计算K_CM_i，K_CCP=10。",
        [
            _chunk(
                metadata={
                    "contains_formula": True,
                    "formulas": [
                        {
                            "text": "K_CM_i=max(K_CCP*((DF_i^pref)/(DF_CCP+DF_CM^pref)),8%*2%*DF_i^pref)",
                            "source_type": "omml",
                            "order_index": 1,
                        }
                    ],
                },
                text=(
                    "合格中央交易对手风险暴露的资本要求公式为：[公式] "
                    "K_CM_i=max(K_CCP*((DF_i^pref)/(DF_CCP+DF_CM^pref)),8%*2%*DF_i^pref)。"
                    "K_CCP、DF_i、DF_CCP 和 DF_CM 为公式变量。"
                ),
            )
        ],
        has_sufficient_context=True,
    )

    assert result.refused is True
    assert result.refusal_code == "unsupported_operation"


def test_word_pdf_formula_answer_refuses_without_formula_context() -> None:
    result = generate_answer(
        "calculate K, A=2, B=3",
        [
            _chunk(
                metadata={
                    "contains_formula": True,
                    "formulas": [{"text": "K=A+B", "source_type": "omml", "order_index": 1}],
                },
                text="[公式] K=A+B",
            )
        ],
        has_sufficient_context=True,
    )

    assert result.refused is True
    assert result.refusal_code == "insufficient_context"


def test_sum_equality_answer_uses_comparison_target() -> None:
    result = generate_answer(
        "资产合计是否等于四个分项之和？",
        [
            _chunk(
                metadata={
                    "dynamic_table_evidence": True,
                    "operation": "sum",
                    "aspect_question": "资产合计是否等于四个分项之和？",
                    "calculation_formula": "CSV!B2 + CSV!B3 + CSV!B4 + CSV!B5",
                    "calculation_display_formula": "1200 + 3500 + 1800 + 500 = 7000",
                    "calculation_result": 7000,
                    "comparison_target": {"label": "资产合计", "value": "7000", "normalized_value": 7000},
                    "comparison_equal": True,
                    "calculation_cells": [
                        {"sheet_name": "CSV", "cell": "B2", "normalized_value": 1200},
                        {"sheet_name": "CSV", "cell": "B3", "normalized_value": 3500},
                        {"sheet_name": "CSV", "cell": "B4", "normalized_value": 1800},
                        {"sheet_name": "CSV", "cell": "B5", "normalized_value": 500},
                    ],
                },
                text="表格计算证据：1200 + 3500 + 1800 + 500 = 7000，资产合计=7000。",
            )
        ],
        has_sufficient_context=True,
    )

    assert result.refused is False
    assert "相等" in result.answer
    assert "1200 + 3500 + 1800 + 500 = 7000" in result.answer
























def test_generation_refuses_without_context() -> None:
    result = generate_answer("不存在的问题", [], has_sufficient_context=False)

    assert result.refused is True
    assert result.refusal_reason == "insufficient_context"






def test_multi_part_question_with_yixia_and_belongs_is_not_treated_as_choice(monkeypatch) -> None:
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", None)

    result = generate_answer(
        "请回答以下两个事项：甲公司属于哪类集团；乙公司应遵守什么要求？",
        [_chunk(text="甲公司属于保险控股型集团。乙公司应当按年度报告。")],
        has_sufficient_context=True,
    )

    assert result.answer_type != "clarification"
    assert result.refusal_reason != "missing_options_for_choice_question"




def test_descriptive_use_of_calculate_is_not_treated_as_formula_request() -> None:
    assert answer_generation_service._is_formula_request(
        "寿险合同负债评估中计算现金流现值所采用的折现率曲线是什么？"
    ) is False
    assert answer_generation_service._is_formula_request("请计算 K，A=2，B=3。") is True


















def test_grounding_uses_cited_document_metadata_for_scope_entities() -> None:
    result = validate_grounded_answer(
        "商业银行应开展有效沟通。[1]",
        [AnswerClaim(text="商业银行应开展有效沟通。", citation_ids=["[1]"])],
        [
            _chunk(
                text="在处置计划实施过程中，如何与监管部门、地方政府、股东、客户、员工和社会公众等开展有效沟通。",
                metadata={"source_title": "处置计划建议示例（商业银行版）"},
            )
        ],
    )

    assert result["passed"] is True
    assert result["unsupported_claim_entities"] == []


def test_grounding_rejects_unsupported_date() -> None:
    result = validate_grounded_answer(
        "期限为2099年1月1日。[1]",
        [AnswerClaim(text="期限为2099年1月1日", citation_ids=["[1]"])],
        [_chunk()],
    )

    assert result["passed"] is False
    assert "2099年1月1日" in result["unsupported_entities"]


def test_grounding_accepts_display_value_rounded_from_evidence() -> None:
    result = validate_grounded_answer(
        "答案为 C：275230.63，核验值为275230.6254。[1]",
        [AnswerClaim(text="答案为 C：275230.63", citation_ids=["[1]"])],
        [_chunk(text="表格单元格原始值：275230.6254")],
    )

    assert result["passed"] is True
    assert result["unsupported_entities"] == []


def test_grounding_rejects_unsupported_regulatory_organization() -> None:
    result = validate_grounded_answer(
        "该事项由国家金融监督管理总局审批。[1]",
        [AnswerClaim(text="该事项由国家金融监督管理总局审批", citation_ids=["[1]"])],
        [_chunk(text="该事项由财政部备案。")],
    )

    assert result["passed"] is False
    assert "国家金融监督管理总局" in result["unsupported_entities"]


def test_grounding_binds_claim_entities_to_its_own_citation() -> None:
    result = validate_grounded_answer(
        "该事项由国家金融监督管理总局审批。[1]",
        [AnswerClaim(text="该事项由国家金融监督管理总局审批", citation_ids=["[1]"])],
        [
            _chunk(text="商业银行应当建立内部控制制度。"),
            _chunk(
                text="该事项由国家金融监督管理总局审批。",
                citation_label="[2]",
                chunk_id="C2",
            ),
        ],
    )

    assert result["unsupported_entities"] == []
    assert result["passed"] is False
    assert result["unsupported_claim_entities"][0]["entities"] == ["国家金融监督管理总局"]


def test_grounding_rejects_nonempty_answer_without_claims() -> None:
    result = validate_grounded_answer("期限为2025年12月31日。[1]", [], [_chunk()])

    assert result["passed"] is False
    assert result["missing_claims"] is True


def test_grounding_rejects_invalid_inline_citation() -> None:
    result = validate_grounded_answer(
        "期限为2025年12月31日。[99]",
        [AnswerClaim(text="期限为2025年12月31日", citation_ids=["[1]"])],
        [_chunk()],
    )

    assert result["passed"] is False
    assert result["invalid_answer_citation_ids"] == ["[99]"]
    assert result["missing_inline_citation_ids"] == ["[1]"]


def test_grounding_does_not_treat_bracketed_document_year_as_citation() -> None:
    result = validate_grounded_answer(
        "依据财会[2009]15号，结论成立。[1]",
        [AnswerClaim(text="依据财会[2009]15号，结论成立", citation_ids=["[1]"])],
        [_chunk(text="依据财会[2009]15号，结论成立。")],
    )

    assert result["invalid_answer_citation_ids"] == []
    assert result["missing_inline_citation_ids"] == []
















































def test_api_unavailable_does_not_guess_choice_from_word_overlap(monkeypatch) -> None:
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", None)
    chunks = [
        _chunk(
            text=(
                "核心一级资本工具的合格标准\n"
                "（一）直接发行且实缴的。\n"
                "（二）按照相关会计准则，实缴资本的数额被列为权益。"
            ),
        )
    ]
    options = [
        "核心一级资本工具应当直接发行且实缴。",
        "消费金融公司是经国家金融监督管理总局批准设立、不吸收公众存款的非银行金融机构。",
        "消费贷款是消费金融公司向借款人发放的以消费为目的的贷款。",
        "消费金融公司名称中应当标明“消费金融”字样。",
    ]

    result = generate_answer(
        "检索《资本工具合格标准》后，以下哪一项与材料内容一致？",
        chunks,
        options=options,
        has_sufficient_context=True,
    )

    assert result.refused is True
    assert result.answer_type == "service_error"
    assert result.grounding_validation["status"] == "error"






def test_pre_generation_gate_refuses_when_context_misses_required_aspect(monkeypatch) -> None:
    chunks = [
        _chunk(
            chunk_id="C1",
            citation_label="[1]",
            text="事项一明确要求保留三年。",
            metadata={"aspect_id": "requirement_one", "prompt_matched_aspects": ["requirement_one"]},
        ),
    ]
    llm_calls = []

    def fail_llm(*args, **kwargs):
        llm_calls.append(args)
        raise AssertionError("LLM must not run when the pre-gate refuses")

    monkeypatch.setattr(answer_generation_service, "_call_llm", fail_llm)
    monkeypatch.setattr(answer_generation_service, "_deterministic_table_answer", lambda *a, **k: None)
    monkeypatch.setattr(
        answer_generation_service,
        "features",
        lambda: replace(RagFeatures(), partial_answer=False),
    )

    generated = generate_answer(
        "两个事项如何处理",
        chunks,
        has_sufficient_context=True,
        answer_mode="text",
        required_aspect_ids=["requirement_one", "requirement_two"],
    )

    assert generated.refused is True
    assert generated.refusal_reason == "missing_required_aspect"
    assert generated.generation_status == "skipped"
    assert generated.grounding_validation["missing_requirements"] == ["requirement_two"]
    assert not llm_calls


def test_pre_generation_gate_allows_partial_answer_when_one_aspect_is_covered(monkeypatch) -> None:
    chunks = [
        _chunk(
            chunk_id="C1",
            citation_label="[1]",
            text="披露期限为2025年12月31日。",
            metadata={"aspect_id": "requirement_one", "prompt_matched_aspects": ["requirement_one"]},
        ),
    ]

    def fake_llm(*args, **kwargs):
        return GeneratedAnswer(
            answer="披露期限为2025年12月31日。[1]",
            answer_type="llm_grounded",
            generation_status="completed",
            claims=[AnswerClaim(text="披露期限为2025年12月31日。[1]", citation_ids=["[1]"],
                                aspect_ids=["requirement_one"])],
        )

    monkeypatch.setattr(answer_generation_service, "_call_llm", fake_llm)
    monkeypatch.setattr(answer_generation_service, "_deterministic_table_answer", lambda *a, **k: None)
    monkeypatch.setattr(
        answer_generation_service,
        "verify_draft",
        lambda ctx, generated: {
            "status": "failed",
            "passed": False,
            "task_complete": False,
            "citation_integrity": True,
            "facts_supported": True,
            "missing_requirements": [],
            "failure_stage": "validation",
        },
    )
    monkeypatch.setattr(
        answer_generation_service,
        "features",
        lambda: replace(RagFeatures(), partial_answer=True),
    )

    generated = generate_answer(
        "披露期限是什么？",
        chunks,
        has_sufficient_context=False,
        answer_mode="text",
        required_aspect_ids=["requirement_one", "requirement_two"],
    )

    assert generated.refused is False
    assert generated.answer_type == "llm_grounded"
    assert generated.grounding_validation["partial_answer"] is True
    assert generated.grounding_validation["missing_requirements"] == ["requirement_two"]
    assert generated.grounding_validation["task_complete"] is False


def test_pre_generation_gate_requires_explicit_choice_evidence() -> None:
    chunks = [
        _chunk(
            chunk_id="C1",
            citation_label="[1]",
            text="选项证据材料。",
            metadata={"aspect_id": "multiple_choice_evidence", "prompt_matched_aspects": ["multiple_choice_evidence"]},
        ),
    ]

    missing = answer_generation_service._missing_aspects_in_context(
        chunks, ["multiple_choice_evidence", "option_evidence"]
    )

    # Choice evidence has the same required-evidence contract as text.
    assert missing == ["option_evidence"]




def test_content_failure_still_runs_llm_repair(monkeypatch) -> None:
    chunks = [
        _chunk(
            chunk_id="C1",
            citation_label="[1]",
            text="事项一明确要求保留三年。",
            metadata={"aspect_id": "requirement_one", "prompt_matched_aspects": ["requirement_one"]},
        ),
        _chunk(
            chunk_id="C2",
            citation_label="[2]",
            text="事项二明确禁止涉及无关第三人。",
            metadata={"aspect_id": "requirement_two", "prompt_matched_aspects": ["requirement_two"]},
        ),
    ]
    llm_calls = []

    def counting_llm(question, context_chunks, normalized_options, normalized_labels, **kwargs):
        llm_calls.append(kwargs.get("correction"))
        return answer_generation_service.GeneratedAnswer(
            answer="事项一明确要求保留三年。[1]",
            answer_type="llm_grounded",
            generation_status="completed",
            claims=[
                AnswerClaim(
                    text="事项一明确要求保留三年。[1]",
                    citation_ids=["[1]"],
                    role="conclusion",
                    aspect_ids=["requirement_one"],
                )
            ],
            refused=False,
        )

    monkeypatch.setattr(answer_generation_service, "_call_llm", counting_llm)
    monkeypatch.setattr(answer_generation_service, "_deterministic_table_answer", lambda *a, **k: None)

    generated = generate_answer(
        "事项一如何处理",
        chunks,
        has_sufficient_context=True,
        answer_mode="text",
        required_aspect_ids=["requirement_one", "requirement_two"],
    )

    # Incomplete answers receive at most one repair, then refusal.
    assert len(llm_calls) == 2
    assert generated.refused














def test_calculation_trace_disambiguates_same_cell_across_two_documents() -> None:
    metadata = {
        "calculation_result": 20.0,
        "calculation_formula": "月度表!C5 - 月度表!C5",
        "operation": "difference",
        "ordered_transition": True,
        "calculation_cells": [
            {"document_id": "DOC-OCT", "sheet_name": "月度表", "cell": "C5", "normalized_value": 100.0},
            {"document_id": "DOC-DEC", "sheet_name": "月度表", "cell": "C5", "normalized_value": 120.0},
        ],
    }

    assert answer_generation_service._calculation_trace_valid(metadata) is True




def test_missing_inline_citation_enters_single_repair_then_refusal(monkeypatch) -> None:
    calls = []

    def fake_call(*args, **kwargs):
        calls.append((args, kwargs))
        return answer_generation_service.GeneratedAnswer(
            answer="监管报送期限为2025年12月31日。",
            answer_type="text_grounded",
            generation_status="completed",
            claims=[AnswerClaim(text="监管报送期限为2025年12月31日。", citation_ids=["[1]"])],
        )

    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_ENABLED", True)
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", "test-key")
    monkeypatch.setattr(answer_generation_service, "_call_llm", fake_call)

    result = generate_answer("期限是什么？", [_chunk()], has_sufficient_context=True)

    assert len(calls) == 2
    assert result.refused is True
    assert result.grounding_validation["passed"] is False
    assert result.grounding_validation["repair_attempted"]


def test_call_llm_reuses_context_package_prompt_in_request_payload(monkeypatch) -> None:
    captured = {}
    response_content = json.dumps(
        {
            "refused": False,
            "refusal_reason": None,
            "claims": [
                {
                    "role": "conclusion",
                    "text": "监管报送期限为2025年12月31日。[1]",
                    "citation_ids": ["[1]"],
                }
            ],
        },
        ensure_ascii=False,
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": response_content}}]},
                ensure_ascii=False,
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(answer_generation_service.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", "test-key")

    llm_prompt = (
        "你是 ReguMate，一个面向银行业监管制度与统计报表的可信 RAG 问答助手。\n\n"
        "【用户问题】\n期限是什么？\n\n"
        "【检索到的知识片段】\n[1] 来源：rules.xlsx / 测试\n监管报送期限为2025年12月31日。"
    )

    result = answer_generation_service._call_llm(
        "期限是什么？",
        [_chunk()],
        [],
        [],
        llm_prompt=llm_prompt,
    )

    user_message = captured["payload"]["messages"][1]["content"]
    assert llm_prompt in user_message
    assert "【结构化生成补充信息】" in user_message
    assert result.answer == "监管报送期限为2025年12月31日。[1]"


def test_parse_keeps_claims_requiring_semantic_verifier(monkeypatch) -> None:
    """需要语义验证器的结论 claim 不被解析层清空（验证是批量后置的）。"""
    monkeypatch.setattr(
        answer_generation_service,
        "validate_grounded_answer",
        lambda answer, claims, chunks: {"passed": True},
    )
    content = json.dumps(
        {
            "refused": False,
            "refusal_reason": None,
            "claims": [
                {
                    "role": "conclusion",
                    "text": "无法生存触发事件指下列两种情形中的较早发生者。[1]",
                    "citation_ids": ["[1]"],
                },
                {
                    "role": "explanation",
                    "text": "依据附件1中的规定。[1]",
                    "citation_ids": ["[1]"],
                },
            ],
        },
        ensure_ascii=False,
    )
    generated = answer_generation_service._parse_generated_answer_content(
        content,
        question="无法生存触发事件包括哪些情形？",
        context_chunks=[_chunk()],
        options=[],
        option_labels=[],
        answer_mode="text",
    )
    assert generated.answer is not None
    assert len(generated.claims) == 2


def test_generation_prompt_requires_citing_all_related_files() -> None:
    messages = answer_generation_service.RAGPromptBuilder().build_generation_messages(
        "对比两个附件的定义是否一致。",
        [_chunk()],
    )
    user_message = messages[1]["content"]
    assert "必须引用每个相关文件" in user_message


def test_calculation_citation_text_joins_operand_chunk_labels() -> None:
    from types import SimpleNamespace

    metadata = {
        "calculation_cells": [
            {"chunk_id": "CH-A"},
            {"chunk_id": "CH-B"},
        ]
    }
    chunks = [
        SimpleNamespace(chunk_id="CH-A", citation_label="[1]"),
        SimpleNamespace(chunk_id="CH-B", citation_label="[2]"),
    ]
    assert answer_generation_service._calculation_citation_text(metadata, chunks) == "[1][2]"


def test_generation_prompt_requires_all_aspects_without_forcing_duplicate_claims() -> None:
    messages = answer_generation_service.RAGPromptBuilder().build_generation_messages(
        "请分别说明两个事项。",
        [_chunk()],
        required_aspect_ids=["aspect_a", "aspect_b"],
    )

    user_message = messages[1]["content"]
    assert "每个必要要求均须在claims中覆盖" in user_message
    assert "同一claim可以覆盖多个要求" in user_message
    assert "aspect_a, aspect_b" in user_message








def test_grounding_accepts_hyphenated_quarter_against_chinese_period_evidence() -> None:
    result = validate_grounded_answer(
        "2023-3季度资金运用余额为275230.6254亿元。[1]",
        [AnswerClaim(text="2023-3季度资金运用余额为275230.6254亿元。", citation_ids=["[1]"])],
        [_chunk(text="2023年3季度资金运用余额为275230.6254亿元。")],
    )
    assert result["passed"] is True
    assert "-3" not in result["unsupported_entities"]


def test_grounding_ignores_rendered_list_ordinals_but_checks_claim_numbers() -> None:
    accepted = validate_grounded_answer(
        "1. 监管报送期限为2025年12月31日。[1]",
        [AnswerClaim(text="监管报送期限为2025年12月31日。", citation_ids=["[1]"])],
        [_chunk()],
    )
    rejected = validate_grounded_answer(
        "- 监管报送期限为2099年1月1日。[1]",
        [AnswerClaim(text="监管报送期限为2099年1月1日。", citation_ids=["[1]"])],
        [_chunk()],
    )

    assert accepted["passed"] is True
    assert "1" not in accepted["unsupported_entities"]
    assert rejected["passed"] is False
    assert "2099年1月1日" in rejected["unsupported_entities"]


def test_grounding_ignores_rendered_ordinals_inside_multiline_claim() -> None:
    text = "1. 绩效薪酬延期支付期限一般不少于3年。[1]\n2. 不得对无关第三人催收。[2]"
    result = validate_grounded_answer(
        text,
        [AnswerClaim(text=text, citation_ids=["[1]", "[2]"])],
        [
            _chunk(
                chunk_id="C1",
                citation_label="[1]",
                text="绩效薪酬延期支付期限一般不少于3年。",
            ),
            _chunk(
                chunk_id="C2",
                citation_label="[2]",
                text="不得对与债务无关的第三人进行催收。",
            ),
        ],
    )

    assert result["passed"] is True
    assert result["unsupported_claim_entities"] == []


def test_grounding_normalizes_full_width_percent_and_spaced_date() -> None:
    result = validate_grounded_answer(
        "费率上限为35%，并于3月31日前报送。[1]",
        [AnswerClaim(text="费率上限为35%，并于3月31日前报送。", citation_ids=["[1]"])],
        [_chunk(text="费率上限为35％，并于3 月 31 日前报送。")],
    )

    assert result["passed"] is True
