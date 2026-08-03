import json

from backend.app.schemas.qa import AnswerClaim, RetrievalResult
from backend.app.services import answer_generation_service
from backend.app.services.answer_generation_service import generate_answer
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


def test_fallback_chunks_by_aspect_prefers_new_document_for_next_aspect() -> None:
    chunks = [
        _chunk(
            metadata={
                "document_id": "D1",
                "aspect_id": "aspect_1",
                "prompt_matched_aspects": ["aspect_1"],
                "aspect_question": "交易账簿头寸",
            },
            text="交易账簿头寸应能够每日进行公允价值计量。",
            citation_label="[1]",
            chunk_id="A1",
        ),
        _chunk(
            metadata={
                "document_id": "D1",
                "aspect_id": "aspect_2",
                "prompt_matched_aspects": ["aspect_2"],
                "aspect_question": "商业银行第三支柱信息披露",
            },
            text="交易账簿中的其他工具说明。",
            citation_label="[2]",
            chunk_id="A2",
        ),
        _chunk(
            metadata={
                "document_id": "D2",
                "aspect_id": "aspect_2",
                "prompt_matched_aspects": ["aspect_2"],
                "aspect_question": "商业银行第三支柱信息披露",
            },
            text="商业银行应按照监管并表范围披露相关信息，表格中另有规定的除外。",
            citation_label="[3]",
            chunk_id="B1",
        ),
    ]

    selected = answer_generation_service._fallback_chunks_by_aspect(
        chunks,
        question="请跨文件分别回答交易账簿头寸和商业银行第三支柱信息披露。",
        limit=4,
    )

    assert [chunk.chunk_id for chunk in selected[:2]] == ["A1", "B1"]


def test_fallback_prefers_exact_definition_over_generic_core_chunk() -> None:
    question = "“流动性资产储备”的完整定义是什么？"
    generic = _chunk(
        chunk_id="GENERIC",
        text="流动性资产储备包括固定收益类资产和权益类资产。",
        metadata={
            "aspect_id": "definition",
            "prompt_matched_aspects": ["definition"],
            "aspect_question": question,
            "prompt_selection_reason": "core",
            "evidence_role": "direct_evidence",
        },
    )
    exact = _chunk(
        chunk_id="EXACT",
        text=(
            "流动性资产储备，是指具备易于交易、易于变现及无变现障碍等特征，"
            "能够以合理成本快速变现并提供流动性的资产。"
        ),
        metadata={
            "aspect_id": "definition",
            "prompt_matched_aspects": ["definition"],
            "aspect_question": question,
            "prompt_selection_reason": "generic",
            "evidence_role": "bounded_lexical_support",
            "lexical_support_phrase": "流动性资产储备",
        },
    )

    selected = answer_generation_service._fallback_chunks_by_aspect(
        [generic, exact],
        question=question,
        limit=2,
    )

    assert selected[0].chunk_id == "EXACT"


def test_option_evidence_matrix_penalizes_unsupported_exclusive_and_negative_relations() -> None:
    chunks = [
        _chunk(
            text="重要实体承载本机构核心业务条线和关键功能，对持续经营、维持关键功能具有重要作用。",
            citation_label="[1]",
            chunk_id="C1",
        ),
        _chunk(
            text="可参考的处置工具包括机构自救、股东注资、引入战略投资者、处置不良资产、接管、收购承接、建立过桥机构、破产清算等。",
            citation_label="[2]",
            chunk_id="C2",
        ),
    ]

    matrix = answer_generation_service._option_evidence_matrix(
        [
            "重要实体与关键功能无关，工具仅限破产清算",
            "重要实体只承载非核心业务，禁止引入战略投资者",
            "重要实体承载核心业务条线和关键功能；工具可包括自救、注资、战略投资、不良资产处置、接管、收购承接、过桥机构和破产清算",
            "处置工具只能由股东注资",
        ],
        ["A", "B", "C", "D"],
        chunks,
    )

    scores = {item["label"]: item for item in matrix}
    assert scores["A"]["minimum_fact_coverage"] == 0.0
    assert scores["B"]["minimum_fact_coverage"] == 0.0
    assert scores["D"]["minimum_fact_coverage"] == 0.0
    assert scores["C"]["minimum_fact_coverage"] >= 0.65


def test_multiple_choice_uses_pre_generation_deterministic_recovery(monkeypatch) -> None:
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", "configured")

    def fail_call_llm(*args, **kwargs):
        raise AssertionError("LLM should not run when option evidence is already decisive")

    monkeypatch.setattr(answer_generation_service, "_call_llm", fail_call_llm)
    chunks = [
        _chunk(
            text="重要实体承载本机构核心业务条线和关键功能，对持续经营、维持关键功能具有重要作用。",
            citation_label="[1]",
            chunk_id="C1",
        ),
        _chunk(
            text="可参考的处置工具包括机构自救、股东注资、引入战略投资者、处置不良资产、接管、收购承接、建立过桥机构、破产清算等。",
            citation_label="[2]",
            chunk_id="C2",
        ),
    ]

    result = generate_answer(
        "关于重要实体与处置工具，下列哪项同时符合材料？",
        chunks,
        options=[
            "重要实体与关键功能无关，工具仅限破产清算",
            "重要实体只承载非核心业务，禁止引入战略投资者",
            "重要实体承载核心业务条线和关键功能；工具可包括自救、注资、战略投资、不良资产处置、接管、收购承接、建立过桥机构和破产清算",
            "处置工具只能由股东注资",
        ],
        option_labels=["A", "B", "C", "D"],
        has_sufficient_context=True,
    )

    assert result.refused is False
    assert result.answer_type == "choice_evidence_deterministic"
    assert "答案为 C" in result.answer
    cited = {
        citation_id
        for claim in result.claims
        for citation_id in claim.citation_ids
    }
    assert cited == {"[1]", "[2]"}


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
    assert "答案为 B、123.45" in result.answer


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


def test_mixed_sum_equality_answer_cites_table_and_regulation(monkeypatch) -> None:
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", None)

    result = generate_answer(
        "结合制度和月报，资产合计是否等于四个分项之和？",
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
                citation_label="[1]",
            ),
            _chunk(
                metadata={"evidence_role": "direct_evidence"},
                text="第四条 资产合计应当等于现金及存放同业、贷款余额、证券投资和其他资产四项金额之和。",
                citation_label="[2]",
                chunk_id="C2",
            ),
        ],
        has_sufficient_context=True,
        answer_mode="mixed",
    )

    assert result.refused is False
    assert result.answer_type == "mixed_table_deterministic"
    assert "相等" in result.answer
    assert "1200 + 3500 + 1800 + 500 = 7000" in result.answer
    assert "[1]" in result.answer and "[2]" in result.answer


def test_mixed_table_fallback_keeps_multiple_regulatory_aspects_and_relevant_tail(monkeypatch) -> None:
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", None)
    long_prefix = "背景说明。" * 80
    result = generate_answer(
        (
            "请联合核验制度文件与统计报表。\n"
            "制度侧：根据《示例制度》，请说明与“资金使用原则”、"
            "“沟通策略关注”相关的规定。\n"
            "报表侧：根据 Excel 附件《汇总表》取数。"
        ),
        [
            _chunk(
                metadata={"dynamic_table_evidence": True, "value": "123.45", "cell": "B2"},
                text="B2=123.45",
                citation_label="[1]",
            ),
            _chunk(
                metadata={"aspect_id": "regulatory_basis_1", "aspect_question": "资金使用原则"},
                text=f"{long_prefix}资金使用上应以自有资产或市场化渠道筹资自救为原则。",
                citation_label="[2]",
                chunk_id="C2",
            ),
            _chunk(
                metadata={"aspect_id": "regulatory_basis_2", "aspect_question": "沟通策略关注"},
                text="沟通策略应关注与监管部门、客户和员工开展有效沟通。",
                citation_label="[3]",
                chunk_id="C3",
            ),
        ],
        has_sufficient_context=True,
        answer_mode="mixed",
    )

    assert result.refused is False
    assert "资金使用上应以自有资产" in (result.answer or "")
    assert "沟通策略应关注" in (result.answer or "")
    assert "[2]" in (result.answer or "") and "[3]" in (result.answer or "")


def test_mixed_table_fallback_keeps_condition_continuation_chunk(monkeypatch) -> None:
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", None)

    result = generate_answer(
        (
            "请联合核验制度文件与统计报表。\n"
            "制度侧：根据《银行函证工作操作指引（PDF）》，请说明与“银行业金融机构公示信息”"
            "相关的明确规定。请完整给出材料明确写明的条件、范围、期限或例外。\n"
            "报表侧：根据 Excel 附件《汇总表》取数。"
        ),
        [
            _chunk(
                metadata={"dynamic_table_evidence": True, "value": "123.45", "cell": "B2"},
                text="B2=123.45",
                citation_label="[1]",
                chunk_id="T1",
            ),
            _chunk(
                metadata={
                    "document_id": "D1",
                    "aspect_id": "regulatory_basis",
                    "prompt_matched_aspects": ["regulatory_basis"],
                    "aspect_question": "银行业金融机构公示信息 条件 范围 期限 例外",
                },
                text=(
                    "银行业金融机构应当在公开渠道就办理函证相关事项进行公示。"
                    "公示信息包括：1.各种函证方式下办理回函工作的机构及其联系方式。"
                ),
                citation_label="[2]",
                chunk_id="C1",
            ),
            _chunk(
                metadata={
                    "document_id": "D1",
                    "aspect_id": "regulatory_basis",
                    "prompt_matched_aspects": ["regulatory_basis"],
                    "prompt_selection_reason": "condition_continuation",
                    "condition_preamble_chunk_id": "C1",
                    "aspect_question": "银行业金融机构公示信息 条件 范围 期限 例外",
                },
                text=(
                    "2.公示后，对于符合公示条件的申请，银行业金融机构不应拒绝。"
                    "3.函证范围和回函用章应当一并公示。"
                ),
                citation_label="[3]",
                chunk_id="C2",
            ),
        ],
        has_sufficient_context=True,
        answer_mode="mixed",
    )

    assert result.refused is False
    assert result.answer_type == "mixed_table_deterministic"
    assert "公示信息包括" in (result.answer or "")
    assert "函证范围和回函用章" in (result.answer or "")
    assert "[2]" in (result.answer or "") and "[3]" in (result.answer or "")


def test_mixed_table_fallback_does_not_let_one_regulatory_aspect_crowd_out_another(
    monkeypatch,
) -> None:
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", None)

    result = generate_answer(
        (
            "联合核验：说明大额风险暴露门槛和处置自救原则；"
            "再比较2025年9月人身险表中意外险、寿险和原保险保费收入，给出最大项、数值和单位。"
        ),
        [
            _chunk(
                metadata={
                    "dynamic_table_evidence": True,
                    "value": "38433.67",
                    "comparison_operation": "max",
                    "row_label": "原保险保费收入",
                    "unit": "亿元",
                    "aspect_id": "table_evidence",
                },
                text="原保险保费收入 38433.67 亿元",
                citation_label="[1]",
                chunk_id="T1",
            ),
            _chunk(
                metadata={
                    "document_id": "D-weak-1",
                    "aspect_id": "regulatory_basis_1",
                    "aspect_question": "说明大额风险暴露门槛",
                    "prompt_matched_aspects": ["regulatory_basis_1"],
                },
                text="小微企业风险暴露是指满足微型和小型企业认定标准等条件的风险暴露。",
                citation_label="[2]",
                chunk_id="W1",
            ),
            _chunk(
                metadata={
                    "document_id": "D-weak-2",
                    "aspect_id": "regulatory_basis_1",
                    "aspect_question": "说明大额风险暴露门槛",
                    "prompt_matched_aspects": ["regulatory_basis_1"],
                },
                text="中小企业风险暴露的有效期限可以采用2.5年。",
                citation_label="[3]",
                chunk_id="W2",
            ),
            _chunk(
                metadata={
                    "document_id": "D-weak-3",
                    "aspect_id": "regulatory_basis_1",
                    "aspect_question": "说明大额风险暴露门槛",
                    "prompt_matched_aspects": ["regulatory_basis_1"],
                },
                text="商业银行应当将大额风险暴露作为集中度风险的评估内容之一。",
                citation_label="[4]",
                chunk_id="W3",
            ),
            _chunk(
                metadata={
                    "document_id": "D-resolution",
                    "aspect_id": "regulatory_basis_1",
                    "aspect_question": "说明大额风险暴露门槛",
                    "prompt_matched_aspects": ["regulatory_basis_1"],
                    "exact_support_score": 1.0,
                },
                text="大额风险暴露是指商业银行对单一客户或一组关联客户超过其一级资本净额2.5%的风险暴露。",
                citation_label="[5]",
                chunk_id="R1",
            ),
            _chunk(
                metadata={
                    "document_id": "D-resolution",
                    "aspect_id": "regulatory_basis_2",
                    "aspect_question": "说明处置自救原则",
                    "prompt_matched_aspects": ["regulatory_basis_2"],
                    "exact_support_score": 1.0,
                },
                text="处置策略建议中可以选择应用多种处置工具，应坚持自救为本的基本原则。",
                citation_label="[6]",
                chunk_id="R2",
            ),
        ],
        has_sufficient_context=True,
        answer_mode="mixed",
    )

    assert result.refused is False
    assert result.answer_type == "mixed_table_deterministic"
    assert "超过其一级资本净额2.5%" in (result.answer or "")
    assert "应坚持自救为本的基本原则" in (result.answer or "")
    assert "[5]" in (result.answer or "") and "[6]" in (result.answer or "")
    assert "regulatory_basis_2" in result.grounding_validation.get("covered_required_aspect_ids", []) or any(
        "regulatory_basis_2" in claim.aspect_ids for claim in result.claims
    )


def test_mixed_table_answer_uses_actual_table_citation_when_not_first(monkeypatch) -> None:
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", None)
    result = generate_answer(
        (
            "制度侧：说明资本约束要求。\n"
            "报表侧：根据 Excel 附件《汇总表》比较并找出最高值。\n"
            "证据边界：能否认定单一机构合规？"
        ),
        [
            _chunk(
                metadata={"aspect_id": "regulatory_basis"},
                text="商业银行应当遵守资本约束要求。",
                citation_label="[1]",
                chunk_id="C1",
            ),
            _chunk(
                metadata={
                    "dynamic_table_evidence": True,
                    "value": "123.45",
                    "comparison_operation": "max",
                    "row_label": "银行业金融机构合计",
                    "aspect_id": "table_evidence",
                },
                text="银行业金融机构合计=123.45",
                citation_label="[2]",
                chunk_id="C2",
            ),
        ],
        has_sufficient_context=True,
        answer_mode="mixed",
    )

    assert result.refused is False
    assert "结果为 123.45" in (result.answer or "")
    assert result.claims[0].citation_ids == ["[2]"]
    assert "[2]" in (result.answer or "")


def test_mixed_table_answer_excludes_spreadsheet_rows_from_regulatory_basis(monkeypatch) -> None:
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", None)
    result = generate_answer(
        (
            "制度侧：根据《银行函证工作操作指引》，说明银行业金融机构公示信息包括哪些内容。\n"
            "报表侧：根据 Excel 附件《汇总表》取数。"
        ),
        [
            _chunk(
                metadata={"dynamic_table_evidence": True, "value": "123.45", "cell": "B2"},
                text="B2=123.45",
                citation_label="[1]",
                chunk_id="T1",
            ),
            _chunk(
                metadata={
                    "spreadsheet_table": True,
                    "table_chunk_role": "row",
                    "chunk_type": "table",
                    "aspect_id": "table_context",
                },
                text="银行业金融机构公示信息 统计值 999.00 亿元",
                citation_label="[2]",
                chunk_id="T2",
            ),
            _chunk(
                metadata={
                    "aspect_id": "regulatory_basis",
                    "aspect_question": "银行业金融机构公示信息包括哪些内容",
                    "prompt_matched_aspects": ["regulatory_basis"],
                },
                text="银行业金融机构公示信息包括函证范围和回函用章。",
                citation_label="[3]",
                chunk_id="R1",
            ),
        ],
        has_sufficient_context=True,
        answer_mode="mixed",
    )

    assert result.refused is False
    assert result.answer_type == "mixed_table_deterministic"
    assert "函证范围和回函用章" in (result.answer or "")
    assert "[3]" in (result.answer or "")
    assert "[2]" not in (result.answer or "")
    assert all("[2]" not in claim.citation_ids for claim in result.claims)


def test_mixed_table_answer_prefers_question_anchor_hit_over_stale_aspect_question(monkeypatch) -> None:
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", None)
    result = generate_answer(
        (
            "制度侧：根据《资本工具合格标准》，请说明与“其他一级资本工具没有到期日，并且”"
            "相关的明确规定。\n"
            "报表侧：根据 Excel 附件《汇总表》取数。"
        ),
        [
            _chunk(
                metadata={"dynamic_table_evidence": True, "value": "8017.85054", "cell": "G5"},
                text="G5=8017.85054",
                citation_label="[1]",
                chunk_id="T1",
            ),
            _chunk(
                metadata={
                    "aspect_id": "regulatory_basis",
                    "aspect_question": "所有其他一级资本工具吸收损失",
                    "prompt_matched_aspects": ["regulatory_basis"],
                },
                text="（十二）所有其他一级资本工具全部吸收损失后，再启动二级资本工具吸收损失。",
                citation_label="[2]",
                chunk_id="R-stale",
            ),
            _chunk(
                metadata={
                    "aspect_id": "regulatory_basis",
                    "aspect_question": "所有其他一级资本工具吸收损失",
                    "prompt_matched_aspects": ["regulatory_basis"],
                },
                text="（四）没有到期日，并且不得含有利率跳升机制及其他赎回激励。",
                citation_label="[3]",
                chunk_id="R-anchor",
            ),
        ],
        has_sufficient_context=True,
        answer_mode="mixed",
    )

    assert result.refused is False
    assert "没有到期日" in (result.answer or "")
    assert "不得含有利率跳升机制及其他赎回激励" in (result.answer or "")
    assert "[3]" in (result.answer or "")


def test_mixed_table_answer_uses_question_anchor_for_long_chunk_excerpt(monkeypatch) -> None:
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", None)
    long_formula_text = "终极利率过渡曲线采用二次插值方法计算得到。rt为第一次插值数值。" * 20
    result = generate_answer(
        (
            "制度侧：根据《寿险合同负债评估折现率曲线》，请说明与"
            "“万能险、投资连结险、变额年金及中短存续期产品适用30BP综合溢价”相关的明确规定。\n"
            "报表侧：根据 Excel 附件《汇总表》取数。"
        ),
        [
            _chunk(
                metadata={"dynamic_table_evidence": True, "value": "45167.98", "cell": "C4"},
                text="C4=45167.98",
                citation_label="[1]",
                chunk_id="T1",
            ),
            _chunk(
                metadata={
                    "aspect_id": "regulatory_basis",
                    "aspect_question": "终极利率过渡曲线第一次插值",
                    "prompt_matched_aspects": ["regulatory_basis"],
                },
                text=(
                    f"{long_formula_text}"
                    "三、综合溢价按以下规则确定：一是高利率保单适用75BP溢价；"
                    "二是万能险、投资连结险、变额年金及中短存续期产品适用30BP综合溢价；"
                    "三是其他产品适用45BP溢价。"
                ),
                citation_label="[2]",
                chunk_id="R1",
            ),
        ],
        has_sufficient_context=True,
        answer_mode="mixed",
    )

    assert result.refused is False
    assert "万能险、投资连结险、变额年金及中短存续期产品适用30BP综合溢价" in (result.answer or "")
    assert "[2]" in (result.answer or "")


def test_extractive_fallback_uses_question_anchor_for_long_chunk_excerpt() -> None:
    long_intro = "保险公司经营意外险业务应遵守信息披露要求。" * 20
    result = answer_generation_service._extractive_fallback(
        [
            _chunk(
                metadata={
                    "aspect_id": "accident_insurance_requirement_2",
                    "aspect_question": "保险公司不得向特定团体成员以外的个人销售团体意外险",
                    "prompt_matched_aspects": ["accident_insurance_requirement_2"],
                },
                text=(
                    f"{long_intro}"
                    "第十九条 保险公司应为客户提供电话、互联网、微信或其他方式的意外险保单信息实时查询服务，"
                    "并在保险合同生效后告知投保人查询途径。"
                    "保单信息查询服务应至少保留至保险责任结束后三个月。"
                ),
                citation_label="[1]",
                chunk_id="R1",
            )
        ],
        question="请说明与“保单信息查询服务应至少保留至保险责任结束后三个月”相关的明确规定。",
    )

    assert "保单信息查询服务应至少保留至保险责任结束后三个月" in (result.answer or "")


def test_mixed_table_fallback_extracts_two_conditions_from_one_shared_chunk(monkeypatch) -> None:
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", None)
    result = generate_answer(
        (
            "制度侧：根据《账簿规则》，请说明与“不得以市场事件为由转换账簿”、"
            "“除非产品性质变化否则转换不可撤销”相关的规定。\n"
            "报表侧：根据 Excel 附件《汇总表》取数。\n"
            "证据边界：能否认定单一机构合规？"
        ),
        [
            _chunk(
                metadata={"dynamic_table_evidence": True, "value": "10", "cell": "B2"},
                text="B2=10",
                citation_label="[1]",
            ),
            _chunk(
                metadata={
                    "aspect_id": "regulatory_basis_1",
                    "prompt_matched_aspects": ["regulatory_basis_1", "regulatory_basis_2"],
                },
                text=(
                    "商业银行不得以市场事件为由转换账簿。"
                    "除非产品性质变化，否则账簿转换不可撤销。"
                    "其他背景说明。"
                ),
                citation_label="[2]",
                chunk_id="C2",
            ),
        ],
        has_sufficient_context=True,
        answer_mode="mixed",
    )

    assert "不得以市场事件为由转换账簿" in (result.answer or "")
    assert "除非产品性质变化" in (result.answer or "")
    assert (result.answer or "").count("[2]") == 1


def test_mixed_sum_equality_recovers_when_llm_validation_fails(monkeypatch) -> None:
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", "test-key")

    def fake_call_llm(*args, **kwargs):
        return answer_generation_service.GeneratedAnswer(
            answer="2026年1月资产合计符合制度要求。[1][2]",
            answer_type="llm",
            generation_status="completed",
            claims=[
                AnswerClaim(
                    text="2026年1月资产合计符合制度要求",
                    citation_ids=["[1]", "[2]"],
                    role="conclusion",
                )
            ],
        )

    monkeypatch.setattr(answer_generation_service, "_call_llm", fake_call_llm)
    monkeypatch.setattr(
        answer_generation_service,
        "_validate_generated_answer",
        lambda *args, **kwargs: {"passed": False, "reason": "unsupported_llm_claim"},
    )
    monkeypatch.setattr(answer_generation_service, "_repair_inline_citation_format", lambda *args, **kwargs: False)

    result = generate_answer(
        "结合制度和月报，资产合计是否等于四个分项之和？",
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
                citation_label="[1]",
            ),
            _chunk(
                metadata={"evidence_role": "direct_evidence"},
                text="第四条 资产合计应当等于现金及存放同业、贷款余额、证券投资和其他资产四项金额之和。",
                citation_label="[2]",
                chunk_id="C2",
            ),
        ],
        has_sufficient_context=True,
        answer_mode="mixed",
    )

    assert result.refused is False
    assert result.answer_type == "mixed_table_deterministic"
    assert result.grounding_validation["recovery_method"] == "deterministic_mixed_after_validation_failed"
    assert "相等" in result.answer
    assert "1200 + 3500 + 1800 + 500 = 7000" in result.answer
    assert "[1]" in result.answer and "[2]" in result.answer


def test_generation_refuses_without_context() -> None:
    result = generate_answer("不存在的问题", [], has_sufficient_context=False)

    assert result.refused is True
    assert result.refusal_reason == "insufficient_context"


def test_choice_question_without_options_uses_open_answer_path() -> None:
    result = generate_answer(
        "关于《账簿划分和名词解释》，下列哪一组选项中的两项表述均属于该材料内容？",
        [_chunk(text="交易账簿包括为交易目的而持有的金融工具。")],
        has_sufficient_context=True,
    )

    assert result.refused is False
    assert result.answer_type == "extractive_fallback"
    assert result.generation_status != "skipped"
    assert result.refusal_reason is None
    return
    assert "请补充选项" in result.answer


def test_normal_question_without_options_is_not_treated_as_choice_question(monkeypatch) -> None:
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", None)

    result = generate_answer(
        "请概述《账簿划分和名词解释》的主要内容。",
        [_chunk(text="交易账簿包括为交易目的而持有的金融工具。")],
        has_sufficient_context=True,
    )

    assert result.refused is False
    assert result.answer_type == "extractive_fallback"


def test_multi_part_question_with_yixia_and_belongs_is_not_treated_as_choice(monkeypatch) -> None:
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", None)

    result = generate_answer(
        "请回答以下两个事项：甲公司属于哪类集团；乙公司应遵守什么要求？",
        [_chunk(text="甲公司属于保险控股型集团。乙公司应当按年度报告。")],
        has_sufficient_context=True,
    )

    assert result.answer_type != "clarification"
    assert result.refusal_reason != "missing_options_for_choice_question"


def test_non_formula_question_is_not_hijacked_by_formula_candidate(monkeypatch) -> None:
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", None)

    result = generate_answer(
        "请说明交易账簿中的金融工具应如何计量。",
        [
            _chunk(
                text="交易账簿中的金融工具原则上应每日进行公允价值计量，变动计入损益。",
                metadata={"contains_formula": True, "formulas": ["R=a+b"]},
            )
        ],
        has_sufficient_context=True,
    )

    assert result.answer_type == "extractive_fallback"
    assert result.refused is False


def test_descriptive_use_of_calculate_is_not_treated_as_formula_request() -> None:
    assert answer_generation_service._is_formula_request(
        "寿险合同负债评估中计算现金流现值所采用的折现率曲线是什么？"
    ) is False
    assert answer_generation_service._is_formula_request("请计算 K，A=2，B=3。") is True


def test_extractive_fallback_selects_query_relevant_sentence_not_chunk_prefix(monkeypatch) -> None:
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", None)
    text = (
        "第一条 本办法规定一般事项。第二条 商业银行应建立治理机制。"
        "第三条 其他说明不影响前述安排。"
        "第四条 商业银行应每年对划分政策和程序开展内部审计，内部审计结果需留档备查。"
    ) * 4

    result = generate_answer(
        "请说明与“商业银行应每年对划分政策和程序开展内部审计”相关的规定。",
        [_chunk(text=text)],
        has_sufficient_context=True,
    )

    assert result.refused is False
    assert "每年对划分政策和程序开展内部审计" in result.answer


def test_relevant_excerpt_removes_unrelated_negative_clause_from_short_chunk() -> None:
    text = (
        "第八条 服务机构不能以任何方式干预客户选择。"
        "第九条 服务机构发现重大违规行为时，应当及时向监督部门报告。"
    )

    excerpt = answer_generation_service._relevant_extractive_excerpt(
        text,
        "根据规定，说明“发现重大违规行为时，应当及时向监督部门报告”的要求。",
    )

    assert "应当及时向监督部门报告" in excerpt
    assert "不能以任何方式干预" not in excerpt


def test_relevant_excerpt_keeps_list_continuation_after_anchored_heading() -> None:
    text = (
        "上一项指标不得连续小于零。"
        "三、报送要求："
        "（一）机构应当报送基础信息表；（二）机构应当报送风险明细表。"
    )

    excerpt = answer_generation_service._relevant_extractive_excerpt(
        text,
        "“报送要求”包括哪些报表？",
    )

    assert "基础信息表" in excerpt
    assert "风险明细表" in excerpt
    assert "不得连续小于零" not in excerpt


def test_relevant_excerpt_trims_negative_clause_before_anchor_in_same_segment() -> None:
    text = (
        "指标不得连续低于零。\n\n"
        "四、报送要求\n\n"
        "机构应当报送基础信息表和风险明细表。"
    )

    excerpt = answer_generation_service._relevant_extractive_excerpt(
        text,
        "说明与“四、报送要求机构”对应的规定。",
    )

    assert "应当报送基础信息表" in excerpt
    assert "不得连续低于零" not in excerpt


def test_extractive_fallback_creates_atomic_claims_for_multi_sentence_excerpt(monkeypatch) -> None:
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", None)
    result = answer_generation_service._extractive_fallback(
        [
            _chunk(
                text=(
                    "机构应至少每半年披露一次关键指标。"
                    "机构应通过官方网站公开披露，披露前应报送监督部门。"
                ),
                citation_label="[1]",
            )
        ],
        question="机构应如何披露关键指标？",
    )

    assert len(result.claims) == 2
    assert all(claim.citation_ids == ["[1]"] for claim in result.claims)
    assert result.grounding_validation["passed"] is True


def test_fallback_relevance_uses_section_heading_with_clause_prefix() -> None:
    aspect_question = "根据《附件甲》，说明与“二、披露要求 （一）机构应”对应的规定。"
    weak = _chunk(
        text="机构应遵循一般管理要求。",
        metadata={"evidence_role": "bounded_lexical_support"},
        chunk_id="weak",
    )
    strong = _chunk(
        text="（一）机构应至少每半年披露一次关键指标。",
        metadata={"evidence_role": "expanded_context"},
        chunk_id="strong",
    )
    strong.section_title = "二、披露要求"

    assert answer_generation_service._fallback_chunk_relevance(
        strong,
        aspect_question,
    ) > answer_generation_service._fallback_chunk_relevance(weak, aspect_question)


def test_excerpt_question_prefers_chunk_specific_aspect_over_generic_global_anchor() -> None:
    chunk = _chunk(
        text="机构不得变更计算方法。",
        metadata={
            "aspect_id": "method_change",
            "retrieval_aspect_id": "method_change",
            "aspect_question": "说明“机构不得变更计算方法”的要求。",
            "prompt_aspect_questions": {
                "method_change": "说明“机构不得变更计算方法”的要求。",
            },
        },
    )

    assert answer_generation_service._fallback_effective_excerpt_question(
        chunk,
        "分别说明“商业银行不得变更方法”和“商业银行应披露报告”的要求。",
    ) == "说明“机构不得变更计算方法”的要求。"


def test_relevant_excerpt_ignores_generic_anchor_contained_in_specific_anchor() -> None:
    excerpt = answer_generation_service._relevant_extractive_excerpt(
        (
            "机构应按一般规则计算风险暴露。"
            "未经监督部门认可，机构不得变更计算方法。"
        ),
        "说明“未经监督部门认可，机构”的明确要求。",
    )

    assert "不得变更计算方法" in excerpt
    assert "一般规则计算风险暴露" not in excerpt


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


def test_evidence_boundary_detection_covers_common_compliance_wording() -> None:
    assert answer_generation_service._requires_evidence_boundary(
        "最后说明能否据此认定单一机构合规。"
    )
    assert answer_generation_service._requires_evidence_boundary(
        "最后判断这些材料能否直接证明某一家机构合规。"
    )
    assert answer_generation_service._requires_evidence_boundary(
        "不得把行业汇总直接用于单一机构结论。"
    )
    assert answer_generation_service._requires_evidence_boundary(
        "行业汇总不得外推到单一机构。"
    )


def test_extractive_fallback_keeps_one_best_chunk_per_planned_aspect(monkeypatch) -> None:
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", None)
    result = answer_generation_service._extractive_fallback(
        [
            _chunk(
                text="薪酬递延期限一般不少于3年。",
                citation_label="[1]",
                metadata={"aspect_id": "salary", "aspect_question": "薪酬递延期限"},
            ),
            _chunk(
                text="与本问题无关的股东资格要求。",
                citation_label="[2]",
                metadata={"aspect_id": "salary", "aspect_question": "薪酬递延期限"},
            ),
            _chunk(
                text="不得对与债务无关的第三人进行催收。",
                citation_label="[3]",
                metadata={"aspect_id": "collection", "aspect_question": "催收禁止对象"},
            ),
        ],
        question="概括薪酬递延期限和催收禁止对象。",
    )

    assert "不少于3年" in result.answer
    assert "无关的第三人" in result.answer
    assert "股东资格" not in result.answer


def test_extractive_fallback_keeps_condition_continuation_chunk(monkeypatch) -> None:
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", None)

    result = answer_generation_service._extractive_fallback(
        [
            _chunk(
                metadata={
                    "document_id": "D1",
                    "aspect_id": "public_info",
                    "prompt_matched_aspects": ["public_info"],
                    "aspect_question": "银行业金融机构公示信息 条件 范围 期限 例外",
                },
                text=(
                    "银行业金融机构应当在公开渠道就办理函证相关事项进行公示。"
                    "公示信息包括：1.各种函证方式下办理回函工作的机构及其联系方式。"
                ),
                citation_label="[1]",
                chunk_id="C1",
            ),
            _chunk(
                metadata={
                    "document_id": "D1",
                    "aspect_id": "public_info",
                    "prompt_matched_aspects": ["public_info"],
                    "aspect_question": "银行业金融机构公示信息 条件 范围 期限 例外",
                },
                text=(
                    "2.公示后，对于符合公示条件的申请，银行业金融机构不应拒绝。"
                    "3.函证范围和回函用章应当一并公示。"
                ),
                citation_label="[2]",
                chunk_id="C2",
            ),
        ],
        question="请说明与“银行业金融机构公示信息”相关的条件、范围、期限或例外。",
    )

    assert result.answer_type == "extractive_fallback"
    assert "公示信息包括" in (result.answer or "")
    assert "函证范围和回函用章" in (result.answer or "")
    assert "[1]" in (result.answer or "") and "[2]" in (result.answer or "")


def test_extractive_fallback_does_not_add_generic_guess_after_core_evidence(monkeypatch) -> None:
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", None)

    result = answer_generation_service._extractive_fallback(
        [
            _chunk(
                metadata={
                    "document_id": "D1",
                    "aspect_id": "public_info",
                    "prompt_matched_aspects": ["public_info"],
                    "prompt_selection_reason": "core",
                    "evidence_role": "direct_evidence",
                    "aspect_question": "银行业金融机构公示信息 条件 范围 期限 例外",
                },
                text="银行业金融机构应当在公开渠道就办理函证相关事项进行公示。公示信息包括办理机构及联系方式。",
                citation_label="[1]",
                chunk_id="C1",
            ),
            _chunk(
                metadata={
                    "document_id": "D1",
                    "aspect_id": "public_info",
                    "prompt_matched_aspects": ["public_info"],
                    "condition_preamble_chunk_id": "C1",
                    "prompt_selection_reason": "condition_continuation",
                    "aspect_question": "银行业金融机构公示信息 条件 范围 期限 例外",
                },
                text="2.公示后，对于符合公示条件的申请，银行业金融机构不应拒绝。",
                citation_label="[2]",
                chunk_id="C2",
            ),
            _chunk(
                metadata={
                    "document_id": "D1",
                    "aspect_id": "public_info",
                    "prompt_matched_aspects": ["public_info"],
                    "prompt_selection_reason": "generic",
                    "evidence_role": "bounded_lexical_support",
                    "exact_support_anchor": "银行业金融机构定义",
                    "aspect_question": "银行业金融机构公示信息 条件 范围 期限 例外",
                },
                text="银行询证函格式由注册会计师填写，银行业金融机构根据本机构掌握的信息核对后回复。",
                citation_label="[3]",
                chunk_id="C3",
            ),
        ],
        question="请说明与“银行业金融机构公示信息”相关的条件、范围、期限或例外。",
    )

    assert "[1]" in (result.answer or "")
    assert "[2]" in (result.answer or "")
    assert "[3]" not in (result.answer or "")


def test_extractive_fallback_covers_distinct_quoted_anchors_in_same_aspect(monkeypatch) -> None:
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", None)

    result = answer_generation_service._extractive_fallback(
        [
            _chunk(
                metadata={"aspect_id": "trade", "aspect_question": "说明“以交易目的持有的头寸”和“每日进行公允价值计量”的规定。"},
                text="2.能够每日进行公允价值计量，变动计入损益；3.能够进行积极的管理。",
                citation_label="[1]",
                chunk_id="C1",
            ),
            _chunk(
                metadata={"aspect_id": "trade", "aspect_question": "说明“以交易目的持有的头寸”和“每日进行公允价值计量”的规定。"},
                text="（三）以下工具若满足前述条件，应划入交易账簿。",
                citation_label="[2]",
                chunk_id="C2",
            ),
            _chunk(
                metadata={"aspect_id": "trade", "aspect_question": "说明“以交易目的持有的头寸”和“每日进行公允价值计量”的规定。"},
                text="前款所称以交易目的持有的头寸是指短期内有目的地持有以便出售，或从价格波动中获利。",
                citation_label="[3]",
                chunk_id="C3",
            ),
        ],
        question="说明“以交易目的持有的头寸”和“每日进行公允价值计量”的规定。",
    )

    assert "每日进行公允价值计量" in (result.answer or "")
    assert "以交易目的持有的头寸是指" in (result.answer or "")
    assert "[1]" in (result.answer or "") and "[3]" in (result.answer or "")
    assert "[2]" not in (result.answer or "")


def test_extractive_fallback_covers_distinct_anchors_even_with_multiple_aspects(monkeypatch) -> None:
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", None)

    result = answer_generation_service._extractive_fallback(
        [
            _chunk(
                metadata={
                    "aspect_id": "trade",
                    "prompt_matched_aspects": ["trade"],
                    "aspect_question": "交易账簿头寸定义：“以交易目的持有的头寸”；公允价值计量条件：“能够每日进行公允价值计量，变动计入损益”",
                    "prompt_selection_reason": "anchor",
                    "evidence_role": "bounded_lexical_support",
                },
                text="2.能够每日进行公允价值计量，变动计入损益；3.能够进行积极的管理。",
                citation_label="[1]",
                chunk_id="C1",
            ),
            _chunk(
                metadata={
                    "aspect_id": "trade",
                    "prompt_matched_aspects": ["trade"],
                    "aspect_question": "交易账簿头寸定义：“以交易目的持有的头寸”；公允价值计量条件：“能够每日进行公允价值计量，变动计入损益”",
                    "prompt_selection_reason": "core",
                    "evidence_role": "bounded_lexical_support",
                },
                text="以交易目的持有的头寸，包括自营业务、做市业务、为满足客户需求提供的对客交易及对冲前述交易相关风险而持有的头寸。",
                citation_label="[2]",
                chunk_id="C2",
            ),
            _chunk(
                metadata={
                    "aspect_id": "disclosure",
                    "prompt_matched_aspects": ["disclosure"],
                    "aspect_question": "商业银行第三支柱信息披露频率和表格例外",
                    "prompt_selection_reason": "core",
                    "evidence_role": "bounded_lexical_support",
                },
                text="商业银行应根据表格要求，分别按照季度、半年和年度的频率披露信息。表格中另有规定的除外。",
                citation_label="[3]",
                chunk_id="C3",
            ),
        ],
        question="请跨文件回答“以交易目的持有的头寸”、“能够每日进行公允价值计量，变动计入损益”和商业银行第三支柱披露要求。",
    )

    assert "能够每日进行公允价值计量" in (result.answer or "")
    assert "以交易目的持有的头寸" in (result.answer or "")
    assert "表格中另有规定的除外" in (result.answer or "")
    assert "[1]" in (result.answer or "") and "[2]" in (result.answer or "") and "[3]" in (result.answer or "")


def test_relevant_extractive_excerpt_keeps_exception_when_question_requests_exceptions() -> None:
    text = (
        "二、披露内容。"
        "（一）商业银行以表格形式进行第三支柱信息披露，包括固定表格和可变表格。"
        "商业银行应按说明填写固定表格。"
        "商业银行应按要求评估披露内容，并确保披露信息对信息使用者具有参考价值。"
        "商业银行应根据表格要求，分别按照季度、半年和年度的频率披露信息。"
        "商业银行应按照监管并表范围披露相关信息，表格中另有规定的除外。"
        "披露报告应由董事会审议。"
    )

    excerpt = answer_generation_service._relevant_extractive_excerpt(
        text,
        "商业银行第三支柱信息披露频率和表格例外",
    )

    assert "季度、半年和年度" in excerpt
    assert "表格中另有规定的除外" in excerpt


def test_relevant_extractive_excerpt_keeps_discount_curve_definition() -> None:
    text = (
        "根据《保险公司偿付能力监管规则第3号：寿险合同负债评估》第十九条规定，"
        "计算现金流现值所采用的折现率曲线由基础利率曲线加综合溢价形成，具体计算方法如下："
        "一、基础利率曲线为即期曲线，由以下三段组成。"
        "其中，t表示年度；终极利率过渡曲线采用二次插值方法计算得到；终极利率暂定为4.5%。"
        "二、终极利率过渡曲线采用二次插值方法计算得到。"
        "t为年度；r为在t年度750天移动平均国债收益率曲线的数值；Rt为在t年度终极利率过渡曲线的数值。"
    )

    excerpt = answer_generation_service._relevant_extractive_excerpt(
        text,
        "寿险合同负债评估中计算现金流现值所采用的折现率曲线",
    )

    assert "折现率曲线由基础利率曲线加综合溢价形成" in excerpt


def test_trusted_extractive_fallback_drops_failed_claim_and_revalidates(monkeypatch) -> None:
    def fake_validate(answer, claims, context_chunks):
        if "坏证据" in answer:
            return {
                "passed": False,
                "invalid_citation_ids": [],
                "invalid_answer_citation_ids": [],
                "missing_inline_citation_ids": [],
                "claim_verdicts": [{"claim_index": 1, "verdict": "contradicted"}],
            }
        return {
            "passed": True,
            "invalid_citation_ids": [],
            "invalid_answer_citation_ids": [],
            "missing_inline_citation_ids": [],
            "claim_verdicts": [{"claim_index": 1, "verdict": "supported"}],
        }

    monkeypatch.setattr(answer_generation_service, "validate_grounded_answer", fake_validate)

    result = answer_generation_service._trusted_extractive_fallback(
        [
            _chunk(
                text="坏证据：银行业金融机构应当留存无关资料。",
                citation_label="[1]",
                chunk_id="C1",
                metadata={"aspect_id": "a", "aspect_question": "银行业金融机构明确规定"},
            ),
            _chunk(
                text="银行业金融机构应当在公开渠道就办理函证相关事项进行公示。",
                citation_label="[2]",
                chunk_id="C2",
                metadata={"aspect_id": "a", "aspect_question": "银行业金融机构明确规定"},
            ),
        ],
        question="请说明银行业金融机构的明确规定。",
        reason="grounding_validation_failed",
    )

    assert result.refused is False
    assert result.grounding_validation["passed"] is True
    assert result.grounding_validation["repair_method"] == "drop_failed_extractive_claims"
    assert "坏证据" not in (result.answer or "")
    assert "[2]" in (result.answer or "")


def test_validation_flags_directory_only_false_insufficient_answer() -> None:
    generated = answer_generation_service.GeneratedAnswer(
        answer=(
            "关于前一份文件：根据知识库仅提供目录，无法获取中资商业银行法人机构开业核准"
            "和中资商业银行分行筹建审批的具体条件。"
        ),
        answer_type="llm_grounded",
        generation_status="completed",
        claims=[
            AnswerClaim(
                text="根据知识库仅提供目录，无法获取中资商业银行法人机构开业核准和中资商业银行分行筹建审批的具体条件",
                citation_ids=["[1]"],
            )
        ],
    )
    chunks = [
        _chunk(
            metadata={
                "aspect_id": "commercial_bank_licensing_requirements",
                "prompt_matched_aspects": ["commercial_bank_licensing_requirements"],
                "aspect_question": "中资商业银行法人机构开业核准 中资商业银行分行筹建审批",
            },
            text=(
                "一、机构设立。（一）法人机构设立。"
                "1.2中资商业银行法人机构开业核准。"
                "（二）境内分支机构设立。1.3中资商业银行分行筹建审批。"
            ),
            citation_label="[1]",
            chunk_id="C1",
        )
    ]

    result = answer_generation_service._validate_no_false_insufficient_aspect_claims(
        generated,
        chunks,
        ["commercial_bank_licensing_requirements"],
    )

    assert result["false_insufficient_aspect_ids"] == ["commercial_bank_licensing_requirements"]
    assert result["false_insufficient_aspect_count"] == 1


def test_fallback_does_not_add_second_bounded_chunk_when_multiple_aspects() -> None:
    chunks = [
        _chunk(
            chunk_id="consumer-correct",
            text="消费金融公司可以在全国范围内开展业务。",
            citation_label="[1]",
            metadata={
                "aspect_id": "consumer_scope",
                "aspect_question": "消费金融公司的业务地域范围是什么？",
                "evidence_role": "bounded_lexical_support",
            },
        ),
        _chunk(
            chunk_id="consumer-nearby",
            text="非银行类金融机构包括消费金融公司、基金公司、理财公司等机构。",
            citation_label="[2]",
            metadata={
                "aspect_id": "consumer_scope",
                "aspect_question": "消费金融公司的业务地域范围是什么？",
                "evidence_role": "bounded_lexical_support",
            },
        ),
        _chunk(
            chunk_id="pillar3",
            text="商业银行以表格形式进行第三支柱信息披露，包括固定表格和可变表格。",
            citation_label="[3]",
            metadata={
                "aspect_id": "pillar3_table_types",
                "aspect_question": "第三支柱表格类型包括哪些？",
                "evidence_role": "bounded_lexical_support",
            },
        ),
    ]

    selected = answer_generation_service._fallback_chunks_by_aspect(
        chunks,
        question="跨制度回答消费金融公司的业务地域范围、第三支柱表格类型。",
        limit=4,
    )

    assert [chunk.chunk_id for chunk in selected] == ["consumer-correct", "pillar3"]


def test_fallback_relevance_prefers_effective_communication_clause() -> None:
    generic = _chunk(
        text="应说明向董事会报告的路线，以及与地方政府的汇报路径和沟通形式。",
        citation_label="[1]",
    )
    governing = _chunk(
        text="六、沟通策略。在处置计划实施过程中，与监管部门、地方政府、股东、客户、员工和社会公众开展有效沟通。",
        citation_label="[2]",
    )

    assert answer_generation_service._fallback_chunk_relevance(
        governing, "说明处置计划的沟通对象"
    ) > answer_generation_service._fallback_chunk_relevance(
        generic, "说明处置计划的沟通对象"
    )
    assert answer_generation_service._fallback_chunk_relevance(
        governing, "说明处置计划沟通策略关注的规定"
    ) > answer_generation_service._fallback_chunk_relevance(
        generic, "说明处置计划沟通策略关注的规定"
    )


def test_fallback_relevance_prioritizes_bounded_direct_support() -> None:
    direct = _chunk(
        text="不得采用暴力、威胁等不正当手段催收，不得对与债务无关的第三人催收。",
        metadata={"evidence_role": "bounded_lexical_support"},
        citation_label="[1]",
    )
    generic = _chunk(
        text="消费金融公司应建立催收管理系统并管理合作机构。",
        metadata={"evidence_role": "related_context", "rerank_score": 0.99},
        citation_label="[2]",
    )

    assert answer_generation_service._fallback_chunk_relevance(
        direct, "消费金融公司催收保护禁止对象的规定"
    ) > answer_generation_service._fallback_chunk_relevance(
        generic, "消费金融公司催收保护禁止对象的规定"
    )


def test_fallback_relevance_prefers_core_direct_evidence_over_generic_bounded_guess() -> None:
    question = "根据《银行函证工作操作指引》，请说明与“银行业金融机构”相关的明确规定。"
    core = _chunk(
        text="银行业金融机构应当在公开渠道就办理函证相关事项进行公示。公示信息包括办理机构及联系方式。",
        metadata={
            "prompt_selection_reason": "core",
            "evidence_role": "direct_evidence",
            "aspect_id": "a",
        },
    )
    bounded_guess = _chunk(
        text="银行询证函格式由注册会计师填写，银行业金融机构根据本机构掌握的信息核对后回复。",
        metadata={
            "prompt_selection_reason": "generic",
            "evidence_role": "bounded_lexical_support",
            "exact_support_anchor": "银行业金融机构定义",
            "lexical_support_phrase": "银行业金融机构定义",
            "aspect_id": "a",
        },
    )

    assert answer_generation_service._fallback_chunk_relevance(
        core, question
    ) > answer_generation_service._fallback_chunk_relevance(bounded_guess, question)


def test_fallback_relevance_prefers_complete_metadata_anchor_over_generic_definition() -> None:
    generic_definition = _chunk(
        text="第二条 本办法所称消费金融公司，是指经批准设立的非银行金融机构。",
        citation_label="[1]",
        metadata={
            "evidence_role": "bounded_lexical_support",
            "exact_support_anchor": "消费金融公司",
            "lexical_support_phrase": "消费金融公司",
            "exact_support_score": 2.35,
        },
    )
    answer_clause = _chunk(
        text="第十四条 消费金融公司可以在全国范围内开展业务。",
        citation_label="[2]",
        metadata={
            "evidence_role": "bounded_lexical_support",
            "exact_support_anchor": "消费金融公司可以在全国范围内开展业务",
            "lexical_support_phrase": "消费金融公司可以在全国范围内开展业务",
            "exact_support_score": 1.55,
        },
    )

    assert answer_generation_service._fallback_chunk_relevance(
        answer_clause, "消费金融公司的业务地域范围是什么？"
    ) > answer_generation_service._fallback_chunk_relevance(
        generic_definition, "消费金融公司的业务地域范围是什么？"
    )


def test_fallback_relevance_uses_quoted_predicate_fragments() -> None:
    target = _chunk(
        text="二、其他一级资本工具的合格标准。（四）没有到期日，并且不得含有利率跳升机制及其他赎回激励。",
        citation_label="[1]",
        metadata={"evidence_role": "related_context"},
    )
    distractor = _chunk(
        text="三、二级资本工具的合格标准。（四）原始期限不低于5年，并且不得含有利率跳升机制及其他赎回激励。",
        citation_label="[2]",
        metadata={"evidence_role": "related_context", "rerank_score": 0.99},
    )

    question = "制度侧：根据《资本工具合格标准》，请说明与“其他一级资本工具没有到期日，并且”相关的明确规定。"

    assert answer_generation_service._fallback_chunk_relevance(
        target, question
    ) > answer_generation_service._fallback_chunk_relevance(distractor, question)


def test_fallback_relevance_prefers_definition_clause_for_quoted_term() -> None:
    definition = _chunk(
        text="第二条 本办法所称意外伤害保险，是以被保险人因遭受意外伤害造成死亡、伤残或者发生保险合同约定的其他事故为给付保险金条件的人身保险。",
        citation_label="[1]",
        metadata={"evidence_role": "bounded_lexical_support"},
    )
    notice = _chunk(
        text="为进一步规范意外伤害保险经营行为，现将《意外伤害保险业务监管办法》印发给你们，请遵照执行。",
        citation_label="[2]",
        metadata={"evidence_role": "bounded_lexical_support", "rerank_score": 0.99},
    )

    question = "根据《意外伤害保险业务监管办法》，找出与意外伤害保险相关的明确规定。"

    assert answer_generation_service._fallback_chunk_relevance(
        definition, question
    ) > answer_generation_service._fallback_chunk_relevance(notice, question)


def test_fallback_relevance_uses_classification_subject_anchor() -> None:
    list_chunk = _chunk(
        text="一、保险控股型集团。下列保险控股型集团应当编制保险集团偿付能力报告：（六）中国太平洋保险（集团）股份有限公司。",
        citation_label="[1]",
        metadata={"evidence_role": "expanded_context"},
    )
    preamble = _chunk(
        text="下列保险集团应当按照有关规定编报保险集团偿付能力报告。",
        citation_label="[2]",
        metadata={"evidence_role": "bounded_lexical_support", "exact_support_score": 1.35},
    )

    question = "请说明与“中国太平洋保险（集团）股份有限公司属于”相关的明确规定。"

    assert answer_generation_service._fallback_chunk_relevance(
        list_chunk, question
    ) > answer_generation_service._fallback_chunk_relevance(preamble, question)


def test_fallback_relevance_uses_elided_prohibition_predicate() -> None:
    list_item = _chunk(
        text="（七）通过保险中介机构为其他机构或者个人谋取不正当利益；（八）距保单到期日前间隔60天以上预收下一保单年度保费；（九）向特定团体成员以外的个人销售团体意外险；",
        citation_label="[1]",
        metadata={"evidence_role": "expanded_context"},
    )
    title_page = _chunk(
        text="银保监办发〔2021〕106号。现将《意外伤害保险业务监管办法》印发给你们，请遵照执行。",
        citation_label="[2]",
        metadata={"evidence_role": "bounded_lexical_support", "rerank_score": 0.99},
    )

    question = "根据《意外伤害保险业务监管办法》，请说明与“保险公司不得向特定团体成员以外的个人销售团体意外险”相关的明确规定。"

    assert answer_generation_service._fallback_chunk_relevance(
        list_item, question
    ) > answer_generation_service._fallback_chunk_relevance(title_page, question)


def test_fallback_relevance_uses_elided_ability_predicate() -> None:
    condition_item = _chunk(
        text="1.不存在实施平盘或完全对冲交易的法律障碍；2.能够每日进行公允价值计量，变动计入损益；3.能够进行积极的管理。",
        citation_label="[1]",
        metadata={"evidence_role": "expanded_context"},
    )
    preamble = _chunk(
        text="（二）交易账簿中的金融工具、外汇和商品头寸原则上应同时满足以下条件：1.不存在实施平盘或完全对冲交易的法律障碍；",
        citation_label="[2]",
        metadata={"evidence_role": "expanded_context", "rerank_score": 0.99},
    )

    question = "根据《账簿划分和名词解释》，请说明与“交易账簿中的金融工具、外汇和商品头寸原则上应能够每日进行公允价值计量，变”相关的明确规定。"

    assert answer_generation_service._fallback_chunk_relevance(
        condition_item, question
    ) > answer_generation_service._fallback_chunk_relevance(preamble, question)


def test_relevant_extractive_excerpt_keeps_leading_normative_sentence() -> None:
    text = (
        "银行业金融机构应当在其总行或总部网站、微信公众号等公开渠道就办理函证相关事项进行公示。"
        "银行业金融机构公示的内容应当符合本指引具体要求。"
        "会计师事务所在此基础上根据银行业金融机构公示信息进行函证。"
        "公示信息包括：1.各种函证方式下办理回函工作的机构及其联系方式。"
        "在实现集约化或数字化的情况下，银行业金融机构应当就询证函的函证范围以及所采用的回函用章的适用范围进行公示。"
    )

    excerpt = answer_generation_service._relevant_extractive_excerpt(
        text,
        "根据《银行函证工作操作指引》，请说明与“银行业金融机构”相关的明确规定。",
        max_chars=180,
    )

    assert "银行业金融机构应当在其总行或总部网站、微信公众号等公开渠道就办理函证相关事项进行公示" in excerpt


def test_api_unavailable_uses_extractive_fallback(monkeypatch) -> None:
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", None)
    result = generate_answer("期限是什么？", [_chunk()], has_sufficient_context=True)

    assert result.degraded is True
    assert result.answer_type == "extractive_fallback"
    assert "2025年12月31日" in result.answer
    assert result.refused is False
    assert result.grounding_validation["passed"] is True
    assert "1. " not in result.answer


def test_api_unavailable_recovers_evidence_ranked_choice(monkeypatch) -> None:
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

    assert result.refused is False
    assert result.degraded is False
    assert result.answer_type == "choice_evidence_deterministic"
    assert result.answer.startswith("答案为 核心一级资本工具应当直接发行且实缴。[1]")
    assert "依据：" in result.answer
    assert result.grounding_validation["selected_option"] == "0"
    assert result.grounding_validation["selected_option_supported"] is True
    assert result.grounding_validation["recovery_method"] == "deterministic_option_evidence"


def test_deepseek_sse_reports_only_complete_verified_claims(monkeypatch) -> None:
    # This test isolates incremental SSE JSON parsing.  Semantic-risk routing
    # is covered separately and may be configured as off/risk_based/all in
    # different environments, so keep it from making this parser test depend
    # on deployment configuration.
    monkeypatch.setattr(
        answer_generation_service,
        "claim_requires_semantic_verifier",
        lambda claim, chunks: False,
    )
    monkeypatch.setattr(
        answer_generation_service,
        "validate_grounded_answer",
        lambda answer, claims, chunks: {"passed": True},
    )
    content_parts = [
        '{"refused":false,"refusal_reason":null,"claims":[',
        '{"role":"conclusion","text":"监管报送期限为2025年12月31日。[1]",',
        '"citation_ids":["[1]"]},',
        '{"role":"explanation","text":"依据材料中的报送期限。[1]","citation_ids":["[1]"]}',
        ']}',
    ]
    stream = _SSEStream([
        ": keep-alive\n",
        *[
            f'data: {json.dumps({"choices": [{"delta": {"content": part}}]}, ensure_ascii=False)}\n'
            for part in content_parts
        ],
        "data: [DONE]\n",
    ])
    reported: list[AnswerClaim] = []

    result = answer_generation_service._read_streaming_llm_content(
        stream,
        question="期限是什么？",
        context_chunks=[_chunk()],
        options=[],
        option_labels=[],
        verified_claim_reporter=reported.append,
        cancellation_checker=None,
    )

    assert json.loads(result)["refused"] is False
    assert [claim.text for claim in reported] == [
        "监管报送期限为2025年12月31日。[1]",
        "依据材料中的报送期限。[1]",
    ]


def test_generated_answer_fails_validation_when_a_required_aspect_is_omitted() -> None:
    chunks = [
        _chunk(
            chunk_id="C1",
            citation_label="[1]",
            text="事项一明确要求保留三年。",
            metadata={"aspect_id": "requirement_one"},
        ),
        _chunk(
            chunk_id="C2",
            citation_label="[2]",
            text="事项二明确禁止涉及无关第三人。",
            metadata={"aspect_id": "requirement_two"},
        ),
    ]
    generated = answer_generation_service.GeneratedAnswer(
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

    validation = answer_generation_service._validate_generated_answer(
        generated,
        chunks,
        [],
        [],
        required_aspect_ids=["requirement_one", "requirement_two"],
    )

    assert validation["passed"] is False
    assert validation["missing_required_aspect_ids"] == ["requirement_two"]
    assert validation["required_aspect_failure_reason"] == "missing_required_aspects"


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
    assert generated.grounding_validation["missing_required_aspect_ids"] == ["requirement_two"]
    assert not llm_calls


def test_pre_generation_gate_exempts_multiple_choice_evidence_aspect() -> None:
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

    # multiple_choice_evidence is exempt; only option_evidence is checked.
    assert missing == ["option_evidence"]


def test_aspect_only_failure_triage_depends_on_retrieval_coverage() -> None:
    missing_validation = {
        "passed": False,
        "missing_required_aspect_ids": ["requirement_two"],
        "required_aspect_failure_reason": "missing_required_aspects",
    }
    content_validation = {
        "passed": False,
        "unsupported_entities": ["2025年12月31日"],
        "missing_required_aspect_ids": ["requirement_two"],
    }
    covered_chunks = [
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
    missing_chunks = covered_chunks[:1]

    # Retrieval lacks the aspect: aspect-only failure → deterministic refusal.
    assert answer_generation_service._is_aspect_only_failure(
        missing_validation, missing_chunks, ["requirement_one", "requirement_two"]
    ) is True
    # Retrieval covers everything: missing claim aspect is a labelling problem
    # → LLM repair round, not a refusal.
    assert answer_generation_service._is_aspect_only_failure(
        missing_validation, covered_chunks, ["requirement_one", "requirement_two"]
    ) is False
    # Content failures always deserve the repair round.
    assert answer_generation_service._is_aspect_only_failure(
        content_validation, missing_chunks, ["requirement_one", "requirement_two"]
    ) is False


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
    monkeypatch.setattr(answer_generation_service, "_deterministic_choice_recovery", lambda *a, **k: None)
    monkeypatch.setattr(answer_generation_service, "_deterministic_mixed_table_answer", lambda *a, **k: None)
    monkeypatch.setattr(answer_generation_service, "_repair_inline_citation_format", lambda *a, **k: False)
    monkeypatch.setattr(
        answer_generation_service,
        "_validate_generated_answer",
        lambda *a, **k: {
            "passed": False,
            "unsupported_entities": ["2025年12月31日"],
            "missing_required_aspect_ids": ["requirement_two"],
        },
    )

    generated = generate_answer(
        "事项一如何处理",
        chunks,
        has_sufficient_context=True,
        answer_mode="text",
        required_aspect_ids=["requirement_one", "requirement_two"],
    )

    # Content failures (unsupported entities) keep the LLM repair round.
    assert len(llm_calls) == 2


def test_required_aspect_validation_does_not_count_chunk_aspect_for_tagged_claim() -> None:
    chunks = [
        _chunk(
            chunk_id="C1",
            citation_label="[1]",
            text="Requirement one and requirement two are both present.",
            metadata={"aspect_id": "requirement_two", "prompt_matched_aspects": ["requirement_one", "requirement_two"]},
        )
    ]
    generated = answer_generation_service.GeneratedAnswer(
        answer="Requirement one is present. [1]",
        answer_type="llm_grounded",
        generation_status="completed",
        claims=[
            AnswerClaim(
                text="Requirement one is present. [1]",
                citation_ids=["[1]"],
                aspect_ids=["requirement_one"],
            )
        ],
    )

    validation = answer_generation_service._validate_generated_answer(
        generated,
        chunks,
        [],
        [],
        required_aspect_ids=["requirement_one", "requirement_two"],
    )

    assert validation["passed"] is False
    assert validation["missing_required_aspect_ids"] == ["requirement_two"]


def test_generated_answer_fails_when_covered_aspect_is_marked_insufficient() -> None:
    chunks = [
        _chunk(
            chunk_id="C1",
            citation_label="[1]",
            text="商业银行应根据表格要求，分别按照季度、半年和年度的频率披露信息。",
            metadata={
                "aspect_id": "pillar3_disclosure_frequency",
                "aspect_question": "第三支柱披露频率",
                "prompt_matched_aspects": ["pillar3_disclosure_frequency"],
            },
        )
    ]
    generated = answer_generation_service.GeneratedAnswer(
        answer="关于第三支柱披露频率，检索到的知识片段中未提及具体要求，因此无法判断。[1]",
        answer_type="llm_grounded",
        generation_status="completed",
        claims=[
            AnswerClaim(
                text="关于第三支柱披露频率，检索到的知识片段中未提及具体要求，因此无法判断。",
                citation_ids=["[1]"],
                role="conclusion",
                aspect_ids=["pillar3_disclosure_frequency"],
            )
        ],
        refused=False,
    )

    validation = answer_generation_service._validate_generated_answer(
        generated,
        chunks,
        [],
        [],
        required_aspect_ids=["pillar3_disclosure_frequency"],
    )

    assert validation["passed"] is False
    assert validation["false_insufficient_aspect_ids"] == ["pillar3_disclosure_frequency"]
    assert validation["false_insufficient_failure_reason"] == "covered_aspect_marked_insufficient"


def test_extractive_fallback_prefers_same_document_for_sibling_aspect() -> None:
    chunks = [
        _chunk(
            chunk_id="D1-C1",
            citation_label="[1]",
            text="大额风险暴露是指商业银行对单一客户或一组关联客户超过其一级资本净额2.5%的风险暴露。",
            metadata={
                "document_id": "D1",
                "aspect_id": "large_exposure",
                "aspect_question": "大额风险暴露",
                "prompt_matched_aspects": ["large_exposure"],
            },
        ),
        _chunk(
            chunk_id="D2-C2",
            citation_label="[2]",
            text="处置策略建议应坚持自救为本的基本原则。",
            metadata={
                "document_id": "D2",
                "aspect_id": "bail_in",
                "aspect_question": "自救原则",
                "prompt_matched_aspects": ["bail_in"],
            },
        ),
        _chunk(
            chunk_id="D1-C2",
            citation_label="[3]",
            text="处置策略建议应坚持自救为本的基本原则。",
            metadata={
                "document_id": "D1",
                "aspect_id": "bail_in",
                "aspect_question": "自救原则",
                "prompt_matched_aspects": ["bail_in"],
            },
        ),
    ]

    selected = answer_generation_service._fallback_chunks_by_aspect(
        chunks,
        question="说明大额风险暴露与自救原则",
        limit=3,
    )

    assert [chunk.chunk_id for chunk in selected[:2]] == ["D1-C1", "D1-C2"]


def test_option_evidence_splits_comma_facts_and_rejects_changed_relation_values() -> None:
    chunks = [
        _chunk(
            citation_label="[1]",
            text="大额风险暴露是指超过一级资本净额2.5%的风险暴露。",
        ),
        _chunk(
            chunk_id="C2",
            citation_label="[2]",
            text="处置策略应坚持自救为本的基本原则。",
        ),
    ]
    options = [
        "门槛为一级资本净额5%，处置以外部救助为本",
        "门槛为一级资本净额2.5%，处置策略应坚持自救为本",
    ]

    matrix = answer_generation_service._option_evidence_matrix(options, ["A", "B"], chunks)

    assert len(matrix[1]["facts"]) == 2
    assert matrix[0]["minimum_fact_coverage"] == 0
    assert matrix[1]["minimum_fact_coverage"] >= 0.3


def test_option_evidence_inherits_one_shared_classification_subject() -> None:
    chunks = [
        _chunk(
            chunk_id="GROUP-CURRENT",
            citation_label="[1]",
            text=(
                "下列保险控股型集团应当编制保险集团偿付能力报告："
                "（六）中国太平洋保险（集团）股份有限公司。"
            ),
        ),
        _chunk(
            chunk_id="GROUP-OTHER",
            citation_label="[2]",
            text=(
                "下列非保险控股型集团应当编制保险集团偿付能力报告："
                "国家电网有限公司或其指定成员公司。"
            ),
        ),
    ]
    options = [
        "太平洋保险集团属于非保险控股型集团",
        "属于应编报的保险控股型集团",
    ]

    matrix = answer_generation_service._option_evidence_matrix(options, ["A", "B"], chunks)

    assert matrix[0]["minimum_fact_coverage"] == 0
    assert matrix[1]["minimum_fact_coverage"] >= 0.3
    assert matrix[1]["facts"][0]["grounded_fact"].startswith("太平洋保险集团属于")
    assert matrix[1]["facts"][0]["citation_ids"] == ["[1]"]


def test_option_evidence_combines_only_linked_modal_preamble_and_list_item() -> None:
    chunks = [
        _chunk(
            chunk_id="RULE-PREAMBLE",
            citation_label="[1]",
            text="保险公司开展业务活动不得存在以下行为：",
            metadata={
                "evidence_role": "mcq_rule_preamble",
                "governs_chunk_ids": ["RULE-ITEM"],
            },
        ),
        _chunk(
            chunk_id="RULE-ITEM",
            citation_label="[2]",
            text="（九）向特定团体成员以外的个人销售团体意外险；",
            metadata={"evidence_role": "mcq_exact_support"},
        ),
        _chunk(
            chunk_id="UNRELATED",
            citation_label="[3]",
            text="其他保险业务可以按照产品备案范围开展。",
        ),
    ]
    options = [
        "可向特定团体成员以外的个人销售团体意外险",
        "不得向特定团体成员以外的个人销售团体意外险",
    ]

    matrix = answer_generation_service._option_evidence_matrix(options, ["A", "B"], chunks)

    assert matrix[0]["minimum_fact_coverage"] == 0
    assert matrix[1]["minimum_fact_coverage"] >= 0.3
    assert matrix[1]["facts"][0]["citation_ids"] == ["[1]", "[2]"]


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


def test_deepseek_sse_does_not_publish_unsupported_number() -> None:
    content = (
        '{"refused":false,"refusal_reason":null,"claims":['
        '{"role":"conclusion","text":"监管报送期限为2099年1月1日。[1]",'
        '"citation_ids":["[1]"]}]}'
    )
    stream = _SSEStream([
        f'data: {json.dumps({"choices": [{"delta": {"content": content}}]}, ensure_ascii=False)}\n',
        "data: [DONE]\n",
    ])
    reported: list[AnswerClaim] = []

    answer_generation_service._read_streaming_llm_content(
        stream,
        question="期限是什么？",
        context_chunks=[_chunk()],
        options=[],
        option_labels=[],
        verified_claim_reporter=reported.append,
        cancellation_checker=None,
    )

    assert reported == []


def test_generation_repairs_missing_inline_citation_without_second_llm_call(monkeypatch) -> None:
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

    assert len(calls) == 1
    assert result.refused is False
    assert result.answer == "监管报送期限为2025年12月31日。[1]"
    assert result.grounding_validation["passed"] is True
    assert result.grounding_validation["repair_method"] == "local_inline_citation_format"


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


def test_generation_prompt_requires_one_claim_per_required_aspect() -> None:
    messages = answer_generation_service.RAGPromptBuilder().build_generation_messages(
        "请分别说明两个事项。",
        [_chunk()],
        required_aspect_ids=["aspect_a", "aspect_b"],
    )

    user_message = messages[1]["content"]
    assert "每个方面至少输出一条独立claim" in user_message
    assert "aspect_a, aspect_b" in user_message


def test_call_llm_recovers_partial_claims_after_stream_parse_failure(monkeypatch) -> None:
    calls = []
    partial_content = (
        '{"refused":false,"refusal_reason":null,"claims":['
        '{"role":"conclusion","text":"Requirement one must be retained for three years. [1]",'
        '"citation_ids":["[1]"],"aspect_ids":["retention"]}'
    )

    class FakeStreamResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return iter(["data: [DONE]\n"])

    class FakeBodyResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": partial_content}}]},
                ensure_ascii=False,
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        calls.append(payload)
        if payload.get("stream") is True:
            return FakeStreamResponse()
        return FakeBodyResponse()

    monkeypatch.setattr(answer_generation_service.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", "test-key")
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_STREAM", True)

    result = answer_generation_service._call_llm(
        "What is the retention requirement?",
        [_chunk(text="Requirement one must be retained for three years.")],
        [],
        [],
        required_aspect_ids=["retention"],
    )

    assert [payload.get("stream") for payload in calls] == [True, None]
    assert result.generation_status == "completed"
    assert result.answer_type == "llm_grounded"
    assert result.answer == "Requirement one must be retained for three years. [1]"
    assert result.claims[0].citation_ids == ["[1]"]
    assert result.claims[0].aspect_ids == ["retention"]


def test_open_question_uses_extractive_fallback_when_generation_validation_fails(monkeypatch) -> None:
    def fake_call(*args, **kwargs):
        return answer_generation_service.GeneratedAnswer(
            answer="该材料规定所有银行必须在2099年完成整改。[1]",
            answer_type="llm_grounded",
            generation_status="completed",
            claims=[AnswerClaim(text="所有银行必须在2099年完成整改", citation_ids=["[1]"])],
        )

    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_ENABLED", True)
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", "test-key")
    monkeypatch.setattr(answer_generation_service, "_call_llm", fake_call)

    result = generate_answer(
        "请概述《账簿划分和名词解释》的主要内容。",
        [_chunk(text="交易账簿包括为交易目的而持有的金融工具。")],
        has_sufficient_context=True,
    )

    # Open questions refuse structurally after validation failure instead of
    # degrading to a conclusion-less excerpt dump.
    assert result.refused is True
    assert result.degraded is True
    assert result.answer_type == "refusal"
    assert result.refusal_reason == "validation_failed_no_reliable_answer"
    assert result.grounding_validation["recovery_method"] == "structural_refusal_after_validation_failed"
    assert result.grounding_validation["passed"] is True


def test_supported_model_refusal_recovers_to_extractive_fallback(monkeypatch) -> None:
    def fake_call(*args, **kwargs):
        return answer_generation_service.GeneratedAnswer(
            answer="根据当前知识库无法判断该摘要要求。",
            answer_type="refusal",
            generation_status="completed",
            refused=True,
            refusal_reason="model_claimed_insufficient_context",
        )

    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_ENABLED", True)
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", "test-key")
    monkeypatch.setattr(answer_generation_service, "_call_llm", fake_call)

    result = generate_answer(
        "请形成一段监管工作摘要。",
        [
            _chunk(
                text="到2027年，试点地区基本建成知识产权金融生态综合试验区。",
                citation_label="[1]",
                metadata={"aspect_id": "ip_target", "prompt_matched_aspects": ["ip_target"]},
            ),
            _chunk(
                text="列入名单的保险集团应当编报保险集团偿付能力报告。",
                citation_label="[2]",
                chunk_id="C2",
                metadata={"aspect_id": "insurance_group", "prompt_matched_aspects": ["insurance_group"]},
            ),
        ],
        has_sufficient_context=True,
        required_aspect_ids=["ip_target", "insurance_group"],
    )

    assert result.refused is False
    assert result.answer_type == "extractive_fallback"
    assert result.grounding_validation["recovery_method"] == "extractive_fallback_after_model_refusal"
    assert {aspect for claim in result.claims for aspect in claim.aspect_ids} == {"ip_target", "insurance_group"}


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


def test_option_evidence_binds_each_numeric_fact_to_supporting_chunk() -> None:
    chunks = [
        _chunk(
            text="个人意外险平均附加费用率上限为35%。",
            citation_label="[1]",
            chunk_id="C1",
        ),
        _chunk(
            text="平均附加费用率上限应当合理确定，团体意外险另有规定。",
            citation_label="[2]",
            chunk_id="C2",
        ),
        _chunk(
            text="团体意外险平均附加费用率上限为25%。",
            citation_label="[3]",
            chunk_id="C3",
        ),
    ]

    matrix = answer_generation_service._option_evidence_matrix(
        ["个人意外险平均附加费用率上限为35%；团体意外险平均附加费用率上限为25%"],
        [None],
        chunks,
    )

    assert [fact["citation_id"] for fact in matrix[0]["facts"]] == ["[1]", "[3]"]


def test_option_evidence_validation_rejects_partially_supported_choice() -> None:
    options = [
        "申请材料包含筹建申请书；寿险责任准备金应使用七百五十日移动平均国债收益率曲线",
        "申请材料包含筹建申请书；中资商业银行法人机构筹建审批属于机构设立类行政许可事项",
    ]
    chunks = [
        _chunk(
            text=(
                "中资商业银行行政许可事项申请材料目录包括筹建申请书。"
                "中资商业银行法人机构筹建审批属于机构设立类行政许可事项。"
            )
        )
    ]

    labels = ["A", "B"]
    invalid = answer_generation_service._validate_selected_option("答案为 A、申请材料包含筹建申请书；寿险责任准备金应使用七百五十日移动平均国债收益率曲线。[1]", options, labels, chunks)
    valid = answer_generation_service._validate_selected_option("答案为 B、申请材料包含筹建申请书；中资商业银行法人机构筹建审批属于机构设立类行政许可事项。[1]", options, labels, chunks)

    assert invalid["evidence_best_option"] == "1"
    assert invalid["selected_option_supported"] is False
    assert valid["selected_option_supported"] is True
    assert valid["selected_option_format_valid"] is True


def test_unlabeled_choice_rejects_invented_option_label() -> None:
    options = ["申请材料包含筹建申请书", "法人机构筹建审批属于机构设立类行政许可事项"]
    chunks = [_chunk(text="法人机构筹建审批属于机构设立类行政许可事项。")]

    result = answer_generation_service._validate_selected_option(
        "答案为 B。[1]",
        options,
        [None, None],
        chunks,
    )

    assert result["selected_option_format_valid"] is False
    assert result["selected_option_format_reason"] == "no_selected_option"


def test_labeled_choice_requires_full_option_text() -> None:
    options = ["申请材料包含筹建申请书", "法人机构筹建审批属于机构设立类行政许可事项"]
    chunks = [_chunk(text="法人机构筹建审批属于机构设立类行政许可事项。")]

    result = answer_generation_service._validate_selected_option(
        "答案为 B。[1]",
        options,
        ["A", "B"],
        chunks,
    )

    assert result["selected_option"] == "1"
    assert result["selected_option_supported"] is True
    assert result["selected_option_format_valid"] is False


def test_option_evidence_failure_uses_deterministic_choice_recovery(monkeypatch) -> None:
    options = [
        "申请材料包含筹建申请书；寿险责任准备金使用折现率曲线",
        "申请材料包含筹建申请书；法人机构筹建审批属于机构设立类行政许可事项",
    ]
    chunks = [
        _chunk(
            text=(
                "申请材料包含筹建申请书，法人机构筹建审批属于机构设立类行政许可事项。"
            )
        )
    ]
    calls = []

    def fake_call(*args, **kwargs):
        calls.append(kwargs.get("correction"))
        choice = "A"
        return answer_generation_service.GeneratedAnswer(
            answer=f"答案为 {choice}。[1]",
            answer_type="llm_grounded",
            generation_status="completed",
            claims=[AnswerClaim(text=f"答案为 {choice}", citation_ids=["[1]"])],
        )

    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_ENABLED", True)
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", "test-key")
    monkeypatch.setattr(answer_generation_service, "_call_llm", fake_call)

    result = generate_answer(
        "哪一项正确？", chunks, options=options, has_sufficient_context=True
    )

    assert len(calls) == 0
    assert result.refused is False
    assert "答案为 申请材料包含筹建申请书；法人机构筹建审批属于机构设立类行政许可事项" in result.answer
    assert "答案为 B" not in result.answer
    assert result.answer_type == "choice_evidence_deterministic"
    assert result.grounding_validation["recovery_method"] == "deterministic_option_evidence"
    assert result.grounding_validation["selected_option_supported"] is True


def test_deterministic_choice_recovery_requires_clear_evidence_margin() -> None:
    result = answer_generation_service._deterministic_choice_recovery(
        ["申请材料包括筹建申请书", "申请材料包括筹建申请书"],
        [_chunk(text="申请材料包括筹建申请书。")],
        {"passed": False},
    )

    assert result is None


def test_deterministic_choice_recovery_accepts_cited_three_fact_option_with_moderate_margin() -> None:
    chunks = [
        _chunk(
            text="不得采用暴力、威胁、恐吓、骚扰等不正当手段进行催收，不得对与债务无关的第三人进行催收。",
            citation_label="[1]",
            chunk_id="C1",
        ),
        _chunk(
            text="银行业金融机构应当在其总行或总部网站、微信公众号等公开渠道就办理函证相关事项进行公示。一份银行询证函只列示一个函证基准日。",
            citation_label="[2]",
            chunk_id="C2",
        ),
    ]

    result = answer_generation_service._deterministic_choice_recovery(
        [
            "可向无关第三人催收；询证函可列多个基准日",
            "禁止暴力等不正当催收且不得催收无关第三人；一函一个基准日；并应通过总行或总部公开渠道公示函证事项",
            "只禁止暴力催收；函证无需公示",
            "可由催收机构自行决定对象；公示仅限纸质公告",
        ],
        ["A", "B", "C", "D"],
        chunks,
        {"passed": False},
    )

    assert result is not None
    assert "答案为 B" in result.answer
    assert result.grounding_validation["selected_option"] == "1"
