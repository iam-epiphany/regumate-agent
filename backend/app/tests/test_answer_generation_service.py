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


def test_choice_question_without_options_returns_clarification() -> None:
    result = generate_answer(
        "关于《账簿划分和名词解释》，下列哪一组选项中的两项表述均属于该材料内容？",
        [_chunk(text="交易账簿包括为交易目的而持有的金融工具。")],
        has_sufficient_context=True,
    )

    assert result.refused is True
    assert result.answer_type == "clarification"
    assert result.generation_status == "skipped"
    assert result.refusal_reason == "missing_options_for_choice_question"
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
    assert result.answer == "答案为 核心一级资本工具应当直接发行且实缴。[1]"
    assert result.grounding_validation["selected_option"] == "0"
    assert result.grounding_validation["selected_option_supported"] is True
    assert result.grounding_validation["recovery_method"] == "deterministic_option_evidence"


def test_deepseek_sse_reports_only_complete_verified_claims() -> None:
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

    assert result.refused is False
    assert result.degraded is True
    assert result.answer_type == "extractive_fallback"
    assert result.generation_status == "validation_degraded"
    assert "交易账簿包括为交易目的而持有的金融工具" in result.answer
    assert result.grounding_validation["degraded_reason"] == "grounding_validation_failed"
    assert result.grounding_validation["passed"] is True


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

    assert len(calls) == 1
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
