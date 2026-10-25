"""Regression tests for the 2026-10-25 refusal-consistency and evidence fixes.

Each test pins one empirically observed failure from the 300+200 evaluation
round (docs/evaluation/official300_selfbuilt200_20261024.md):

- salvaged/partial answers whose text is an evidence-boundary statement carried
  ``refused=false`` while refusing in substance (11 self-built refusal cases);
- the grounding validator rejected a document number embedded in a sentence
  (“严格遵守财会〔2022〕39号”) even though the cited excerpt contains the same
  number (official Q119 over-refusal);
- percent-scaled ratio calculations showed only “17.37%” while the gold
  convention was the decimal ratio (4 self-built table_calculation cases);
- comparison rewrites that dropped the candidate rows left “以下哪一项数值最高”
  without a referent (11 official table-comparison questions).
"""

from __future__ import annotations

from backend.app.schemas.qa import AnswerClaim
from backend.app.services.answer_generation_service import (
    GeneratedAnswer,
    _claims_all_boundary,
    _ensure_refusal_flag_consistency,
    _is_evidence_boundary_lead,
    _ratio_display_note,
)
from backend.app.services.grounding_validation_service import (
    _entity_is_supported,
    _normalize,
)
from scripts.evaluation.rewrite_official300_open import ensure_candidate_enumeration


def test_boundary_lead_answer_is_flagged_refused() -> None:
    answer = GeneratedAnswer(
        answer="无法提供中国工商银行2025年年度报告中的净利润数据。\n\n现有证据仅为商业银行分机构类净利润汇总数据，未包含中国工商银行年度报告或该行净利润数据。",
        answer_type="llm_grounded",
        generation_status="partial",
        refusal_reason="partial_answer",
    )
    flagged = _ensure_refusal_flag_consistency(answer)
    assert flagged.refused is True
    assert flagged.answer_type == "refusal"
    assert flagged.refusal_reason == "partial_answer"
    assert flagged.refusal_code == "evidence_boundary"


def test_advice_boundary_lead_answer_is_flagged_refused() -> None:
    answer = GeneratedAnswer(
        answer="依据不足，不能建议该财务公司扩大同业拆借规模。\n\n2023年版目录规定，财务公司从事同业拆借等专项业务需进行资格审批。",
        answer_type="llm_grounded",
        generation_status="completed",
    )
    flagged = _ensure_refusal_flag_consistency(answer)
    assert flagged.refused is True
    assert flagged.refusal_code == "evidence_boundary"


def test_substantive_answers_keep_refused_false() -> None:
    substantive = [
        "2024年9月人身险公司原保险保费收入（本年累计）为4505亿元。[1]",
        "计算结果为 16.09%（比值0.1609）。计算式：3887.79602 / 24155.58512 × 100 = 16.09。[1][2]",
        "《商业银行资本管理办法》规定，交易账簿包括为交易目的或对冲交易账簿其他项目风险而持有的金融工具、外汇和商品头寸。[1]",
    ]
    for text in substantive:
        answer = _ensure_refusal_flag_consistency(
            GeneratedAnswer(
                answer=text,
                answer_type="llm_grounded",
                generation_status="completed",
            )
        )
        assert answer.refused is False, text


def test_claims_all_boundary_detects_refusal_in_substance() -> None:
    boundary = [
        AnswerClaim(
            text="无法提供最新一期贷款市场报价利率的数值和日期。", citation_ids=["[1]"]
        ),
        AnswerClaim(
            text="证据仅提到贷款市场报价利率属于常见市场利率，未给出最新报价。",
            citation_ids=["[1]"],
        ),
    ]
    assert _claims_all_boundary(boundary) is True
    mixed = [
        AnswerClaim(text="无法提供2025年数据。", citation_ids=["[1]"]),
        AnswerClaim(text="2024年原保险保费收入为49951亿元。", citation_ids=["[2]"]),
    ]
    assert _claims_all_boundary(mixed) is False
    assert _claims_all_boundary([]) is False


def test_boundary_lead_matcher_is_false_for_period_statements() -> None:
    assert (
        _is_evidence_boundary_lead(
            "口径说明：期间标签未指定具体日期，按表内顺序取首项。"
        )
        is False
    )
    assert (
        _is_evidence_boundary_lead(
            "《寿险合同负债评估折现率曲线》规定：折现率曲线由基础利率曲线加综合溢价形成。"
        )
        is False
    )


def test_document_number_core_is_supported_from_cited_excerpt() -> None:
    evidence = _normalize(
        "会计师事务所、注册会计师应当严格遵守《关于加快推进银行函证规范化、集约化、数字化建设的通知》"
        "（财会〔2022〕39号）等文件和本操作指引的要求"
    )
    assert (
        _entity_is_supported("注册会计师应当严格遵守财会〔2022〕39号", evidence) is True
    )
    assert _entity_is_supported("财会〔2022〕39号", evidence) is True
    assert _entity_is_supported("财会〔2021〕39号", evidence) is False


def test_ratio_display_note_only_for_percent_scaled_calculations() -> None:
    assert (
        _ratio_display_note({"result_scale": 100, "calculation_result": 17.37})
        == "（比值0.1737）"
    )
    assert _ratio_display_note({"calculation_result": 17.37}) == ""
    assert _ratio_display_note({"result_scale": 100, "calculation_result": None}) == ""
    assert _ratio_display_note({}) == ""


def test_comparison_rewrite_names_the_candidate_rows() -> None:
    question = (
        "根据 Excel 附件《2024年9月人身险公司经营情况表》（工作表：人身保险公司（月度） ），"
        "在“本年累计/截至当期”口径下，以下哪一项数值最高？"
    )
    options = ["原保险保费收入", "意外险", "寿险", "新增保险金额"]
    updated = ensure_candidate_enumeration(question, options)
    assert "以下哪一项" not in updated
    for option in options:
        assert option in updated
    assert "新增保险金额四项的哪一项数值最高" in updated
    assert ensure_candidate_enumeration(updated, options) == updated


def test_statement_options_are_not_enumerated() -> None:
    question = (
        "请根据《中国银保监会关于实施 保险公司偿付能力监管规则（Ⅱ） 有关事项的通知》回答："
        "根据《寿险合同负债评估折现率曲线》，下列哪项表述正确。请直接给出规定内容，不要要求提供选项。"
    )
    options = [
        "750日移动平均国债收益率曲线适用于0年到20年的区间。",
        "列入名单的保险集团应当按照《保险公司偿付能力监管规则第19号：保险集团》有关规定编报保险集团偿付能力报告。",
        "其他符合保险集团定义的保险集团暂不编报偿付能力报告。",
        "中国人民保险集团股份有限公司属于应当编制保险集团偿付能力报告的保险控股型集团。",
    ]
    assert ensure_candidate_enumeration(question, options) == question
