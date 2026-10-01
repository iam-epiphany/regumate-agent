from backend.app.services.answer_generation_service import (
    _question_soft_anchors,
    _relevant_extractive_excerpt,
    _soft_anchor_score,
)


def test_question_soft_anchors_splits_on_function_words() -> None:
    anchors = _question_soft_anchors(
        "保险公司偿付能力监管规则第3号：寿险合同负债评估和市场风险（利率风险）监管规则中，"
        "寿险合同负债评估的折现率曲线是如何构成的？"
    )
    assert "折现率曲线" in anchors
    assert "寿险合同负债评估" in anchors


def test_soft_anchor_score_requires_co_occurrence() -> None:
    anchors = ("折现率曲线", "寿险合同负债评估")
    assert _soft_anchor_score("折现率曲线由多个要素构成", anchors) == 0.0
    assert _soft_anchor_score("折现率曲线是寿险合同负债评估的组成", anchors) == 4.0
    assert _soft_anchor_score("与问题无关的句子", anchors) == 0.0


def test_soft_anchor_score_single_anchor_question_neutral() -> None:
    # A question yielding one fragment cannot earn the co-occurrence bonus.
    assert _soft_anchor_score("折现率曲线内容", ("折现率曲线",)) == 0.0


def test_excerpt_prefers_multi_fragment_sentences() -> None:
    text = (
        "商业银行应当建立健全操作风险管理体系。"
        "寿险合同负债评估的折现率曲线由基础利率曲线和综合溢价构成。"
        "其他无关内容不计入评估范围。"
    )
    question = "寿险合同负债评估的折现率曲线是如何构成的？"
    excerpt = _relevant_extractive_excerpt(text, question)
    assert "折现率曲线由基础利率曲线" in excerpt
