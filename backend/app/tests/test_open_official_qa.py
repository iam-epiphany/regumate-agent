from pathlib import Path

from scripts.evaluate_contest_qa import Case, read_cases
from scripts.evaluation.open_official_qa import (
    audit_records,
    audit_records_v3,
    build_records,
    build_records_v3,
    transform_question,
    transform_question_v3,
)


def _case(*, case_id: str = "T001", qa_type: str, question: str, answer_text: str = "x", options: tuple[str, ...] = ("选项甲内容", "选项乙内容", "选项丙内容", "选项丁内容")) -> Case:
    return Case(
        id=case_id,
        source_type="excel",
        difficulty="",
        qa_type=qa_type,
        question=question,
        options=options,
        answer="A",
        answer_text=answer_text,
        evidence="",
        file_label="fixture.xlsx",
    )


def test_table_comparison_becomes_open_question_without_option_text() -> None:
    case = _case(
        qa_type="\u8868\u683c\u6bd4\u8f83",
        question="\u6839\u636e Excel \u9644\u4ef6\uff0c\u4ee5\u4e0b\u54ea\u4e00\u9879\u6570\u503c\u6700\u9ad8\uff1f",
    )

    question, kind = transform_question(case)

    assert kind == "table_comparison"
    assert "\u54ea\u4e00\u9879" not in question
    assert "\u6307\u51fa" in question


def test_choice_stem_becomes_open_regulatory_fact_request() -> None:
    case = _case(
        qa_type="\u5355\u4e8b\u5b9e\u68c0\u7d22",
        question="\u6839\u636e\u300a\u67d0\u529e\u6cd5\u300b\uff0c\u4e0b\u5217\u54ea\u9879\u8868\u8ff0\u6b63\u786e\uff1f",
    )

    question, kind = transform_question(case)

    assert kind == "regulatory_fact"
    assert "\u76f4\u63a5\u7ed9\u51fa" in question


def test_compound_choice_stem_becomes_open_regulatory_fact_request() -> None:
    case = _case(
        qa_type="\u591a\u4e8b\u5b9e\u68c0\u7d22",
        question="\u5173\u4e8e\u300a\u67d0\u529e\u6cd5\u300b\uff0c\u4e0b\u5217\u54ea\u4e00\u7ec4\u9009\u9879\u4e2d\u7684\u4e24\u9879\u8868\u8ff0\u5747\u5c5e\u4e8e\u8be5\u6750\u6599\u5185\u5bb9\uff1f",
    )

    question, kind = transform_question(case)

    assert kind == "regulatory_fact"
    assert "\u9009\u9879" not in question


def test_all_official_open_questions_pass_transform_audit() -> None:
    root = Path(__file__).resolve().parents[3]
    cases = read_cases(root / "data" / "contest_dataset" / "QA数据.xlsx")
    public, reference = build_records(cases)

    audit = audit_records(cases, public, reference)

    assert audit["passed"] is True
    assert audit["case_count"] == 300


# ---------------------------------------------------------------------------
# open-official-v3
# ---------------------------------------------------------------------------


def test_v3_fact_question_anchors_gold_statement() -> None:
    case = _case(
        qa_type="单事实检索",
        question="根据《某办法》，下列哪项表述正确？",
        answer_text="某办法规定借款人以消费为目的的贷款不包含住房贷款。",
    )

    question, kind = transform_question_v3(case)

    assert kind == "verify_fact"
    assert "判断以下表述是否符合该材料内容" in question
    assert "某办法规定借款人以消费为目的的贷款不包含住房贷款。" in question
    assert "下列" not in question
    assert "哪项" not in question


def test_v3_multi_fact_question_splits_statements() -> None:
    case = _case(
        qa_type="多事实检索",
        question="关于《某办法》，下列哪一组选项中的两项表述均属于该材料内容？",
        answer_text="第一项表述内容。；第二项表述内容。",
    )

    question, kind = transform_question_v3(case)

    assert kind == "verify_fact"
    assert "判断以下2项表述是否符合该材料内容" in question
    assert "（1）“第一项表述内容。”" in question
    assert "（2）“第二项表述内容。”" in question


def test_v3_table_comparison_embeds_candidates() -> None:
    case = _case(
        qa_type="表格比较",
        question="根据 Excel 附件《某表》（工作表：某工作表），在“某口径”口径下，以下哪一项数值最高？",
        options=("甲指标", "乙指标", "丙指标", "丁指标"),
        answer_text="甲指标",
    )

    question, kind = transform_question_v3(case)

    assert kind == "table_comparison"
    assert "比较以下指标：甲指标、乙指标、丙指标、丁指标" in question
    assert "哪一项数值最高？请给出该指标的名称和对应数值。" in question
    assert "以下哪一项" not in question


def test_v3_direct_table_keeps_original_question() -> None:
    case = _case(
        qa_type="表格取数",
        question="根据 Excel 附件《某表》，“甲指标”的数值是多少？",
    )

    question, kind = transform_question_v3(case)

    assert kind == "direct_table"
    assert question == "根据 Excel 附件《某表》，“甲指标”的数值是多少？"


def test_v3_same_regulation_cases_get_unique_questions() -> None:
    """v3 anchors the gold statement, so same-regulation cases must not share
    an identical question (the v2 collapse defect)."""

    cases = [
        _case(case_id="T001", qa_type="单事实检索", question="根据《某办法》，下列哪项表述正确？", answer_text="表述甲。"),
        _case(case_id="T002", qa_type="单事实检索", question="根据《某办法》，下列哪项表述正确？", answer_text="表述乙。"),
    ]
    public, reference = build_records_v3(cases)

    questions = {item["question"] for item in public}

    assert len(questions) == 2
    assert audit_records_v3(cases, public, reference)["passed"] is True


def test_v3_audit_rejects_identical_question_with_different_golds() -> None:
    """An identical transformed question (here a direct_table question whose
    stem carries no gold) with different golds must be rejected."""

    cases = [
        _case(case_id="T001", qa_type="表格取数", question="根据 Excel 附件《某表》，“甲指标”的数值是多少？", answer_text="100"),
        _case(case_id="T002", qa_type="表格取数", question="根据 Excel 附件《某表》，“甲指标”的数值是多少？", answer_text="200"),
    ]
    public, reference = build_records_v3(cases)

    audit = audit_records_v3(cases, public, reference)

    assert audit["passed"] is False
    assert any("duplicated transformed question carries different golds" in error for error in audit["errors"])


def test_v3_audit_rejects_leaked_non_answer_option() -> None:
    case = _case(
        qa_type="单事实检索",
        question="根据《某办法》，下列哪项表述正确？",
        answer_text="正确表述。",
        options=("正确表述。", "错误表述甲。", "错误表述乙。", "错误表述丙。"),
    )
    public, reference = build_records_v3([case])
    # Inject the wrong-option text into the question to simulate leakage.
    public[0]["question"] = public[0]["question"] + "错误表述乙。"
    reference[0]["question"] = public[0]["question"]

    audit = audit_records_v3([case], public, reference)

    assert audit["passed"] is False
    assert any("non-answer option text leaked" in error for error in audit["errors"])


def test_v3_all_official_open_questions_pass_audit_with_gold() -> None:
    root = Path(__file__).resolve().parents[3]
    cases = read_cases(root / "data" / "contest_dataset" / "QA数据.xlsx")
    gold_path = root / "data" / "evaluation" / "open_v3_gold_probe.json"
    assert gold_path.exists(), "run build_open_v3_gold.py before this test"
    import json

    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    public, reference = build_records_v3(cases, gold)

    audit = audit_records_v3(cases, public, reference, gold)

    assert audit["passed"] is True, audit["errors"][:5]
    assert audit["case_count"] == 300
