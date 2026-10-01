from scripts.evaluate_contest_qa import (
    Case,
    assign_splits,
    build_ood_cases,
    extract_option,
    grounded_option,
    format_ordered_options,
    parse_case_ids,
    score_excel_evidence,
    score_manifest_evidence,
    score_text_evidence,
    source_matches,
    summarize,
    timing_summary,
)
from scripts.validate_official_excel_cells import validate_cells, values_equal
import sqlite3


def _case(index: int, *, source_type: str = "excel") -> Case:
    return Case(
        id=f"Q{index:03d}",
        source_type=source_type,
        difficulty=("简单", "中等", "困难")[index % 3],
        qa_type="表格取数" if source_type == "excel" else "单事实检索",
        question=f"2025年测试问题{index}",
        options=("选项甲", "选项乙", "选项丙", "选项丁"),
        answer="B",
        answer_text="选项乙",
        evidence="指标甲=1（B2）；指标乙=2（C2）",
        file_label=f"source-{source_type}.xlsx",
    )


def test_assign_splits_is_deterministic_and_exactly_240_60() -> None:
    cases = [
        _case(index, source_type=("excel", "word", "pdf")[index % 3])
        for index in range(300)
    ]

    first = assign_splits(cases)
    second = assign_splits(cases)

    assert sum(case.split == "dev" for case in first) == 240
    assert sum(case.split == "holdout" for case in first) == 60
    assert [case.split for case in first] == [case.split for case in second]


def test_build_ood_cases_creates_30_explicit_refusal_questions() -> None:
    cases = [
        _case(index, source_type="excel" if index < 100 else "word" if index < 200 else "pdf")
        for index in range(300)
    ]

    ood = build_ood_cases(cases)

    assert len(ood) == 30
    assert all(case.answer == "REFUSE" and case.split == "ood" for case in ood)
    excel_questions = [case.question for case in ood if case.source_type == "excel"]
    assert sum("2099年" in question for question in excel_questions) == 5
    assert sum("虚构监管指标" in question for question in excel_questions) == 5
    assert sum("不存在的监管条款" in case.question for case in ood) == 20


def test_extract_option_accepts_letter_and_answer_text() -> None:
    options = ("资本充足率", "流动性比例", "拨备覆盖率", "杠杆率")

    assert extract_option("答案为 C。", options) == "C"
    assert extract_option("根据证据，应选择流动性比例。", options) == "B"


def test_parse_case_ids_accepts_generic_regression_subset() -> None:
    assert parse_case_ids("Q003, Q001,Q003") == {"Q001", "Q003"}
    assert parse_case_ids(None) == set()
    import pytest

    with pytest.raises(ValueError, match="QNNN"):
        parse_case_ids("Q1")


def test_format_ordered_options_preserves_order_with_selection_labels() -> None:
    assert format_ordered_options(("first", "second")) == ["A. first", "B. second"]


def test_extract_option_ignores_citation_numbers_and_rounding() -> None:
    options = ("1", "0.1091", "275230.63", "0.0292")

    assert extract_option("答案为 275230.63。依据表格证据，核验值为 275230.6254。[1]", options) == "C"


def test_extract_option_ignores_terminal_punctuation_for_text_options() -> None:
    options = (
        "消费金融公司是经批准设立、不吸收公众存款的非银行金融机构。",
        "不相关表述。",
        "另一项。",
        "其他项。",
    )

    assert extract_option("消费金融公司是经批准设立、不吸收公众存款的非银行金融机构[1]", options) == "A"


def test_grounded_option_wins_when_explanation_mentions_other_long_options() -> None:
    body = {
        "grounding_validation": {
            "passed": True,
            "selected_option": "0",
            "selected_option_supported": True,
            "selected_option_format_valid": True,
        }
    }

    assert grounded_option(body) == "A"


def test_manifest_evidence_scores_final_rerank_and_citations() -> None:
    result = {
        "citations": [{"chunk_id": "chunk-good"}, {"chunk_id": "chunk-extra"}],
        "context_package": {
            "context_chunks": [{"chunk_id": "chunk-good"}],
            "retrieval_summary": {
                "aspect_retrievals": [
                    {
                        "diagnostics": [
                            {
                                "query_type": "aspect_fused",
                                "reranked_ranking": [
                                    {"rank": 1, "chunk_id": "chunk-good"},
                                    {"rank": 2, "chunk_id": "chunk-extra"},
                                ],
                            }
                        ]
                    }
                ]
            },
        },
    }
    manifest = {
        "review_status": "reviewed",
        "required_aspects": [
            {"aspect_id": "a1", "acceptable_chunk_ids": ["chunk-good"]}
        ],
    }

    score = score_manifest_evidence(result, manifest)

    assert score["final_recall"] == 1.0
    assert score["rerank_retention"] == 1.0
    assert score["citation_completeness"] == 1.0
    assert score["citation_accuracy"] == 0.5
    assert score["best_acceptable_rerank_rank"] == 1


def test_source_match_tolerates_official_truncated_parenthesis_filename() -> None:
    case = Case(
        **{
            **_case(2, source_type="pdf").__dict__,
            "file_label": "附件：中资商业银行行政许可事项申请材料目录及格式要求（2023年版）.pdf",
            "source_title": "中国银保监会关于印发中资商业银行行政许可事项申请材料目录及格式要求的通知",
        }
    )

    assert source_matches(
        case,
        "430_中国银保监会关于印发中资商业银行行政许可事项申请材料_目录及格式要求的通知_"
        "附件：中资商业银行行政许可事项申请材料目录及格式要求（2023年.pdf",
    )


def test_score_excel_evidence_counts_cell_level_recall() -> None:
    case = _case(1)
    citations = [
        {
            "metadata": {
                "cell": "B2",
                "calculation_cells": [{"cell": "C2"}],
            }
        }
    ]

    score = score_excel_evidence(case, citations)

    assert score["expected_cells"] == ["B2", "C2"]
    assert score["actual_cells"] == ["B2", "C2"]
    assert score["cell_recall"] == 1.0


def test_score_text_evidence_requires_each_fact_in_citations() -> None:
    case = Case(
        **{
            **_case(2, source_type="word").__dict__,
            "evidence": "资本工具应当直接发行且实缴；资本工具没有到期日。",
        }
    )
    citations = [{"excerpt": "监管规则明确，资本工具应当直接发行且实缴。"}]

    score = score_text_evidence(case, citations)

    assert score["aspect_count"] == 2
    assert score["covered_aspect_count"] == 1
    assert score["aspect_coverage"] == 0.5


def test_comparison_evidence_ignores_later_duplicate_business_group() -> None:
    case = Case(
        **{
            **_case(3).__dict__,
            "qa_type": "表格比较",
            "options": ("原保险保费收入", "财产险", "人身险", "总资产"),
            "evidence": (
                "原保险保费收入=47945.35(C6)；财产险=10844.7(C7)；"
                "人身险=37100.65(C8)；原保险赔付支出=17310.82(C9)；"
                "财产险=6939.25(C10)；人身险=10371.57(C11)。"
            ),
        }
    )
    citations = [
        {
            "metadata": {
                "comparison_cells": [
                    {"cell": "C6"},
                    {"cell": "C7"},
                    {"cell": "C8"},
                ]
            }
        }
    ]

    score = score_excel_evidence(case, citations)

    assert score["expected_cells"] == ["C6", "C7", "C8"]
    assert score["cell_recall"] == 1.0


def test_official_cell_validator_trims_sheet_and_accepts_display_rounding(tmp_path) -> None:
    database = tmp_path / "cells.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE documents (document_id TEXT PRIMARY KEY, filename TEXT);
        CREATE TABLE spreadsheet_cells (
            document_id TEXT, sheet_name TEXT, coordinate TEXT, value TEXT,
            numeric_value REAL, row_label TEXT, column_label TEXT
        );
        INSERT INTO documents VALUES ('D1', 'rules.xlsx');
        INSERT INTO spreadsheet_cells VALUES (
            'D1', '保险业数据  ', 'B7', '0.02234890541', 0.02234890541, '收益率', '当期'
        );
        """
    )
    connection.commit()
    connection.close()

    results = validate_cells(
        database,
        [
            {
                "question_id": "Q1",
                "filename": "rules.xlsx",
                "sheet_name": "保险业数据",
                "coordinate": "B7",
                "label": "收益率",
                "expected_value": "0.0223",
            }
        ],
    )

    assert results[0]["located"] is True
    assert results[0]["value_matched"] is True
    assert values_equal("0.0223", "0.0231", 0.0231) is False


def test_timing_summary_reports_maximum() -> None:
    assert timing_summary([1.0, 2.0, 10.0]) == {
        "avg": 4.33,
        "p50": 2.0,
        "p95": 10.0,
        "max": 10.0,
    }


def test_evaluator_does_not_mislabel_first_case_as_cold_start() -> None:
    result = {
        "error": None,
        "elapsed_ms": 125.0,
        "answer_correct": True,
        "source_hit": True,
        "expected": "A",
        "source_type": "word",
        "qa_type": "单事实检索",
        "difficulty": "简单",
        "answer_type": "deterministic_selection",
        "generation_status": "completed",
        "excel_evidence": None,
        "text_evidence": None,
        "grounding_validation": {"passed": True},
        "refused": False,
        "citation_count": 1,
        "error_type": None,
    }

    summary = summarize([result])

    assert summary["first_case_ms"] == 125.0
    assert summary["cold_start_ms"] is None
    assert summary["elapsed_ms"]["max"] == 125.0
    assert summary["by_slice"]["answer_type"]["deterministic_selection"]["elapsed_ms"]["p95"] == 125.0
