import json
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.core.database import Base
from backend.app.models.document import Document, DocumentChunk
from backend.app.services.query_planner_service import QueryAspect, QuerySearchQuery
from backend.app.services.spreadsheet_retrieval_service import (
    SpreadsheetCandidate,
    _calculation_match_from_selected,
    _indexed_cell_score,
    _row_target_matches,
    _specific_column_target_from_question,
    _source_title_matches,
    get_last_spreadsheet_diagnostic,
    retrieve_spreadsheet_matches,
)
from backend.app.services.spreadsheet_cell_index_service import rebuild_spreadsheet_cell_index


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as session:
        yield session


def test_spreadsheet_retrieval_returns_cell_level_lookup(db_session) -> None:
    _add_table_row(
        db_session,
        chunk_id="DOC-TEST-0001-CHUNK-0001",
        row_label="全国合计 / 原保险保费收入",
        value=123.45,
        coordinate="C5",
    )
    aspect = _aspect(
        "全国合计的原保险保费收入是多少？",
        table_task="lookup",
        table_filters={
            "source_title": "2024年一季度全国各地区原保险保费收入情况表",
            "year": 2024,
            "quarter": 1,
            "row_label": "全国合计",
            "indicator": "原保险保费收入",
            "column_label": "本年累计 / 原保险保费收入",
        },
    )

    matches = retrieve_spreadsheet_matches(db_session, aspect)

    assert len(matches) == 1
    match = matches[0]
    assert match.citation.chunk_type == "table_cell"
    assert match.citation.metadata["sheet_name"] == "2024年一季度"
    assert match.citation.metadata["cell"] == "C5"
    assert match.citation.metadata["value"] == "123.45"
    assert match.citation.metadata["unit"] == "亿元"
    assert match.citation.metadata["row_label"] == "全国合计 / 原保险保费收入"
    assert match.citation.metadata["column_label"] == "本年累计 / 原保险保费收入"
    assert "单元格 C5" in match.citation.excerpt


def test_spreadsheet_retrieval_supports_max_compare(db_session) -> None:
    _add_table_row(db_session, chunk_id="DOC-TEST-0001-CHUNK-0001", row_label="北京 / 原保险保费收入", value=11.2, coordinate="C5")
    _add_table_row(db_session, chunk_id="DOC-TEST-0001-CHUNK-0002", row_label="上海 / 原保险保费收入", value=18.8, coordinate="C6")
    aspect = _aspect(
        "哪个地区原保险保费收入最高？",
        table_task="compare",
        operation="max",
        table_filters={"column_label": "本年累计 / 原保险保费收入", "indicator": "原保险保费收入"},
    )

    matches = retrieve_spreadsheet_matches(db_session, aspect)

    assert len(matches) == 1
    assert matches[0].citation.metadata["row_label"] == "上海 / 原保险保费收入"
    assert matches[0].citation.metadata["cell"] == "C6"
    assert matches[0].citation.metadata["value"] == "18.8"


def test_indexed_cell_score_keeps_row_total_separate_from_total_column() -> None:
    from types import SimpleNamespace

    total = SimpleNamespace(
        row_label_norm="全国合计",
        column_label_norm="2023年10月全国各地区原保险保费收入情况表合计",
        column_label="2023年10月全国各地区原保险保费收入情况表 / 合计",
        numeric_value=45167.98,
    )
    property_insurance = SimpleNamespace(
        row_label_norm="全国合计",
        column_label_norm="2023年10月全国各地区原保险保费收入情况表财产保险",
        column_label="2023年10月全国各地区原保险保费收入情况表 / 财产保险",
        numeric_value=11366.02,
    )
    filters = {
        "row_label": "全国合计",
        "column_label": "合计",
        "indicator": "原保险保费收入",
    }

    total_score = _indexed_cell_score(total, "全国合计的合计值", filters)
    property_score = _indexed_cell_score(property_insurance, "全国合计的合计值", filters)

    assert total_score > property_score


def test_spreadsheet_compare_row_selector_ignores_table_title_inside_column_label(db_session) -> None:
    column_label = "2025年3月全国各地区原保险保费收入情况表 / 健康险"
    _add_table_row(
        db_session,
        chunk_id="DOC-TEST-0001-CHUNK-0001",
        row_label="全  国",
        value=3781.8,
        coordinate="G4",
        column_label=column_label,
    )
    _add_table_row(
        db_session,
        chunk_id="DOC-TEST-0001-CHUNK-0002",
        row_label="北  京",
        value=224.92,
        coordinate="G5",
        column_label=column_label,
    )
    _add_table_row(
        db_session,
        chunk_id="DOC-TEST-0001-CHUNK-0003",
        row_label="天  津",
        value=38.44,
        coordinate="G6",
        column_label=column_label,
    )
    aspect = _aspect(
        "比较北京、全国、天津的健康险，给出最大项。",
        table_task="compare",
        operation="max",
        table_filters={"column_label": "健康险"},
        selectors=({"label": "北京"}, {"label": "全国"}, {"label": "天津"}),
    )

    matches = retrieve_spreadsheet_matches(db_session, aspect)

    assert len(matches) == 1
    assert matches[0].citation.metadata["row_label"] == "全  国"
    assert matches[0].citation.metadata["cell"] == "G4"
    assert [item["row_label"] for item in matches[0].citation.metadata["comparison_cells"]] == [
        "北  京",
        "全  国",
        "天  津",
    ]


def test_spreadsheet_compare_prefers_question_specific_column_over_generic_total(db_session) -> None:
    for row_label, total, health, row in (
        ("全  国", 52145.77, 8426.99, 4),
        ("北  京", 3267.58, 512.04, 5),
        ("天  津", 708.01, 97.29, 6),
    ):
        _add_table_row(
            db_session,
            chunk_id=f"DOC-TEST-TOTAL-CHUNK-{row:04d}",
            row_label=row_label,
            value=total,
            coordinate=f"C{row}",
            column_label="2025年9月全国各地区原保险保费收入情况表 / 合计",
        )
        _add_table_row(
            db_session,
            chunk_id=f"DOC-TEST-HEALTH-CHUNK-{row:04d}",
            row_label=row_label,
            value=health,
            coordinate=f"G{row}",
            column_label="2025年9月全国各地区原保险保费收入情况表 / 健康险",
        )
    aspect = _aspect(
        "比较2025年9月全国健康险合计、北京健康险和天津健康险，给出最大项。",
        table_task="compare",
        operation="max",
        table_filters={"column_label": "合计"},
        selectors=({"label": "全国"}, {"label": "北京"}, {"label": "天津"}),
    )

    matches = retrieve_spreadsheet_matches(db_session, aspect)

    assert len(matches) == 1
    assert matches[0].citation.metadata["row_label"] == "全  国"
    assert matches[0].citation.metadata["cell"] == "G4"
    assert matches[0].citation.metadata["value"] == "8426.99"
    assert {item["column_label"].rsplit(" / ", 1)[-1] for item in matches[0].citation.metadata["comparison_cells"]} == {
        "健康险"
    }


def test_spreadsheet_calculation_keeps_all_selector_operand_columns(db_session) -> None:
    """Calculation operands must be selected before any single-column recovery."""

    for value, coordinate, column_label in (
        (100.0, "C5", "metrics / total"),
        (30.0, "D5", "metrics / category"),
    ):
        _add_table_row(
            db_session,
            chunk_id=f"DOC-CALC-CHUNK-{coordinate}",
            row_label="region total",
            value=value,
            coordinate=coordinate,
            column_label=column_label,
        )
    aspect = _aspect(
        "What is the change from total to category for the region total?",
        table_task="calculate",
        operation="difference",
        table_filters={"row_label": "region total", "column_label": "category"},
        selectors=(
            {"row_label": "region total", "column_label": "total", "ordered_transition": True},
            {"row_label": "region total", "column_label": "category"},
        ),
    )

    matches = retrieve_spreadsheet_matches(db_session, aspect)

    assert len(matches) == 1
    assert matches[0].citation.metadata["calculation_result"] == pytest.approx(-70.0)
    assert [cell["cell"] for cell in matches[0].citation.metadata["calculation_cells"]] == ["C5", "D5"]


def test_spreadsheet_ratio_returns_percentage_with_auditable_scale(db_session) -> None:
    for value, coordinate, column_label in (
        (25.0, "C5", "健康险"),
        (100.0, "D5", "合计"),
    ):
        _add_table_row(
            db_session,
            chunk_id=f"DOC-RATIO-CHUNK-{coordinate}",
            row_label="全国合计",
            value=value,
            coordinate=coordinate,
            column_label=column_label,
        )
    aspect = _aspect(
        "健康险数值占合计数值的百分比是多少？结果保留两位小数。",
        table_task="calculate",
        operation="ratio",
        selectors=(
            {"row_label": "全国合计", "column_label": "健康险"},
            {"row_label": "全国合计", "column_label": "合计"},
        ),
    )

    matches = retrieve_spreadsheet_matches(db_session, aspect)

    assert len(matches) == 1
    metadata = matches[0].citation.metadata
    assert metadata["calculation_result"] == pytest.approx(25.0)
    assert metadata["unit"] == "%"
    assert metadata["result_scale"] == 100
    assert metadata["display_decimal_places"] == 2
    assert metadata["calculation_display_formula"] == "25 / 100 × 100 = 25.00"


def test_spreadsheet_difference_preserves_source_operand_precision(db_session) -> None:
    for value, coordinate, column_label in (
        (0.109398205216, "C5", "一季度"),
        (0.09928, "D5", "四季度"),
    ):
        _add_table_row(
            db_session,
            chunk_id=f"DOC-PRECISE-CHUNK-{coordinate}",
            row_label="同比增长率",
            value=value,
            coordinate=coordinate,
            column_label=column_label,
        )
    aspect = _aspect(
        "计算同比增长率四季度比一季度的差额，并列出原始值。",
        table_task="calculate",
        operation="difference",
        selectors=(
            {"row_label": "同比增长率", "column_label": "一季度", "ordered_transition": True},
            {"row_label": "同比增长率", "column_label": "四季度"},
        ),
    )

    matches = retrieve_spreadsheet_matches(db_session, aspect)

    assert len(matches) == 1
    assert (
        matches[0].citation.metadata["calculation_display_formula"]
        == "0.09928 - 0.109398205216 = -0.010118205216"
    )


def test_question_specific_column_recovery_respects_selector_source_scope() -> None:
    document = SimpleNamespace(
        document_id="DOC-TABLE",
        filename="2025年9月全国各地区原保险保费收入情况表.xlsx",
    )
    regional_candidate = SpreadsheetCandidate(
        document=document,
        chunk=SimpleNamespace(chunk_id="DOC-TABLE-CHUNK-0001"),
        metadata={"source_title": "2025年9月全国各地区原保险保费收入情况表"},
        score=0.9,
        selected_cell={"column_label": "2025年9月全国各地区原保险保费收入情况表 / 人身险"},
    )
    scoped_candidate = SpreadsheetCandidate(
        document=document,
        chunk=SimpleNamespace(chunk_id="DOC-TABLE-CHUNK-0002"),
        metadata={"source_title": "2025年9月人身险公司经营情况表"},
        score=0.9,
        selected_cell={"column_label": "2025年9月人身险公司经营情况表 / 本年累计/截至当期"},
    )

    recovered = _specific_column_target_from_question(
        "比较2025年9月人身险公司原保险保费收入本年累计值。",
        [regional_candidate, scoped_candidate],
        selectors=[{"source_title": "2025年9月人身险公司经营情况表"}],
    )

    assert recovered == ""


def test_question_specific_column_recovery_ignores_structural_header_names() -> None:
    document = SimpleNamespace(
        document_id="DOC-TABLE",
        filename="sample.xlsx",
    )
    chunk = SimpleNamespace(chunk_id="DOC-TABLE-CHUNK-0001")
    candidates = [
        SpreadsheetCandidate(
            document=document,
            chunk=chunk,
            metadata={"source_title": "sample"},
            score=0.9,
            selected_cell={"column_label": "项目"},
        ),
        SpreadsheetCandidate(
            document=document,
            chunk=chunk,
            metadata={"source_title": "sample"},
            score=0.9,
            selected_cell={"column_label": "指标"},
        ),
        SpreadsheetCandidate(
            document=document,
            chunk=chunk,
            metadata={"source_title": "sample"},
            score=0.9,
            selected_cell={"column_label": "本年累计/截至当期"},
        ),
    ]

    recovered = _specific_column_target_from_question(
        "请比较多个项目，找出数值最高的项目名称、指标名称和单位。",
        candidates,
    )

    assert recovered == ""


def test_source_title_match_accepts_insurance_alias_and_zero_padded_month() -> None:
    assert _source_title_matches(
        "2024年9月人身险公司经营情况表",
        "2024年09月人身保险公司经营情况表",
    )
    assert _source_title_matches(
        "2025年3月财产保险公司经营情况表",
        "2025年03月财产险公司经营情况表",
    )
    assert _source_title_matches(
        "148_2023年3季度保险业资金运用情况表",
        "2023年3季度保险业资金运用情况表_2023年三季度保险业资金运用情况表",
    )
    assert _source_title_matches(
        "2022年商业银行主要指标分机构类情况表(法人)",
        "商业银行主要指标分机构类情况表(法人)",
    )


def test_nationwide_row_label_accepts_total_alias() -> None:
    assert _row_target_matches("全国合计", "全国") is True
    assert _row_target_matches("全国", "全国合计") is True
    assert _row_target_matches("北京", "全国") is False


def test_spreadsheet_retrieval_supports_difference_calculation(db_session) -> None:
    _add_table_row(db_session, chunk_id="DOC-TEST-0001-CHUNK-0001", row_label="北京 / 原保险保费收入", value=11.2, coordinate="C5")
    _add_table_row(db_session, chunk_id="DOC-TEST-0001-CHUNK-0002", row_label="上海 / 原保险保费收入", value=18.8, coordinate="C6")
    aspect = _aspect(
        "北京和上海的原保险保费收入差值是多少？",
        table_task="calculate",
        operation="difference",
        table_filters={"column_label": "本年累计 / 原保险保费收入", "indicator": "原保险保费收入"},
    )

    matches = retrieve_spreadsheet_matches(db_session, aspect)

    assert len(matches) == 1
    match = matches[0]
    assert match.citation.chunk_type == "table_calculation"
    assert match.citation.metadata["operation"] == "difference"
    assert match.citation.metadata["calculation_formula"] == "2024年一季度!C5 - 2024年一季度!C6"
    assert match.citation.metadata["calculation_result"] == pytest.approx(-7.6)
    assert len(match.citation.metadata["calculation_cells"]) == 2


def test_spreadsheet_retrieval_refuses_missing_period(db_session) -> None:
    _add_table_row(
        db_session,
        chunk_id="DOC-TEST-0001-CHUNK-0001",
        row_label="全国合计 / 原保险保费收入",
        value=123.45,
        coordinate="C5",
    )
    aspect = _aspect(
        "2099年全国合计的原保险保费收入是多少？",
        table_task="lookup",
        table_filters={
            "source_title": "2099年一季度全国各地区原保险保费收入情况表",
            "year": 2099,
            "quarter": 1,
            "row_label": "全国合计",
            "indicator": "原保险保费收入",
        },
    )

    matches = retrieve_spreadsheet_matches(db_session, aspect)

    assert matches == []
    diagnostic = get_last_spreadsheet_diagnostic()
    assert diagnostic["match_status"] == "period_not_found"
    assert "指定年份" in diagnostic["refusal_reason"]


def test_spreadsheet_retrieval_refuses_missing_indicator(db_session) -> None:
    _add_table_row(
        db_session,
        chunk_id="DOC-TEST-0001-CHUNK-0001",
        row_label="全国合计 / 原保险保费收入",
        value=123.45,
        coordinate="C5",
    )
    aspect = _aspect(
        "请查询不存在的指标“虚构监管指标Z-002”的数值。",
        table_task="lookup",
        table_filters={
            "source_title": "2024年一季度全国各地区原保险保费收入情况表",
            "year": 2024,
            "quarter": 1,
            "indicator": "虚构监管指标Z-002",
        },
        selectors=({"row_or_indicator": "虚构监管指标Z-002"},),
    )

    matches = retrieve_spreadsheet_matches(db_session, aspect)

    assert matches == []
    diagnostic = get_last_spreadsheet_diagnostic()
    assert diagnostic["match_status"] == "indicator_not_found"
    assert "指定指标" in diagnostic["refusal_reason"]


def _aspect(
    question: str,
    *,
    table_task: str,
    operation: str = "none",
    table_filters: dict | None = None,
    selectors: tuple[dict, ...] = (),
) -> QueryAspect:
    return QueryAspect(
        aspect_id="table_evidence",
        question=question,
        search_queries=(QuerySearchQuery(question, "semantic_question", ""),),
        evidence_need="表格工作表、行列标签、单元格坐标和值",
        keywords=("原保险保费收入",),
        modality="table",
        table_task=table_task,
        table_filters=table_filters or {},
        operation=operation,
        selectors=selectors,
    )


def _add_table_row(
    db_session,
    *,
    chunk_id: str,
    row_label: str,
    value: float,
    coordinate: str,
    column_label: str = "本年累计 / 原保险保费收入",
    source_title: str = "2024年一季度全国各地区原保险保费收入情况表",
    year: int = 2024,
    quarter: int | None = 1,
) -> None:
    document_id = "DOC-TEST-0001"
    period_raw = f"{year}年{quarter}季度" if quarter is not None else f"{year}年"
    if db_session.get(Document, 1) is None:
        db_session.add(
            Document(
                document_id=document_id,
                filename=f"{source_title}.xlsx",
                content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                file_type="xlsx",
                size=1024,
                storage_path=f"{source_title}.xlsx",
                document_metadata=json.dumps({"source_title": source_title}, ensure_ascii=False),
                status="indexed",
                index_version="test",
                chunk_count=1,
            )
        )
    sheet_name = "2024年一季度" if source_title == "2024年一季度全国各地区原保险保费收入情况表" else source_title
    metadata = {
        "spreadsheet_table": True,
        "table_chunk_role": "row",
        "source_title": source_title,
        "source_format": "xlsx",
        "sheet_name": sheet_name,
        "table_title": source_title,
        "unit": "亿元",
        "period": {"year": year, "quarter": quarter, "month": None, "raw": period_raw},
        "row_label": row_label,
        "row_cells": {
            "地区": row_label.split(" / ")[0],
            "指标": "原保险保费收入",
            column_label: str(value),
        },
        "cells": [
            {
                "coordinate": "A" + coordinate[1:],
                "row_label": row_label,
                "column_label": "地区",
                "column_path": ["地区"],
                "value": row_label.split(" / ")[0],
                "normalized_value": row_label.split(" / ")[0],
                "unit": "亿元",
            },
            {
                "coordinate": "B" + coordinate[1:],
                "row_label": row_label,
                "column_label": "指标",
                "column_path": ["指标"],
                "value": "原保险保费收入",
                "normalized_value": "原保险保费收入",
                "unit": "亿元",
            },
            {
                "coordinate": coordinate,
                "row_label": row_label,
                "column_label": column_label,
                "column_path": [part for part in column_label.split(" / ") if part],
                "value": str(value),
                "normalized_value": value,
                "unit": "亿元",
            },
        ],
    }
    db_session.add(
        DocumentChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            text=f"表格行证据：在《2024年一季度全国各地区原保险保费收入情况表》中，行标签为“{row_label}”。本年累计 / 原保险保费收入=“{value}”。",
            embedding_text=f"{row_label} 原保险保费收入 {value}",
            chunk_metadata=json.dumps(metadata, ensure_ascii=False),
            token_count=20,
            index_status="indexed",
            index_version="test",
            title="2024年一季度全国各地区原保险保费收入情况表",
            source_file="2024年一季度全国各地区原保险保费收入情况表.xlsx",
            page_number=None,
            section_title="2024年一季度全国各地区原保险保费收入情况表",
        )
    )
    db_session.flush()
    document = db_session.query(Document).filter_by(document_id=document_id).one()
    rebuild_spreadsheet_cell_index(db_session, document)
    db_session.commit()


def test_three_operand_reconciliation_uses_first_minus_remaining_operands() -> None:
    document = SimpleNamespace(
        document_id="DOC-RECON",
        filename="reconciliation.xlsx",
        source_url=None,
        attachment_url=None,
        title=None,
        issuing_authority=None,
        publication_date=None,
        document_number=None,
        version_status="unknown",
    )
    candidates = []
    for index, (value, coordinate) in enumerate(((21745, "C5"), (16590.18, "C6"), (5155, "C7")), start=1):
        chunk = SimpleNamespace(
            chunk_id=f"DOC-RECON-CHUNK-{index:04d}",
            section_title="原保险保费收入",
            page_number=None,
        )
        candidates.append(
            SpreadsheetCandidate(
                document=document,
                chunk=chunk,
                metadata={"sheet_name": "2025年3月", "unit": "亿元"},
                score=0.9,
                selected_cell={
                    "coordinate": coordinate,
                    "value": str(value),
                    "normalized_value": value,
                    "unit": "亿元",
                },
            )
        )
    aspect = SimpleNamespace(aspect_id="table_evidence", question="计算勾稽差额")

    match = _calculation_match_from_selected(
        candidates,
        aspect,
        "difference",
        ordered_transition=False,
    )

    assert match is not None
    assert match.metadata["calculation_result"] == pytest.approx(-0.18)
    assert match.metadata["calculation_display_formula"] == "21745 - 16590.18 - 5155 = -0.18"
    assert len(match.metadata["calculation_cells"]) == 3


def _add_fund_table_rows(db_session) -> None:
    """资金运用情况表三行：保险公司合计 / 其中：财产险公司 / 人身险公司。"""
    _add_table_row(
        db_session,
        chunk_id="DOC-TEST-0001-CHUNK-0001",
        row_label="保险公司 / 资金运用余额",
        value=384799.2195,
        coordinate="C5",
        column_label="截至当期 / 账面余额",
    )
    _add_table_row(
        db_session,
        chunk_id="DOC-TEST-0001-CHUNK-0002",
        row_label="其中：财产险公司 / 资金运用余额",
        value=24155.58512,
        coordinate="C6",
        column_label="截至当期 / 账面余额",
    )
    _add_table_row(
        db_session,
        chunk_id="DOC-TEST-0001-CHUNK-0003",
        row_label="人身险公司 / 资金运用余额",
        value=346645.0491,
        coordinate="C7",
        column_label="截至当期 / 账面余额",
    )


def test_lookup_prefers_institution_specific_row_over_aggregate(db_session) -> None:
    """机构类别行定位词必须选中财产险子行，而不是只匹配指标的合计行。"""
    _add_fund_table_rows(db_session)
    aspect = _aspect(
        "财产险公司资金运用余额是多少？",
        table_task="lookup",
        table_filters={
            "source_title": "2024年一季度全国各地区原保险保费收入情况表",
            "year": 2024,
            "quarter": 1,
            "row_label": "财产险公司 资金运用余额",
            "indicator": "资金运用余额",
            "column_label": "截至当期 / 账面余额",
        },
    )

    matches = retrieve_spreadsheet_matches(db_session, aspect)

    assert len(matches) == 1
    assert "财产险公司" in matches[0].citation.metadata["row_label"]
    assert matches[0].citation.metadata["value"] == "24155.58512"


def test_lookup_aggregate_row_with_institution_word(db_session) -> None:
    """问'保险公司'（合计行定位词）时仍应选中保险公司合计行。"""
    _add_fund_table_rows(db_session)
    aspect = _aspect(
        "保险公司资金运用余额是多少？",
        table_task="lookup",
        table_filters={
            "source_title": "2024年一季度全国各地区原保险保费收入情况表",
            "year": 2024,
            "quarter": 1,
            "row_label": "保险公司 资金运用余额",
            "indicator": "资金运用余额",
            "column_label": "截至当期 / 账面余额",
        },
    )

    matches = retrieve_spreadsheet_matches(db_session, aspect)

    assert len(matches) == 1
    assert "保险公司" in matches[0].citation.metadata["row_label"]
    assert "财产险" not in matches[0].citation.metadata["row_label"]
    assert matches[0].citation.metadata["value"] == "384799.2195"


def test_lookup_bare_indicator_refuses_ambiguous_rows(db_session) -> None:
    """单词 row_label（无机构类别）命中多行时按可信原则歧义拒绝，而非任取一行。"""
    _add_fund_table_rows(db_session)
    aspect = _aspect(
        "资金运用余额是多少？",
        table_task="lookup",
        table_filters={
            "source_title": "2024年一季度全国各地区原保险保费收入情况表",
            "year": 2024,
            "quarter": 1,
            "row_label": "资金运用余额",
            "indicator": "资金运用余额",
            "column_label": "截至当期 / 账面余额",
        },
    )

    matches = retrieve_spreadsheet_matches(db_session, aspect)

    assert matches == []
    assert get_last_spreadsheet_diagnostic().get("match_status") == "ambiguous_candidates"


def test_indexed_cell_score_requires_all_row_label_words() -> None:
    from types import SimpleNamespace

    property_row = SimpleNamespace(
        row_label_norm="其中:财产险公司资金运用余额",
        column_label_norm="截至当期账面余额",
        column_label="截至当期 / 账面余额",
        numeric_value=24155.58512,
    )
    aggregate_row = SimpleNamespace(
        row_label_norm="保险公司资金运用余额",
        column_label_norm="截至当期账面余额",
        column_label="截至当期 / 账面余额",
        numeric_value=384799.2195,
    )
    filters = {
        "row_label": "财产险公司 资金运用余额",
        "indicator": "资金运用余额",
    }

    property_score = _indexed_cell_score(property_row, "财产险公司资金运用余额", filters)
    aggregate_score = _indexed_cell_score(aggregate_row, "财产险公司资金运用余额", filters)

    # 多词 row_label 必须全部覆盖：财产险行 +5，合计行（缺财产险公司）不得分。
    assert property_score > aggregate_score


def test_row_target_matches_requires_all_words() -> None:
    assert _row_target_matches("财产险公司 资金运用余额", "其中:财产险公司资金运用余额")
    assert not _row_target_matches("财产险公司 资金运用余额", "保险公司资金运用余额")
    assert _row_target_matches("资金运用余额", "其中:财产险公司资金运用余额")
    assert _row_target_matches("全国", "全国合计")


def _add_transposed_bank_rows(db_session) -> None:
    """转置布局表：机构类别在列、“X季度 / 指标”组合在行。"""
    for quarter, coord, value in (
        ("一季度", "G5", 8234.537443),
        ("二季度", "G6", 8051.364405),
        ("三季度", "G7", 8274.705191),
        ("四季度", "G8", 8064.958648),
    ):
        _add_table_row(
            db_session,
            chunk_id=f"DOC-TEST-0001-CHUNK-{coord}",
            row_label=f"{quarter} / 不良贷款余额",
            value=value,
            coordinate=coord,
            column_label="农村商业银行",
            source_title="2025年商业银行主要指标分机构类情况表（季度）",
            year=2025,
            quarter=None,
        )
    _add_table_row(
        db_session,
        chunk_id="DOC-TEST-0001-CHUNK-OTHER",
        row_label="一季度 / 不良贷款率",
        value=0.02862,
        coordinate="G9",
        column_label="农村商业银行",
        source_title="2025年商业银行主要指标分机构类情况表（季度）",
        year=2025,
        quarter=None,
    )


def test_transposed_lookup_institution_column(db_session) -> None:
    """转置表（机构在列、时间/指标在行）：时间+指标匹配行、机构词匹配列。"""
    _add_transposed_bank_rows(db_session)
    aspect = _aspect(
        "根据《2025年商业银行主要指标分机构类情况表（季度）》，农村商业银行一季度不良贷款余额是多少？",
        table_task="lookup",
        table_filters={
            "source_title": "2025年商业银行主要指标分机构类情况表（季度）",
            "year": 2025,
            "quarter": 1,
            "row_label": "农村商业银行",
            "indicator": "不良贷款余额",
        },
    )

    matches = retrieve_spreadsheet_matches(db_session, aspect)

    assert len(matches) == 1
    assert matches[0].citation.metadata["cell"] == "G5"
    assert matches[0].citation.metadata["value"] == "8234.537443"


def test_transposed_calculate_cross_period_difference(db_session) -> None:
    """转置表跨期差值：一季度与四季度按文本序（先 − 后）。"""
    _add_transposed_bank_rows(db_session)
    aspect = _aspect(
        "根据《2025年商业银行主要指标分机构类情况表（季度）》，农村商业银行一季度与四季度不良贷款余额的差值是多少？",
        table_task="calculate",
        operation="difference",
        table_filters={
            "source_title": "2025年商业银行主要指标分机构类情况表（季度）",
            "row_label": "农村商业银行",
            "indicator": "不良贷款余额",
        },
        selectors=(
            {"year": 2025, "quarter": 1, "row_label": "不良贷款余额"},
            {"year": 2025, "quarter": 4, "row_label": "不良贷款余额"},
        ),
    )

    matches = retrieve_spreadsheet_matches(db_session, aspect)

    assert len(matches) == 1
    assert matches[0].citation.metadata["calculation_result"] == pytest.approx(169.578795)
    assert matches[0].citation.metadata["calculation_display_formula"] == "8234.537443 - 8064.958648 = 169.578795"


def test_selector_label_word_and_with_structural_prefix(db_session) -> None:
    """“财产险公司 银行存款” 在行 “其中：财产险公司 / 其中：银行存款” 中
    被 “其中：” 隔开，应按词级 AND 匹配而非连续子串。"""
    _add_table_row(
        db_session,
        chunk_id="DOC-TEST-0001-CHUNK-D1",
        row_label="其中：财产险公司 / 其中：银行存款",
        value=3887.79602,
        coordinate="C7",
        column_label="截至当期 / 账面余额",
    )
    _add_table_row(
        db_session,
        chunk_id="DOC-TEST-0001-CHUNK-D2",
        row_label="其中：财产险公司 / 资金运用余额",
        value=24155.58512,
        coordinate="C6",
        column_label="截至当期 / 账面余额",
    )
    aspect = _aspect(
        "财产险公司银行存款占资金运用余额的比例约为多少？",
        table_task="calculate",
        operation="ratio",
        table_filters={
            "source_title": "2024年一季度全国各地区原保险保费收入情况表",
            "year": 2024,
            "quarter": 1,
            "indicator": "银行存款",
            "row_label": "财产险公司 银行存款",
        },
        selectors=(
            {"row_or_indicator": "财产险公司 银行存款"},
            {"row_or_indicator": "财产险公司 资金运用余额"},
        ),
    )

    matches = retrieve_spreadsheet_matches(db_session, aspect)

    assert len(matches) == 1
    assert matches[0].citation.metadata["calculation_result"] == pytest.approx(0.1609481203, rel=1e-6)


def test_operand_decimal_prefers_displayed_value(db_session) -> None:
    """计算操作数优先取显示值字符串，避免 float 精度噪声。"""
    from decimal import Decimal

    from backend.app.services.spreadsheet_retrieval_service import _operand_decimal

    candidate = SimpleNamespace(
        selected_cell={
            "value": "24155.58512",
            "normalized_value": 24155.58512172,
        },
    )
    assert _operand_decimal(candidate) == Decimal("24155.58512")
