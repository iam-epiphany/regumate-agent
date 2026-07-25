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
) -> None:
    document_id = "DOC-TEST-0001"
    if db_session.get(Document, 1) is None:
        db_session.add(
            Document(
                document_id=document_id,
                filename="2024年一季度全国各地区原保险保费收入情况表.xlsx",
                content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                file_type="xlsx",
                size=1024,
                storage_path="2024年一季度全国各地区原保险保费收入情况表.xlsx",
                document_metadata=json.dumps({"source_title": "2024年一季度全国各地区原保险保费收入情况表"}, ensure_ascii=False),
                status="indexed",
                index_version="test",
                chunk_count=1,
            )
        )
    metadata = {
        "spreadsheet_table": True,
        "table_chunk_role": "row",
        "source_title": "2024年一季度全国各地区原保险保费收入情况表",
        "source_format": "xlsx",
        "sheet_name": "2024年一季度",
        "table_title": "2024年一季度全国各地区原保险保费收入情况表",
        "unit": "亿元",
        "period": {"year": 2024, "quarter": 1, "month": None, "raw": "2024年一季度"},
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
