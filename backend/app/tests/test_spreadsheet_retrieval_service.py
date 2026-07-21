import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.core.database import Base
from backend.app.models.document import Document, DocumentChunk
from backend.app.services.query_planner_service import QueryAspect, QuerySearchQuery
from backend.app.services.spreadsheet_retrieval_service import (
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
            "本年累计 / 原保险保费收入": str(value),
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
                "column_label": "本年累计 / 原保险保费收入",
                "column_path": ["本年累计", "原保险保费收入"],
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
