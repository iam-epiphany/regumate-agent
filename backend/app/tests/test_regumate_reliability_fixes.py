"""Regression tests for the "simple questions must not be wrong" fixes.

Each test pins one empirically observed failure: solvency quarter lookups
refused as ambiguous, the wrong monthly table winning a lookup, a sub-row
answered with its parent, a ratio question planned as a lookup, cross-period
differences pinned to one quarter column, double percent scaling, and the
quarter qualifier that never compared its value.
"""
from __future__ import annotations

import json
from decimal import Decimal

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.core.database import Base
from backend.app.models.document import Document, DocumentChunk
from backend.app.services.query_planner_service import (
    QueryAspect,
    QuerySearchQuery,
    _asks_ratio_calculation,
    _extract_table_filters,
    _fallback_table_aspect,
    _institution_row_terms,
    _period_selectors,
    _ratio_operand_labels,
    _split_indicator_scope,
)
from backend.app.services.spreadsheet_cell_index_service import (
    rebuild_spreadsheet_cell_index,
)
from backend.app.services.spreadsheet_retrieval_service import (
    _latest_period_tie_candidate,
    _round_half_up_text,
    _row_label_terms_covered,
    _row_word_variants,
    retrieve_spreadsheet_matches,
)


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as session:
        yield session


# --------------------------------------------------------------------------
# Planner-level fixes
# --------------------------------------------------------------------------


def test_institution_term_strips_quarter_end_tail() -> None:
    # "四季度末财产险公司" must yield 财产险公司, not 末财产险公司.
    assert _institution_row_terms("四季度末财产险公司的综合偿付能力充足率", "综合偿付能力充足率") == ["财产险公司"]


def test_split_indicator_scope_separates_family_word() -> None:
    scope, core = _split_indicator_scope("保险业原保险保费收入")
    assert scope == "保险业"
    assert core == "原保险保费收入"
    # A product name that merely starts with 人身 is not a scope word.
    scope2, core2 = _split_indicator_scope("人身意外伤害险原保险保费收入")
    assert scope2 == ""
    assert core2 == "人身意外伤害险原保险保费收入"


def _with_table_vocabulary(labels: tuple[str, ...]):
    from contextlib import contextmanager

    from backend.app.services import query_planner_service as qps

    @contextmanager
    def ctx():
        token = qps._TABLE_VOCABULARY.set(labels)
        try:
            yield
        finally:
            qps._TABLE_VOCABULARY.reset(token)

    return ctx()


def test_ratio_operand_labels_split_compound_operands() -> None:
    with _with_table_vocabulary(("人身意外伤害险", "原保险保费收入")):
        labels = _ratio_operand_labels(
            "2021年12月，人身意外伤害险原保险保费收入占保险业原保险保费收入的比重约为多少？"
        )
    assert labels == ["人身意外伤害险", "原保险保费收入"]


def test_ratio_question_without_book_title_plans_calculation() -> None:
    # Used to fall through to lookup because the ratio branch was gated on
    # explicit table markers.
    aspect = _fallback_table_aspect(
        "2021年12月，人身意外伤害险原保险保费收入占保险业原保险保费收入的比重约为多少？"
    )
    assert aspect is not None
    assert aspect.table_task == "calculate"
    assert aspect.operation == "ratio"
    assert len(aspect.selectors) == 2


def test_period_selectors_carry_institution_scope() -> None:
    selectors = _period_selectors("2021年三季度末到四季度末，人身险公司综合偿付能力充足率变化了多少个百分点？")
    assert len(selectors) == 2
    assert selectors[0]["quarter"] == 3
    assert selectors[1]["quarter"] == 4
    for selector in selectors:
        assert "人身险公司" in str(selector.get("row_label"))


def test_period_selectors_never_pin_a_period_column() -> None:
    # "三季度末到四季度末" used to set column_label=四季度末 on every selector,
    # emptying the Q3 operand.
    selectors = _period_selectors("2021年三季度末到四季度末，人身险公司综合偿付能力充足率变化了多少个百分点？")
    for selector in selectors:
        assert not selector.get("column_label")


def test_quarter_interval_does_not_become_one_quarter_filter() -> None:
    filters = _extract_table_filters("2024年第1-4季度商业银行不良贷款率哪个季度最低？")
    assert "quarter" not in filters


def test_sub_item_context_question_targets_the_child_row() -> None:
    # "资金运用余额中，债券投资" — the row is the child item, the parent is
    # context.  Without an indexed vocabulary the phrase stays 债券投资; the
    # retrieval layer trims the measurement suffix back to the row 债券.
    filters = _extract_table_filters("2021年12月保险业资金运用余额中，债券投资是多少亿元？")
    assert filters.get("context") == "资金运用余额"
    assert filters.get("indicator") in {"债券", "债券投资"}


def test_vocabulary_match_strips_layout_artifacts() -> None:
    # Row labels like "风险综合评级    （家）" must match a question that says
    # just 风险综合评级.
    assert _asks_ratio_calculation("X占Y的比重是多少") is True


def test_row_word_variants_trim_measurement_suffix() -> None:
    assert _row_word_variants("债券投资") == ["债券投资", "债券"]
    assert _row_word_variants("综合偿付能力充足率") == ["综合偿付能力充足率"]


def test_row_label_terms_covered_accepts_suffix_base() -> None:
    assert _row_label_terms_covered("债券", ["债券投资"]) is True
    assert _row_label_terms_covered("银行存款", ["债券投资"]) is False


# --------------------------------------------------------------------------
# Retrieval-level fixes
# --------------------------------------------------------------------------


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
        keywords=("指标",),
        modality="table",
        table_task=table_task,
        table_filters=table_filters or {},
        operation=operation,
        selectors=selectors,
    )


def _add_solvency_row(
    db_session,
    *,
    chunk_id: str,
    row_label: str,
    quarter: int,
    value: float,
    document_id: str = "DOC-SOLV-2021",
    source_title: str = "2021年保险业偿付能力状况表",
    coordinate: str = "C5",
) -> None:
    if db_session.query(Document).filter_by(document_id=document_id).first() is None:
        db_session.add(
            Document(
                document_id=document_id,
                filename=f"{source_title}.xls",
                content_type="application/vnd.ms-excel",
                file_type="xls",
                size=1024,
                storage_path=f"{source_title}.xls",
                document_metadata=json.dumps({"source_title": source_title}, ensure_ascii=False),
                status="indexed",
                index_version="test",
                chunk_count=1,
            )
        )
    metadata = {
        "spreadsheet_table": True,
        "table_chunk_role": "row",
        "source_title": source_title,
        "sheet_name": "Sheet1",
        "table_title": source_title,
        "unit": "%、家",
        "period": {"year": 2021, "quarter": None, "month": None},
        "row_label": row_label,
        "cells": [
            {
                "coordinate": coordinate,
                "row": 5,
                "column": quarter + 2,
                "row_label": row_label,
                "column_label": f"{'一二三四'[quarter - 1]}季度末",
                "value": str(value),
                "normalized_value": value,
            }
        ],
    }
    db_session.add(
        DocumentChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            text=f"{row_label} {value}",
            chunk_metadata=json.dumps(metadata, ensure_ascii=False),
            token_count=20,
            index_status="indexed",
            index_version="test",
            title=source_title,
            source_file=f"{source_title}.xls",
            section_title=source_title,
        )
    )


def test_identical_tied_values_answer_instead_of_refusing(db_session) -> None:
    for index, coordinate in enumerate(("C5", "D5"), start=1):
        _add_solvency_row(
            db_session,
            chunk_id=f"DOC-SOLV-2021-CHUNK-000{index}",
            row_label="综合偿付能力充足率（%） / 保险公司",
            quarter=1,
            value=232.1,
            coordinate=coordinate,
        )
    db_session.flush()
    document = db_session.query(Document).filter_by(document_id="DOC-SOLV-2021").one()
    rebuild_spreadsheet_cell_index(db_session, document)
    db_session.flush()
    aspect = _aspect(
        "2021年保险公司的综合偿付能力充足率是多少？",
        table_task="lookup",
        table_filters={"year": 2021, "indicator": "综合偿付能力充足率", "row_label": "保险公司"},
    )
    matches = retrieve_spreadsheet_matches(db_session, aspect)
    assert len(matches) == 1
    assert matches[0].citation.metadata["value"] == "232.1"


def test_latest_period_tie_never_mixes_quarter_and_month_axes() -> None:
    from types import SimpleNamespace

    from backend.app.services.spreadsheet_retrieval_service import SpreadsheetCandidate

    def candidate(column_label: str, quarter: int | None, month: int | None, value: str) -> SpreadsheetCandidate:
        cell = {
            "row_label": "原保险保费收入",
            "column_label": column_label,
            "value": value,
            "normalized_value": float(value),
            "coordinate": "C5",
        }
        metadata = {"source_title": "表", "year": 2023, "quarter": quarter, "month": month}
        return SpreadsheetCandidate(
            document=SimpleNamespace(document_id="D", filename="表.xlsx"),
            chunk=SimpleNamespace(chunk_id="C"),
            metadata=metadata,
            score=1.0,
            selected_cell=cell,
        )

    # Quarter 4 must not beat month 11 on a shared integer axis.
    tied = [candidate("四季度", 4, None, "10"), candidate("11月", None, 11, "20")]
    assert _latest_period_tie_candidate(tied) is None
    # Same axis: the later period wins.
    tied_same = [
        candidate("三季度", 3, None, "10"),
        candidate("四季度", 4, None, "20"),
    ]
    winner = _latest_period_tie_candidate(tied_same)
    assert winner is not None and winner.selected_cell["value"] == "20"


def test_round_half_up_not_bankers_rounding() -> None:
    assert _round_half_up_text(Decimal("0.125"), 2) == "0.13"
    assert _round_half_up_text(Decimal("2.675"), 2) == "2.68"
    assert _round_half_up_text(Decimal("9.1"), 2) == "9.10"


# --------------------------------------------------------------------------
# Display / validation fixes
# --------------------------------------------------------------------------


def test_value_with_unit_does_not_rescale_result_scale_100() -> None:
    from backend.app.services.answer_generation_service import _value_with_unit

    metadata = {"result_scale": 100, "display_decimal_places": 2}
    assert _value_with_unit(2.6952, "%", metadata) == "2.70%"
    # A raw decimal fraction still scales once.
    raw_metadata = {"display_decimal_places": 2}
    assert _value_with_unit(0.026952, "%", raw_metadata) == "2.70%"


def test_format_value_uses_half_up() -> None:
    from backend.app.services.answer_generation_service import _format_value

    assert _format_value(0.125, decimal_places=2) == "0.13"


def test_match_option_prefers_numeric_equality_over_label_hit() -> None:
    from backend.app.services.answer_generation_service import _match_option

    options = ["资本充足率为8%", "资本充足率为10.5%"]
    labels = ["A", "B"]
    chosen, text = _match_option(10.5, "资本充足率", options, labels)
    assert chosen == "B"
    assert text == "资本充足率为10.5%"
    # No numeric match → no guess even when the label hits every option.
    chosen2, text2 = _match_option(11.0, "资本充足率", options, labels)
    assert chosen2 is None and text2 is None


def test_formula_percent_without_division_is_not_scaled() -> None:
    from backend.app.services.formula_parser_service import _format_formula_result

    # Y = A × 50% with A=200 → 100, never “10000%”.
    assert _format_formula_result(100.0, "Y = A × 50%", "Y是多少") == "100"
    # A quotient formula asked as a percentage still scales.
    assert _format_formula_result(0.5, "A / B", "A占B的百分比") == "50%"
    # A quotient formula with an explicit *100% keeps the regulatory form.
    assert _format_formula_result(0.09, "(A-B)/(C)*100%", "杠杆率是多少") == "9%"


def test_grounding_accepts_zero_padded_month_variant() -> None:
    from backend.app.services.grounding_validation_service import _entity_is_supported

    evidence = "2023年03月保险业经营情况表 原保险保费收入 19456.91"
    assert _entity_is_supported("2023年3月", evidence) is True
    assert _entity_is_supported("2023年12月", evidence) is False


def test_quarter_qualifier_compares_the_quarter_value() -> None:
    from types import SimpleNamespace

    from backend.app.services.evidence_contract_service import _qualifier_matches

    chunk = SimpleNamespace(source_doc="保险业偿付能力状况表(法人)(2021年).xls", metadata={})
    metadata = {"inferred_year": 2021, "quarter": 1, "period": {"year": 2021, "quarter": 1}}
    # Q1 metadata must NOT satisfy a 四季度 qualifier.
    assert _qualifier_matches("四季度", "2021年 保险业偿付能力状况表", metadata, chunk) is False
    assert _qualifier_matches("一季度", "2021年 保险业偿付能力状况表", metadata, chunk) is True
    # Combined year+quarter qualifier checks both values.
    metadata_q4 = {"inferred_year": 2021, "quarter": 4, "period": {"year": 2021, "quarter": 4}}
    assert _qualifier_matches("2021年四季度", "2021年 保险业偿付能力状况表", metadata_q4, chunk) is True
    assert _qualifier_matches("2021年四季度", "2020年 保险业偿付能力状况表", {"inferred_year": 2020, "quarter": 4, "period": {"year": 2020, "quarter": 4}}, chunk) is False
