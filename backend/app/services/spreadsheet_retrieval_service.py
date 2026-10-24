from __future__ import annotations
from backend.app.core.rag_features import features, feature_gate, event

from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
import json
import re
from typing import Any

from sqlalchemy import Select, or_, select
from sqlalchemy.orm import Session

from backend.app.core.config import TABLE_STRICT_EVIDENCE_VALIDATION
from backend.app.models.document import Document, DocumentChunk, SpreadsheetCell
from backend.app.schemas.qa import Citation
from backend.app.services.retrieval_service import RetrievalMatch
from backend.app.services.spreadsheet_cell_index_service import normalize_table_text


@dataclass
class SpreadsheetCandidate:
    document: Document
    chunk: DocumentChunk
    metadata: dict[str, Any]
    score: float
    selected_cell: dict[str, Any] | None = None


# Per-request diagnostic slot (ContextVar, like retrieval_service's
# diagnostics): a module-level dict would race between concurrent requests.
_SPREADSHEET_DIAGNOSTIC: ContextVar[dict[str, Any] | None] = ContextVar(
    "regumate_last_spreadsheet_diagnostic",
    default=None,
)


def get_last_spreadsheet_diagnostic() -> dict[str, Any]:
    current = _SPREADSHEET_DIAGNOSTIC.get()
    return dict(current or {})


@feature_gate("structured_index")
def retrieve_spreadsheet_matches(db: Session, aspect: Any, *, limit: int = 5) -> list[RetrievalMatch]:
    _set_last_diagnostic(match_status="not_run", refusal_reason=None)
    filters = dict(getattr(aspect, "table_filters", {}) or {})
    question = str(getattr(aspect, "question", "") or "")
    table_task = str(getattr(aspect, "table_task", "none") or "none")
    operation = str(getattr(aspect, "operation", "none") or "none")

    indexed = _indexed_spreadsheet_matches(
        db,
        aspect,
        question=question,
        filters=filters,
        table_task=table_task,
        operation=operation,
        limit=limit,
    )
    if indexed is not None:
        return indexed

    # Compatibility path for old databases and focused unit tests. New uploads
    # and rebuilt indexes always use the normalized SpreadsheetCell table.
    candidates = _spreadsheet_candidates(db, question, filters)
    if not candidates:
        _set_last_diagnostic(match_status="no_candidates", refusal_reason="未检索到可用表格候选。")
        return []

    if table_task == "compare" and operation in {"max", "min"}:
        comparison = _comparison_candidate(candidates, operation)
        return [_to_match(comparison, aspect, table_task, operation)] if comparison else []

    if table_task == "calculate" and operation in {"difference", "sum", "ratio"}:
        if not features().calculation:
            return _operand_matches(candidates, aspect)
        calculation = _calculation_match(candidates, aspect, operation)
        return [calculation] if calculation else []

    lookup_limit = 1 if table_task == "lookup" else limit
    matches = [_to_match(candidate, aspect, table_task, operation) for candidate in candidates[:lookup_limit]]
    _set_valid_diagnostic(matches)
    return matches


def _indexed_base_statement(filters: dict[str, Any], selectors: tuple | list = ()) -> Select:
    """Build the spreadsheet query without period (year/month/quarter) filters.

    Period filters are applied separately so a transposed-layout fallback
    (period words live inside row labels such as “一季度 / 不良贷款余额”) can
    re-run the same query without them and match periods at the row level.
    """
    statement = (
        select(SpreadsheetCell, DocumentChunk, Document)
        .join(DocumentChunk, SpreadsheetCell.chunk_id == DocumentChunk.chunk_id)
        .join(Document, SpreadsheetCell.document_id == Document.document_id)
        .where(Document.status.in_(["indexed", "table_indexed"]))
    )
    if str(filters.get("version_status") or "") not in {"repealed", "superseded"}:
        statement = statement.where(Document.version_status.notin_(("repealed", "superseded")))
    source = normalize_table_text(filters.get("source_title") or filters.get("filename"))
    selector_sources = {
        normalize_table_text(item.get("source_title"))
        for item in selectors
        if isinstance(item, dict) and item.get("source_title")
    }
    # A cross-period question can produce a human-readable composite title
    # such as “2023年10月与12月…表”.  The per-period selectors are the actual
    # hard locators, so the composite phrase must not eliminate both files.
    if len(selector_sources) >= 2:
        source = ""
    sheet = normalize_table_text(filters.get("sheet"))
    if source:
        source_variants = _source_title_variants(source)
        statement = statement.where(
            or_(*(SpreadsheetCell.source_title_norm.contains(value) for value in source_variants if value))
        )
    if sheet:
        statement = statement.where(SpreadsheetCell.sheet_name_norm.contains(sheet))
    for filter_name, document_field in (
        ("external_doc_id", Document.external_doc_id),
        ("issuing_authority", Document.issuing_authority),
        ("publication_date", Document.publication_date),
        ("document_number", Document.document_number),
        ("regulatory_topic", Document.regulatory_topic),
        ("business_domain", Document.business_domain),
        ("version_status", Document.version_status),
    ):
        value = filters.get(filter_name)
        if value not in (None, ""):
            statement = statement.where(document_field == str(value))
    return statement


def _indexed_spreadsheet_matches(
    db: Session,
    aspect: Any,
    *,
    question: str,
    filters: dict[str, Any],
    table_task: str,
    operation: str,
    limit: int,
) -> list[RetrievalMatch] | None:
    if not db.scalar(select(SpreadsheetCell.id).limit(1)):
        return None

    selectors = [item for item in getattr(aspect, "selectors", ()) if isinstance(item, dict)]
    base_statement = _indexed_base_statement(filters, selectors)
    # Quarter/month are CELL-level hard constraints: a cell satisfies the
    # requested period either through the index column (title-inferred periods)
    # or, when that column is NULL, through a period-shaped COLUMN header
    # ("一季度/…/四季度", "1月..12月" — the dominant statistical layout).
    # Cells with neither (a different period, or no period at all) stay
    # excluded, so an explicit constraint still never falls back to the whole
    # corpus.  The year always lives on the index column.
    period_statement = base_statement
    if filters.get("year") is not None:
        period_statement = period_statement.where(SpreadsheetCell.year == int(filters["year"]))
    for field_name in ("month", "quarter"):
        value = filters.get(field_name)
        if value is not None:
            column = getattr(SpreadsheetCell, field_name)
            period_statement = period_statement.where(or_(column == int(value), column.is_(None)))

    rows = db.execute(period_statement).all()
    rows = [row for row in rows if _cell_period_satisfies(row[0], filters)]
    # An explicit hard constraint that matches nothing must never fall back to
    # a whole-corpus maximum or calculation.  One legitimate exception keeps
    # its period outside both the index columns and the column headers: a
    # transposed layout (institution classes on columns, "一季度 / 不良贷款
    # 余额" period/indicator combinations on rows) keeps periods in row labels.
    # That layout is re-queried without the period filters and matched at the
    # row level instead.
    if not rows:
        transposed_base = base_statement
        if filters.get("year") is not None:
            transposed_base = transposed_base.where(SpreadsheetCell.year == int(filters["year"]))
        transposed = _transposed_spreadsheet_matches(
            db,
            aspect,
            base_statement=transposed_base,
            question=question,
            filters=filters,
            table_task=table_task,
            operation=operation,
            limit=limit,
        )
        if transposed is not None:
            return transposed
        _set_last_diagnostic(
            match_status=_missing_row_status(filters),
            refusal_reason=_missing_row_reason(filters),
            table_filters=filters,
        )
        return []

    candidates = [_candidate_from_index(cell, chunk, document, question, filters) for cell, chunk, document in rows]
    _patch_column_period_metadata(candidates)
    label_filters = dict(filters)
    question_column_target = _specific_column_target_from_question(
        question,
        candidates,
        selectors=selectors,
    )
    if table_task in {"compare", "calculate"} and getattr(aspect, "selectors", ()):
        # In questions such as “全国合计”从“合计”到“健康险”, the parser also
        # extracts the last quoted column as table_filters["column_label"].
        # Treating operands as global hard filters removes the other operands
        # before selector matching.  Each comparison/calculation operand is
        # checked by its own selector below.  Keep a row scope such as
        # “农村商业银行”, which is independent from the operand labels.
        selector_labels = {
            normalize_table_text(
                item.get("label") or item.get("row_or_indicator") or item.get("row_label") or item.get("column_label")
            )
            for item in getattr(aspect, "selectors", ())
            if isinstance(item, dict)
        }
        if table_task == "calculate":
            label_filters.pop("row_label", None)
            label_filters.pop("column_label", None)
        elif normalize_table_text(label_filters.get("row_label")) in selector_labels:
            label_filters.pop("row_label", None)
        if normalize_table_text(label_filters.get("column_label")) in selector_labels:
            label_filters.pop("column_label", None)
        # A calculation's selectors define separate operands.  Applying a
        # single column inferred from the question here would discard the
        # other operand before selector resolution (for example, a change
        # from one column to another).  Column recovery remains useful for a
        # comparison, where every selected row shares one metric column.
        if (
            table_task != "calculate"
            and question_column_target
            and _is_generic_column_target(label_filters.get("column_label"))
        ):
            label_filters["column_label"] = question_column_target
        label_filters.pop("indicator", None)
        label_filters.pop("metric", None)
    candidates = _filter_candidates_by_required_labels(
        candidates,
        label_filters,
        getattr(aspect, "selectors", ()) if table_task == "lookup" else (),
    )
    if not candidates:
        _set_last_diagnostic(
            match_status="indicator_not_found",
            refusal_reason="指定指标、行标签或列口径未在候选表格中找到精确证据。",
            table_filters=filters,
            selectors=list(getattr(aspect, "selectors", ())),
        )
        return []
    candidates = [candidate for candidate in candidates if candidate.score > 0 or candidate.selected_cell]
    candidates.sort(key=lambda item: item.score, reverse=True)
    if table_task == "compare" and operation in {"max", "min"}:
        usable_selectors = [item for item in selectors if not _generic_comparison_selector(item)]
        if usable_selectors:
            selected = _cells_for_selectors(
                candidates,
                usable_selectors,
                question=question,
                question_column_target=question_column_target,
                keep_all_periods=True,
            )
        else:
            selected = _numeric_candidates_for_column(
                candidates,
                label_filters.get("column_label") or filters.get("column_label"),
            )
        winner = _comparison_candidate(selected, operation)
        if winner:
            winner.metadata["comparison_operation"] = operation
            winner.metadata["comparison_cells"] = [_cell_metadata(candidate) for candidate in selected]
            matches = [_to_match(winner, aspect, table_task, operation, match_status="valid_comparison")]
            _set_valid_diagnostic(matches)
            return matches
        _set_last_diagnostic(match_status="ambiguous_candidates", refusal_reason="候选单元格不足或不可比较。")
        return []

    if table_task == "calculate" and operation in {"difference", "sum", "ratio"}:
        if len(selectors) < 2:
            _set_last_diagnostic(match_status="missing_operands",
                                 refusal_reason="尚未明确全部操作数，请给出每项的来源、期间及行列定位。")
            return []
        if operation == "ratio":
            # A ratio is only meaningful inside one source table: several
            # monthly families (保险业 / 人身险公司 / 财产保险公司) carry the
            # same row labels, and per-selector best picks can mix tables.
            selected = _ratio_same_source_selection(
                candidates,
                selectors,
                question=question,
                question_column_target=question_column_target,
                filters=filters,
            )
        else:
            selected = _cells_for_selectors(
                candidates,
                selectors,
                question=question,
                question_column_target=question_column_target,
                preserve_explicit_generic_columns=True,
            )
        if not features().calculation and selected:
            return _operand_matches(selected, aspect)
        calculation = _calculation_match_from_selected(
            selected,
            aspect,
            operation,
            # Only cross-period or reversed-operand ("A比B多") selectors mark
            # themselves as ordered transitions (second - first); ordinary
            # "X与Y的差值" keeps the selector order (first - second).
            ordered_transition=any(
                isinstance(selector, dict) and selector.get("ordered_transition")
                for selector in selectors
            ),
        )
        if calculation:
            _attach_sum_comparison_target(calculation, candidates, selected, filters, operation)
            matches: list[RetrievalMatch] = [calculation]
            # Cross-table calculations (“《表1》和《表2》…的差值”) draw operands
            # from different source files.  Return one cell-level evidence per
            # operand table as well, so the answer can cite every involved
            # source instead of only the anchor table.
            anchor_chunk_id = str(calculation.citation.chunk_id or "")
            operand_source_titles = {
                normalize_table_text(candidate.metadata.get("source_title") or "")
                for candidate in selected
            }
            if len(operand_source_titles) > 1:
                for candidate in selected:
                    cell_chunk_id = str(candidate.chunk.chunk_id or "")
                    if cell_chunk_id == anchor_chunk_id:
                        continue
                    matches.append(
                        _to_match(candidate, aspect, "lookup", operation, match_status="valid_cell")
                    )
            _set_valid_diagnostic(matches)
            return matches
        # Transposed-layout fallback: period selectors keep their quarters in
        # the row labels (“一季度 / 不良贷款余额”), so the standard selector
        # year/quarter equality checks always fail.  Retry with transposed
        # semantics when the candidate rows carry that structure.
        if any(
            _transposed_row_parts((candidate.selected_cell or {}).get("row_label"))
            for candidate in candidates
        ):
            transposed = _transposed_spreadsheet_matches(
                db,
                aspect,
                base_statement=base_statement,
                question=question,
                filters=filters,
                table_task=table_task,
                operation=operation,
                limit=limit,
            )
            if transposed is not None:
                return transposed
        _set_last_diagnostic(match_status="ambiguous_candidates", refusal_reason="计算所需的操作数不足、单位不一致或除零。")
        return []

    if table_task == "lookup" and selectors:
        selected = _cells_for_selectors(
            candidates,
            selectors,
            question=question,
            question_column_target=question_column_target,
        )
        if not selected:
            _set_last_diagnostic(
                match_status="indicator_not_found",
                refusal_reason="指定指标或列口径未在表格中找到唯一可用单元格。",
                table_filters=filters,
                selectors=selectors,
            )
            return []
        if TABLE_STRICT_EVIDENCE_VALIDATION and len(selected) > 1:
            _set_last_diagnostic(
                match_status="ambiguous_candidates",
                refusal_reason="指定条件命中了多个候选单元格，无法确定唯一答案。",
                table_filters=filters,
                selectors=selectors,
            )
            return []
        matches = [_to_match(selected[0], aspect, table_task, operation, match_status="valid_cell")]
        _set_valid_diagnostic(matches)
        return matches

    lookup_limit = 1 if table_task == "lookup" else limit
    if table_task == "lookup" and TABLE_STRICT_EVIDENCE_VALIDATION:
        top = _unique_numeric_candidates(candidates)
        if not top:
            _set_last_diagnostic(match_status="empty_value", refusal_reason="未找到带数值的有效单元格。")
            return []
        if _has_explicit_label_filters(filters) and len(top) > 1:
            best_score = top[0].score
            tied = [candidate for candidate in top if abs(candidate.score - best_score) < 0.0001]
            if len(tied) > 1:
                # Identical values are not an ambiguity: several layouts
                # repeat one figure across equivalent rows/columns.
                tied_values = {
                    round(float((candidate.selected_cell or {}).get("normalized_value")), 6)
                    for candidate in tied
                }
                if len(tied_values) == 1:
                    tied[0].metadata["selection_assumptions"] = [
                        "多个候选单元格取值相同，已按其一作答："
                        + f"{(tied[0].selected_cell or {}).get('row_label')} / "
                        + f"{(tied[0].selected_cell or {}).get('column_label')}。"
                    ]
                    top = [tied[0]]
                else:
                    resolved = (
                        _first_block_tie_candidate(tied)
                        or _latest_period_tie_candidate(tied)
                        or _question_signal_tie_candidate(question, tied)
                    )
                    if resolved is None:
                        summary = "；".join(
                            f"{(candidate.selected_cell or {}).get('row_label')}"
                            f"/{(candidate.selected_cell or {}).get('column_label')}"
                            f"={(candidate.selected_cell or {}).get('normalized_value')}"
                            for candidate in tied[:4]
                        )
                        _set_last_diagnostic(
                            match_status="ambiguous_candidates",
                            refusal_reason=f"指定条件命中多个等价候选单元格（{summary}），请补充期间或口径。",
                        )
                        return []
                    resolved.metadata["selection_assumptions"] = [
                        "存在多个不同取值的候选单元格，已按问题限定择一："
                        + f"{(resolved.selected_cell or {}).get('row_label')} / "
                        + f"{(resolved.selected_cell or {}).get('column_label')}。"
                    ]
                    top = [resolved]
        matches = [_to_match(top[0], aspect, table_task, operation, match_status="valid_cell")]
        _set_valid_diagnostic(matches)
        return matches

    matches = [_to_match(candidate, aspect, table_task, operation) for candidate in candidates[:lookup_limit]]
    _set_valid_diagnostic(matches)
    return matches


_TRANSPOSED_ROW_PATTERN = re.compile(
    r"^\s*(20\d{2}年)?\s*(?:(\d{1,2})月|第?([一二三四1-4])季度)\s*[/|｜]\s*(.+?)\s*$"
)


def _transposed_row_parts(row_label: Any) -> tuple[str, str] | None:
    """Split a transposed row label “一季度 / 不良贷款余额” into (period, metric).

    Transposed layouts put institution classes on columns and period/metric
    combinations on rows.  Returns None for ordinary row labels so the fallback
    only fires for genuinely transposed tables.  The raw label is used because
    ``normalize_table_text`` strips the “/” separator.
    """
    match = _TRANSPOSED_ROW_PATTERN.match(str(row_label or ""))
    if not match:
        return None
    if match.group(2):
        period_part = f"{match.group(1) or ''}{match.group(2)}月"
    else:
        period_part = f"{match.group(1) or ''}{match.group(3)}季度"
    return period_part, match.group(4)


def _transposed_period_matches(period_part: str, filters: dict[str, Any]) -> bool:
    """Match a row-label period against requested year/month/quarter filters."""
    quarter = filters.get("quarter")
    month = filters.get("month")
    year = filters.get("year")
    if quarter is not None:
        cjk = "一二三四"[int(quarter) - 1]
        if f"{cjk}季度" not in period_part and f"{int(quarter)}季度" not in period_part:
            return False
    if month is not None and f"{int(month)}月" not in period_part:
        return False
    if year is not None and re.search(r"20\d{2}年", period_part):
        if f"{int(year)}年" not in period_part:
            return False
    return True


_COLUMN_PERIOD_PATTERN = re.compile(r"(?:第?([一二三四1-4])季度|(\d{1,2})月)")


def _parse_column_period(column_label: Any) -> dict[str, int] | None:
    """Parse a period expressed as a COLUMN header (“四季度”, “3月”, “2024年四季度”).

    Returns ``{"quarter": n}`` or ``{"month": n}``, or None when the header
    carries no period (institution names, indicator names, …).
    """
    match = _COLUMN_PERIOD_PATTERN.search(str(column_label or ""))
    if not match:
        return None
    if match.group(1):
        token = match.group(1)
        quarter = int(token) if token.isdigit() else "一二三四".find(token) + 1
        return {"quarter": quarter} if 1 <= quarter <= 4 else None
    month = int(match.group(2))
    return {"month": month} if 1 <= month <= 12 else None


def _cell_period_satisfies(cell: SpreadsheetCell, filters: dict[str, Any]) -> bool:
    """Cell-level quarter/month test against the requested period filters.

    A cell passes when its index period column carries the requested value
    (title-inferred periods) or, when that column is NULL, when its COLUMN
    header parses to the requested period (“四季度”, “3月”).  NULL cells with
    no period-shaped header fail, preserving the hard-constraint semantics.
    """
    for field_name in ("quarter", "month"):
        value = filters.get(field_name)
        if value is None:
            continue
        stored = getattr(cell, field_name, None)
        if stored is not None:
            if int(stored) != int(value):
                return False
            continue
        parsed = _parse_column_period(cell.column_label)
        if parsed is None or parsed.get(field_name) != int(value):
            return False
    return True


def _patch_column_period_metadata(candidates: list[SpreadsheetCandidate]) -> None:
    """Fill missing quarter/month metadata from period-shaped column headers.

    Selector matching compares ``year/month/quarter`` equality against the
    candidate metadata; for column-header periods the index column is NULL, so
    without this patch cross-period selectors (“一季度与四季度…变化”) exclude
    every cell of the dominant layout.
    """
    for candidate in candidates:
        parsed = _parse_column_period((candidate.selected_cell or {}).get("column_label"))
        if not parsed:
            continue
        for key, value in parsed.items():
            if candidate.metadata.get(key) is None:
                candidate.metadata[key] = value


def _transposed_institution_terms(filters: dict[str, Any]) -> list[str]:
    """Institution words of the row locator (the non-indicator part)."""
    indicator = normalize_table_text(filters.get("indicator") or "")
    return [term for term in _row_label_terms(filters.get("row_label")) if term != indicator]


def _transposed_row_matches(cell_row: Any, filters: dict[str, Any]) -> bool:
    """Transposed row test: requested period and indicator must match the row."""
    parts = _transposed_row_parts(cell_row)
    if parts is None:
        return False
    period_part, metric_part = parts
    if not _transposed_period_matches(period_part, filters):
        return False
    indicator = filters.get("indicator")
    if indicator and not _row_label_terms_covered(metric_part, _row_label_terms(indicator)):
        return False
    return True


def _transposed_column_matches(cell_column: Any, filters: dict[str, Any]) -> bool:
    """Transposed column test: the institution word must match the column."""
    terms = _row_label_terms(filters.get('column_label')) or _transposed_institution_terms(filters)
    if not terms:
        return False
    return _row_label_terms_covered(normalize_table_text(str(cell_column or "")), terms)


def _transposed_selector_cell(
    candidates: list[SpreadsheetCandidate],
    selector: dict[str, Any],
    filters: dict[str, Any],
) -> SpreadsheetCandidate | None:
    """One transposed calculation operand: period/metric from the selector
    against the row, institution word from the shared filters against the
    column."""
    metric = normalize_table_text(selector.get("row_or_indicator") or selector.get("row_label") or "")
    if not metric:
        metric = normalize_table_text(filters.get("indicator") or "")
    best: SpreadsheetCandidate | None = None
    for candidate in candidates:
        cell = candidate.selected_cell or {}
        if not isinstance(cell.get("normalized_value"), (int, float)):
            continue
        # The year lives in index metadata (quarter/month are in row labels).
        if selector.get("year") is not None:
            meta_year = int(candidate.metadata.get("year") or 0)
            if not meta_year or meta_year != int(selector["year"]):
                continue
        parts = _transposed_row_parts(cell.get("row_label"))
        if parts is None:
            continue
        period_part, metric_part = parts
        if not _transposed_period_matches(period_part, selector):
            continue
        if metric and not _row_label_terms_covered(metric_part, _row_label_terms(metric)):
            continue
        column_filters = {**filters, 'column_label': selector.get('column_label') or filters.get('column_label')}
        if not _transposed_column_matches(cell.get("column_label"), column_filters):
            continue
        if best is None:
            best = candidate
        elif _candidate_cell_key(best) != _candidate_cell_key(candidate):
            return None  # ambiguous
    return best


def _transposed_spreadsheet_matches(
    db: Session,
    aspect: Any,
    *,
    base_statement: Select,
    question: str,
    filters: dict[str, Any],
    table_task: str,
    operation: str,
    limit: int,
) -> list[RetrievalMatch] | None:
    """Fallback for transposed layouts (institution classes on columns,
    “一季度 / 不良贷款余额” period/metric combinations on rows).

    The standard path applies year/month/quarter equality filters on index
    metadata columns, which are empty for transposed tables.  This path
    re-queries without period filters and matches periods inside row labels.
    Returns None when the table is not transposed (caller keeps its refusal).
    """
    rows = db.execute(base_statement).all()
    if not rows:
        return None
    if not any(_transposed_row_parts(cell.row_label) is not None for cell, _chunk, _doc in rows):
        return None
    selectors = [item for item in getattr(aspect, "selectors", ()) if isinstance(item, dict)]
    candidates = [_candidate_from_index(cell, chunk, document, question, filters) for cell, chunk, document in rows]

    if table_task == "calculate" and operation in {"difference", "sum", "ratio"} and selectors:
        selected: list[SpreadsheetCandidate] = []
        for selector in selectors:
            chosen = _transposed_selector_cell(candidates, selector, filters)
            if chosen is None:
                break
            selected.append(chosen)
        if len(selected) == len(selectors):
            if not features().calculation:
                return _operand_matches(selected, aspect)
            calculation = _calculation_match_from_selected(
                selected,
                aspect,
                operation,
                ordered_transition=any(selector.get("ordered_transition") for selector in selectors),
            )
            if calculation:
                _set_valid_diagnostic([calculation])
                return [calculation]
        _set_last_diagnostic(match_status="ambiguous_candidates", refusal_reason="计算所需的操作数不足、单位不一致或除零。")
        return []

    if table_task == "lookup":
        matches = [
            candidate
            for candidate in candidates
            if isinstance((candidate.selected_cell or {}).get("normalized_value"), (int, float))
            and _transposed_row_matches(candidate.selected_cell.get("row_label"), filters)
            and _transposed_column_matches(candidate.selected_cell.get("column_label"), filters)
        ]
        if len(matches) == 1:
            match = _to_match(matches[0], aspect, table_task, operation, match_status="valid_cell")
            _set_valid_diagnostic([match])
            return [match]
        _set_last_diagnostic(
            match_status="indicator_not_found" if not matches else "ambiguous_candidates",
            refusal_reason=(
                "指定指标、行标签或列口径未在候选表格中找到精确证据。"
                if not matches
                else "指定条件命中多个候选单元格，无法确定唯一答案。"
            ),
            table_filters=filters,
        )
        return []

    return None


def _candidate_from_index(
    cell: SpreadsheetCell,
    chunk: DocumentChunk,
    document: Document,
    question: str,
    filters: dict[str, Any],
) -> SpreadsheetCandidate:
    metadata = _metadata(chunk)
    metadata.update(
        {
            "year": cell.year,
            "month": cell.month,
            "quarter": cell.quarter,
            "source_title": cell.source_title or metadata.get("source_title"),
            "sheet_name": cell.sheet_name or metadata.get("sheet_name"),
        }
    )
    selected_cell = {
        "coordinate": cell.coordinate,
        "row": cell.row_index,
        "column": cell.column_index,
        "row_label": cell.row_label,
        "column_label": cell.column_label,
        "column_path": [part for part in cell.column_label.split(" / ") if part],
        "value": cell.value,
        "normalized_value": cell.numeric_value if cell.numeric_value is not None else cell.value,
        "unit": cell.unit,
        "is_formula": cell.is_formula,
        "formula": cell.formula,
    }
    score = _indexed_cell_score(cell, question, filters)
    return SpreadsheetCandidate(document, chunk, metadata, score, selected_cell)


def _indexed_cell_score(cell: SpreadsheetCell, question: str, filters: dict[str, Any]) -> float:
    row = cell.row_label_norm
    column = cell.column_label_norm
    combined = f"{row}{column}"
    source_norm = normalize_table_text(getattr(cell, "source_title_norm", None) or "")
    score = 0.25 if cell.numeric_value is not None else 0.0
    column_leaf = normalize_table_text(re.split(r"\s*/\s*|[|｜>]", str(cell.column_label or ""))[-1])
    locator_row_label = normalize_table_text(
        " ".join(
            str(filters.get(key) or "")
            for key in ("row_label", "indicator", "metric", "scope")
        )
    )
    # Derived ratio rows (“占银行业金融机构比例”) are never the subject a
    # locator word names — they must not collect row/scope hits through a
    # bare substring of the institution word.
    derived_ratio_row = bool(
        re.search(r"占.{0,14}(?:比例|比重)$|占比$", row)
    ) and not re.search(r"(?:比例|比重|占比)$", locator_row_label)
    for key, weight in (("row_label", 5.0), ("column_label", 5.0), ("indicator", 4.0), ("metric", 4.0), ("scope", 3.0)):
        target = normalize_table_text(filters.get(key))
        if key == "row_label":
            if derived_ratio_row and target:
                continue
            # Multi-word row labels ("财产险公司 资金运用余额") must ALL be
            # covered by the row, otherwise an aggregate row that only matches
            # the indicator would tie with (or beat) the institution-specific
            # sub-row.  Single-word labels keep the historical substring test.
            if not _row_label_terms_covered(row, _row_label_terms(filters.get(key))):
                # Whole-table indicators ("银行业金融机构普惠型小微企业贷款余额"
                # on rows named 银行业金融机构合计) leave the locator word
                # unmatched: a partial word hit still separates that family
                # from unrelated tables, just with less weight.
                if any(
                    normalize_table_text(term) in row
                    for term in _row_label_terms(filters.get(key))
                ):
                    score += 1.0
                continue
            score += weight
            # An exact row label outranks substring relatives: 资本充足率 must
            # not tie with 核心一级资本充足率 when both rows exist in the table.
            if target and row == target:
                score += 2.0
            # A single-word institution locator (“保险公司”) is a substring of
            # every sibling row (“再保险公司”, “财产险公司”).  When one of the
            # locator words equals a row's TAIL segment, that row is the named
            # row and the substring siblings stay behind it.
            if target:
                for term in _row_label_terms(filters.get(key)):
                    raw_row_label = getattr(cell, "row_label", None) or (cell.row_label_norm or "")
                    if normalize_table_text(term) == normalize_table_text(
                        str(raw_row_label).split("/")[-1]
                    ):
                        score += 2.0
                        break
        elif key == "column_label":
            # A row such as “全国合计” must not make every numeric cell look
            # like the requested “合计” column.  Compare column constraints
            # only with the terminal header, excluding the repeated table
            # title prefix and the row label.
            structural_text = column_leaf
            if target and target in structural_text:
                score += weight
        else:
            structural_text = combined
            if target and target in structural_text:
                if derived_ratio_row and key == "scope":
                    continue
                score += weight
                if key in {"indicator", "metric"} and row == target:
                    score += 2.0
            # Whole-table indicators (“普惠型小微企业贷款…情况表”) live in the
            # source title, not in any row/column label.  Crediting the source
            # keeps that table family ahead of unrelated ones.  A derived
            # ratio row must not ride along: it is not the queried subject.
            elif target and target in source_norm and key in {"indicator", "metric"} and not derived_ratio_row:
                score += 2.5
    score += _token_overlap(question, combined)
    # Table-family scope (“保险业” vs “人身险公司经营情况表”): the family word
    # names the source table, not a row or column, so it scores the source
    # title.  This breaks the exact tie where several monthly tables carry
    # identically labelled rows (原保险保费收入) for the same period.  A long
    # family word whose full form is absent (“银行业金融机构” on sources titled
    # “银行业总资产、总负债”) still credits its longest matching prefix.
    scope = normalize_table_text(filters.get("scope"))
    if scope:
        if scope in source_norm:
            score += 4.0
        elif len(scope) >= 4:
            prefix = scope[: len(scope) - 1]
            while len(prefix) >= 3:
                if prefix in source_norm:
                    score += 3.0
                    break
                prefix = prefix[:-1]
    # Parent context (“资金运用余额中，债券投资”): a source whose title or the
    # row family carries the parent indicator is preferred over unrelated
    # tables that happen to contain the sub-row label.
    context = normalize_table_text(filters.get("context"))
    if context and (context in row or context in source_norm):
        score += 2.0
    return score


def _filter_candidates_by_required_labels(
    candidates: list[SpreadsheetCandidate],
    filters: dict[str, Any],
    selectors: tuple[dict[str, Any], ...],
) -> list[SpreadsheetCandidate]:
    if not TABLE_STRICT_EVIDENCE_VALIDATION:
        return candidates
    required_terms = _required_label_terms(filters, selectors)
    if not required_terms:
        return candidates
    result: list[SpreadsheetCandidate] = []
    for candidate in candidates:
        cell = candidate.selected_cell or {}
        searchable = normalize_table_text(
            " ".join(
                str(part)
                for part in [
                    cell.get("row_label"),
                    cell.get("column_label"),
                    " ".join(cell.get("column_path") or []),
                    candidate.metadata.get("row_label"),
                    candidate.metadata.get("table_title"),
                    candidate.metadata.get("source_title"),
                    # The filename is the strongest surviving locator when a
                    # chunk's table_title is degraded (a document id, a bare
                    # sheet name): “249_2021年银行业金融机构普惠型小微企业
                    # 贷款情况表” names both the family and the rows.
                    candidate.document.filename,
                ]
                if part
            )
        )
        if all(_required_term_hit(term, searchable) for term in required_terms):
            result.append(candidate)
    return result


def _required_term_hit(term: str, searchable: str) -> bool:
    """A required term matches when it (or its long-word prefix) appears.

    Some indexed chunks carry a degraded ``table_title`` (a document id), so a
    family word like “银行业金融机构” only reaches the searchable text through
    the source title “2026年银行业总资产、总负债（月度）”.  Matching the same
    longest-prefix semantics as the scope score keeps those rows in play
    without letting short fragments through.
    """

    if term in searchable:
        return True
    # Period-shaped terms match by period VALUE, not literally: the question
    # says “四季度末” (a solvency-table header) while the target table's
    # column is “四季度” — the 末 suffix is layout, not semantics.
    period = _parse_column_period(term)
    if period is not None:
        if period.get("quarter") is not None:
            cjk = "一二三四"[int(period["quarter"]) - 1]
            return bool(
                re.search(rf"{cjk}\s*季度|{int(period['quarter'])}\s*季度", searchable)
            )
        if period.get("month") is not None:
            return bool(re.search(rf"{int(period['month'])}\s*月", searchable))
    if len(term) >= 5:
        prefix = term[:-1]
        while len(prefix) >= 3:
            if prefix in searchable:
                return True
            prefix = prefix[:-1]
    return False


def _required_label_terms(filters: dict[str, Any], selectors: tuple[dict[str, Any], ...]) -> list[str]:
    terms: list[str] = []
    for key in ("indicator", "row_label", "column_label", "metric"):
        value = normalize_table_text(filters.get(key))
        if value and not _is_generic_table_scope(value):
            if key == "row_label":
                # A composite row label ("财产险公司 资金运用余额") makes every
                # word a hard constraint: rows missing the institution category
                # are excluded instead of tying on the bare indicator.
                for term in _row_label_terms(filters.get(key)):
                    normalized = normalize_table_text(term)
                    if normalized and not _is_generic_table_scope(normalized):
                        terms.append(normalized)
            else:
                terms.append(value)
    for selector in selectors:
        if not isinstance(selector, dict):
            continue
        for key in ("row_or_indicator", "label", "row_label", "column_label"):
            value = normalize_table_text(selector.get(key))
            if value and not _is_generic_table_scope(value):
                terms.append(value)
    return list(dict.fromkeys(terms))


def _is_generic_table_scope(value: str) -> bool:
    return value in {"本年累计", "截至当期", "本年累计截至当期", "年季度", "季度", "月度", "年月度"}


def _row_label_terms(value: Any) -> list[str]:
    """Split a row_label filter into required row-location words.

    The planner may join an institution category with the indicator
    ("财产险公司 资金运用余额"); every word must then be covered by the
    candidate row's label (AND semantics), so a category term excludes
    aggregate rows that only match the indicator.  A single word keeps the
    historical substring semantics.
    """

    return [
        term
        for term in re.split(r"[\s、，,]+", str(value or "").strip())
        if term
    ]


_MEASUREMENT_SUFFIXES = ("投资", "余额", "收入", "支出", "金额", "发生额", "总额", "净额", "总量")


def _row_word_variants(word: str) -> list[str]:
    """A row word plus its base form with a trailing measurement suffix removed.

    Questions say “债券投资” while the indexed row is “债券”; requiring the
    full word would exclude the exact row.  The base form only fires when the
    full word fails, and never for single-character bases.
    """

    variants = [word]
    for suffix in _MEASUREMENT_SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            base = word[: -len(suffix)]
            if base not in variants:
                variants.append(base)
    return variants


def _row_label_terms_covered(row_text: str, terms: list[str]) -> bool:
    """True when every normalized row-label word is a substring of the row."""

    if not terms:
        return True
    for term in terms:
        normalized = normalize_table_text(term)
        if not normalized:
            continue
        if normalized in row_text:
            continue
        # A measurement-suffix base (“债券投资” → “债券”) only matches when it
        # ends a word in the row: “债券” matches the row “债券” but must not
        # match “不良贷款率” through “不良贷款”.
        if any(
            re.search(rf"{re.escape(variant)}(?![\u4e00-\u9fffA-Za-z0-9])", row_text)
            for variant in _row_word_variants(normalized)[1:]
        ):
            continue
        return False
    return True


def _has_explicit_label_filters(filters: dict[str, Any]) -> bool:
    return any(normalize_table_text(filters.get(key)) for key in ("indicator", "row_label", "column_label", "metric"))


def _source_title_variants(source: str) -> set[str]:
    """Return conservative aliases for a user-written spreadsheet title."""

    variants = {source}
    # Corpus exports often prefix attachment titles with an ordinal such as
    # ``148_``.  The runtime index deliberately stores the document title
    # without that catalogue-only prefix, so use the title both with and
    # without it.  This is identifier normalization, not a document map.
    # ``normalize_table_text`` removes the separator before this helper is
    # called, hence also accept a short digit run immediately before a dated
    # title (``1482023年…``).
    stripped_prefix = re.sub(r"^\d{1,4}(?:\s*[_\-－—]\s*|(?=20\d{2}年))", "", source)
    if stripped_prefix:
        variants.add(stripped_prefix)
    # A user-facing title often combines a reporting period from the landing
    # page with the shorter attachment title stored in SpreadsheetCell.  The
    # period remains a separate hard filter, while this alias resolves the
    # attachment name without requiring a corpus-specific filename map.
    for value in list(variants):
        without_period = re.sub(
            r"^20\d{2}年(?:\d{1,2}月|第?[一二三四1-4]季度|[一二三四1-4]季度)?",
            "",
            value,
        )
        if without_period:
            variants.add(without_period)
    for value in list(variants):
        variants.add(value.replace("财产险", "财产保险"))
        variants.add(value.replace("人身险", "人身保险"))
        variants.add(value.replace("财产保险", "财产险"))
        variants.add(value.replace("人身保险", "人身险"))
    for value in list(variants):
        variants.add(re.sub(r"(\d{4})年0([1-9])月", r"\1年\2月", value))
        variants.add(re.sub(r"(\d{4})年([1-9])月", r"\1年0\2月", value))
    for value in list(variants):
        if "情况表" in value:
            variants.add(value.replace("情况表", "表"))
        elif value.endswith("表"):
            variants.add(f"{value[:-1]}情况表")
    for value in list(variants):
        # A user question may append “（季度）”(“2025年…情况表（季度）”) while
        # the indexed attachment title omits it.  Both forms stay valid; the
        # stripped variant widens the contains match without requiring the
        # index to carry the suffix.
        stripped_suffix = re.sub(r"\(季度\)$", "", value)
        if stripped_suffix and stripped_suffix != value:
            variants.add(stripped_suffix)
    return {value for value in variants if value}


def _source_title_matches(requested_source: str, actual_source: str) -> bool:
    if not requested_source:
        return True
    if requested_source in actual_source:
        return True
    return any(variant and variant in actual_source for variant in _source_title_variants(requested_source))


def _first_block_tie_candidate(tied: list[SpreadsheetCandidate]) -> SpreadsheetCandidate | None:
    """Resolve identical row labels in one table by first occurrence.

    Block-structured monthly tables (“1. 银行业金融机构 / 其中：商业银行合计 /
    2. 大型商业银行…”) repeat one row label (“总资产”) per block with different
    values.  Statistical convention puts the whole-scope block first, so the
    question's unnamed subject (“…总资产是多少”) reads as the first block.
    Returns None unless every tied candidate shares one row label and one
    source table.
    """

    if len(tied) < 2:
        return None
    rows = {normalize_table_text((candidate.selected_cell or {}).get("row_label")) for candidate in tied}
    sources = {
        normalize_table_text(candidate.metadata.get("source_title") or candidate.document.filename)
        for candidate in tied
    }
    if len(rows) != 1 or len(sources) != 1:
        return None
    values = {
        round(float((candidate.selected_cell or {}).get("normalized_value")), 6)
        for candidate in tied
        if isinstance((candidate.selected_cell or {}).get("normalized_value"), (int, float))
    }
    if len(values) == 1:
        return None  # identical values already answered upstream
    return min(
        tied,
        key=lambda candidate: int((candidate.selected_cell or {}).get("row") or 0),
    )


def _latest_period_tie_candidate(tied: list[SpreadsheetCandidate]) -> SpreadsheetCandidate | None:
    """Resolve a strict lookup tie that differs only by the period column.

    “2024年商业银行拨备覆盖率是多少？” ties all four quarterly cells of the
    same row.  Picking the latest period with a disclosed assumption answers
    the user instead of refusing.  Returns None unless EVERY tied candidate
    shares one row label and one source table and each carries a DISTINCT
    period column — same-period duplicates (genuinely indistinguishable
    values) stay ambiguous.
    """
    if len(tied) < 2:
        return None
    rows = {normalize_table_text((candidate.selected_cell or {}).get("row_label")) for candidate in tied}
    sources = {
        normalize_table_text(candidate.metadata.get("source_title") or candidate.document.filename)
        for candidate in tied
    }
    if len(rows) != 1 or len(sources) != 1:
        return None
    keyed: list[tuple[tuple[int, str], SpreadsheetCandidate]] = []
    for candidate in tied:
        parsed = _parse_column_period((candidate.selected_cell or {}).get("column_label"))
        if not parsed:
            return None
        # Quarter and month are different axes: comparing "Q4" against "11月"
        # on one integer silently prefers the month.  The kind keeps them
        # apart, and the year keeps cross-year ties ordered correctly.
        kind = "quarter" if parsed.get("quarter") is not None else "month"
        value = int(parsed.get(kind) or 0)
        year = int(candidate.metadata.get("year") or 0)
        keyed.append(((year, kind, value), candidate))
    periods = [key for key, _candidate in keyed]
    if len(set(periods)) != len(periods):
        return None
    if len({kind for _year, kind, _value in periods}) != 1:
        return None
    return max(keyed, key=lambda item: item[0])[1]


_QUESTION_TERM_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]+")


def _question_signal_tie_candidate(
    question: str,
    tied: list[SpreadsheetCandidate],
) -> SpreadsheetCandidate | None:
    """Break a value tie with question wording that only one row satisfies.

    “综合偿付能力充足率” vs “核心偿付能力充足率” rows tie when the row filter
    only matched the shared tail (保险公司).  The question names one block —
    “综合…” — so the candidate whose row label covers more distinct question
    terms wins.  Returns None unless exactly one candidate leads.
    """

    terms = [
        normalize_table_text(term)
        for term in _QUESTION_TERM_PATTERN.findall(str(question or ""))
        if len(term) >= 2
    ]
    # Chinese runs are whole clauses; their 2-grams carry the discriminating
    # block words (综合/核心).
    grams: list[str] = []
    for term in terms:
        grams.append(term)
        if re.fullmatch(r"[\u4e00-\u9fff]+", term):
            grams.extend(term[i : i + 2] for i in range(len(term) - 1))
    if not grams:
        return None
    best: list[tuple[int, SpreadsheetCandidate]] = []
    for candidate in tied:
        row_text = normalize_table_text(str((candidate.selected_cell or {}).get("row_label") or ""))
        column_text = normalize_table_text(str((candidate.selected_cell or {}).get("column_label") or ""))
        signal = sum(1 for gram in set(grams) if gram in row_text)
        signal_column = sum(1 for gram in set(grams) if gram in column_text)
        best.append((signal * 2 + signal_column, candidate))
    top = max(score for score, _candidate in best)
    if top <= 0:
        return None
    leaders = [candidate for score, candidate in best if score == top]
    if len(leaders) != 1:
        return None
    return leaders[0]


def _cells_for_selectors(
    candidates: list[SpreadsheetCandidate],
    selectors: list[dict[str, Any]],
    *,
    question: str = "",
    question_column_target: str | None = None,
    preserve_explicit_generic_columns: bool = False,
    keep_all_periods: bool = False,
) -> list[SpreadsheetCandidate]:
    selected: list[SpreadsheetCandidate] = []
    if question_column_target is None:
        question_column_target = _specific_column_target_from_question(question, candidates)
    for selector in selectors:
        label_words = [
            normalize_table_text(word)
            for word in re.split(r"[\s、，,]+", str(selector.get("label") or selector.get("row_or_indicator") or "").strip())
            if word
        ]
        # Keep the space-joined form: multi-word labels (“财产险公司 银行存款”)
        # are matched word-wise by _table_label_match_specificity, because the
        # row may interleave structural words (“其中：”).
        label = " ".join(label_words)
        # Keep the raw selector row_label: _row_target_matches splits it into
        # words itself (normalize_table_text would collapse the spaces).
        row_target = str(selector.get("row_label") or "").strip()
        column_target = normalize_table_text(selector.get("column_label"))
        # An explicit selector column always wins; the question-derived target
        # only fills in an absent selector column.  A bare "合计" operand in
        # "从'合计'到'健康险'" must not be overwritten by another column name
        # found in the question.  A selector that carries its own period
        # (year/quarter/month) selects its column through that period — the
        # question's period-shaped mention (“三季度末到四季度末”) belongs to
        # the OTHER selectors and must not pin this one to a single column.
        selector_has_period = any(selector.get(key) is not None for key in ("year", "month", "quarter"))
        if question_column_target and not column_target and not selector_has_period:
            column_target = question_column_target
        elif (
            question_column_target
            and not selector_has_period
            and _is_generic_column_target(column_target)
            and not (preserve_explicit_generic_columns and column_target)
        ):
            column_target = question_column_target
        generic_period = bool(re.fullmatch(r"(?:年|季度|月度|月|季|度)+", column_target))
        period_pattern = r'[一二三四1-4]季度' if '季' in column_target else r'\d{1,2}月'
        if generic_period:
            column_target = ""
        scope = normalize_table_text(selector.get("scope"))
        if scope in {"季度", "年季度", "月度", "年月度"}:
            scope = ""
        matches: list[tuple[int, float, SpreadsheetCandidate]] = []
        for candidate in candidates:
            cell = candidate.selected_cell or {}
            if not isinstance(cell.get("normalized_value"), (int, float)):
                continue
            row = normalize_table_text(cell.get("row_label"))
            column = normalize_table_text(cell.get("column_label"))
            column_index = int(cell.get("column") or 0)
            combined = f"{row}{column}"
            if generic_period and not re.search(period_pattern, str(cell.get('column_label') or '')):
                continue
            if any(
                selector.get(key) is not None and int(candidate.metadata.get(key) or 0) != int(selector[key])
                for key in ("year", "month", "quarter")
            ):
                continue
            requested_source = normalize_table_text(selector.get("source_title"))
            actual_source = normalize_table_text(candidate.metadata.get("source_title") or candidate.document.filename)
            if requested_source and not _source_title_matches(requested_source, actual_source):
                continue
            label_specificity = _table_label_match_specificity(label, row, str(cell.get("column_label") or ""))
            if label and label_specificity <= 0:
                continue
            if row_target and not _row_target_matches(row_target, row):
                continue
            unlabeled_scope_fallback = False
            if column_target and column_target not in column:
                if column_target == "本年累计截至当期" and column.startswith("列"):
                    unlabeled_scope_fallback = True
                else:
                    continue
            if scope and scope not in combined:
                continue
            specificity = sum(bool(value) for value in (label, row_target, column_target, scope))
            row_index = int(cell.get("row") or 0)
            row_preference = -row_index * 0.01
            column_preference = column_index * 0.01 if unlabeled_scope_fallback else 0.0
            matches.append(
                (
                    label_specificity,
                    candidate.score + specificity * 10.0 + row_preference + column_preference,
                    candidate,
                )
            )
        if not matches and row_target and " " in row_target.strip():
            # A composite locator (“商业银行 不良贷款余额”) can overshoot when
            # the matched table has no institution prefix at all — the 商业银行
            # regulatory-indicator tables ARE the commercial-bank aggregate and
            # name rows “不良贷款余额” bare.  Retry with the non-institution
            # words only; the source scope still separates the right table.
            non_institution = [
                word for word in _row_label_terms(row_target)
                if not re.search(r"(?:公司|银行|机构|合计)$", normalize_table_text(word))
            ]
            retry_words = non_institution or _row_label_terms(row_target)
            retry_target = " ".join(retry_words)
            if retry_target and retry_target != row_target:
                for candidate in candidates:
                    cell = candidate.selected_cell or {}
                    if not isinstance(cell.get("normalized_value"), (int, float)):
                        continue
                    row = normalize_table_text(cell.get("row_label"))
                    column = normalize_table_text(cell.get("column_label"))
                    if generic_period and not re.search(period_pattern, str(cell.get('column_label') or '')):
                        continue
                    if any(
                        selector.get(key) is not None and int(candidate.metadata.get(key) or 0) != int(selector[key])
                        for key in ("year", "month", "quarter")
                    ):
                        continue
                    requested_source = normalize_table_text(selector.get("source_title"))
                    actual_source = normalize_table_text(candidate.metadata.get("source_title") or candidate.document.filename)
                    if requested_source and not _source_title_matches(requested_source, actual_source):
                        continue
                    if retry_target and not _row_target_matches(retry_target, row):
                        continue
                    if column_target and column_target not in column:
                        continue
                    if scope and scope not in f"{row}{column}":
                        continue
                    row_index = int(cell.get("row") or 0)
                    matches.append(
                        (
                            0,
                            candidate.score - row_index * 0.01,
                            candidate,
                        )
                    )
        if not matches:
            return []
        matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
        chosen = matches[0][2]
        if generic_period:
            # Temporal placeholders do not identify a quarter. Use actual table
            # order, never fixed spreadsheet columns. A transition with two
            # placeholders uses the first and last available records.
            best_specificity = matches[0][0]
            ordered = sorted(
                (item[2] for item in matches if item[0] == best_specificity),
                key=lambda item: (
                    str(item.metadata.get("source_title") or ""),
                    str(item.metadata.get("sheet_name") or ""),
                    int((item.selected_cell or {}).get("row") or 0),
                    int((item.selected_cell or {}).get("column") or 0),
                ),
            )
            transition = any(item.get("ordered_transition") for item in selectors)
            if transition and selected and row_target == str(selectors[0].get('row_label') or '').strip():
                anchor = selected[0]
                ordered = [item for item in ordered if
                    item.document.document_id == anchor.document.document_id and
                    item.metadata.get('sheet_name') == anchor.metadata.get('sheet_name') and
                    (item.selected_cell or {}).get('row') == (anchor.selected_cell or {}).get('row')]
                if not ordered:
                    return []
            chosen = ordered[-1] if transition and selected else ordered[0]
            chosen.metadata["selection_assumptions"] = [
                "期间标签未指定具体日期，按表内顺序取"
                + ("末项" if transition and selected else "首项")
                + f"：{(chosen.selected_cell or {}).get('row_label')} / "
                + f"{(chosen.selected_cell or {}).get('column_label')}。"
            ]
        elif not any(selector.get(key) is not None for key in ("month", "quarter")):
            period_labels = {
                token for _, _, item in matches
                for token in re.findall(r"[一二三四1-4]季度|\d{1,2}月", str(
                    (item.selected_cell or {}).get("row_label", "")
                ) + str((item.selected_cell or {}).get("column_label", "")))
            }
            if len(period_labels) > 1:
                if keep_all_periods:
                    # compare（max/min）把期间当作分组维度：保留全部期间的
                    # 候选，由 _comparison_candidate 在完整集合上挑极值并显示
                    # 获胜期间，而不是先截断到首个记录再比较。
                    best_specificity = matches[0][0]
                    period_matched = [item[2] for item in matches if item[0] == best_specificity]
                    for candidate in period_matched:
                        if not any(
                            _candidate_cell_key(candidate) == _candidate_cell_key(existing)
                            for existing in selected
                        ):
                            selected.append(candidate)
                    continue
                chosen.metadata["selection_assumptions"] = [
                    "未指定具体期间，采用表内首个匹配记录："
                    + f"{(chosen.selected_cell or {}).get('row_label')} / "
                    + f"{(chosen.selected_cell or {}).get('column_label')}。"
                ]
        if any(_candidate_cell_key(chosen) == _candidate_cell_key(existing) for existing in selected):
            # A single cell cannot stand in for two different requested operands.
            return []
        selected.append(chosen)
    return selected


def _specific_column_target_from_question(
    question: str,
    candidates: list[SpreadsheetCandidate],
    *,
    selectors: list[dict[str, Any]] | None = None,
) -> str:
    question_norm = normalize_table_text(question)
    if not question_norm:
        return ""
    selector_sources = {
        normalize_table_text(selector.get("source_title"))
        for selector in selectors or []
        if isinstance(selector, dict) and selector.get("source_title")
    }
    column_leafs: list[str] = []
    for candidate in candidates:
        if selector_sources:
            candidate_source = normalize_table_text(
                candidate.metadata.get("source_title") or candidate.document.filename
            )
            if not any(source and source in candidate_source for source in selector_sources):
                continue
        cell = candidate.selected_cell or {}
        leaf = _column_leaf(cell.get("column_label"))
        if leaf and not _is_generic_column_target(leaf):
            column_leafs.append(leaf)
    matches = [leaf for leaf in dict.fromkeys(column_leafs) if leaf in question_norm]
    if not matches:
        return ""
    matches.sort(key=len, reverse=True)
    return matches[0]


def _column_leaf(column_label: Any) -> str:
    return normalize_table_text(str(column_label or "").rsplit(" / ", 1)[-1])


def _is_generic_column_target(value: str) -> bool:
    return normalize_table_text(value) in {
        "",
        "项目",
        "名称",
        "项目名称",
        "指标",
        "指标名称",
        "地区",
        "合计",
        "总计",
        "本年累计",
        "截至当期",
        "本年累计截至当期",
    }


def _table_label_match_specificity(requested: str, row: str, column_label: str) -> int:
    """Match a selector against structural labels without table-title leakage.

    Spreadsheet column labels often contain the full table title before the
    final header (for example ``...全国各地区... / 健康险``).  Matching a row
    selector such as ``全国`` against that entire string makes every row look
    like the nationwide row.  Compare only the normalized row label and the
    terminal column header, with exact row matches taking precedence.
    """

    if not requested:
        return 0
    row_value = normalize_table_text(row)
    column_leaf = normalize_table_text(column_label.rsplit(" / ", 1)[-1])
    if requested == row_value:
        return 40
    if requested == column_leaf:
        return 35
    if requested in row_value or row_value in requested:
        return 20
    # Space-joined multi-word labels (“财产险公司 银行存款”) may be separated
    # inside the row by structural words (“其中：银行存款”), so fall back to
    # word-wise AND coverage.
    terms = [term for term in re.split(r"\s+", requested) if term]
    if len(terms) > 1 and all(term in row_value for term in terms):
        return 12
    if requested in column_leaf or column_leaf in requested:
        return 15
    requested_base = re.sub(r"(?:合计|总计)$", "", requested)
    row_base = re.sub(r"(?:合计|总计)$", "", row_value)
    column_base = re.sub(r"(?:合计|总计)$", "", column_leaf)
    if requested_base and row_base and (requested_base in row_base or row_base in requested_base):
        return 10
    if requested_base and column_base and (requested_base in column_base or column_base in requested_base):
        return 5
    return 0


def _row_target_matches(requested: str, row: str) -> bool:
    if not requested:
        return True
    terms = _row_label_terms(requested)
    if _row_label_terms_covered(row, terms):
        return True
    nationwide_aliases = {"全国", "全国合计"}
    return (
        len(terms) == 1
        and normalize_table_text(terms[0]) in nationwide_aliases
        and row in nationwide_aliases
    )


def _unique_numeric_candidates(candidates: list[SpreadsheetCandidate]) -> list[SpreadsheetCandidate]:
    result: list[SpreadsheetCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        cell = candidate.selected_cell or {}
        if not isinstance(cell.get("normalized_value"), (int, float)):
            continue
        key = _candidate_cell_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def _candidate_cell_key(candidate: SpreadsheetCandidate) -> tuple[str, str, str]:
    cell = candidate.selected_cell or {}
    return candidate.document.document_id, str(candidate.metadata.get("sheet_name") or ""), str(cell.get("coordinate") or "")


def _round_half_up_text(value: Decimal, decimal_places: int | None) -> str:
    """Render a Decimal with regulatory 四舍五入 (half-up), not half-even."""

    if decimal_places is None:
        return format(value, "f")
    return str(value.quantize(Decimal(1).scaleb(-decimal_places), rounding=ROUND_HALF_UP))


def _operand_decimal(candidate: SpreadsheetCandidate) -> Decimal:
    """Calculation operand from the displayed value string when possible.

    The index keeps both the displayed string (``value``, e.g. “24155.58512”)
    and the full-precision number (``numeric_value``).  A calculation must be
    reproducible from what the user can see, and gold answers are computed
    from the displayed values, so prefer the string; fall back to the numeric
    value when the string is not a plain decimal.
    """
    raw = (candidate.selected_cell or {}).get("value")
    if isinstance(raw, str) and re.fullmatch(r"-?\d+(?:\.\d+)?", raw.strip()):
        return Decimal(raw.strip())
    return Decimal(str((candidate.selected_cell or {})["normalized_value"]))


def _ratio_same_source_selection(
    candidates: list[SpreadsheetCandidate],
    selectors: list[dict[str, Any]],
    *,
    question: str,
    question_column_target: str | None,
    filters: dict[str, Any],
) -> list[SpreadsheetCandidate]:
    """Pick ratio operands from ONE source table, preferring the question's family.

    Each source group must satisfy every selector; among the satisfying
    groups, a source whose title carries the question's family scope word
    (“保险业”) wins, then the group matching the most selectors.
    """

    groups: dict[str, list[SpreadsheetCandidate]] = {}
    for candidate in candidates:
        key = normalize_table_text(
            str(candidate.metadata.get("source_title") or candidate.document.filename or "")
        )
        groups.setdefault(key, []).append(candidate)
    scope = normalize_table_text(filters.get("scope") or "")
    best: tuple[tuple[int, int], list[SpreadsheetCandidate]] | None = None
    for key, group in groups.items():
        selected = _cells_for_selectors(
            group,
            selectors,
            question=question,
            question_column_target=question_column_target,
            preserve_explicit_generic_columns=True,
        )
        if len(selected) != len(selectors):
            continue
        # Distinct operands must stay distinct cells.
        keys = {_candidate_cell_key(candidate) for candidate in selected}
        if len(keys) != len(selected):
            continue
        rank = (1 if scope and scope in key else 0, len(selected))
        if best is None or rank > best[0]:
            best = (rank, selected)
    return best[1] if best else []


def _calculation_match_from_selected(
    candidates: list[SpreadsheetCandidate],
    aspect: Any,
    operation: str,
    *,
    ordered_transition: bool,
) -> RetrievalMatch | None:
    if not candidates or operation == "ratio" and len(candidates) != 2:
        return None
    if operation == "difference" and len(candidates) < 2:
        return None
    units = {str((candidate.selected_cell or {}).get("unit") or candidate.metadata.get("unit") or "") for candidate in candidates}
    units.discard("")
    if len(units) > 1:
        return None
    decimal_values = [_operand_decimal(candidate) for candidate in candidates]
    question_text = str(getattr(aspect, "question", "") or "")
    ratio_is_percent = False
    if operation == "difference":
        if ordered_transition and len(decimal_values) == 2:
            decimal_result = decimal_values[1] - decimal_values[0]
            formula = f"{_cell_ref(candidates[1])} - {_cell_ref(candidates[0])}"
        else:
            # Reconciliation expressions commonly use A-B-C.  Operand order is
            # preserved by selectors, so a difference with more than two cells
            # means the first value minus every subsequent value.
            decimal_result = decimal_values[0] - sum(decimal_values[1:], Decimal("0"))
            formula = " - ".join(_cell_ref(candidate) for candidate in candidates)
    elif operation == "sum":
        decimal_result = sum(decimal_values, Decimal("0"))
        formula = " + ".join(_cell_ref(candidate) for candidate in candidates)
    elif operation == "ratio":
        if decimal_values[1] == 0:
            return None
        # A ratio is normally the raw quotient (0.1609 / 2.50); an explicit
        # percent-shaped request (百分比/比重/占比/比例) reads as a percentage.
        quotient = decimal_values[0] / decimal_values[1]
        if re.search(r"百分比|百分数|比重|占比|比例", question_text):
            decimal_result = quotient * Decimal("100")
            formula = f"{_cell_ref(candidates[0])} / {_cell_ref(candidates[1])} * 100"
            ratio_is_percent = True
        else:
            decimal_result = quotient
            formula = f"{_cell_ref(candidates[0])} / {_cell_ref(candidates[1])}"
            ratio_is_percent = False
    else:
        return None
    result = float(decimal_result)
    display_decimal_places = _requested_decimal_places(question_text)
    if display_decimal_places is None and operation == "ratio":
        # “多少倍” reads as a multiplier with two decimals (2.50); a percent
        # answer keeps the regulatory convention of two decimals (2.70%);
        # a plain proportion keeps four (0.1609).
        if re.search(r"[多少几]倍", question_text) or ratio_is_percent:
            # 倍数与百分数都保留两位小数（2.50 倍 / 2.70%）。
            display_decimal_places = 2
        else:
            display_decimal_places = 4
    display_result = _round_half_up_text(decimal_result, display_decimal_places)
    anchor = candidates[0]
    metadata = {
        **anchor.metadata,
        "dynamic_table_evidence": True,
        "table_task": "calculate",
        "operation": operation,
        "ordered_transition": ordered_transition,
        "match_status": "valid_calculation",
        "refusal_reason": None,
        "calculation_formula": formula,
        "calculation_display_formula": _display_calculation_formula(
            candidates,
            operation,
            result,
            ordered_transition=ordered_transition,
            decimal_places=display_decimal_places,
            result_text=display_result,
            ratio_percent=ratio_is_percent,
        ),
        "calculation_result": result,
        "calculation_result_text": display_result,
        "calculation_cells": [_cell_metadata(candidate) for candidate in candidates],
        "selection_assumptions": list(dict.fromkeys(
            assumption for candidate in candidates
            for assumption in candidate.metadata.get("selection_assumptions", [])
        )),
        **({"result_scale": 100, "unit": "%"} if ratio_is_percent else {}),
        **({"display_decimal_places": display_decimal_places} if display_decimal_places is not None else {}),
        "evidence_id": f"{anchor.chunk.chunk_id}::CALC::{operation}",
        "aspect_id": getattr(aspect, "aspect_id", None),
        "aspect_question": getattr(aspect, "question", None),
    }
    excerpt = f"表格计算证据：{formula} = {result}。参与单元格：" + "；".join(
        f"{candidate.metadata.get('sheet_name')}!{candidate.selected_cell.get('coordinate')}={candidate.selected_cell.get('value')}"
        for candidate in candidates
    )
    return RetrievalMatch(
        citation=Citation(
            document_id=anchor.document.document_id,
            chunk_id=anchor.chunk.chunk_id,
            filename=anchor.document.filename,
            source_url=anchor.document.source_url,
            attachment_url=anchor.document.attachment_url,
            source_title=anchor.document.title,
            issuing_authority=anchor.document.issuing_authority,
            publication_date=anchor.document.publication_date,
            document_number=anchor.document.document_number,
            version_status=anchor.document.version_status,
            section_title=anchor.chunk.section_title,
            section_path=[anchor.chunk.section_title] if anchor.chunk.section_title else [],
            page_number=anchor.chunk.page_number,
            excerpt=excerpt,
            score=anchor.score,
            rerank_score=anchor.score,
            chunk_type="table_calculation",
            evidence_role="table_evidence",
            metadata=metadata,
        ),
        score=anchor.score,
        rerank_score=anchor.score,
        coverage_score=1.0,
        evidence_role="table_evidence",
        evidence_text=excerpt,
        metadata=metadata,
    )


def _attach_sum_comparison_target(
    calculation: RetrievalMatch,
    candidates: list[SpreadsheetCandidate],
    selected: list[SpreadsheetCandidate],
    filters: dict[str, Any],
    operation: str,
) -> None:
    if operation != "sum":
        return
    target = normalize_table_text(filters.get("indicator") or filters.get("row_label"))
    if not target:
        return
    selected_keys = {_candidate_cell_key(candidate) for candidate in selected}
    matches = [
        candidate
        for candidate in candidates
        if _candidate_cell_key(candidate) not in selected_keys
        and target in normalize_table_text((candidate.selected_cell or {}).get("row_label") or candidate.metadata.get("row_label"))
        and isinstance((candidate.selected_cell or {}).get("normalized_value"), (int, float))
    ]
    if not matches:
        return
    matches.sort(key=lambda item: item.score, reverse=True)
    target_candidate = matches[0]
    target_cell = target_candidate.selected_cell or {}
    target_value = float(target_cell.get("normalized_value"))
    result = float(calculation.metadata.get("calculation_result"))
    equal = abs(result - target_value) < 0.000001
    comparison = {
        "label": target_cell.get("row_label"),
        "cell": target_cell.get("coordinate"),
        "value": target_cell.get("value"),
        "normalized_value": target_value,
        "equal": equal,
    }
    calculation.metadata["comparison_target"] = comparison
    calculation.metadata["comparison_equal"] = equal
    calculation.citation.metadata["comparison_target"] = comparison
    calculation.citation.metadata["comparison_equal"] = equal


def _display_calculation_formula(
    candidates: list[SpreadsheetCandidate],
    operation: str,
    result: float,
    *,
    ordered_transition: bool,
    decimal_places: int | None = None,
    result_text: str | None = None,
    ratio_percent: bool = False,
) -> str:
    values = [_source_number_for_formula(candidate) for candidate in candidates]
    result_text = result_text or _format_number_for_formula(
        result,
        decimal_places=decimal_places,
    )
    if operation == "difference":
        if ordered_transition and len(values) == 2:
            return f"{values[1]} - {values[0]} = {result_text}"
        return f"{' - '.join(values)} = {result_text}"
    if operation == "sum":
        return f"{' + '.join(values)} = {result_text}"
    if operation == "ratio" and len(values) >= 2:
        if ratio_percent:
            return f"{values[0]} / {values[1]} × 100 = {result_text}"
        return f"{values[0]} / {values[1]} = {result_text}"
    return result_text


def _source_number_for_formula(candidate: SpreadsheetCandidate) -> str:
    cell = candidate.selected_cell or {}
    raw_value = str(cell.get("value") or "").strip()
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?", raw_value):
        decimal_value = Decimal(raw_value)
        if decimal_value == decimal_value.to_integral_value():
            return str(int(decimal_value))
        return raw_value
    return _format_number_for_formula(float(cell["normalized_value"]))


def _format_number_for_formula(value: float, *, decimal_places: int | None = None) -> str:
    if decimal_places is not None:
        return f"{value:.{decimal_places}f}"
    return str(int(value)) if float(value).is_integer() else f"{value:.4f}".rstrip("0").rstrip(".")


def _requested_decimal_places(question: str) -> int | None:
    match = re.search(r"保留([零〇一二两三四五六七八九十\d]+)位小数", question)
    if not match:
        return None
    value = match.group(1)
    if value.isdigit():
        places = int(value)
    else:
        places = {
            "零": 0,
            "〇": 0,
            "一": 1,
            "二": 2,
            "两": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
            "十": 10,
        }.get(value)
    return places if places is not None and 0 <= places <= 10 else None


def _spreadsheet_candidates(db: Session, question: str, filters: dict[str, Any]) -> list[SpreadsheetCandidate]:
    rows = db.execute(
        select(DocumentChunk, Document)
        .join(Document, DocumentChunk.document_id == Document.document_id)
        .where(Document.status == "indexed", DocumentChunk.chunk_metadata.is_not(None))
    ).all()
    candidates: list[SpreadsheetCandidate] = []
    for chunk, document in rows:
        metadata = _metadata(chunk)
        if not metadata.get("spreadsheet_table") or metadata.get("table_chunk_role") != "row":
            continue
        sheet_filter = normalize_table_text(filters.get("sheet"))
        if sheet_filter and sheet_filter not in normalize_table_text(metadata.get("sheet_name")):
            continue
        if any(filters.get(key) is not None for key in ("year", "month", "quarter")) and not _period_matches(
            filters, metadata, document.filename
        ):
            continue
        score = _candidate_score(question, filters, document, chunk, metadata)
        if score <= 0:
            continue
        selected_cell = _select_cell(question, filters, metadata)
        if selected_cell:
            score += 2.0
        candidates.append(SpreadsheetCandidate(document, chunk, metadata, score, selected_cell))
    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


def _candidate_score(
    question: str,
    filters: dict[str, Any],
    document: Document,
    chunk: DocumentChunk,
    metadata: dict[str, Any],
) -> float:
    haystack = _normalize(
        " ".join(
            str(part)
            for part in [
                document.filename,
                metadata.get("source_title"),
                metadata.get("table_title"),
                metadata.get("sheet_name"),
                metadata.get("row_label"),
                chunk.text,
                json.dumps(metadata.get("row_cells") or {}, ensure_ascii=False),
            ]
            if part
        )
    )
    score = 0.0
    for key in ("source_title", "filename"):
        value = str(filters.get(key) or "").strip()
        if value and _normalize(value) in haystack:
            score += 4.0
    for key, weight in [("sheet", 3.0), ("row_label", 3.0), ("indicator", 2.0), ("column_label", 2.0), ("metric", 2.0)]:
        value = str(filters.get(key) or "").strip()
        if key == "row_label":
            # Multi-word row labels must all be covered (AND semantics), the
            # same rule as the indexed path.
            if not _row_label_terms_covered(haystack, _row_label_terms(value)):
                continue
            score += weight
        elif value and _normalize(value) in haystack:
            score += weight
    if _period_matches(filters, metadata, document.filename):
        score += 3.0
    score += _token_overlap(question, haystack)
    return score


def _select_cell(question: str, filters: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any] | None:
    cells = [cell for cell in metadata.get("cells") or [] if isinstance(cell, dict)]
    if not cells:
        return None
    target = _normalize(
        " ".join(
            str(filters.get(key) or "")
            for key in ("column_label", "indicator", "metric", "scope")
            if filters.get(key)
        )
    )
    if not target:
        quoted = re.findall(r"“([^”]+)”", question)
        target = _normalize(" ".join(quoted))

    scored: list[tuple[float, dict[str, Any]]] = []
    for cell in cells:
        label = _normalize(" ".join(str(part) for part in [cell.get("column_label"), " ".join(cell.get("column_path") or [])] if part))
        value = str(cell.get("value") or "")
        score = 0.0
        for key in ("column_label", "indicator", "metric", "scope"):
            filter_value = str(filters.get(key) or "").strip()
            if filter_value and _normalize(filter_value) in label:
                score += 3.0
        if target and target in label:
            score += 4.0
        if target and label and label in target:
            score += 2.0
        score += _token_overlap(question, label)
        if isinstance(cell.get("normalized_value"), (int, float)):
            score += 0.25
        if value and _normalize(value) in _normalize(question):
            score += 0.5
        scored.append((score, cell))
    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored or scored[0][0] <= 0:
        numeric = [cell for cell in cells if isinstance(cell.get("normalized_value"), (int, float))]
        return numeric[0] if numeric else cells[0]
    return scored[0][1]


def _comparison_candidate(candidates: list[SpreadsheetCandidate], operation: str) -> SpreadsheetCandidate | None:
    units = {str((item.selected_cell or {}).get("unit") or item.metadata.get("unit") or "")
             for item in candidates}
    if len(units) > 1:
        return None
    comparable: list[tuple[float, SpreadsheetCandidate]] = []
    for candidate in candidates:
        cell = candidate.selected_cell or _first_numeric_cell(candidate.metadata)
        if not cell or not isinstance(cell.get("normalized_value"), (int, float)):
            continue
        candidate.selected_cell = cell
        comparable.append((float(cell["normalized_value"]), candidate))
    if not comparable:
        return None
    comparable.sort(key=lambda item: item[0], reverse=operation == "max")
    return comparable[0][1]


def _calculation_match(candidates: list[SpreadsheetCandidate], aspect: Any, operation: str) -> RetrievalMatch | None:
    numeric: list[SpreadsheetCandidate] = []
    for candidate in candidates:
        cell = candidate.selected_cell or _first_numeric_cell(candidate.metadata)
        if cell and isinstance(cell.get("normalized_value"), (int, float)):
            candidate.selected_cell = cell
            numeric.append(candidate)
    if not numeric or operation in {"difference", "ratio"} and len(numeric) < 2:
        return None

    values = [float(candidate.selected_cell["normalized_value"]) for candidate in numeric]
    if operation == "sum":
        result = sum(values)
        formula = " + ".join(_cell_ref(candidate) for candidate in numeric)
    elif operation == "difference":
        result = values[0] - values[1]
        formula = f"{_cell_ref(numeric[0])} - {_cell_ref(numeric[1])}"
    elif operation == "ratio":
        if values[1] == 0:
            return None
        quotient = values[0] / values[1]
        if re.search(r"百分比", str(getattr(aspect, "question", "") or "")):
            result = quotient * 100
            formula = f"{_cell_ref(numeric[0])} / {_cell_ref(numeric[1])} * 100"
            ratio_is_percent = True
        else:
            result = quotient
            formula = f"{_cell_ref(numeric[0])} / {_cell_ref(numeric[1])}"
            ratio_is_percent = False
    else:
        return None

    anchor = numeric[0]
    metadata = {
        **anchor.metadata,
        "dynamic_table_evidence": True,
        "table_task": "calculate",
        "operation": operation,
        "match_status": "valid_calculation",
        "refusal_reason": None,
        "calculation_formula": formula,
        "calculation_result": result,
        "calculation_cells": [_cell_metadata(candidate) for candidate in numeric],
        **({"result_scale": 100, "unit": "%"} if ratio_is_percent else {}),
        "evidence_id": f"{anchor.chunk.chunk_id}::CALC::{operation}",
        "aspect_id": getattr(aspect, "aspect_id", None),
    }
    excerpt = f"表格计算证据：{formula} = {result}。参与单元格：" + "；".join(
        f"{candidate.metadata.get('sheet_name')}!{candidate.selected_cell.get('coordinate')}={candidate.selected_cell.get('value')}"
        for candidate in numeric
    )
    return RetrievalMatch(
        citation=Citation(
            document_id=anchor.document.document_id,
            chunk_id=anchor.chunk.chunk_id,
            filename=anchor.document.filename,
            source_url=anchor.document.source_url,
            attachment_url=anchor.document.attachment_url,
            source_title=anchor.document.title,
            issuing_authority=anchor.document.issuing_authority,
            publication_date=anchor.document.publication_date,
            document_number=anchor.document.document_number,
            version_status=anchor.document.version_status,
            section_title=anchor.chunk.section_title,
            section_path=[anchor.chunk.section_title] if anchor.chunk.section_title else [],
            page_number=anchor.chunk.page_number,
            excerpt=excerpt,
            score=anchor.score,
            rerank_score=anchor.score,
            chunk_type="table_calculation",
            evidence_role="table_evidence",
            metadata=metadata,
        ),
        score=anchor.score,
        rerank_score=anchor.score,
        coverage_score=1.0,
        evidence_role="table_evidence",
        evidence_text=excerpt,
        metadata=metadata,
    )


def _to_match(
    candidate: SpreadsheetCandidate,
    aspect: Any,
    table_task: str,
    operation: str,
    *,
    match_status: str = "valid_cell",
    refusal_reason: str | None = None,
) -> RetrievalMatch:
    cell = candidate.selected_cell or _select_cell(str(getattr(aspect, "question", "") or ""), getattr(aspect, "table_filters", {}) or {}, candidate.metadata)
    metadata = {
        **candidate.metadata,
        "dynamic_table_evidence": bool(cell),
        "table_task": table_task,
        "operation": operation,
        "match_status": match_status,
        "refusal_reason": refusal_reason,
        "selected_cell": cell,
        "sheet_name": candidate.metadata.get("sheet_name"),
        "cell": cell.get("coordinate") if cell else None,
        "value": cell.get("value") if cell else None,
        "unit": cell.get("unit") or candidate.metadata.get("unit") if cell else candidate.metadata.get("unit"),
        "row_label": cell.get("row_label") if cell else candidate.metadata.get("row_label"),
        "column_label": cell.get("column_label") if cell else None,
        "evidence_id": f"{candidate.chunk.chunk_id}::{cell.get('coordinate')}" if cell else candidate.chunk.chunk_id,
        "aspect_id": getattr(aspect, "aspect_id", None),
        "aspect_question": getattr(aspect, "question", None),
    }
    excerpt = _cell_excerpt(candidate, cell) if cell else candidate.chunk.text
    return RetrievalMatch(
        citation=Citation(
            document_id=candidate.document.document_id,
            chunk_id=candidate.chunk.chunk_id,
            filename=candidate.document.filename,
            source_url=candidate.document.source_url,
            attachment_url=candidate.document.attachment_url,
            source_title=candidate.document.title,
            issuing_authority=candidate.document.issuing_authority,
            publication_date=candidate.document.publication_date,
            document_number=candidate.document.document_number,
            version_status=candidate.document.version_status,
            section_title=candidate.chunk.section_title,
            section_path=[candidate.chunk.section_title] if candidate.chunk.section_title else [],
            page_number=candidate.chunk.page_number,
            excerpt=excerpt,
            score=candidate.score,
            rerank_score=candidate.score,
            chunk_type="table_cell" if cell else "table",
            evidence_role="table_evidence",
            metadata=metadata,
        ),
        score=candidate.score,
        rerank_score=candidate.score,
        coverage_score=1.0,
        evidence_role="table_evidence",
        evidence_text=excerpt,
        metadata=metadata,
    )


def _cell_excerpt(candidate: SpreadsheetCandidate, cell: dict[str, Any]) -> str:
    unit = cell.get("unit") or candidate.metadata.get("unit") or ""
    unit_text = f"，单位：{unit}" if unit else ""
    return (
        f"表格单元格证据：文件《{candidate.document.filename}》，工作表“{candidate.metadata.get('sheet_name')}”，"
        f"表格《{candidate.metadata.get('table_title')}》，行标签“{cell.get('row_label')}”，"
        f"列“{cell.get('column_label')}”，单元格 {cell.get('coordinate')} 的原始值为 {cell.get('value')}{unit_text}。"
    )


def _first_numeric_cell(metadata: dict[str, Any]) -> dict[str, Any] | None:
    for cell in metadata.get("cells") or []:
        if isinstance(cell, dict) and isinstance(cell.get("normalized_value"), (int, float)):
            return cell
    return None


def _generic_comparison_selector(selector: dict[str, Any]) -> bool:
    label = normalize_table_text(
        selector.get("label") or selector.get("row_label") or selector.get("row_or_indicator") or ""
    )
    return label in {"数值", "值", "指标", "哪一项", "项目", "项目名称", "指标名称"}


def _numeric_candidates_for_column(
    candidates: list[SpreadsheetCandidate], column_target: Any
) -> list[SpreadsheetCandidate]:
    """Pick one numeric cell per row in the requested column.

    A "which item is highest" question is a column scan, not a set of
    independent row selectors.  The generic selector produced by the fallback
    planner (for example ``{"label": "数值"}``) must not collapse the scan to
    one arbitrary cell.
    """

    target = normalize_table_text(column_target)
    selected: list[SpreadsheetCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        cell = None
        if target:
            for item in candidate.metadata.get("cells") or []:
                if not isinstance(item, dict) or not isinstance(item.get("normalized_value"), (int, float)):
                    continue
                label = normalize_table_text(
                    " ".join(
                        str(part)
                        for part in [item.get("column_label"), *(item.get("column_path") or [])]
                        if part
                    )
                )
                if target in label or label in target:
                    cell = item
                    break
        if cell is None:
            cell = _first_numeric_cell(candidate.metadata)
        if not cell or not isinstance(cell.get("normalized_value"), (int, float)):
            continue
        candidate.selected_cell = cell
        key = _candidate_cell_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        selected.append(candidate)
    return selected


def _metadata(chunk: DocumentChunk) -> dict[str, Any]:
    if not chunk.chunk_metadata:
        return {}
    try:
        value = json.loads(chunk.chunk_metadata)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _period_matches(filters: dict[str, Any], metadata: dict[str, Any], filename: str) -> bool:
    period = metadata.get("period") if isinstance(metadata.get("period"), dict) else {}
    matched = False
    for key in ("year", "month", "quarter"):
        if filters.get(key) is None:
            continue
        matched = True
        if period.get(key) == filters.get(key):
            continue
        if str(filters.get(key)) not in filename:
            return False
    return matched


def _token_overlap(question: str, text: str) -> float:
    terms = [term for term in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]+", question) if len(term) >= 2]
    if not terms:
        return 0.0
    normalized_text = _normalize(text)
    hit = 0.0
    total = 0.0
    for term in terms:
        normalized_term = _normalize(term)
        total += 1
        if normalized_term in normalized_text:
            hit += 1
            continue
        # A Chinese clause almost never appears verbatim in a label, so whole
        # clauses score 0 and cannot break ties.  Their character bigrams
        # carry the discriminating words (综合 vs 核心, 债券 vs 存款).
        if re.fullmatch(r"[\u4e00-\u9fff]+", normalized_term) and len(normalized_term) > 2:
            grams = [normalized_term[i : i + 2] for i in range(len(normalized_term) - 1)]
            covered = sum(1 for gram in grams if gram in normalized_text)
            hit += covered / len(grams) * 0.5
    return hit / total if total else 0.0


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _cell_ref(candidate: SpreadsheetCandidate) -> str:
    cell = candidate.selected_cell or {}
    return f"{candidate.metadata.get('sheet_name')}!{cell.get('coordinate')}"


def _cell_metadata(candidate: SpreadsheetCandidate) -> dict[str, Any]:
    cell = candidate.selected_cell or {}
    return {
        "document_id": candidate.document.document_id,
        "chunk_id": candidate.chunk.chunk_id,
        "filename": candidate.document.filename,
        "source_title": candidate.metadata.get("source_title") or candidate.document.title,
        "sheet_name": candidate.metadata.get("sheet_name"),
        "cell": cell.get("coordinate"),
        "value": cell.get("value"),
        "normalized_value": cell.get("normalized_value"),
        "unit": cell.get("unit") or candidate.metadata.get("unit"),
        "row_label": cell.get("row_label"),
        "column_label": cell.get("column_label"),
        "period": candidate.metadata.get("period"),
    }


def _set_last_diagnostic(**values: Any) -> None:
    current = _SPREADSHEET_DIAGNOSTIC.get()
    if current is None:
        current = {}
        _SPREADSHEET_DIAGNOSTIC.set(current)
    current.clear()
    current.update(values)


def _set_valid_diagnostic(matches: list[RetrievalMatch]) -> None:
    statuses = [
        str((match.metadata or match.citation.metadata or {}).get("match_status") or "valid_cell")
        for match in matches
    ]
    _set_last_diagnostic(
        match_status=statuses[0] if len(set(statuses)) == 1 and statuses else "valid",
        refusal_reason=None,
        match_count=len(matches),
    )


def _missing_row_status(filters: dict[str, Any]) -> str:
    if any(filters.get(key) is not None for key in ("year", "month", "quarter")):
        return "period_not_found"
    if filters.get("sheet"):
        return "sheet_not_found"
    if filters.get("source_title") or filters.get("filename"):
        return "source_not_found"
    return "no_candidates"


def _missing_row_reason(filters: dict[str, Any]) -> str:
    status = _missing_row_status(filters)
    if status == "period_not_found":
        return "指定年份、月份或季度在表格索引中不存在。"
    if status == "sheet_not_found":
        return "指定工作表在候选文件中不存在。"
    if status == "source_not_found":
        return "指定文件或报表标题未在知识库中找到。"
    return "未检索到可用表格候选。"


def _operand_matches(candidates: list[SpreadsheetCandidate], aspect: Any) -> list[RetrievalMatch]:
    """Expose original cells, never compute or inject the answer in this path."""
    matches = [_to_match(item, aspect, "lookup", "none", match_status="valid_cell")
               for item in candidates]
    for match in matches:
        match.metadata["calculation_operand_only"] = True
    _set_valid_diagnostic(matches)
    event("calculation", enabled=False, operands=len(matches))
    return matches
