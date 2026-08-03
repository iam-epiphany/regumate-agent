from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import json
import re
from typing import Any

from sqlalchemy import or_, select
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


_LAST_DIAGNOSTIC: dict[str, Any] = {}


def get_last_spreadsheet_diagnostic() -> dict[str, Any]:
    return dict(_LAST_DIAGNOSTIC)


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
        calculation = _calculation_match(candidates, aspect, operation)
        return [calculation] if calculation else []

    lookup_limit = 1 if table_task == "lookup" else limit
    matches = [_to_match(candidate, aspect, table_task, operation) for candidate in candidates[:lookup_limit]]
    _set_valid_diagnostic(matches)
    return matches


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
        for item in getattr(aspect, "selectors", ())
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
    for field_name in ("year", "month", "quarter"):
        value = filters.get(field_name)
        if value is not None:
            statement = statement.where(getattr(SpreadsheetCell, field_name) == int(value))
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

    rows = db.execute(statement).all()
    # An explicit hard constraint that matches nothing must never fall back to
    # a whole-corpus maximum or calculation.
    if not rows:
        _set_last_diagnostic(
            match_status=_missing_row_status(filters),
            refusal_reason=_missing_row_reason(filters),
            table_filters=filters,
        )
        return []

    candidates = [_candidate_from_index(cell, chunk, document, question, filters) for cell, chunk, document in rows]
    label_filters = dict(filters)
    selectors = [item for item in getattr(aspect, "selectors", ()) if isinstance(item, dict)]
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
        selected = (
            _cells_for_selectors(
                candidates,
                selectors,
                question=question,
                question_column_target=question_column_target,
            )
            if selectors
            else _unique_numeric_candidates(candidates)
        )
        selected = _comparable_metric_candidates(selected)
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
        selected = (
            _cells_for_selectors(
                candidates,
                selectors,
                question=question,
                question_column_target=question_column_target,
                preserve_explicit_generic_columns=True,
            )
            if selectors
            else _unique_numeric_candidates(candidates)[:2]
        )
        calculation = _calculation_match_from_selected(
            selected,
            aspect,
            operation,
            ordered_transition=bool(selectors),
        )
        if calculation:
            _attach_sum_comparison_target(calculation, candidates, selected, filters, operation)
            _set_valid_diagnostic([calculation])
            return [calculation]
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
                _set_last_diagnostic(match_status="ambiguous_candidates", refusal_reason="指定条件命中多个等价候选单元格。")
                return []
        matches = [_to_match(top[0], aspect, table_task, operation, match_status="valid_cell")]
        _set_valid_diagnostic(matches)
        return matches

    matches = [_to_match(candidate, aspect, table_task, operation) for candidate in candidates[:lookup_limit]]
    _set_valid_diagnostic(matches)
    return matches


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
    score = 0.25 if cell.numeric_value is not None else 0.0
    column_leaf = normalize_table_text(re.split(r"\s*/\s*|[|｜>]", str(cell.column_label or ""))[-1])
    for key, weight in (("row_label", 5.0), ("column_label", 5.0), ("indicator", 4.0), ("metric", 4.0), ("scope", 3.0)):
        target = normalize_table_text(filters.get(key))
        if key == "row_label":
            structural_text = row
        elif key == "column_label":
            # A row such as “全国合计” must not make every numeric cell look
            # like the requested “合计” column.  Compare column constraints
            # only with the terminal header, excluding the repeated table
            # title prefix and the row label.
            structural_text = column_leaf
        else:
            structural_text = combined
        if target and target in structural_text:
            score += weight
    score += _token_overlap(question, combined)
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
                ]
                if part
            )
        )
        if all(term in searchable for term in required_terms):
            result.append(candidate)
    return result


def _required_label_terms(filters: dict[str, Any], selectors: tuple[dict[str, Any], ...]) -> list[str]:
    terms: list[str] = []
    for key in ("indicator", "row_label", "column_label", "metric"):
        value = normalize_table_text(filters.get(key))
        if value and not _is_generic_table_scope(value):
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
    return {value for value in variants if value}


def _source_title_matches(requested_source: str, actual_source: str) -> bool:
    if not requested_source:
        return True
    if requested_source in actual_source:
        return True
    return any(variant and variant in actual_source for variant in _source_title_variants(requested_source))


def _cells_for_selectors(
    candidates: list[SpreadsheetCandidate],
    selectors: list[dict[str, Any]],
    *,
    question: str = "",
    question_column_target: str | None = None,
    preserve_explicit_generic_columns: bool = False,
) -> list[SpreadsheetCandidate]:
    selected: list[SpreadsheetCandidate] = []
    if question_column_target is None:
        question_column_target = _specific_column_target_from_question(question, candidates)
    for selector in selectors:
        label = normalize_table_text(selector.get("label") or selector.get("row_or_indicator"))
        row_target = normalize_table_text(selector.get("row_label"))
        column_target = normalize_table_text(selector.get("column_label"))
        # An explicit selector column always wins; the question-derived target
        # only fills in an absent selector column.  A bare "合计" operand in
        # "从'合计'到'健康险'" must not be overwritten by another column name
        # found in the question.
        if question_column_target and not column_target:
            column_target = question_column_target
        elif (
            question_column_target
            and _is_generic_column_target(column_target)
            and not (preserve_explicit_generic_columns and column_target)
        ):
            column_target = question_column_target
        quarter_mode = None
        if column_target == "年季度":
            quarter_mode, column_target = "first", ""
        elif column_target == "季度":
            quarter_mode, column_target = "last", ""
        elif column_target == "季度季度":
            quarter_mode, column_target = "last_later_table", ""
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
            if any(
                selector.get(key) is not None and int(candidate.metadata.get(key) or 0) != int(selector[key])
                for key in ("year", "month", "quarter")
            ):
                continue
            requested_source = normalize_table_text(selector.get("source_title"))
            actual_source = normalize_table_text(candidate.metadata.get("source_title") or candidate.document.filename)
            if requested_source and not _source_title_matches(requested_source, actual_source):
                continue
            if label and any(term in label for term in ["总资产", "总负债"]):
                requested_scope = normalize_table_text(selector.get("scope"))
                if "本年累计" in requested_scope:
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
            effective_quarter_mode = quarter_mode
            if effective_quarter_mode == "last" and len(selectors) == 1:
                effective_quarter_mode = "first"
            if effective_quarter_mode == "first" and column_index != 2:
                continue
            if effective_quarter_mode in {"last", "last_later_table"} and column_index != 5:
                continue
            if scope and scope not in combined:
                continue
            specificity = sum(bool(value) for value in (label, row_target, column_target, scope))
            row_index = int(cell.get("row") or 0)
            if quarter_mode == "last_later_table" and selected:
                anchor_row = int((selected[0].selected_cell or {}).get("row") or 0)
                if row_index <= anchor_row:
                    continue
            row_preference = -row_index * 0.01
            column_preference = column_index * 0.01 if unlabeled_scope_fallback else 0.0
            matches.append(
                (
                    label_specificity,
                    candidate.score + specificity * 10.0 + row_preference + column_preference,
                    candidate,
                )
            )
        if matches:
            matches.sort(key=lambda item: (item[0], item[1]), reverse=True)
            chosen = matches[0][2]
            if all(_candidate_cell_key(chosen) != _candidate_cell_key(existing) for existing in selected):
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
    if requested in row:
        return True
    nationwide_aliases = {"全国", "全国合计"}
    return requested in nationwide_aliases and row in nationwide_aliases


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


def _comparable_metric_candidates(candidates: list[SpreadsheetCandidate]) -> list[SpreadsheetCandidate]:
    """Avoid comparing monetary flows with policy counts or insured amounts."""

    ordinary = [candidate for candidate in candidates if _metric_family(candidate) == "ordinary"]
    return ordinary if len(ordinary) >= 2 else candidates


def _metric_family(candidate: SpreadsheetCandidate) -> str:
    cell = candidate.selected_cell or {}
    label = normalize_table_text(f"{cell.get('row_label') or ''}{cell.get('column_label') or ''}")
    if any(term in label for term in ["件数", "保单数", "承保数量"]):
        return "count"
    if "保险金额" in label and "保费" not in label:
        return "insured_amount"
    return "ordinary"


def _candidate_cell_key(candidate: SpreadsheetCandidate) -> tuple[str, str, str]:
    cell = candidate.selected_cell or {}
    return candidate.document.document_id, str(candidate.metadata.get("sheet_name") or ""), str(cell.get("coordinate") or "")


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
    decimal_values = [
        Decimal(str((candidate.selected_cell or {})["normalized_value"]))
        for candidate in candidates
    ]
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
        decimal_result = decimal_values[0] / decimal_values[1] * Decimal("100")
        formula = f"{_cell_ref(candidates[0])} / {_cell_ref(candidates[1])} * 100"
    else:
        return None
    result = float(decimal_result)
    display_decimal_places = _requested_decimal_places(str(getattr(aspect, "question", "") or ""))
    display_result = (
        f"{decimal_result:.{display_decimal_places}f}"
        if display_decimal_places is not None
        else format(decimal_result, "f")
    )
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
        ),
        "calculation_result": result,
        "calculation_result_text": display_result,
        "calculation_cells": [_cell_metadata(candidate) for candidate in candidates],
        **({"result_scale": 100, "unit": "%"} if operation == "ratio" else {}),
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
        return f"{values[0]} / {values[1]} × 100 = {result_text}"
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
        if value and _normalize(value) in haystack:
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
        result = values[0] / values[1] * 100
        formula = f"{_cell_ref(numeric[0])} / {_cell_ref(numeric[1])} * 100"
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
        **({"result_scale": 100, "unit": "%"} if operation == "ratio" else {}),
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
    return sum(1 for term in terms if _normalize(term) in normalized_text) / len(terms)


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
    }


def _set_last_diagnostic(**values: Any) -> None:
    _LAST_DIAGNOSTIC.clear()
    _LAST_DIAGNOSTIC.update(values)


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
