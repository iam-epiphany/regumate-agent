from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from sqlalchemy import select
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
    sheet = normalize_table_text(filters.get("sheet"))
    if source:
        statement = statement.where(SpreadsheetCell.source_title_norm.contains(source))
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
    if table_task == "calculate" and getattr(aspect, "selectors", ()):
        # In questions such as “全国合计”从“合计”到“健康险”, the parser also
        # extracts the last quoted column as table_filters["column_label"].
        # Treating that as a global hard filter removes the first operand
        # before selector matching.  For structured calculations, row/indicator
        # constraints remain hard, while each operand column is checked by its
        # own selector below.
        label_filters.pop("row_label", None)
        label_filters.pop("column_label", None)
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
    selectors = [item for item in getattr(aspect, "selectors", ()) if isinstance(item, dict)]

    if table_task == "compare" and operation in {"max", "min"}:
        selected = _cells_for_selectors(candidates, selectors) if selectors else _unique_numeric_candidates(candidates)
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
        selected = _cells_for_selectors(candidates, selectors) if selectors else _unique_numeric_candidates(candidates)[:2]
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
        selected = _cells_for_selectors(candidates, selectors)
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
    for key, weight in (("row_label", 5.0), ("column_label", 5.0), ("indicator", 4.0), ("metric", 4.0), ("scope", 3.0)):
        target = normalize_table_text(filters.get(key))
        if target and target in combined:
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


def _cells_for_selectors(
    candidates: list[SpreadsheetCandidate],
    selectors: list[dict[str, Any]],
) -> list[SpreadsheetCandidate]:
    selected: list[SpreadsheetCandidate] = []
    for selector in selectors:
        label = normalize_table_text(selector.get("label") or selector.get("row_or_indicator"))
        row_target = normalize_table_text(selector.get("row_label"))
        column_target = normalize_table_text(selector.get("column_label"))
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
        matches: list[tuple[float, SpreadsheetCandidate]] = []
        for candidate in candidates:
            cell = candidate.selected_cell or {}
            if not isinstance(cell.get("normalized_value"), (int, float)):
                continue
            row = normalize_table_text(cell.get("row_label"))
            column = normalize_table_text(cell.get("column_label"))
            column_index = int(cell.get("column") or 0)
            combined = f"{row}{column}"
            if label and any(term in label for term in ["总资产", "总负债"]):
                requested_scope = normalize_table_text(selector.get("scope"))
                if "本年累计" in requested_scope:
                    continue
            if label and label not in combined:
                continue
            if row_target and row_target not in row:
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
            matches.append((candidate.score + specificity * 10.0 + row_preference + column_preference, candidate))
        if matches:
            matches.sort(key=lambda item: item[0], reverse=True)
            chosen = matches[0][1]
            if all(_candidate_cell_key(chosen) != _candidate_cell_key(existing) for existing in selected):
                selected.append(chosen)
    return selected


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
    if not candidates or operation in {"difference", "ratio"} and len(candidates) != 2:
        return None
    units = {str((candidate.selected_cell or {}).get("unit") or candidate.metadata.get("unit") or "") for candidate in candidates}
    units.discard("")
    if len(units) > 1:
        return None
    values = [float((candidate.selected_cell or {})["normalized_value"]) for candidate in candidates]
    if operation == "difference":
        if ordered_transition:
            result = values[1] - values[0]
            formula = f"{_cell_ref(candidates[1])} - {_cell_ref(candidates[0])}"
        else:
            result = values[0] - values[1]
            formula = f"{_cell_ref(candidates[0])} - {_cell_ref(candidates[1])}"
    elif operation == "sum":
        result = sum(values)
        formula = " + ".join(_cell_ref(candidate) for candidate in candidates)
    elif operation == "ratio":
        if values[1] == 0:
            return None
        result = values[0] / values[1]
        formula = f"{_cell_ref(candidates[0])} / {_cell_ref(candidates[1])}"
    else:
        return None
    anchor = candidates[0]
    metadata = {
        **anchor.metadata,
        "dynamic_table_evidence": True,
        "table_task": "calculate",
        "operation": operation,
        "match_status": "valid_calculation",
        "refusal_reason": None,
        "calculation_formula": formula,
        "calculation_display_formula": _display_calculation_formula(candidates, operation, result, ordered_transition=ordered_transition),
        "calculation_result": result,
        "calculation_cells": [_cell_metadata(candidate) for candidate in candidates],
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
) -> str:
    values = [_format_number_for_formula(float((candidate.selected_cell or {})["normalized_value"])) for candidate in candidates]
    result_text = _format_number_for_formula(result)
    if operation == "difference":
        if ordered_transition:
            return f"{values[1]} - {values[0]} = {result_text}"
        return f"{values[0]} - {values[1]} = {result_text}"
    if operation == "sum":
        return f"{' + '.join(values)} = {result_text}"
    if operation == "ratio" and len(values) >= 2:
        return f"{values[0]} / {values[1]} = {result_text}"
    return result_text


def _format_number_for_formula(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.4f}".rstrip("0").rstrip(".")


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
        result = values[0] / values[1]
        formula = f"{_cell_ref(numeric[0])} / {_cell_ref(numeric[1])}"
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
