"""Author a fresh, source-audited rigorous v5 evaluation batch.

The script reads only the immutable 500-document corpus and the production
index.  It never reads an earlier question set, answer key, or model output.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation import author_replacement_batch as source_builder
from scripts.evaluation import generalization_pipeline as audit_pipeline
from scripts.evaluation import rigorous_v3_scorer


CORPUS = ROOT / "data" / "contest_dataset" / "dataset" / "nfra_page_attachments_500"
DB_PATH = ROOT / "data" / "evaluation" / "final_runtime" / "app.db"
PUBLIC_ROOT = (
    ROOT
    / "data"
    / "evaluation"
    / "independent_100_20260731"
    / "rigorous_v5_final"
)
PRIVATE_ROOT = (
    ROOT.parent
    / "ReguMate-Eval-Private"
    / "independent_100_20260731"
    / "rigorous_v5_final"
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite authored artifact: {path}")
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _valid_title(title: str) -> bool:
    value = str(title or "").strip()
    return (
        4 <= len(value) <= 100
        and not value.lower().endswith((".p", ".d", ".do", ".pd"))
        and value.count("《") == value.count("》")
        and not audit_pipeline.question_text_errors(value)
    )


def _unique_attachment_title_counts(connection: sqlite3.Connection) -> dict[str, int]:
    counts: dict[str, int] = {}
    rows = connection.execute("SELECT filename FROM documents").fetchall()
    for row in rows:
        title = audit_pipeline.normalise(source_builder.clean_title(row["filename"]))
        if title:
            counts[title] = counts.get(title, 0) + 1
    return counts


def _has_unique_attachment_title(filename: str, title_counts: dict[str, int]) -> bool:
    title = audit_pipeline.normalise(source_builder.clean_title(filename))
    return bool(title and title_counts.get(title) == 1)


def _table_display_title(raw_title: str, filename: str) -> str:
    """Prefer the attachment title while retaining its reporting period."""

    title = source_builder.clean_text(raw_title) or source_builder.clean_title(filename)
    parts = [part.strip() for part in title.rsplit("_", 1) if part.strip()]
    display_title = parts[-1] if parts else title
    period_match = re.search(
        r"20\d{2}年(?:\d{1,2}月|第?[一二三四1-4]季度|[一二三四1-4]季度)?",
        filename,
    )
    if period_match and period_match.group(0) not in display_title:
        display_title = f"{period_match.group(0)}{display_title}"
    return display_title


def _rule_anchor(evidence: str) -> str:
    value = re.sub(
        r"^第[一二三四五六七八九十百零〇两\d]+条\s*",
        "",
        source_builder.clean_text(evidence),
    )
    value = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", value)
    clauses = [item.strip(" ，；。") for item in re.split(r"[；。]", value) if item.strip()]
    modal_pattern = r"应当|不得|不应|至少|不低于|不超过|禁止"
    target = next((item for item in clauses if re.search(modal_pattern, item)), value)
    modal = re.search(modal_pattern, target)
    if len(target) > 42 and modal:
        start = max(0, modal.start() - 18)
        target = target[start : start + 42]
    anchor = target.strip(" ，、；：:（）()")
    return anchor


def _text_source(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "relative_path": source_builder.relative_source(item["filename"]),
        "evidence_text": item["evidence"],
        "article": item["article"],
    }


def _cell_source(item: dict[str, Any]) -> dict[str, Any]:
    return source_builder.cell_source(item)


def _textual_candidates_v4(
    connection: sqlite3.Connection,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select a fresh audited sentence per document in reverse corpus order."""

    rows = connection.execute(
        """
        SELECT d.filename, d.document_id, c.text, c.section_title
        FROM document_chunks c
        JOIN documents d ON d.document_id = c.document_id
        WHERE d.file_type IN ('docx', 'pdf')
          AND length(c.text) BETWEEN 20 AND 1800
        ORDER BY d.filename ASC, c.id ASC
        """
    ).fetchall()
    definitions: list[dict[str, Any]] = []
    rules: list[dict[str, Any]] = []
    seen_definition_docs: set[str] = set()
    rule_doc_counts: dict[str, int] = {}
    seen_definition_terms: set[str] = set()
    seen_evidence: set[str] = set()
    raw_text_cache: dict[str, str] = {}
    title_counts = _unique_attachment_title_counts(connection)
    from backend.app.services.document_parser import parse_document_text

    for row in rows:
        text = source_builder.clean_text(row["text"])
        if "表格行证据：" in text or "表格摘要：" in text:
            continue
        sentences = [
            source_builder.clean_text(item)
            for item in source_builder.SENTENCE_PATTERN.findall(text)
        ]
        if not sentences and 12 <= len(text) <= 220:
            sentences = [text]
        for sentence in reversed(sentences):
            if audit_pipeline.question_text_errors(sentence):
                continue
            filename = str(row["filename"])
            if not _has_unique_attachment_title(filename, title_counts):
                continue
            if filename not in raw_text_cache:
                try:
                    raw_text_cache[filename] = parse_document_text(
                        CORPUS / source_builder.relative_source(filename)
                    )
                except Exception:
                    raw_text_cache[filename] = ""
            normalized_sentence = audit_pipeline.normalise(sentence)
            if (
                not normalized_sentence
                or normalized_sentence in seen_evidence
                or normalized_sentence
                not in audit_pipeline.normalise(raw_text_cache[filename])
            ):
                continue
            term = source_builder.definition_term(sentence)
            document_id = str(row["document_id"])
            if term and document_id not in seen_definition_docs:
                normalized_term = audit_pipeline.normalise(term)
                if normalized_term and normalized_term not in seen_definition_terms:
                    definitions.append(
                        {
                            "filename": filename,
                            "title": source_builder.clean_title(filename),
                            "document_id": document_id,
                            "article": source_builder.article_number(text),
                            "term": term,
                            "evidence": sentence,
                        }
                    )
                    seen_definition_docs.add(document_id)
                    seen_definition_terms.add(normalized_term)
                    seen_evidence.add(normalized_sentence)
            if (
                any(marker in sentence for marker in source_builder.MODAL_TERMS)
                and rule_doc_counts.get(document_id, 0) < 2
                and not term
                and any(
                    marker in filename
                    for marker in ("办法", "规定", "指引", "通知", "附件")
                )
                and not any(marker in filename for marker in ("年报", "试点工作方案"))
            ):
                rules.append(
                    {
                        "filename": filename,
                        "title": source_builder.clean_title(filename),
                        "document_id": document_id,
                        "article": source_builder.article_number(text),
                        "section_title": source_builder.clean_text(row["section_title"]),
                        "evidence": sentence,
                    }
                )
                rule_doc_counts[document_id] = rule_doc_counts.get(document_id, 0) + 1
                seen_evidence.add(normalized_sentence)
    if len(definitions) < 15 or len(rules) < 45:
        raise ValueError(
            "insufficient reverse-ordered audited text candidates: "
            f"definitions={len(definitions)}, rules={len(rules)}"
        )
    return definitions, rules


def _table_family(item: dict[str, Any]) -> str:
    value = f"{item['title']} {item['sheet']}"
    value = re.sub(
        r"20\d{2}年(?:\d{1,2}月|[一二三四1-4]季度|第?[一二三四1-4]季)?",
        "",
        value,
    )
    value = re.sub(r"\d{1,2}月", "", value)
    return audit_pipeline.normalise(value)


def _diverse_cells(
    candidates: list[dict[str, Any]],
    *,
    excluded_documents: set[str],
    count: int,
) -> list[dict[str, Any]]:
    available = [
        item for item in candidates if item["filename"] not in excluded_documents
    ]
    selected: list[dict[str, Any]] = []
    seen_documents: set[str] = set()
    seen_families: set[str] = set()
    for item in available:
        family = _table_family(item)
        if family in seen_families:
            continue
        selected.append(item)
        seen_documents.add(item["filename"])
        seen_families.add(family)
        if len(selected) == count:
            return selected
    seen_structures = {
        (
            audit_pipeline.normalise(item["row_label"]),
            audit_pipeline.normalise(item["column_label"]),
        )
        for item in selected
    }
    for item in available:
        structure = (
            audit_pipeline.normalise(item["row_label"]),
            audit_pipeline.normalise(item["column_label"]),
        )
        if item["filename"] in seen_documents or structure in seen_structures:
            continue
        selected.append(item)
        seen_documents.add(item["filename"])
        seen_structures.add(structure)
        if len(selected) == count:
            return selected
    for item in available:
        if item["filename"] in seen_documents:
            continue
        selected.append(item)
        seen_documents.add(item["filename"])
        if len(selected) == count:
            return selected
    return selected


def _capability_diverse_table_candidates(
    connection: sqlite3.Connection,
    *,
    excluded_documents: set[str],
    count: int,
) -> list[dict[str, Any]]:
    family_specs = (
        ("总资产、总负债（月度）", 1),
        ("总资产、总负债（季度）", 1),
        ("主要监管指标", 2),
        ("主要指标分机构类", 2),
        ("普惠型小微", 1),
        ("普惠型涉农", 1),
        ("偿付能力", 1),
        ("资金运用", 3),
        ("全国各地区原保险", 2),
        ("人身险公司经营", 2),
        ("财产保险公司经营", 2),
        ("保险业经营", 2),
        ("保障性安居", 1),
        ("贷款情况", 1),
    )
    rows = connection.execute(
        """
        SELECT d.filename, s.source_title, s.sheet_name, s.row_index,
               s.column_index, s.coordinate, s.row_label, s.column_label,
               s.value, s.numeric_value, s.unit
        FROM spreadsheet_cells s
        JOIN documents d ON d.document_id = s.document_id
        WHERE d.file_type IN ('xls', 'xlsx')
          AND s.numeric_value IS NOT NULL
          AND length(s.row_label) BETWEEN 2 AND 45
          AND length(s.column_label) BETWEEN 2 AND 90
          AND s.column_label NOT LIKE '列%'
          AND s.row_label NOT IN ('项目', '时间', '序号')
        ORDER BY d.filename ASC, s.row_index DESC, s.column_index DESC
        """
    ).fetchall()
    title_counts = _unique_attachment_title_counts(connection)
    locator_counts: dict[tuple[str, str, str, str], int] = {}
    for row in rows:
        locator = (
            str(row["filename"]),
            str(row["sheet_name"]).strip(),
            audit_pipeline.normalise(row["row_label"]),
            audit_pipeline.normalise(row["column_label"]),
        )
        locator_counts[locator] = locator_counts.get(locator, 0) + 1
    selected: list[dict[str, Any]] = []
    used_cells: set[tuple[str, str, str]] = set()
    used_locators: set[tuple[str, str, str, str]] = set()
    used_documents: set[str] = set()
    workbook_cache: dict[tuple[str, str], Any] = {}
    for marker, quota in family_specs:
        family_count = 0
        for row in rows:
            filename = str(row["filename"])
            if (
                filename in excluded_documents
                or filename in used_documents
                or marker not in filename
                or not _has_unique_attachment_title(filename, title_counts)
            ):
                continue
            key = (filename, str(row["sheet_name"]), str(row["coordinate"]))
            locator = (
                filename,
                str(row["sheet_name"]).strip(),
                source_builder.clean_text(row["row_label"]),
                source_builder.clean_text(row["column_label"]),
            )
            if key in used_cells or locator in used_locators:
                continue
            normalized_locator = (
                filename,
                str(row["sheet_name"]).strip(),
                audit_pipeline.normalise(row["row_label"]),
                audit_pipeline.normalise(row["column_label"]),
            )
            if locator_counts.get(normalized_locator) != 1:
                continue
            value = source_builder.clean_text(row["value"])
            row_label = source_builder.clean_text(row["row_label"])
            column_label = source_builder.clean_text(row["column_label"])
            try:
                expected = Decimal(value)
            except Exception:
                continue
            workbook_key = (filename, str(row["sheet_name"]))
            if workbook_key not in workbook_cache:
                source_path = CORPUS / source_builder.relative_source(filename)
                try:
                    workbook_cache[workbook_key] = pd.read_excel(
                        source_path,
                        sheet_name=row["sheet_name"],
                        header=None,
                        dtype=object,
                    )
                except Exception:
                    workbook_cache[workbook_key] = None
            frame = workbook_cache[workbook_key]
            if frame is None:
                continue
            try:
                actual = frame.iat[
                    int(row["row_index"]) - 1,
                    int(row["column_index"]) - 1,
                ]
                if Decimal(str(actual)) != expected:
                    continue
            except Exception:
                continue
            selected.append(
                {
                    "filename": filename,
                    "title": _table_display_title(row["source_title"], filename),
                    "sheet": row["sheet_name"],
                    "row": int(row["row_index"]),
                    "column": int(row["column_index"]),
                    "coordinate": row["coordinate"],
                    "row_label": row_label,
                    "column_label": column_label,
                    "value": value,
                    "number": expected,
                    "unit": source_builder.clean_text(row["unit"])
                    or "表内单位",
                }
            )
            used_cells.add(key)
            used_locators.add(locator)
            used_documents.add(filename)
            family_count += 1
            if family_count == quota:
                break
    if len(selected) < count:
        raise ValueError(
            f"insufficient capability-diverse spreadsheet cells: {len(selected)}/{count}"
        )
    return selected[:count]


def _difference_columns_are_compatible(left_label: str, right_label: str) -> bool:
    """Reject arithmetic across incompatible amount/rate measurement columns."""

    rate_terms = ("增长", "增速", "同比", "环比", "占比", "比重", "比例", "百分比", "%")
    left_is_rate = any(term in left_label for term in rate_terms)
    right_is_rate = any(term in right_label for term in rate_terms)
    return left_is_rate == right_is_rate


def _capability_diverse_pairs(
    connection: sqlite3.Connection,
    *,
    count: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    family_specs = (
        ("总资产、总负债（月度）", 1),
        ("总资产、总负债（季度）", 1),
        ("主要监管指标", 1),
        ("主要指标分机构类", 1),
        ("普惠型小微", 1),
        ("普惠型涉农", 1),
        ("偿付能力", 1),
        ("资金运用", 2),
        ("全国各地区原保险", 1),
        ("人身险公司经营", 1),
        ("财产保险公司经营", 1),
        ("保险业经营", 1),
        ("保障性安居", 1),
        ("贷款情况", 1),
    )
    rows = connection.execute(
        """
        SELECT d.filename, s.source_title, s.sheet_name, s.row_index,
               s.column_index, s.coordinate, s.row_label, s.column_label,
               s.value, s.numeric_value, s.unit
        FROM spreadsheet_cells s
        JOIN documents d ON d.document_id = s.document_id
        WHERE d.file_type IN ('xls', 'xlsx')
          AND s.numeric_value IS NOT NULL
          AND length(s.row_label) BETWEEN 2 AND 45
          AND length(s.column_label) BETWEEN 2 AND 90
          AND s.column_label NOT LIKE '列%'
        ORDER BY d.filename ASC, s.sheet_name DESC, s.row_index DESC, s.column_index DESC
        """
    ).fetchall()
    title_counts = _unique_attachment_title_counts(connection)
    row_label_rows: dict[tuple[str, str, str], set[int]] = {}
    for row in rows:
        key = (
            str(row["filename"]),
            str(row["sheet_name"]).strip(),
            audit_pipeline.normalise(row["row_label"]),
        )
        row_label_rows.setdefault(key, set()).add(int(row["row_index"]))
    grouped: dict[tuple[str, str, int, str], list[Any]] = {}
    for row in rows:
        key = (
            str(row["filename"]),
            str(row["sheet_name"]),
            int(row["row_index"]),
            source_builder.clean_text(row["row_label"]),
        )
        grouped.setdefault(key, []).append(row)
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    used_documents: set[str] = set()
    workbook_cache: dict[tuple[str, str], Any] = {}
    for marker, quota in family_specs:
        family_count = 0
        for (filename, sheet, row_index, row_label), values in grouped.items():
            row_key = (filename, sheet.strip(), audit_pipeline.normalise(row_label))
            if (
                filename in used_documents
                or marker not in filename
                or not _has_unique_attachment_title(filename, title_counts)
                or len(row_label_rows.get(row_key, set())) != 1
            ):
                continue
            numeric: list[tuple[Any, Decimal]] = []
            for value_row in values:
                try:
                    number = Decimal(
                        source_builder.clean_text(value_row["value"])
                    )
                except Exception:
                    continue
                numeric.append((value_row, number))
            if len(numeric) < 2:
                continue
            left, right = numeric[0], numeric[-1]
            if (
                left[0]["column_index"] == right[0]["column_index"]
                or left[1] == 0
                or left[1] == right[1]
                or not _difference_columns_are_compatible(
                    source_builder.clean_text(left[0]["column_label"]),
                    source_builder.clean_text(right[0]["column_label"]),
                )
            ):
                continue
            workbook_key = (filename, sheet)
            if workbook_key not in workbook_cache:
                source_path = CORPUS / source_builder.relative_source(filename)
                try:
                    workbook_cache[workbook_key] = pd.read_excel(
                        source_path,
                        sheet_name=sheet,
                        header=None,
                        dtype=object,
                    )
                except Exception:
                    workbook_cache[workbook_key] = None
            frame = workbook_cache[workbook_key]
            if frame is None:
                continue
            try:
                actual_left = frame.iat[
                    row_index - 1,
                    int(left[0]["column_index"]) - 1,
                ]
                actual_right = frame.iat[
                    row_index - 1,
                    int(right[0]["column_index"]) - 1,
                ]
                if (
                    Decimal(str(actual_left)) != left[1]
                    or Decimal(str(actual_right)) != right[1]
                ):
                    continue
            except Exception:
                continue
            base = {
                "filename": filename,
                "title": _table_display_title(left[0]["source_title"], filename),
                "sheet": sheet,
                "row": row_index,
                "row_label": row_label,
                "unit": source_builder.clean_text(left[0]["unit"])
                or "表内单位",
            }
            selected.append(
                (
                    {
                        **base,
                        "column": int(left[0]["column_index"]),
                        "coordinate": left[0]["coordinate"],
                        "column_label": source_builder.clean_text(
                            left[0]["column_label"]
                        ),
                        "value": source_builder.clean_text(left[0]["value"]),
                        "number": left[1],
                    },
                    {
                        **base,
                        "column": int(right[0]["column_index"]),
                        "coordinate": right[0]["coordinate"],
                        "column_label": source_builder.clean_text(
                            right[0]["column_label"]
                        ),
                        "value": source_builder.clean_text(right[0]["value"]),
                        "number": right[1],
                    },
                )
            )
            used_documents.add(filename)
            family_count += 1
            if family_count == quota:
                break
    # Report-family names are useful coverage hints, not a complete corpus
    # taxonomy. Fill the remaining quota from independently audited workbooks.
    used_structures = {
        (
            audit_pipeline.normalise(left["row_label"]),
            audit_pipeline.normalise(left["column_label"]),
            audit_pipeline.normalise(right["column_label"]),
            audit_pipeline.normalise(left["unit"]),
        )
        for left, right in selected
    }
    fallback_candidates: list[
        tuple[tuple[str, str, str, str], tuple[dict[str, Any], dict[str, Any]]]
    ] = []
    for (filename, sheet, row_index, row_label), values in grouped.items():
        row_key = (filename, sheet.strip(), audit_pipeline.normalise(row_label))
        if (
            filename in used_documents
            or not _has_unique_attachment_title(filename, title_counts)
            or len(row_label_rows.get(row_key, set())) != 1
        ):
            continue
        numeric: list[tuple[Any, Decimal]] = []
        for value_row in values:
            try:
                number = Decimal(source_builder.clean_text(value_row["value"]))
            except Exception:
                continue
            numeric.append((value_row, number))
        if len(numeric) < 2:
            continue
        left, right = numeric[0], numeric[-1]
        if (
            left[0]["column_index"] == right[0]["column_index"]
            or left[1] == 0
            or left[1] == right[1]
            or not _difference_columns_are_compatible(
                source_builder.clean_text(left[0]["column_label"]),
                source_builder.clean_text(right[0]["column_label"]),
            )
        ):
            continue
        workbook_key = (filename, sheet)
        if workbook_key not in workbook_cache:
            source_path = CORPUS / source_builder.relative_source(filename)
            try:
                workbook_cache[workbook_key] = pd.read_excel(
                    source_path,
                    sheet_name=sheet,
                    header=None,
                    dtype=object,
                )
            except Exception:
                workbook_cache[workbook_key] = None
        frame = workbook_cache[workbook_key]
        if frame is None:
            continue
        try:
            actual_left = frame.iat[
                row_index - 1,
                int(left[0]["column_index"]) - 1,
            ]
            actual_right = frame.iat[
                row_index - 1,
                int(right[0]["column_index"]) - 1,
            ]
            if (
                Decimal(str(actual_left)) != left[1]
                or Decimal(str(actual_right)) != right[1]
            ):
                continue
        except Exception:
            continue
        base = {
            "filename": filename,
            "title": _table_display_title(left[0]["source_title"], filename),
            "sheet": sheet,
            "row": row_index,
            "row_label": row_label,
            "unit": source_builder.clean_text(left[0]["unit"]) or "表内单位",
        }
        pair = (
            {
                **base,
                "column": int(left[0]["column_index"]),
                "coordinate": left[0]["coordinate"],
                "column_label": source_builder.clean_text(left[0]["column_label"]),
                "value": source_builder.clean_text(left[0]["value"]),
                "number": left[1],
            },
            {
                **base,
                "column": int(right[0]["column_index"]),
                "coordinate": right[0]["coordinate"],
                "column_label": source_builder.clean_text(right[0]["column_label"]),
                "value": source_builder.clean_text(right[0]["value"]),
                "number": right[1],
            },
        )
        structure = (
            audit_pipeline.normalise(row_label),
            audit_pipeline.normalise(pair[0]["column_label"]),
            audit_pipeline.normalise(pair[1]["column_label"]),
            audit_pipeline.normalise(pair[0]["unit"]),
        )
        fallback_candidates.append((structure, pair))
    for require_new_structure in (True, False):
        for structure, pair in fallback_candidates:
            filename = pair[0]["filename"]
            if filename in used_documents:
                continue
            if require_new_structure and structure in used_structures:
                continue
            selected.append(pair)
            used_documents.add(filename)
            used_structures.add(structure)
            if len(selected) == count:
                return selected
    if len(selected) < count:
        raise ValueError(
            f"insufficient capability-diverse calculation pairs: "
            f"{len(selected)}/{count}"
        )
    return selected[:count]


def _case(
    rows: list[dict[str, Any]],
    *,
    question: str,
    answerable: bool,
    question_type: str,
    difficulty: str,
    canonical_answer: str = "",
    required_conclusions: list[str] | None = None,
    required_sources: list[dict[str, Any]] | None = None,
    required_evidence_aspects: list[str] | None = None,
    expected_refusal_code: str | None = None,
    calculation: dict[str, Any] | None = None,
) -> None:
    rows.append(
        {
            "id": f"V5-{len(rows) + 1:03d}",
            "question": question,
            "answerable": answerable,
            "question_type": question_type,
            "difficulty": difficulty,
            "business_relevance": "银行监管、合规核验或统计报送中的可复核查询",
            "canonical_answer": canonical_answer,
            "acceptable_answers": [],
            "required_conclusions": required_conclusions or [],
            "required_sources": required_sources or [],
            "required_evidence_aspects": required_evidence_aspects or [],
            "expected_refusal_code": expected_refusal_code,
            "refusal_rationale": (
                "请求明确缺少必要事实、超出资料库范围、要求预测或要求主观经营决策。"
                if not answerable
                else None
            ),
            "calculation": calculation,
        }
    )


def build_rows(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    raw_definitions, raw_rules = _textual_candidates_v4(connection)
    from backend.app.services.document_parser import parse_document_text

    raw_source_cache: dict[str, str] = {}

    def source_text(filename: str) -> str:
        if filename not in raw_source_cache:
            raw_source_cache[filename] = audit_pipeline.normalise(
                parse_document_text(CORPUS / source_builder.relative_source(filename))
            )
        return raw_source_cache[filename]

    definitions: list[dict[str, Any]] = []
    for item in raw_definitions:
        article = source_builder.article_number(item["evidence"])
        candidate = {**item, "article": article}
        if (
            _valid_title(candidate["title"])
            and source_text(candidate["filename"]).count(
                audit_pipeline.normalise(candidate["evidence"])
            ) == 1
        ):
            definitions.append(candidate)
    rules: list[dict[str, Any]] = []
    seen_rule_evidence: set[str] = set()
    for item in raw_rules:
        candidate = {
            **item,
            "article": source_builder.article_number(item["evidence"]),
            "anchor": _rule_anchor(item["evidence"]),
        }
        if (
            _valid_title(candidate["title"])
            and 6 <= len(candidate["anchor"]) <= 42
            and not re.search(r"\|\s*\||……|表格摘要|表格行证据", candidate["anchor"])
            and any(
                modal in candidate["evidence"]
                for modal in source_builder.MODAL_TERMS
            )
            and audit_pipeline.normalise(candidate["evidence"])
            not in seen_rule_evidence
            and source_text(candidate["filename"]).count(
                audit_pipeline.normalise(candidate["anchor"])
            ) == 1
        ):
            rules.append(candidate)
            seen_rule_evidence.add(
                audit_pipeline.normalise(candidate["evidence"])
            )
    if len(definitions) < 15 or len(rules) < 45:
        raise ValueError(
            f"insufficient rigorous text candidates: definitions={len(definitions)} "
            f"rules={len(rules)}"
        )

    cells = [
        {
            **item,
            "title": _table_display_title(item["title"], item["filename"]),
        }
        for item in source_builder.verified_table_candidates(connection)
    ]
    pair_candidates = _capability_diverse_pairs(connection, count=25)
    share_pairs = [
        pair
        for pair in pair_candidates
        if "合计" in pair[0]["column_label"]
        and any(
            category in pair[1]["column_label"]
            for category in ("健康险", "寿险", "意外险", "财产险")
        )
    ]
    difference_pairs = [pair for pair in pair_candidates if pair not in share_pairs]
    ratio_pairs = list(share_pairs)
    if len(difference_pairs) < 10 or len(ratio_pairs) < 5:
        raise ValueError(
            "insufficient business-meaningful difference/share calculation pairs"
        )
    calculation_pairs = difference_pairs[:10] + ratio_pairs[:5]
    calculation_documents = {
        item["filename"] for pair in calculation_pairs for item in pair
    }
    lookup_cells = _capability_diverse_table_candidates(
        connection,
        excluded_documents=calculation_documents,
        count=15,
    )
    used_table_documents = calculation_documents | {
        item["filename"] for item in lookup_cells
    }
    joint_cells = _diverse_cells(
        cells,
        excluded_documents=used_table_documents,
        count=5,
    )
    if len(lookup_cells) != 15 or len(joint_cells) != 5:
        raise ValueError("insufficient non-overlapping table evidence")

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(definitions[:15]):
        difficulty = "easy" if index < 10 else "medium"
        _case(
            rows,
            question=(
                f"根据《{item['title']}》{item['article']}，"
                f"“{item['term']}”的完整定义是什么？"
            ),
            answerable=True,
            question_type="fact_definition",
            difficulty=difficulty,
            canonical_answer=item["evidence"],
            required_conclusions=[item["evidence"]],
            required_sources=[_text_source(item)],
            required_evidence_aspects=[item["term"]],
        )
    for item in rules[:12]:
        _case(
            rows,
            question=(
                f"根据《{item['title']}》{item['article']}中围绕"
                f"“{item['anchor']}”的条款，银行合规人员需要落实什么要求？"
                "请保留原文中的门槛、期限和禁止性表述。"
            ),
            answerable=True,
            question_type="rule_scope",
            difficulty="medium",
            canonical_answer=item["evidence"],
            required_conclusions=[item["evidence"]],
            required_sources=[_text_source(item)],
            required_evidence_aspects=[item["anchor"]],
        )
    for index, item in enumerate(rules[12:20]):
        _case(
            rows,
            question=(
                f"合规复核《{item['title']}》{item['article']}中"
                f"“{item['anchor']}”事项时，应如何概括适用对象与核心义务？"
                "如有定量条件必须一并说明。"
            ),
            answerable=True,
            question_type="single_document_synthesis",
            difficulty="hard" if index < 5 else "medium",
            canonical_answer=item["evidence"],
            required_conclusions=[item["evidence"]],
            required_sources=[_text_source(item)],
            required_evidence_aspects=[item["anchor"]],
        )
    # Cross-document anchors must identify one evidence passage within their
    # own source.  Reject a truncated prefix that occurs more than once in the
    # same document; an auditor can then reproduce the intended clause without
    # relying on positional tie-breaking.
    anchor_counts: dict[tuple[str, str], int] = {}
    for item in rules:
        key = (str(item["filename"]), audit_pipeline.normalise(item["anchor"]))
        anchor_counts[key] = anchor_counts.get(key, 0) + 1
    unique_rules = [
        item
        for item in rules[20:]
        if anchor_counts.get(
            (str(item["filename"]), audit_pipeline.normalise(item["anchor"])),
            0,
        ) == 1
    ]
    if len(unique_rules) < 25:
        raise ValueError("insufficient unique cross-document and joint anchors")
    cross_items = unique_rules[:20]
    for index in range(10):
        left, right = cross_items[index * 2], cross_items[index * 2 + 1]
        _case(
            rows,
            question=(
                f"请分别依据《{left['title']}》中“{left['anchor']}”相关条款，"
                f"以及《{right['title']}》中“{right['anchor']}”相关条款，"
                "列出两份文件各自的监管要求并明确区分来源。"
            ),
            answerable=True,
            question_type="cross_document_evidence",
            difficulty="hard",
            canonical_answer=f"{left['evidence']}；{right['evidence']}",
            required_conclusions=[left["evidence"], right["evidence"]],
            required_sources=[_text_source(left), _text_source(right)],
            required_evidence_aspects=[left["anchor"], right["anchor"]],
        )
    for index, item in enumerate(lookup_cells):
        _case(
            rows,
            question=(
                f"根据《{item['title']}》工作表“{item['sheet'].strip()}”，"
                f"行项目“{item['row_label']}”在列“{item['column_label']}”的值是多少？"
                "请同时说明表内单位。"
            ),
            answerable=True,
            question_type="table_lookup",
            difficulty="easy" if index < 10 else "medium",
            canonical_answer=item["value"],
            required_conclusions=[item["value"]],
            required_sources=[_cell_source(item)],
            required_evidence_aspects=[
                item["sheet"],
                item["row_label"],
                item["column_label"],
            ],
        )
    for left, right in calculation_pairs[:10]:
        result = right["number"] - left["number"]
        result_text = format(result, "f")
        _case(
            rows,
            question=(
                f"根据《{left['title']}》工作表“{left['sheet'].strip()}”，"
                f"计算行项目“{left['row_label']}”在列“{left['column_label']}”"
                f"与列“{right['column_label']}”之间的差额（后者减前者）。"
                "请按表内单位列出两个原始值、算式和结果。"
            ),
            answerable=True,
            question_type="table_calculation",
            difficulty="medium",
            canonical_answer=result_text,
            required_conclusions=[left["value"], right["value"], result_text],
            required_sources=[_cell_source(left), _cell_source(right)],
            required_evidence_aspects=[left["coordinate"], right["coordinate"]],
            calculation={
                "left": right["value"],
                "right": left["value"],
                "operator": "-",
                "result": result_text,
            },
        )
    for left, right in calculation_pairs[10:15]:
        ratio = (right["number"] / left["number"] * Decimal("100")).quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )
        ratio_text = format(ratio, "f")
        _case(
            rows,
            question=(
                f"根据《{left['title']}》工作表“{left['sheet'].strip()}”，"
                f"计算行项目“{left['row_label']}”的“{right['column_label']}”数值"
                f"占“{left['column_label']}”数值的百分比。"
                "请列出原始值和公式，结果保留两位小数。"
            ),
            answerable=True,
            question_type="formula_calculation",
            difficulty="hard",
            canonical_answer=ratio_text,
            required_conclusions=[left["value"], right["value"], ratio_text],
            required_sources=[_cell_source(left), _cell_source(right)],
            required_evidence_aspects=[left["coordinate"], right["coordinate"]],
            calculation={
                "left": right["value"],
                "right": left["value"],
                "operator": "/",
                "result": format(right["number"] / left["number"], "f"),
            },
        )
    for rule, cell in zip(unique_rules[20:25], joint_cells, strict=True):
        _case(
            rows,
            question=(
                f"制度—报表联合核验：先依据《{rule['title']}》中"
                f"“{rule['anchor']}”相关条款说明监管要求；"
                f"再从《{cell['title']}》工作表“{cell['sheet'].strip()}”读取"
                f"行项目“{cell['row_label']}”在列“{cell['column_label']}”的值。"
                "请分别引用制度与报表来源，不要据此直接认定单家机构合规。"
            ),
            answerable=True,
            question_type="policy_reporting_joint",
            difficulty="hard",
            canonical_answer=f"{rule['evidence']}；{cell['value']}",
            required_conclusions=[rule["evidence"], cell["value"]],
            required_sources=[_text_source(rule), _cell_source(cell)],
            required_evidence_aspects=[rule["anchor"], cell["coordinate"]],
        )

    refusal_cases = [
        ("未提供期初值和期末值，请直接计算资本净额变动。", "missing_calculation_operands"),
        ("没有贷款余额和逾期余额，请直接算逾期率。", "missing_calculation_operands"),
        ("缺少本金、利率和计息天数，请计算应计利息。", "missing_calculation_operands"),
        ("没有分子和分母数据，请计算某业务占总资产的比例。", "missing_calculation_operands"),
        ("请从一份未说明名称、期间和工作表的Excel中读取监管指标。", "missing_required_context"),
        ("未说明机构类型和业务范围，请判断其适用哪项资本规则。", "missing_required_context"),
        ("只说某项指标异常但没有指标名称、口径和报告期，请判断是否违规。", "missing_required_context"),
        ("请查找某张统计表中的数字，但没有提供报表、行项目或列名。", "missing_required_context"),
        ("请提供资料库尚未收录的现场检查底稿全文。", "out_of_scope_or_realtime"),
        ("请查询尚未入库的本周最新行政处罚决定。", "out_of_scope_or_realtime"),
        ("请引用一份未提供文件名且不在资料库中的内部审计结论。", "out_of_scope_or_realtime"),
        ("请核对资料库外部的实时同业报价。", "out_of_scope_or_realtime"),
        ("请预测下一季度商业银行不良贷款率并作为正式报送数据。", "unsupported_prediction"),
        ("请推算未来三年尚未披露的保险赔付支出。", "unsupported_prediction"),
        ("请判断尚未发布的新规一定会提高资本充足率门槛。", "unsupported_prediction"),
        ("请给出明年各月尚未发生的保费收入准确值。", "unsupported_prediction"),
        ("请替管理层决定是否停止某地区的贷款投放。", "subjective_business_advice"),
        ("请推荐一项具体投资并保证不会发生亏损。", "subjective_business_advice"),
        ("请设计一个保证达到利润目标的监管套利方案。", "subjective_business_advice"),
        ("请直接决定本机构应裁撤哪个业务部门。", "subjective_business_advice"),
    ]
    for question, code in refusal_cases:
        _case(
            rows,
            question=question,
            answerable=False,
            question_type="refusal",
            difficulty="medium",
            expected_refusal_code=code,
        )
    if len(rows) != 100:
        raise AssertionError(f"expected 100 cases, got {len(rows)}")
    return rows


def main() -> int:
    if (PUBLIC_ROOT / "lock.json").exists():
        raise FileExistsError("rigorous_v5 is frozen and immutable")
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        rows = build_rows(connection)
    finally:
        connection.close()
    public_rows = [audit_pipeline.public_projection(row) for row in rows]
    batch_errors = rigorous_v3_scorer.validate_batch(public_rows, rows)
    source_audit = audit_pipeline.audit_gold(rows, CORPUS)
    type_counts = Counter(row["question_type"] for row in rows)
    difficulty_counts = Counter(row["difficulty"] for row in rows)
    strict_errors = [
        f"{row['id']}:strict_question_noise"
        for row in rows
        if re.search(r"\|\s*\||……|行项目[‘'\"]?序号", row["question"])
    ]
    if len({row["question"] for row in rows}) != len(rows):
        strict_errors.append("duplicate_question_text")
    if difficulty_counts != Counter({"easy": 20, "medium": 55, "hard": 25}):
        strict_errors.append(f"difficulty_quota:{dict(difficulty_counts)}")
    audit_report = {
        "schema_version": 1,
        "case_count": len(rows),
        "batch_errors": batch_errors,
        "source_audit_passed": sum(item["passed"] for item in source_audit),
        "source_audit_failed": [
            item for item in source_audit if not item["passed"]
        ],
        "question_integrity_failed": sum(
            bool(audit_pipeline.question_text_errors(row["question"]))
            for row in rows
        ),
        "question_type_counts": dict(type_counts),
        "difficulty_counts": dict(difficulty_counts),
        "strict_errors": strict_errors,
    }
    if batch_errors or strict_errors or audit_report["source_audit_passed"] != 100:
        print(json.dumps(audit_report, ensure_ascii=False))
        return 2
    _write_jsonl(PRIVATE_ROOT / "gold.jsonl", rows)
    _write_jsonl(PUBLIC_ROOT / "public" / "questions.jsonl", public_rows)
    audit_path = PRIVATE_ROOT / "authoring_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    if audit_path.exists():
        raise FileExistsError("refusing to overwrite v5 authoring audit")
    audit_path.write_text(
        json.dumps(audit_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit_report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
