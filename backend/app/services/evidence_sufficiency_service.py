"""Deterministic evidence-set audit.

The audit answers one question that relevance scores cannot: *does the
selected evidence cover the information facets the question asked for?*  It
is intentionally rule-based and callable without a model, so the result can
be replayed and audited.  It never changes retrieval or turns a proxy
coverage signal into ``facts_supported``.

Design note (BFR-inspired, not a BFR reproduction): the paper separates
"relevant records" from "a sufficient evidence set".  ReguMate already has
requirement objects and an ``EvidenceBundle``; this module adds the missing
per-requirement facet report and the expansion/cost trace.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

SCHEMA_VERSION = "1.0"

_NUMERIC_QUESTION = re.compile(
    r"\d|百分之|比例|比率|阈值|限额|期限|天数|金额|余额|增速|增幅|资本|权重"
)
_NUMERIC_VALUE = re.compile(r"\d+(?:\.\d+)?\s*[%％]?")
_TABLE_FILTER_KEYS = {
    "row_label", "column_label", "sheet_name", "cell", "coordinate",
    "table_id", "year", "month", "quarter", "period",
}
# A missing source, threshold, definition or table operand changes whether a
# conclusion can be supported at all; a missing soft facet only makes the set
# partial.  Soft facets remain visible instead of forcing a refusal.
_CRITICAL_FACETS = {"source", "numeric", "definition", "table"}
_CONDITION_QUESTION = re.compile(r"条件|情形|例外|除外|不适用|不得|应当|可以|前提|边界")
_CONDITION_MARKER = re.compile(r"条件|情形|例外|除外|不适用|不得|应当|可以|前提|但书|但是|除")
_LIST_QUESTION = re.compile(r"哪些|包括|分别|列出|情形|项目|清单")
_LIST_MARKER = re.compile(r"[（(]\s*[一二三四五六七八九十\d]+\s*[）)]|\n\s*\d+[.、]|[；;]")
_DEFINITION_MARKER = re.compile(r"是指|系指|定义为|所称|指的是")
_TABLE_MARKERS = ("sheet_name", "cell", "coordinate", "table_id")


def _text(value: Any) -> str:
    return str(value or "")


def _normalize(value: Any) -> str:
    text = _text(value).lower()
    text = text.replace("《", "").replace("》", "")
    text = re.sub(r"[\s·•、，,。.;；:：()（）\"'“”‘’]+", "", text)
    return text


def _metadata(chunk: Any) -> dict[str, Any]:
    metadata = getattr(chunk, "metadata", None)
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _chunk_id(chunk: Any) -> str:
    return _text(getattr(chunk, "chunk_id", "") or _metadata(chunk).get("chunk_id"))


def _chunk_text(chunk: Any) -> str:
    parts = [
        _text(getattr(chunk, "section_title", "")),
        _text(getattr(chunk, "text", "")),
        _text(getattr(chunk, "excerpt", "")),
    ]
    return "\n".join(part for part in parts if part)


def _source_text(chunk: Any) -> str:
    metadata = _metadata(chunk)
    return " ".join(
        part for part in (
            _text(getattr(chunk, "source_doc", "")),
            _text(getattr(chunk, "filename", "")),
            _text(getattr(chunk, "source_title", "")),
            _text(metadata.get("source_title")),
            _text(metadata.get("filename")),
            _text(metadata.get("title")),
        ) if part
    )


def _is_table_chunk(chunk: Any) -> bool:
    metadata = _metadata(chunk)
    if any(metadata.get(key) not in (None, "") for key in _TABLE_MARKERS):
        return True
    return _text(getattr(chunk, "chunk_type", "")) in {"table", "table_cell", "table_calculation"}


def _chunks_for_requirement(
    requirement_id: str,
    coverage: Mapping[str, Iterable[str]],
    by_id: Mapping[str, Any],
) -> list[Any]:
    ids = coverage.get(requirement_id) or []
    return [by_id[chunk_id] for chunk_id in ids if chunk_id in by_id]


def _facet(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"facet": name, "ok": bool(ok), "detail": detail}


def _source_facet(requirement: Any, chunks: list[Any]) -> dict[str, Any] | None:
    titles = [str(title) for title in (getattr(requirement, "source_titles", None) or []) if str(title)]
    if not titles:
        return None
    available = " ".join(_source_text(chunk) for chunk in chunks)
    normalized = _normalize(available)
    missing = [title for title in titles if _normalize(title) not in normalized]
    return _facet(
        "source",
        not missing,
        "来源未命中：" + "、".join(missing) if missing else "来源已绑定",
    )


def _target_facet(requirement: Any, chunks: list[Any]) -> dict[str, Any] | None:
    terms = [str(term) for term in (getattr(requirement, "target_terms", None) or []) if str(term)]
    if not terms:
        return None
    text = _normalize("\n".join(_chunk_text(chunk) for chunk in chunks))
    missing = [term for term in terms if _normalize(term) not in text]
    return _facet(
        "target",
        not missing,
        "目标词未出现：" + "、".join(missing) if missing else "目标词已出现",
    )


def _numbers(text: str) -> set[str]:
    values: set[str] = set()
    for match in _NUMERIC_VALUE.finditer(text):
        token = match.group(0).replace("%", "").replace("％", "").strip()
        try:
            values.add(str(float(token)))
        except ValueError:
            continue
    return values


def _numeric_facet(requirement: Any, chunks: list[Any]) -> dict[str, Any] | None:
    question = _text(getattr(requirement, "question", ""))
    if not _NUMERIC_QUESTION.search(question):
        return None
    expected = _numbers(question)
    text = "\n".join(_chunk_text(chunk) for chunk in chunks)
    actual = _numbers(text)
    if not expected:
        return _facet("numeric", bool(actual), "未出现可核对数值" if not actual else "已出现可核对数值")
    missing = sorted(expected - actual)
    return _facet(
        "numeric",
        not missing,
        "问题数值未在证据中出现：" + "、".join(missing) if missing else "问题数值均可核对",
    )


def _condition_facet(requirement: Any, chunks: list[Any]) -> dict[str, Any] | None:
    question = _text(getattr(requirement, "question", ""))
    if not _CONDITION_QUESTION.search(question):
        return None
    text = "\n".join(_chunk_text(chunk) for chunk in chunks)
    return _facet(
        "condition",
        bool(_CONDITION_MARKER.search(text)),
        "未出现适用条件/例外表述" if not _CONDITION_MARKER.search(text) else "已出现适用条件/例外表述",
    )


def _enumeration_facet(requirement: Any, chunks: list[Any]) -> dict[str, Any] | None:
    question = _text(getattr(requirement, "question", ""))
    intent = _text(getattr(requirement, "intent", ""))
    if intent != "list" and not _LIST_QUESTION.search(question):
        return None
    text = "\n".join(_chunk_text(chunk) for chunk in chunks)
    markers = len(_LIST_MARKER.findall(text))
    return _facet(
        "enumeration",
        markers >= 2,
        f"列举标记 {markers} 个，少于 2 个" if markers < 2 else f"列举标记 {markers} 个",
    )


def _definition_facet(requirement: Any, chunks: list[Any]) -> dict[str, Any] | None:
    intent = _text(getattr(requirement, "intent", ""))
    question = _text(getattr(requirement, "question", ""))
    if intent != "define" and not re.search(r"定义|是什么|含义|界定|所称", question):
        return None
    text = "\n".join(_chunk_text(chunk) for chunk in chunks)
    return _facet(
        "definition",
        bool(_DEFINITION_MARKER.search(text)),
        "未出现定义句式" if not _DEFINITION_MARKER.search(text) else "已出现定义句式",
    )


def _table_facet(requirement: Any, chunks: list[Any]) -> dict[str, Any] | None:
    constraints = getattr(requirement, "constraints", None) or {}
    selectors = getattr(requirement, "selectors", None) or []
    table_constraints = {
        key: value for key, value in dict(constraints).items()
        if key in _TABLE_FILTER_KEYS and value not in (None, "", [])
    }
    if not selectors and not table_constraints:
        return None
    ok = any(_is_table_chunk(chunk) for chunk in chunks)
    return _facet("table", ok, "未绑定表格/单元格证据" if not ok else "已绑定表格/单元格证据")


def _requirement_report(requirement: Any, chunks: list[Any]) -> dict[str, Any]:
    requirement_id = _text(getattr(requirement, "requirement_id", ""))
    if not chunks:
        return {
            "requirement_id": requirement_id,
            "status": "unbound",
            "facets": [],
            "evidence_ids": [],
            "reason_codes": ["no_evidence"],
        }
    facets = [
        facet for facet in (
            _source_facet(requirement, chunks),
            _target_facet(requirement, chunks),
            _numeric_facet(requirement, chunks),
            _condition_facet(requirement, chunks),
            _enumeration_facet(requirement, chunks),
            _definition_facet(requirement, chunks),
            _table_facet(requirement, chunks),
        ) if facet is not None
    ]
    if not facets:
        return {
            "requirement_id": requirement_id,
            "status": "covered",
            "facets": [],
            "evidence_ids": [_chunk_id(chunk) for chunk in chunks],
            "reason_codes": [],
        }
    passed = [facet for facet in facets if facet["ok"]]
    critical_missing = [
        facet for facet in facets
        if not facet["ok"] and str(facet["facet"]) in _CRITICAL_FACETS
    ]
    if len(passed) == len(facets):
        status = "covered"
    elif critical_missing or not passed:
        status = "missing"
    else:
        status = "partial"
    reason_codes = [str(facet["facet"]) + "_missing" for facet in facets if not facet["ok"]]
    return {
        "requirement_id": requirement_id,
        "status": status,
        "facets": facets,
        "evidence_ids": [_chunk_id(chunk) for chunk in chunks],
        "reason_codes": reason_codes,
    }


def _expansion_diagnostics(chunks: list[Any]) -> dict[str, Any]:
    reasons: Counter[str] = Counter()
    added_tokens = 0
    expanded_ids: list[str] = []
    for chunk in chunks:
        metadata = _metadata(chunk)
        reason = _text(metadata.get("expansion_reason"))
        parent = _text(metadata.get("expanded_from_chunk_id"))
        if not reason and not parent:
            continue
        expanded_ids.append(_chunk_id(chunk))
        reasons[reason or "expanded"] += 1
        token_count = metadata.get("token_count")
        try:
            added_tokens += int(token_count)
        except (TypeError, ValueError):
            added_tokens += max(len(_chunk_text(chunk)) // 2, 1)
    return {
        "attempted": bool(expanded_ids),
        "expanded_chunks": len(expanded_ids),
        "expanded_chunk_ids": expanded_ids,
        "reasons": dict(sorted(reasons.items())),
        "added_token_estimate": added_tokens,
    }


def _candidate_ids(aspect_retrievals: Iterable[Any] | None) -> set[str]:
    ids: set[str] = set()
    for item in aspect_retrievals or []:
        for candidate in getattr(item, "candidates", None) or []:
            chunk_id = _chunk_id(candidate)
            if chunk_id:
                ids.add(chunk_id)
    return ids


def audit_evidence_sufficiency(
    *,
    requirements: Iterable[Any] | None,
    context_chunks: Iterable[Any] | None,
    coverage: Mapping[str, Iterable[str]] | None = None,
    missing_requirements: Iterable[str] | None = None,
    aspect_retrievals: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Return a replayable sufficiency report for one answer context.

    ``coverage`` defaults to the bundle's requirement -> chunk-id mapping.  A
    requirement not present in ``coverage`` is reported as ``unbound``; it is
    never converted into a silent pass.
    """

    requirement_list = list(requirements or [])
    chunk_list = list(context_chunks or [])
    if not requirement_list:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "not_applicable",
            "requirements": [],
            "missing_requirements": [],
            "missing_facets": [],
            "expansion": _expansion_diagnostics(chunk_list),
            "counts": {
                "requirements": 0,
                "candidates": len(_candidate_ids(aspect_retrievals)),
                "selected": len(chunk_list),
            },
            "notes": ["未形成证据要求，无法判断集合充分性。"],
        }

    by_id = {_chunk_id(chunk): chunk for chunk in chunk_list if _chunk_id(chunk)}
    coverage_map = {str(key): list(value or []) for key, value in (coverage or {}).items()}
    reports = [
        _requirement_report(requirement, _chunks_for_requirement(
            _text(getattr(requirement, "requirement_id", "")), coverage_map, by_id,
        ))
        for requirement in requirement_list
    ]
    mandatory_ids = {
        _text(getattr(requirement, "requirement_id", ""))
        for requirement in requirement_list
        if bool(getattr(requirement, "mandatory", True))
    }
    declared_missing = {str(item) for item in (missing_requirements or [])}
    missing_requirements = [
        report["requirement_id"] for report in reports
        if report["requirement_id"] in mandatory_ids
        and (report["status"] in {"unbound", "missing"} or report["requirement_id"] in declared_missing)
    ]
    partial_requirements = [
        report["requirement_id"] for report in reports
        if report["requirement_id"] in mandatory_ids and report["status"] == "partial"
    ]
    missing_facets = [
        f"{report['requirement_id']}:{code}"
        for report in reports
        if report["requirement_id"] in mandatory_ids
        for code in report["reason_codes"]
    ]

    if not chunk_list or missing_requirements:
        status = "insufficient"
    elif partial_requirements or missing_facets:
        status = "partial"
    else:
        status = "sufficient"

    notes: list[str] = []
    if any(report["status"] == "unbound" for report in reports):
        notes.append("存在未绑定任何证据的要求；该状态不等同于已满足。")
    if status != "sufficient":
        notes.append("充分性为确定性代理判断，最终事实支持仍由统一核验裁定。")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "requirements": reports,
        "missing_requirements": missing_requirements,
        "partial_requirements": partial_requirements,
        "missing_facets": missing_facets,
        "expansion": _expansion_diagnostics(chunk_list),
        "counts": {
            "requirements": len(requirement_list),
            "candidates": len(_candidate_ids(aspect_retrievals)),
            "selected": len(chunk_list),
        },
        "notes": notes,
    }
