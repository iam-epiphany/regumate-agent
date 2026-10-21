"""Deterministic structural coordinates for regulatory evidence.

A locator is derived only from data that already exists on a chunk
(document title, ``section_path``, ``section_number`` and the leading text).
It is deliberately independent of retrieval: the same function is used to
label a citation at answer time and to compare a retrieved citation with a
gold citation at evaluation time.

Nothing in this module calls a model, touches the database or the vector
store, or changes ranking.  Missing information stays missing; the locator
never guesses a chapter, article or version.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

SCHEMA_VERSION = "1.0"

# Ordered from coarser to finer.  ``heading`` is a layout heading that does
# not carry a statutory level; it is kept for display but never matched as a
# legal coordinate.
LEVEL_ORDER = ("part", "chapter", "section", "article", "paragraph", "item")
LEVEL_LABELS = {
    "part": "编",
    "chapter": "章",
    "section": "节",
    "article": "条",
    "paragraph": "款",
    "item": "项",
}
DEPTHS = ("document", *LEVEL_ORDER)

_CN_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}
_NUMERAL = r"[零〇一二三四五六七八九十百千\d]+"

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("part", re.compile(rf"第\s*({_NUMERAL})\s*编")),
    ("chapter", re.compile(rf"第\s*({_NUMERAL})\s*章")),
    ("section", re.compile(rf"第\s*({_NUMERAL})\s*节")),
    ("article", re.compile(rf"第\s*({_NUMERAL})\s*条(?:\s*之\s*({_NUMERAL}))?")),
    ("paragraph", re.compile(rf"第\s*({_NUMERAL})\s*款")),
)
_ITEM_RE = re.compile(rf"^\s*[（(]\s*({_NUMERAL})\s*[）)]")


def chinese_number(value: Any) -> int | None:
    """Convert a Chinese or Arabic numeral to an integer, or return ``None``."""

    text = str(value or "").strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    total = 0
    current = 0
    matched = False
    for char in text:
        if char in _CN_DIGITS:
            current = _CN_DIGITS[char]
            matched = True
        elif char in _CN_UNITS:
            unit = _CN_UNITS[char]
            if current == 0:
                current = 1
            total += current * unit
            current = 0
            matched = True
        elif char in {"零", "〇"}:
            matched = True
        else:
            return None
    total += current
    return total if matched else None


def _level(kind: str, label: str, number: int | None, suffix: str | None = None) -> dict[str, Any]:
    return {
        "kind": kind,
        "label": label,
        "number": number,
        "suffix": suffix,
        "raw": label,
    }


def parse_structure_level(value: Any, *, start_only: bool = False) -> dict[str, Any] | None:
    """Parse one heading or clause title into a typed structural level.

    Returns ``None`` for empty text.  A title without a statutory marker is
    returned as a ``heading`` level so the display can still show the section
    path, but it is not treated as a legal coordinate.  ``start_only`` is used
    for chunk text, where a reference such as "根据第十条" must not be mistaken
    for the chunk's own coordinate.
    """

    text = " ".join(str(value or "").split())
    if not text:
        return None
    for kind, pattern in _PATTERNS:
        match = pattern.match(text) if start_only else pattern.search(text)
        if not match:
            continue
        number = chinese_number(match.group(1))
        suffix = match.group(2) if kind == "article" and match.lastindex and match.lastindex >= 2 else None
        label = match.group(0)
        return _level(kind, label.strip(), number, str(suffix) if suffix else None)
    item = _ITEM_RE.match(text)
    if item:
        return _level("item", item.group(0).strip(), chinese_number(item.group(1)))
    return _level("heading", text[:120], None)


def _normalize_document(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\.(?:pdf|docx?|xlsx?|xls|csv|txt|md)$", "", text)
    text = text.replace("《", "").replace("》", "")
    text = re.sub(r"^\d+[_-]+", "", text)
    text = re.sub(r"[\s·•、，,。.;；:：()（）\"'“”‘’]+", "", text)
    return text


def _display_document(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\.(?:pdf|docx?|xlsx?|xls|csv|txt|md)$", "", text, flags=re.IGNORECASE)
    text = text.strip("《》 ")
    return text


def build_structure_locator(
    *,
    document_id: str,
    document_title: str | None = None,
    section_path: Iterable[Any] | None = None,
    section_title: str | None = None,
    section_number: str | None = None,
    text: str | None = None,
    index_version: str | None = None,
    parser_version: str | None = None,
) -> dict[str, Any]:
    """Build one JSON-serialisable structural coordinate.

    ``section_path`` is the parser-provided heading chain.  ``text`` is only
    inspected for a leading ``第X条`` / ``第X款`` marker so a chunk that starts
    mid-clause does not borrow the previous heading's coordinate by accident.
    """

    path_values = [value for value in (section_path or []) if str(value or "").strip()]
    parsed: list[dict[str, Any]] = []
    for value in path_values:
        level = parse_structure_level(value)
        if level is not None:
            parsed.append(level)
    title_level = parse_structure_level(section_title)
    if title_level is not None and title_level["kind"] != "heading":
        parsed.append(title_level)

    has_article = any(level["kind"] == "article" for level in parsed)
    if not has_article and section_number:
        level = parse_structure_level(section_number)
        if level is not None and level["kind"] == "article":
            parsed.append(level)
            has_article = True
    if not has_article and text:
        level = parse_structure_level(str(text)[:120], start_only=True)
        if level is not None and level["kind"] == "article":
            parsed.append(level)
            has_article = True

    # Keep the last value per statutory kind: a nested heading later in the
    # path is the more specific coordinate.
    latest: dict[str, dict[str, Any]] = {}
    headings: list[dict[str, Any]] = []
    for level in parsed:
        if level["kind"] == "heading":
            headings.append(level)
        else:
            latest[level["kind"]] = level
    ordered = [latest[kind] for kind in LEVEL_ORDER if kind in latest]
    if not ordered:
        ordered = headings[:1]

    display_title = _display_document(document_title)
    labels = [level["label"] for level in ordered]
    display = " · ".join(([f"《{display_title}》"] if display_title else []) + labels) or "定位未记录"

    key_parts = [f"doc:{_normalize_document(document_title) or document_id}"]
    for kind in LEVEL_ORDER:
        level = latest.get(kind)
        if level is None:
            continue
        number = level.get("number")
        value = str(number) if number is not None else str(level.get("label") or "")
        suffix = level.get("suffix")
        if suffix:
            value = f"{value}-{suffix}"
        key_parts.append(f"{kind}:{value}")

    if has_article:
        status = "resolved"
    elif any(level["kind"] in {"part", "chapter", "section"} for level in ordered):
        status = "partial"
    else:
        status = "unresolved"

    return {
        "schema_version": SCHEMA_VERSION,
        "document_id": str(document_id or ""),
        "document_title": display_title or None,
        "index_version": index_version or None,
        "parser_version": parser_version or None,
        "path": ordered,
        "display": display,
        "key": "|".join(key_parts),
        "resolution_status": status,
    }


def locator_from_citation(citation: Any) -> dict[str, Any] | None:
    """Build a locator from an object shaped like ``schemas.qa.Citation``."""

    metadata = getattr(citation, "metadata", None) or {}
    document_title = (
        getattr(citation, "source_title", None)
        or metadata.get("source_title")
        or metadata.get("title")
        or getattr(citation, "filename", None)
    )
    locator = build_structure_locator(
        document_id=str(getattr(citation, "document_id", "") or ""),
        document_title=str(document_title) if document_title else None,
        section_path=getattr(citation, "section_path", None) or [],
        section_title=getattr(citation, "section_title", None),
        section_number=getattr(citation, "section_number", None),
        text=getattr(citation, "excerpt", None),
        index_version=str(metadata.get("index_version") or "") or None,
        parser_version=str(metadata.get("parser_version") or "") or None,
    )
    return locator


def locator_from_chunk(chunk: Any) -> dict[str, Any] | None:
    """Build a locator from an object shaped like ``RetrievalResult``."""

    metadata = getattr(chunk, "metadata", None) or {}
    document_title = (
        metadata.get("source_title")
        or metadata.get("title")
        or getattr(chunk, "source_doc", None)
        or getattr(chunk, "filename", None)
    )
    return build_structure_locator(
        document_id=str(
            getattr(chunk, "document_id", "")
            or metadata.get("document_id")
            or ""
        ),
        document_title=str(document_title) if document_title else None,
        section_path=getattr(chunk, "section_path", None) or [],
        section_title=getattr(chunk, "section_title", None),
        section_number=(metadata.get("article_number") or getattr(chunk, "section_number", None)),
        text=getattr(chunk, "text", None),
        index_version=str(metadata.get("index_version") or "") or None,
        parser_version=str(metadata.get("parser_version") or "") or None,
    )


def _locator_value(locator: Mapping[str, Any] | None, kind: str) -> dict[str, Any] | None:
    if not locator:
        return None
    if kind == "document":
        return None
    for level in locator.get("path") or []:
        if isinstance(level, Mapping) and level.get("kind") == kind:
            return dict(level)
    return None


def _same_level(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_number = left.get("number")
    right_number = right.get("number")
    if left_number is not None and right_number is not None:
        if int(left_number) != int(right_number):
            return False
        return str(left.get("suffix") or "") == str(right.get("suffix") or "")
    return str(left.get("label") or "").strip() == str(right.get("label") or "").strip()


def match_structure(
    gold: Mapping[str, Any] | None,
    retrieved: Mapping[str, Any] | None,
    *,
    depth: str = "article",
    mode: str = "ancestor",
) -> dict[str, Any]:
    """Compare two locators down to ``depth``.

    ``mode="ancestor"`` treats a level that only one side specifies as
    implicit (the paper's prefix rule).  ``mode="strict"`` reports that case
    as ``level_unavailable`` instead of silently crediting it.  Both results
    are returned as a diagnostic dict so an evaluator can report strict and
    ancestor hit rates separately.
    """

    if depth not in DEPTHS:
        raise ValueError(f"Unknown structure depth: {depth}")
    if mode not in {"strict", "ancestor"}:
        raise ValueError(f"Unknown structure match mode: {mode}")

    result: dict[str, Any] = {
        "matched": False,
        "depth": depth,
        "mode": mode,
        "reason": None,
        "mismatch_level": None,
        "implicit_levels": [],
        "compared_levels": [],
    }
    if not gold or not retrieved:
        result["reason"] = "locator_missing"
        return result

    gold_doc = str(gold.get("key") or "").split("|", maxsplit=1)[0]
    retrieved_doc = str(retrieved.get("key") or "").split("|", maxsplit=1)[0]
    if gold_doc and retrieved_doc and gold_doc != retrieved_doc:
        result["reason"] = "document_mismatch"
        result["mismatch_level"] = "document"
        return result
    result["compared_levels"].append("document")

    allowed = LEVEL_ORDER[: LEVEL_ORDER.index(depth) + 1] if depth != "document" else ()
    for kind in allowed:
        gold_level = _locator_value(gold, kind)
        retrieved_level = _locator_value(retrieved, kind)
        if gold_level is None or retrieved_level is None:
            result["implicit_levels"].append(kind)
            continue
        result["compared_levels"].append(kind)
        if not _same_level(gold_level, retrieved_level):
            result["reason"] = "level_mismatch"
            result["mismatch_level"] = kind
            return result

    if mode == "strict" and depth != "document":
        target = _locator_value(gold, depth) is not None and _locator_value(retrieved, depth) is not None
        if not target:
            result["reason"] = "level_unavailable"
            return result

    result["matched"] = True
    result["reason"] = "matched"
    return result


def structure_prefix_match(
    gold: Mapping[str, Any] | None,
    retrieved: Mapping[str, Any] | None,
    *,
    depth: str = "article",
) -> bool:
    """Convenience wrapper for the ancestor/prefix rule."""

    return bool(match_structure(gold, retrieved, depth=depth, mode="ancestor")["matched"])
