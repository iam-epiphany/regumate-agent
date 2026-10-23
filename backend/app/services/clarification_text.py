"""Turn model-supplied clarification payloads into user-facing text.

Planner and answer models occasionally return clarification as structured
data such as ``{"needed": true, "question": "..."}`` or prefix the text with
an internal marker such as ``needed:``.  Those values are machine-facing:
rendering them verbatim leaks field names and JSON/Python syntax into the
answer.  This module is the single normalisation point for both paths.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

DEFAULT_CLARIFICATION_TEXT = (
    "请补充更具体的问题信息"
    "（如制度文件、统计报表、指标名称、期间或业务范围），以便继续检索。"
)

_MARKER_RE = re.compile(
    r"^\s*(?:needed|need|required|missing(?:\s+information|_information)?|"
    r"clarification|question)\s*[:：]\s*",
    re.IGNORECASE,
)
# A human question is more specific than a boolean flag, so it comes first.
_TEXT_KEYS = (
    "question",
    "message",
    "text",
    "clarification",
    "needed",
    "missing_information",
    "missing",
    "required_information",
    "required",
    "reason",
    "detail",
    "items",
    "fields",
)
_MACHINE_WORD_RE = re.compile(
    r"^(?:true|false|none|null|needed|required|missing|clarification)$",
    re.IGNORECASE,
)
_MACHINE_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_CLARIFICATION_HINT_RE = re.compile(
    r"请(?:补充|提供|明确|说明|确认)|需要?(?:补充|提供|明确)|缺少|"
    r"needed|missing|provide|specify|clarify|identify",
    re.IGNORECASE,
)
_BARE_PREFIX = "请补充以下信息："
_SENTENCE_ENDINGS = "。！？；;.!?：:"


def normalize_clarification_text(value: Any) -> str:
    """Return a clean, user-facing clarification sentence.

    Structured payloads are unwrapped, internal markers are removed, and a
    bare noun phrase is turned into an explicit request.  An empty string
    means the payload carries no usable user-facing text.
    """

    text = _render(value, reject_machine=True)
    if not text:
        return ""
    if _CLARIFICATION_HINT_RE.search(text) or text.endswith(tuple(_SENTENCE_ENDINGS)):
        return text
    return f"{_BARE_PREFIX}{text}"


def normalize_structured_text(value: Any) -> str:
    """Extract plain text from a structured payload without rewriting it."""

    return _render(value, reject_machine=False)


def normalize_refusal_text(value: Any) -> str:
    """Return user-facing refusal text while rejecting internal codes.

    A structured or marker-prefixed value keeps the clarification styling
    (for example ``needed: X`` becomes ``请补充以下信息：X``).  A plain
    human sentence is returned unchanged so a legitimate refusal such as
    ``证据不足`` is not rewritten as a clarification request.
    """

    plain = normalize_structured_text(value)
    if not plain or not normalize_clarification_text(plain):
        return ""
    marked = plain != str(value or "").strip() or not isinstance(value, str)
    return normalize_clarification_text(value) if marked else plain


def is_clarification_request(value: Any) -> bool:
    text = normalize_clarification_text(value)
    return bool(text and _CLARIFICATION_HINT_RE.search(text))


def _render(value: Any, *, reject_machine: bool) -> str:
    parts = _collect_parts(value, depth=0, reject_machine=reject_machine)
    text = _strip_markers("；".join(parts))
    if not text:
        return ""
    if reject_machine and (
        _MACHINE_WORD_RE.match(text) or _MACHINE_TOKEN_RE.match(text)
    ):
        return ""
    return text


def _collect_parts(value: Any, *, depth: int, reject_machine: bool) -> list[str]:
    if depth > 4:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        parsed = _parse_structured(text)
        if parsed is not None:
            return _collect_parts(
                parsed, depth=depth + 1, reject_machine=reject_machine
            )
        text = _strip_markers(text)
        if not text:
            return []
        if reject_machine and (
            _MACHINE_WORD_RE.match(text) or _MACHINE_TOKEN_RE.match(text)
        ):
            return []
        if text.startswith(("{", "[")):
            # Unparseable structured syntax is not user-facing prose.
            return []
        return [text]
    if isinstance(value, dict):
        parts: list[str] = []
        for key in _TEXT_KEYS:
            if key in value:
                parts.extend(
                    _collect_parts(
                        value[key], depth=depth + 1, reject_machine=reject_machine
                    )
                )
        if not parts:
            for item in value.values():
                parts.extend(
                    _collect_parts(item, depth=depth + 1, reject_machine=reject_machine)
                )
        return _dedupe(parts)
    if isinstance(value, (list, tuple, set)):
        parts = []
        for item in value:
            parts.extend(
                _collect_parts(item, depth=depth + 1, reject_machine=reject_machine)
            )
        return _dedupe(parts)
    return []


def _parse_structured(text: str) -> Any | None:
    stripped = text.strip().strip("`").strip()
    if len(stripped) < 2 or len(stripped) > 8000:
        return None
    if stripped[0] not in "{[\"'" or stripped[-1] not in "}]\"'":
        return None
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(stripped)
        except (ValueError, SyntaxError, TypeError):
            continue
        if isinstance(parsed, (dict, list, tuple, str)):
            return parsed
    return None


def _strip_markers(text: str) -> str:
    cleaned = text.strip().strip("\"'“”").strip()
    previous = None
    while cleaned and cleaned != previous:
        previous = cleaned
        cleaned = _MARKER_RE.sub("", cleaned).strip()
        cleaned = cleaned.strip("\"'“”").strip()
    return cleaned


def _dedupe(parts: list[str]) -> list[str]:
    result: list[str] = []
    for part in parts:
        if not part:
            continue
        if any(part == existing or part in existing for existing in result):
            continue
        result = [existing for existing in result if existing not in part]
        result.append(part)
    return result
