"""Shared helpers for the Banking Workbench evaluation toolchain.

Evaluation-only.  Production code never imports this package.
"""

from __future__ import annotations

from decimal import Decimal
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


PUBLIC_FIELDS = ("id", "question", "answerable", "question_type", "difficulty", "business_relevance")
PRIVATE_FIELDS = (
    "canonical_answer",
    "acceptable_answers",
    "required_conclusions",
    "required_evidence_aspects",
    "required_sources",
    "calculation",
    "expected_refusal_code",
    "refusal_rationale",
    "evidence_text",
    "sheet",
    "sheet_raw",
    "row",
    "column",
    "coordinate",
    "expected_value",
)

QUESTION_TYPE_QUOTA = {
    "fact_definition": 15,
    "rule_scope": 20,
    "threshold_rule": 13,
    "scope_list": 8,
    "table_lookup": 10,
    "table_calculation": 10,
    "cross_document_evidence": 4,
    "refusal": 20,
}
DIFFICULTY_QUOTA = {"easy": 20, "medium": 55, "hard": 25}
ANSWERABLE_QUOTA = {"answerable": 80, "refusal": 20}
MIN_SOURCE_DOCUMENTS = 20


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise(text: Any) -> str:
    """Normalise for containment matching: strip whitespace, fold full-width
    punctuation, letters and digits to ASCII, lower-case."""

    value = str(text or "")
    fullwidth = {
        "％": "%", "－": "-", "−": "-", "﹣": "-", "　": "", "：": ":", "（": "(", "）": ")",
        "，": ",", "。": ".", "；": ";", "？": "?", "！": "!", "、": ",", "“": '"', "”": '"',
        "‘": "'", "’": "'",
    }
    for index in range(ord("０"), ord("９") + 1):
        fullwidth[chr(index)] = chr(index - ord("０") + ord("0"))
    for index in range(ord("Ａ"), ord("Ｚ") + 1):
        fullwidth[chr(index)] = chr(index - ord("Ａ") + ord("A"))
    for index in range(ord("ａ"), ord("ｚ") + 1):
        fullwidth[chr(index)] = chr(index - ord("ａ") + ord("a"))
    value = value.translate(str.maketrans(fullwidth))
    value = re.sub(r"\s+", "", value).lower()
    return value


def normalise_title(text: Any) -> str:
    """Normalise a document title: drop numbering prefix and extension."""

    value = str(text or "").strip()
    value = re.sub(r"^\d+_", "", value)
    value = re.sub(r"\.(?:xls|xlsx|doc|docx|pdf|csv)$", "", value, flags=re.IGNORECASE)
    return normalise(value)


def corpus_manifest_hash(corpus_root: Path) -> str:
    """Deterministic hash over the corpus file list and per-file sha256."""

    files = sorted(
        path
        for path in corpus_root.iterdir()
        if path.is_file() and not path.name.startswith("~$")
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def bigrams(text: str) -> list[str]:
    cleaned = re.sub(r"[^\w一-鿿]+", "", text)
    if len(cleaned) < 2:
        return [cleaned] if cleaned else []
    return [cleaned[index : index + 2] for index in range(len(cleaned) - 1)]


def bigram_recall(answer: str, gold: str) -> float:
    gold_grams = set(bigrams(gold))
    if not gold_grams:
        return 1.0 if normalise(gold) in normalise(answer) else 0.0
    answer_grams = set(bigrams(answer))
    matched = sum(1 for gram in gold_grams if gram in answer_grams)
    return matched / len(gold_grams)


_NEGATION_MARKERS = ("不得", "禁止", "不应", "不允许", "不可", "不得以", "不得高于", "不得低于", "不超过", "不高于", "不低于", "严禁")


def polarity_conflict(answer: str, gold: str) -> bool:
    """True when the gold mandates a negative and the answer is affirmative."""

    gold_negated = any(marker in gold for marker in _NEGATION_MARKERS)
    answer_negated = any(marker in answer for marker in _NEGATION_MARKERS)
    return gold_negated and not answer_negated


def cell_value_numeric(raw: Any) -> Decimal | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return Decimal(str(raw))
    text = str(raw).strip().replace(",", "").replace("，", "")
    text = text.replace("%", "")
    try:
        return Decimal(text)
    except Exception:
        return None
