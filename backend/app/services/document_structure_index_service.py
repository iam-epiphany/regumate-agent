"""Ingest-time structural lookup indexes: definition dictionary + clause index.

Two additive tables mirror the spreadsheet_cells pattern:

- ``document_definitions``: term -> chunk rows extracted from definition
  sentences ("所称X…是指 / X是指 / （简称X）"), so the definition retrieval
  channel can resolve a quoted term without a full-corpus LIKE scan.
- ``document_clauses``: "第X条" clause-number -> chunk rows derived from chunk
  section titles, so clause references resolve against a table instead of
  re-deriving the mapping per request.

Chunk text remains the source of evidence shown to users; both tables are
safe to rebuild at any time and degrade silently to the existing query-time
scans when they hold no rows for a term.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.app.models.document import (
    Document,
    DocumentChunk,
    DocumentClause,
    DocumentDefinition,
)

# Chinese + Latin + digits + full/half width parens; bounded to plausible
# regulatory term lengths so prose fragments do not match.
_TERM_CHARS = r"\u4e00-\u9fffA-Za-z0-9（）()"

NAMED_AS_DEFINITION_RE = re.compile(
    rf"所称([{_TERM_CHARS}]{{2,24}}?)(?=，是指|是指|，指|指的)"
)
ABBREVIATION_DEFINITION_RE = re.compile(
    rf"简称[“\"‘’]?([{_TERM_CHARS}]{{2,20}}?)(?=[）)”\"‘’。；;，,\n]|$)"
)
MEANS_DEFINITION_RE = re.compile(rf"([{_TERM_CHARS}]{{2,24}})(?=是指)")

# Regulatory terms are usually introduced after these prefixes; strip them so
# "本办法所称贷款是指" indexes "贷款" and not "本办法所称贷款".
DEFINITION_LEADING_PREFIXES = (
    "本办法所称",
    "本法所称",
    "本条例所称",
    "所称之",
    "所称",
    "本办法",
    "本法",
    "本条例",
    "前述",
    "上述",
    "下列",
    "之",
    "的",
)

_CLAUSE_LABEL_RE = re.compile(r"第([一二三四五六七八九十百零\d]{1,4})条")

_CN_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100}


def normalize_definition_term(value: Any) -> str:
    """Normalize a term for dictionary lookups without destroying meaning."""

    text = str(value or "").strip().lower()
    text = text.translate(
        str.maketrans({"（": "(", "）": ")", "：": ":", "“": "", "”": ""})
    )
    return re.sub(r"\s+", "", text)


def clause_number_from_label(label: Any) -> int | None:
    """Convert "第四条"/"第105条" style labels to a normalized integer."""

    text = str(label or "").strip()
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
        elif char == "零":
            matched = True
        else:
            return None
    total += current
    return total if matched else None


def _strip_definition_prefixes(term: str) -> str:
    stripped = term.lstrip("的")
    changed = True
    while changed:
        changed = False
        for prefix in DEFINITION_LEADING_PREFIXES:
            if stripped.startswith(prefix) and len(stripped) - len(prefix) >= 2:
                stripped = stripped[len(prefix) :]
                changed = True
    return stripped


def _snippet(text: str, start: int, limit: int = 120) -> str:
    return re.sub(r"\s+", " ", text[start : start + limit]).strip()


def extract_definition_terms(text: str) -> list[tuple[str, str, str]]:
    """Extract (pattern_kind, term, snippet) triples from one chunk text."""

    results: list[tuple[str, str, str]] = []
    for match in NAMED_AS_DEFINITION_RE.finditer(text):
        term = match.group(1).lstrip("的")
        if len(term) >= 2:
            results.append(("所称", term, _snippet(text, match.start())))
    for match in ABBREVIATION_DEFINITION_RE.finditer(text):
        term = match.group(1).rstrip("）)").strip()
        if len(term) >= 2:
            results.append(("简称", term, _snippet(text, match.start())))
    for match in MEANS_DEFINITION_RE.finditer(text):
        term = _strip_definition_prefixes(match.group(1))
        if len(term) >= 2:
            results.append(("是指", term, _snippet(text, match.start())))
    return results


def clause_labels_in_title(section_title: Any) -> list[str]:
    """Return the bare clause numerals ("二", "105") in one section title."""

    return list(_CLAUSE_LABEL_RE.findall(str(section_title or "")))


def rebuild_document_structure_index(
    db: Session,
    document: Document,
    chunks: Iterable[DocumentChunk] | None = None,
) -> tuple[int, int]:
    """Replace one document's definition dictionary and clause index rows."""

    db.execute(
        delete(DocumentDefinition).where(
            DocumentDefinition.document_id == document.document_id
        )
    )
    db.execute(
        delete(DocumentClause).where(DocumentClause.document_id == document.document_id)
    )
    source_chunks = (
        list(chunks)
        if chunks is not None
        else list(
            db.scalars(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document.document_id)
                .order_by(DocumentChunk.id.asc())
            )
        )
    )
    definition_count = 0
    clause_count = 0
    for chunk in source_chunks:
        text = str(chunk.text or "") or str(chunk.embedding_text or "")
        seen: set[str] = set()
        for kind, term, snippet in extract_definition_terms(text):
            term_norm = normalize_definition_term(term)
            # One chunk defines one term once: "所称X…是指" also matches the
            # generic "X是指" scan, so dedupe across pattern kinds.
            if not term_norm or term_norm in seen:
                continue
            seen.add(term_norm)
            db.add(
                DocumentDefinition(
                    document_id=document.document_id,
                    chunk_id=chunk.chunk_id,
                    term=term,
                    term_norm=term_norm,
                    pattern_kind=kind,
                    snippet=snippet,
                    section_title=chunk.section_title,
                    source_file=chunk.source_file,
                )
            )
            definition_count += 1
        for label in clause_labels_in_title(chunk.section_title):
            number = clause_number_from_label(label)
            if number is None:
                continue
            db.add(
                DocumentClause(
                    document_id=document.document_id,
                    chunk_id=chunk.chunk_id,
                    clause_number=number,
                    clause_label=f"第{label}条",
                    section_title=chunk.section_title,
                    source_file=chunk.source_file,
                )
            )
            clause_count += 1
    db.flush()
    return definition_count, clause_count


def lookup_definition_chunks(
    db: Session,
    terms: Iterable[str],
    document_ids: set[str] | None = None,
    limit: int = 24,
) -> list[tuple[DocumentChunk, str]]:
    """Resolve quoted terms against the definition dictionary.

    Returns (chunk, pattern) pairs shaped like the query-time scan results so
    the definition channel can consume both interchangeably.  The pattern is
    rebuilt from the question's own term surface form.
    """

    norms: dict[str, str] = {}
    for term in terms:
        normalized = normalize_definition_term(term)
        if normalized and normalized not in norms:
            norms[normalized] = str(term)
    if not norms:
        return []
    statement = (
        select(DocumentDefinition, DocumentChunk)
        .join(DocumentChunk, DocumentChunk.chunk_id == DocumentDefinition.chunk_id)
        .where(
            DocumentChunk.index_status == "indexed",
            DocumentDefinition.term_norm.in_(list(norms)),
        )
        .order_by(DocumentDefinition.id.asc())
    )
    if document_ids:
        statement = statement.where(DocumentDefinition.document_id.in_(document_ids))
    rows = db.execute(statement.limit(limit * 2)).all()
    found: list[tuple[DocumentChunk, str]] = []
    seen: set[str] = set()
    for definition, chunk in rows:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        question_term = norms.get(definition.term_norm, definition.term)
        if definition.pattern_kind == "所称":
            pattern = f"所称{question_term}"
        elif definition.pattern_kind == "简称":
            pattern = f"简称{question_term}"
        else:
            pattern = f"{question_term}是指"
        found.append((chunk, pattern))
        if len(found) >= limit:
            break
    return found


def normalize_document_title(value: Any) -> str:
    """Normalize a document title/filename for cross-document resolution."""

    text = str(value or "").strip().lower()
    text = text.translate(str.maketrans({"（": "(", "）": ")", "：": ":", "＿": "_"}))
    text = re.sub(r"\.(docx?|pdf|xls|xlsx|csv)$", "", text)
    return re.sub(r"\s+", "", text)


def resolve_document_ids_by_titles(
    db: Session,
    titles: Iterable[str],
    *,
    max_documents: int = 2,
) -> dict[str, str]:
    """Resolve bracketed regulation titles to document ids.

    Matching order per title: normalized filename equality, then containment
    in filename/title with the shortest filename winning (the main regulation
    body is shorter than its attachments).  Unresolvable titles are skipped.
    """

    normalized_titles: list[str] = []
    for title in titles:
        normalized = normalize_document_title(title)
        if normalized and normalized not in normalized_titles:
            normalized_titles.append(normalized)
    if not normalized_titles:
        return {}
    resolved: dict[str, str] = {}
    documents = list(db.scalars(select(Document).order_by(Document.id.asc())))
    for normalized in normalized_titles:
        exact = next(
            (
                d
                for d in documents
                if normalize_document_title(d.filename_norm) == normalized
            ),
            None,
        )
        if exact is None:
            exact = next(
                (
                    d
                    for d in documents
                    if normalize_document_title(d.title) == normalized
                ),
                None,
            )
        if exact is None:
            candidates = [
                d
                for d in documents
                if normalized in normalize_document_title(d.filename_norm)
                or (d.title and normalized in normalize_document_title(d.title))
            ]
            exact = min(
                candidates, key=lambda d: len(d.filename_norm or ""), default=None
            )
        if exact is not None:
            resolved[normalized] = exact.document_id
            if len(resolved) >= max_documents:
                break
    return resolved
