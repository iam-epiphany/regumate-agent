"""Recognize references with a model; resolve only against stored document data.

No model-supplied chunk IDs or evidence text are accepted. Missing/ambiguous
targets remain unresolved. Parser table_index is an ingest ordinal, NOT a
printed table number. Figure pixels and external attachment files are not text.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.app.core.rag_features import event
from backend.app.services.document_structure_index_service import clause_number_from_label
from backend.app.services.llm_client import ChatCompletionError
from backend.app.services.retrieval_advisor_service import request_advice

MAX_SOURCES = 8
MAX_REFERENCES = 6
MAX_TARGETS = 2
NUMBER = r"[零〇一二三四五六七八九十百千万两\d]+(?:[.．-]\d+)*"
LABELS = {"clause": rf"第\s*({NUMBER})\s*条", "page": rf"第\s*({NUMBER})\s*页",
          "table": rf"(?:附表|表)\s*({NUMBER})", "attachment": rf"(?:附件|附录)\s*({NUMBER})",
          "figure": rf"图\s*({NUMBER})"}


class ReferenceHint(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    kind: Literal["clause", "page", "table", "attachment", "figure", "context", "section"]
    quote: str = Field(min_length=1, max_length=160)
    number: str | None = Field(default=None, max_length=32)
    page_number: int | None = Field(default=None, ge=1)
    title: str | None = Field(default=None, max_length=120)
    direction: Literal["previous", "next"] | None = None
    document_title: str | None = Field(default=None, max_length=120)


class SourceReferences(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source: int = Field(ge=0, lt=MAX_SOURCES)
    references: list[ReferenceHint] = Field(max_length=MAX_REFERENCES)


class ReferenceBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    sources: list[SourceReferences] = Field(max_length=MAX_SOURCES)


@dataclass(frozen=True)
class ResolvedReference:
    chunk: Any
    hint: ReferenceHint
    method: str


def metadata(chunk: Any) -> dict[str, Any]:
    try:
        value = json.loads(chunk.chunk_metadata or "{}")
        return value if isinstance(value, dict) else {}
    except (ValueError, TypeError):
        return {}


def _text(chunk: Any) -> str:
    # Embedding prefixes can repeat titles and references from other paragraphs.
    return str(chunk.text or "")


def _grounded_hint(hint: ReferenceHint, text: str) -> bool:
    if hint.quote not in text:
        return False
    numbers = {_number(match[0]) for match in re.finditer(NUMBER, hint.quote)}
    if hint.number and _number(hint.number) not in numbers:
        return False
    if hint.page_number is not None and str(hint.page_number) not in numbers:
        return False
    if hint.kind in LABELS and hint.number:
        explicit = {_number(match[1]) for match in re.finditer(LABELS[hint.kind], hint.quote)}
        if explicit and _number(hint.number) not in explicit:
            return False
    return True


def recognize_references(anchors: list[Any]) -> dict[str, list[ReferenceHint]]:
    """One batch per expansion, with explicit citations retained on model failure."""
    result = {anchor.chunk_id: fallback_references(_text(anchor)) for anchor in anchors}
    selected = anchors[:MAX_SOURCES]
    if not selected:
        return result
    try:
        batch = ReferenceBatch.model_validate(request_advice(
            '识别每个候选文本中真实存在的引用（不要把自身标题当引用）。返回 '
            '{"sources":[{"source":0,"references":[{"kind":"clause|page|table|attachment|figure|context|section",'
            '"quote":"原文中的完整引用短语","number":"目标编号或null","page_number":18,'
            '"title":null,"direction":"previous|next或null","document_title":null}]}]}。'
            'quote必须逐字来自输入。复合引用如“第18页表3”用table+number=3+page_number=18；'
            '“下表/上图/上述章节”用对应类型和direction；纯上下文引用用context。'
            '跨文档引用填原文书名。不得输出目标chunk或编造内容。每个文本最多6项。',
            [{"source": index, "text": _text(anchor)[:2400]} for index, anchor in enumerate(selected)],
        ))
        model_hints: dict[str, list[ReferenceHint]] = {}
        for source in batch.sources:
            if source.source >= len(selected):
                continue
            anchor = selected[source.source]
            for hint in source.references:
                if not _grounded_hint(hint, _text(anchor)[:2400]):
                    continue
                # An external book title cannot become a same-document edge.
                if "《" in hint.quote and not hint.document_title:
                    continue
                model_hints.setdefault(anchor.chunk_id, [])
                if hint not in model_hints[anchor.chunk_id]:
                    model_hints[anchor.chunk_id].append(hint)
        for key, hints in model_hints.items():
            result[key] = hints + [hint for hint in result[key] if hint not in hints]
        event("reference_recognition", status="model", sources=len(selected))
    except (ChatCompletionError, ValueError, TypeError):
        event("reference_recognition", status="rule_fallback", sources=len(selected))
    return {key: hints[:MAX_REFERENCES] for key, hints in result.items()}


def fallback_references(text: str) -> list[ReferenceHint]:
    """Small compatibility fallback; semantic/implicit forms belong to the model."""
    hints: list[ReferenceHint] = []
    # Exclude external citations from same-document fallback, preserving offsets.
    compound = rf"第\s*({NUMBER})\s*页\s*[，,、]?\s*(?:的)?\s*表\s*({NUMBER})"
    target_labels = "|".join(LABELS.values())
    local = re.sub(rf"《[^》\n]+》\s*(?:的|中)?\s*(?:{compound}|{target_labels})",
                   lambda match: " " * len(match[0]), text)
    compound_spans: list[tuple[int, int]] = []
    for match in re.finditer(compound, local):
        page = _number(match[1])
        if not page.isdigit() or int(page) < 1:
            continue
        hints.append(ReferenceHint(kind="table", quote=match[0], number=match[2], page_number=int(page)))
        compound_spans.append(match.span())
    for kind, pattern in LABELS.items():
        for match in re.finditer(pattern, local):
            if any(start <= match.start() < end for start, end in compound_spans):
                continue
            hints.append(ReferenceHint(kind=kind, quote=match[0], number=match[1]))
    for match in re.finditer(r"[上下](?:表|图|文)", local):
        kind = {"表": "table", "图": "figure", "文": "context"}[match[0][-1]]
        hints.append(ReferenceHint(kind=kind, quote=match[0], direction="next" if match[0][0] == "下" else "previous"))
    return hints[:MAX_REFERENCES]


def _number(value: Any) -> str:
    raw = str(value or "").strip().replace("．", ".")
    normalized = clause_number_from_label(raw)
    return str(normalized) if normalized is not None else raw.casefold()


def _label_numbers(text: str, kind: str) -> set[str]:
    if kind not in LABELS:
        return set()
    return {_number(match[1]) for line in text.splitlines()
            if (match := re.match(rf"^\s*{LABELS[kind]}", line))}


def _locator_text(chunk: Any, data: dict[str, Any], kind: str) -> str:
    keys = {"table": ("table_title",), "attachment": ("attachment_title",),
            "figure": ("figure_title",), "section": ("section_title",)}.get(kind, ())
    parts = [str(chunk.section_title or ""), *(str(data.get(key) or "") for key in keys)]
    path = data.get("section_path")
    if isinstance(path, list):
        parts.extend(item for item in path if isinstance(item, str))
    # Only caption/heading lines, never an arbitrary mention of the target.
    pattern = LABELS.get(kind)
    if pattern:
        parts.extend(line.strip() for line in _text(chunk).splitlines() if re.match(rf"^\s*{pattern}", line))
    return "\n".join(parts)


def _is_kind(chunk: Any, data: dict[str, Any], kind: str) -> bool:
    if kind == "table":
        return bool(data.get("table_id") or data.get("table_chunk_role") or data.get("chunk_type") == "table"
                    or "\n|" in _text(chunk) or _text(chunk).startswith("表格"))
    if kind in {"figure", "attachment"}:
        return bool(data.get(f"{kind}_number") or data.get(f"{kind}_title")
                    or _label_numbers(_locator_text(chunk, data, kind), kind))
    return True


def resolve_references(anchor: Any, chunks: list[Any], hints: list[ReferenceHint]) -> list[ResolvedReference]:
    """Resolve bounded one-hop edges in the source document, with no guessed IDs."""
    ordered = [chunk for chunk in chunks if chunk.document_id == anchor.document_id]
    if anchor.index_status != "indexed" or not any(chunk.chunk_id == anchor.chunk_id for chunk in ordered):
        return []
    anchor_index = next(i for i, chunk in enumerate(ordered) if chunk.chunk_id == anchor.chunk_id)
    hits: list[ResolvedReference] = []
    seen = {anchor.chunk_id}
    for hint in hints[:MAX_REFERENCES]:
        if hint.document_title or not _grounded_hint(hint, _text(anchor)):
            continue
        anchor_data = metadata(anchor)
        table_context = anchor_data.get("table_context")
        if (hint.kind == "table" and hint.direction and isinstance(table_context, str)
                and hint.quote in table_context and _is_kind(anchor, anchor_data, "table")):
            # The parser already merged “see below” and that very table.
            continue
        candidates: list[tuple[int, Any, dict[str, Any]]] = []
        for index, chunk in enumerate(ordered):
            if chunk.chunk_id == anchor.chunk_id:
                continue
            data = metadata(chunk)
            if hint.page_number is not None and chunk.page_number != hint.page_number:
                continue
            if hint.direction == "next" and index <= anchor_index:
                continue
            if hint.direction == "previous" and index >= anchor_index:
                continue
            if hint.kind == "page":
                target_page = hint.page_number or (_number(hint.number) if hint.number else None)
                if target_page is None or str(chunk.page_number) != str(target_page):
                    continue
            elif hint.kind == "context":
                if not hint.direction or abs(index - anchor_index) != 1:
                    continue
            else:
                if not _is_kind(chunk, data, hint.kind):
                    continue
                locators = _locator_text(chunk, data, hint.kind)
                if hint.number:
                    numbers = _label_numbers(locators, hint.kind)
                    for key in {"clause": ("article_number", "section_number"), "section": ("section_number",), "table": ("table_number",),
                                "attachment": ("attachment_number",), "figure": ("figure_number",)}.get(hint.kind, ()):
                        value = data.get(key)
                        if value:
                            numbers.add(_number(value))
                            numbers.update(_label_numbers(str(value), hint.kind))
                    if hint.kind == "section":
                        heading_number = re.match(r"^\s*(\d+(?:\.\d+)*)[.、\s]", str(chunk.section_title or ""))
                        if heading_number:
                            numbers.add(heading_number[1])
                    if _number(hint.number) not in numbers:
                        continue
                if hint.title and hint.title not in locators:
                    continue
                if not hint.number and not hint.title and not hint.direction:
                    continue
            candidates.append((index, chunk, data))
        if hint.direction:
            candidates.sort(key=lambda item: abs(item[0] - anchor_index))
            # Direction means the nearest matching object; include split rows of it.
            if candidates:
                nearest = candidates[0]
                object_id = nearest[2].get("table_id") if hint.kind == "table" else None
                candidates = [item for item in candidates if object_id and item[2].get("table_id") == object_id] if object_id else [nearest]
        elif hint.kind not in {"page", "context"} and len(candidates) > 1:
            # Split chunks can share a locator. Repeated numbered objects in
            # different sections/pages without a shared identity are ambiguous.
            identities = set()
            for _, chunk, data in candidates:
                path = data.get("section_path")
                if hint.kind == "table":
                    identity = data.get("table_id") or (chunk.page_number, chunk.section_title, data.get("table_title"))
                else:
                    identity = tuple(path) if isinstance(path, list) and all(isinstance(p, str) for p in path) and path else (
                        chunk.section_title or (chunk.page_number, _text(chunk).partition("\n")[0]))
                identities.add(identity)
            if len(identities) > 1:
                candidates = []
        candidates = [item for item in candidates if item[1].index_status == "indexed"]
        if not candidates:
            event("reference_resolution", source=anchor.chunk_id, reference_kind=hint.kind, status="unresolved")
        for _, chunk, _ in candidates[:MAX_TARGETS]:
            if chunk.chunk_id not in seen:
                hits.append(ResolvedReference(chunk, hint, "stored_locator"))
                seen.add(chunk.chunk_id)
        if len(hits) >= MAX_TARGETS:
            break
    return hits[:MAX_TARGETS]
