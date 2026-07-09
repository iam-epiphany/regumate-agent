from dataclasses import dataclass
import re
from typing import Any

from backend.app.core.config import (
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE,
    CHUNK_TARGET_TOKENS,
    SEMANTIC_BREAK_THRESHOLD,
)
from backend.app.services.embedding_service import cosine_similarity, embed_for_semantic_split
from backend.app.services.document_types import ParsedBlock, ParsedDocument


@dataclass
class ChunkDraft:
    chunk_id: str
    document_id: str
    text: str
    embedding_text: str
    token_count: int
    title: str | None
    section_title: str | None
    page_number: int | None
    chunk_type: str = "paragraph"
    section_path: list[str] | None = None
    section_number: str | None = None
    parent_section_number: str | None = None
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
    metadata: dict[str, Any] | None = None


def build_chunks(document_id: str, text: str) -> list[ChunkDraft]:
    """Compatibility wrapper for plain text callers."""

    parsed = ParsedDocument(
        text=text.strip(),
        blocks=[ParsedBlock(text=text.strip(), block_type="paragraph", order_index=1)] if text.strip() else [],
        metadata={"source_format": "plain_text", "parser_version": "plain-text-compat"},
    )
    return build_chunks_from_parsed(document_id=document_id, parsed=parsed)


def build_chunks_from_parsed(
    document_id: str,
    parsed: ParsedDocument,
    source_file: str | None = None,
) -> list[ChunkDraft]:
    """Build retrievable chunks from structured parser blocks."""

    source_format = _metadata_text(parsed.metadata.get("source_format"))
    groups = _group_blocks(parsed.blocks)
    chunks: list[ChunkDraft] = []
    for group in groups:
        for piece in _split_group(group, document_id=document_id):
            token_count = count_tokens(piece.text)
            chunks.append(
                ChunkDraft(
                    chunk_id=f"{document_id}-CHUNK-{len(chunks) + 1:04d}",
                    document_id=document_id,
                    text=piece.text,
                    embedding_text=build_contextual_embedding_text(
                        text=piece.text,
                        source_file=source_file,
                        source_format=source_format,
                        section_title=group.section_title,
                        section_path=group.section_path,
                        section_number=group.section_number,
                        parent_section_number=group.parent_section_number,
                        page_number=group.page_number,
                        chunk_type=group.block_type,
                        chunk_metadata=piece.metadata,
                    ),
                    token_count=token_count,
                    title=group.section_title,
                    section_title=group.section_title,
                    page_number=group.page_number,
                    chunk_type=group.block_type,
                    section_path=group.section_path,
                    section_number=group.section_number,
                    parent_section_number=group.parent_section_number,
                    metadata=piece.metadata,
                )
            )
    for previous, current, next_chunk in zip([None, *chunks[:-1]], chunks, [*chunks[1:], None], strict=True):
        current.previous_chunk_id = previous.chunk_id if previous else None
        current.next_chunk_id = next_chunk.chunk_id if next_chunk else None
    return chunks


@dataclass
class _ChunkGroup:
    text: str
    section_title: str | None
    page_number: int | None
    block_type: str = "paragraph"
    section_path: list[str] | None = None
    section_number: str | None = None
    parent_section_number: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class _ChunkPiece:
    text: str
    metadata: dict[str, Any]


def _group_blocks(blocks: list[ParsedBlock]) -> list[_ChunkGroup]:
    groups: list[_ChunkGroup] = []
    buffer: list[str] = []
    current_section: str | None = None
    current_path: list[str] = []
    current_section_number: str | None = None
    current_parent_section_number: str | None = None
    current_page: int | None = None
    has_body = False

    def flush() -> None:
        nonlocal buffer, has_body
        text = "\n\n".join(part for part in buffer if part.strip()).strip()
        if text and has_body:
            groups.append(
                _ChunkGroup(
                    text=text,
                    section_title=current_section,
                    page_number=current_page,
                    section_path=list(current_path),
                    section_number=current_section_number,
                    parent_section_number=current_parent_section_number,
                )
            )
        buffer = []
        has_body = False

    for block in blocks:
        page_changed = current_page is not None and block.page_number is not None and block.page_number != current_page
        if page_changed:
            flush()

        if block.block_type == "heading":
            if has_body:
                flush()
            current_section = block.text
            current_section_number = _section_number(block.text)
            current_parent_section_number = _parent_section_number(current_section_number)
            current_path = _updated_section_path(current_path, block.text, block.level, current_section_number)
            buffer = [block.text]
            current_page = block.page_number
            has_body = False
            continue

        if block.block_type == "table":
            table_context = ""
            if has_body and _should_merge_table_context(buffer):
                table_context = "\n\n".join(part for part in buffer if part.strip()).strip()
                buffer = []
                has_body = False
            else:
                flush()
            if block.section_title:
                current_section = block.section_title
                current_section_number = _section_number(block.section_title)
                current_parent_section_number = _parent_section_number(current_section_number)
                current_path = _updated_section_path(
                    current_path,
                    block.section_title,
                    block.level,
                    current_section_number,
                )
            if block.page_number is not None:
                current_page = block.page_number
            table_text = f"表格：\n{block.text}"
            if table_context:
                table_text = f"{table_context}\n\n{table_text}"
            groups.append(
                _ChunkGroup(
                    text=table_text,
                    section_title=current_section,
                    page_number=current_page,
                    block_type="table",
                    section_path=list(current_path),
                    section_number=current_section_number,
                    parent_section_number=current_parent_section_number,
                    metadata={
                        **(block.metadata or {}),
                        "table_context": table_context,
                        "table_title": _table_title(table_context, current_section),
                    },
                )
            )
            continue

        if block.section_title and block.section_title != current_section and has_body:
            flush()

        if block.section_title:
            current_section = block.section_title
            current_section_number = _section_number(block.section_title)
            current_parent_section_number = _parent_section_number(current_section_number)
            if not current_path or current_path[-1] != block.section_title:
                current_path = _updated_section_path(
                    current_path,
                    block.section_title,
                    block.level,
                    current_section_number,
                )
        if block.page_number is not None:
            current_page = block.page_number

        buffer.append(block.text)
        has_body = True

    flush()
    return groups


def _section_number(section_title: str | None) -> str | None:
    if not section_title:
        return None
    match = re.match(r"^\s*(\d+(?:\.\d+)*)[\.、\s]", section_title)
    return match.group(1) if match else None


def _parent_section_number(section_number: str | None) -> str | None:
    if not section_number or "." not in section_number:
        return None
    return section_number.rsplit(".", maxsplit=1)[0]


def _updated_section_path(
    current_path: list[str],
    title: str,
    level: int | None,
    section_number: str | None,
) -> list[str]:
    if level is None and section_number:
        level = section_number.count(".") + 1
    if level is None:
        return [title]
    prefix = current_path[: max(level - 1, 0)]
    return [*prefix, title]


def _should_merge_table_context(buffer: list[str]) -> bool:
    text = "\n\n".join(part for part in buffer if part.strip()).strip()
    return bool(text) and count_tokens(text) <= 120


def _split_group(group: _ChunkGroup, *, document_id: str) -> list[_ChunkPiece]:
    if group.block_type == "table":
        return _split_table_text(group, document_id=document_id)

    cleaned = group.text.strip()
    if not cleaned:
        return []
    if count_tokens(cleaned) <= CHUNK_MAX_TOKENS:
        return [_ChunkPiece(text=cleaned, metadata={})]

    paragraphs = [paragraph.strip() for paragraph in cleaned.split("\n\n") if paragraph.strip()]
    if len(paragraphs) >= 2:
        semantic_pieces = _split_by_semantic_breaks(paragraphs)
    else:
        semantic_pieces = [cleaned]

    pieces: list[str] = []
    for piece in semantic_pieces:
        pieces.extend(_split_by_token_limit(piece))
    return [_ChunkPiece(text=piece, metadata={}) for piece in _add_overlap([piece for piece in pieces if piece])]


def _split_by_semantic_breaks(paragraphs: list[str]) -> list[str]:
    vectors = embed_for_semantic_split(paragraphs)
    pieces: list[str] = []
    buffer: list[str] = []

    for index, paragraph in enumerate(paragraphs):
        if not buffer:
            buffer.append(paragraph)
            continue

        current_text = "\n\n".join(buffer)
        similarity = cosine_similarity(vectors[index - 1], vectors[index]) if index < len(vectors) else 1.0
        should_break = (
            count_tokens(current_text) >= CHUNK_TARGET_TOKENS
            or similarity < SEMANTIC_BREAK_THRESHOLD
            or count_tokens(f"{current_text}\n\n{paragraph}") > CHUNK_MAX_TOKENS
        )
        if should_break:
            pieces.append(current_text.strip())
            buffer = [paragraph]
        else:
            buffer.append(paragraph)

    if buffer:
        pieces.append("\n\n".join(buffer).strip())
    return pieces


def _split_by_token_limit(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    if count_tokens(cleaned) <= CHUNK_MAX_TOKENS:
        return [cleaned]

    pieces: list[str] = []
    buffer = ""
    for unit in _semantic_units(cleaned):
        if count_tokens(unit) > CHUNK_MAX_TOKENS:
            if buffer.strip():
                pieces.append(buffer.strip())
                buffer = ""
            pieces.extend(_hard_split_token_units(unit))
            continue

        candidate = f"{buffer}{unit}" if buffer else unit.lstrip()
        if count_tokens(candidate) <= CHUNK_MAX_TOKENS:
            buffer = candidate
            continue

        if buffer.strip():
            pieces.append(buffer.strip())
        buffer = unit.lstrip()

    if buffer.strip():
        pieces.append(buffer.strip())
    return [piece for piece in pieces if piece]


def _add_overlap(pieces: list[str]) -> list[str]:
    if len(pieces) <= 1:
        return pieces

    overlapped = [pieces[0]]
    for previous, current in zip(pieces, pieces[1:]):
        tail = _semantic_overlap_tail(previous)
        candidate = f"{tail}\n\n{current}".strip() if tail else current
        if count_tokens(candidate) > CHUNK_MAX_TOKENS:
            candidate = current
        overlapped.append(candidate.strip())
    return overlapped


def count_tokens(text: str) -> int:
    return len(_token_units(text))


def _token_units(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+|[^\s]", text)


def _join_token_units(tokens: list[str]) -> str:
    text = ""
    previous_ascii = False
    for token in tokens:
        current_ascii = bool(re.fullmatch(r"[a-zA-Z0-9_]+", token))
        if text and previous_ascii and current_ascii:
            text += " "
        text += token
        previous_ascii = current_ascii
    return text


def build_contextual_embedding_text(
    *,
    text: str,
    source_file: str | None = None,
    source_format: str | None = None,
    section_title: str | None,
    section_path: list[str] | None = None,
    section_number: str | None = None,
    parent_section_number: str | None = None,
    page_number: int | None = None,
    chunk_type: str = "paragraph",
    chunk_metadata: dict[str, Any] | None = None,
) -> str:
    """Prepend deterministic retrieval context while keeping chunk text unchanged."""

    chunk_metadata = chunk_metadata or {}
    labels: list[str] = []
    if source_file:
        labels.append(f"来源文件：{source_file}")
    if source_format:
        labels.append(f"文档格式：{source_format}")
    if section_path:
        labels.append("章节路径：" + " > ".join(section_path))
    if section_title:
        labels.append(f"章节：{section_title}")
    if section_number:
        labels.append(f"条款号：{section_number}")
    if parent_section_number:
        labels.append(f"父条款号：{parent_section_number}")
    if page_number is not None:
        labels.append(f"页码：{page_number}")
    if chunk_type == "table":
        labels.append("内容类型：表格")
        labels.append(_table_header_label(text))
        table_title = _metadata_text(chunk_metadata.get("table_title"))
        headers = [
            str(header)
            for header in chunk_metadata.get("table_headers") or chunk_metadata.get("headers") or []
            if str(header).strip()
        ]
        row_cells = chunk_metadata.get("row_cells") or {}
        labels.append("内容类型：表格")
        if table_title:
            labels.append(f"表格标题：{table_title}")
        if headers:
            labels.append("表头：" + "、".join(headers))
        if chunk_metadata.get("row_index") is not None:
            labels.append(f"表格行号：{chunk_metadata.get('row_index')}")
        if isinstance(row_cells, dict) and row_cells:
            labels.append("行数据：" + "；".join(f"{key}={value}" for key, value in row_cells.items()))
    else:
        labels.append("内容类型：正文")
    labels = [label for label in labels if label]
    label_text = "\n".join(labels)
    return f"{label_text}\n\n{text}" if labels else text


def _metadata_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _split_table_text(group: _ChunkGroup, *, document_id: str) -> list[_ChunkPiece]:
    cleaned = group.text.strip()
    if not cleaned:
        return []

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    table_indexes = [index for index, line in enumerate(lines) if _looks_like_table_line(line)]
    if not table_indexes:
        metadata = _base_table_metadata(group, document_id)
        return [_ChunkPiece(text=piece, metadata=metadata) for piece in _split_by_token_limit(cleaned)]

    first_table_index = table_indexes[0]
    prefix_lines = lines[:first_table_index]
    table_lines = lines[first_table_index:]
    header_count = 2 if len(table_lines) >= 2 and _is_table_separator_line(table_lines[1]) else 1
    header = table_lines[:header_count]
    rows = table_lines[header_count:]
    if not rows:
        metadata = _base_table_metadata(group, document_id)
        return [_ChunkPiece(text=piece, metadata=metadata) for piece in _split_by_token_limit(cleaned)]

    pieces: list[_ChunkPiece] = []
    header_cells = [_normalize_table_header(cell, index) for index, cell in enumerate(_table_cells(header[0]), start=1)]
    context = _table_context_text(prefix_lines)
    base_metadata = _base_table_metadata(group, document_id)
    base_metadata["table_headers"] = header_cells
    base_metadata["raw_table_preview"] = "\n".join(table_lines[: min(len(table_lines), 6)])
    data_rows = [row for row in rows if not _is_table_separator_line(row)]

    if data_rows:
        pieces.append(
            _ChunkPiece(
                text=_table_summary_text(context, base_metadata, len(data_rows)),
                metadata={**base_metadata, "table_chunk_role": "summary", "row_index": None},
            )
        )

    for row_index, row in enumerate(data_rows, start=1):
        row_pairs = _table_row_pairs(header_cells, _table_cells(row))
        row_metadata = {
            **base_metadata,
            "table_chunk_role": "row",
            "row_index": row_index,
            "row_cells": {header: cell for header, cell in row_pairs},
        }
        row_text = _table_row_to_evidence(context, base_metadata, row_pairs)
        pieces.append(_ChunkPiece(text=row_text, metadata=row_metadata))
    return [piece for piece in pieces if piece.text.strip()]


def _table_context_text(prefix_lines: list[str]) -> str:
    context_lines = [line for line in prefix_lines if line != "表格："]
    return "\n\n".join(context_lines).strip()


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip("|").split("|")]


def _table_row_to_evidence(context: str, headers: list[str], cells: list[str]) -> str:
    width = max(len(headers), len(cells))
    padded_headers = headers + [f"列{index + 1}" for index in range(len(headers), width)]
    padded_cells = cells + [""] * (width - len(cells))
    pairs = [
        (header, cell)
        for header, cell in zip(padded_headers, padded_cells, strict=True)
        if header.strip() and cell.strip()
    ]
    if not pairs:
        return context

    if len(pairs) == 2:
        row_sentence = f"表格行证据：{pairs[0][0]}为“{pairs[0][1]}”时，{pairs[1][0]}为“{pairs[1][1]}”。"
    else:
        row_sentence = "表格行证据：" + "；".join(f"{header}为“{cell}”" for header, cell in pairs) + "。"
    if context:
        return f"{context}\n\n{row_sentence}"
    return row_sentence


def _normalize_table_header(header: str, index: int) -> str:
    cleaned = header.strip()
    return cleaned or f"列{index}"


def _table_row_pairs(headers: list[str], cells: list[str]) -> list[tuple[str, str]]:
    width = max(len(headers), len(cells))
    padded_headers = headers + [f"列{index}" for index in range(len(headers) + 1, width + 1)]
    padded_cells = cells + [""] * (width - len(cells))
    return [
        (header, cell)
        for header, cell in zip(padded_headers, padded_cells, strict=True)
        if header.strip() and cell.strip()
    ]


def _table_row_to_evidence(
    context: str,
    table_metadata: dict[str, Any],
    pairs: list[tuple[str, str]],
) -> str:
    if not pairs:
        return context

    table_title = _metadata_text(table_metadata.get("table_title"))
    title_text = f"《{table_title}》" if table_title else "该表"
    if len(pairs) == 2:
        row_sentence = f"表格行证据：在{title_text}中，{pairs[0][0]}为“{pairs[0][1]}”时，{pairs[1][0]}为“{pairs[1][1]}”。"
    else:
        row_sentence = f"表格行证据：在{title_text}中，" + "；".join(f"{header}=“{cell}”" for header, cell in pairs) + "。"
    if context:
        return f"{context}\n\n{row_sentence}"
    return row_sentence


def _table_summary_text(context: str, table_metadata: dict[str, Any], row_count: int) -> str:
    table_title = _metadata_text(table_metadata.get("table_title")) or "未命名表格"
    headers = [str(header) for header in table_metadata.get("table_headers") or [] if str(header).strip()]
    header_text = "、".join(headers) if headers else "未识别表头"
    summary = f"表格摘要：{table_title}。表头：{header_text}。共{row_count}行数据。"
    return f"{context}\n\n{summary}" if context else summary


def _base_table_metadata(group: _ChunkGroup, document_id: str) -> dict[str, Any]:
    source = dict(group.metadata or {})
    table_index = int(source.get("table_index") or 1)
    table_id = source.get("table_id") or f"{document_id}-TABLE-{table_index:04d}"
    table_title = _metadata_text(source.get("table_title")) or _metadata_text(group.section_title) or "未命名表格"
    raw_table_text = _metadata_text(source.get("raw_table_text")) or group.text
    return {
        **source,
        "table_id": table_id,
        "table_title": table_title,
        "raw_table_text": raw_table_text,
        "raw_table_preview": raw_table_text[:500],
    }


def _table_title(context: str, section_title: str | None) -> str | None:
    for line in reversed([line.strip() for line in context.splitlines() if line.strip()]):
        if _looks_like_table_title(line):
            return line.rstrip("：:")
    return section_title


def _looks_like_table_title(text: str) -> bool:
    compact = text.strip()
    if len(compact) > 80:
        return False
    return bool(re.search(r"(表|附表|指标表)", compact))


def _looks_like_table_line(line: str) -> bool:
    if "|" not in line:
        return False
    cells = [cell.strip() for cell in line.strip("|").split("|")]
    return len(cells) >= 2


def _is_table_separator_line(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip("|").split("|")]
    return bool(cells) and all(cell and set(cell) <= {"-", ":"} for cell in cells)


def _table_header_label(text: str) -> str:
    if "表格行证据：" in text:
        return "内容形态：表格行级证据"
    for line in text.splitlines():
        if _looks_like_table_line(line) and not _is_table_separator_line(line):
            cells = [cell.strip() for cell in line.strip("|").split("|") if cell.strip()]
            if cells:
                return "表格字段：" + "、".join(cells)
    return ""


def _semantic_units(text: str) -> list[str]:
    units: list[str] = []
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    for paragraph_index, paragraph in enumerate(paragraphs):
        prefix = "\n\n" if paragraph_index > 0 else ""
        if count_tokens(paragraph) <= CHUNK_TARGET_TOKENS:
            units.append(f"{prefix}{paragraph}")
            continue

        sentences = _split_sentences(paragraph)
        if not sentences:
            units.append(f"{prefix}{paragraph}")
            continue
        for sentence_index, sentence in enumerate(sentences):
            sentence_prefix = prefix if sentence_index == 0 else ""
            units.append(f"{sentence_prefix}{sentence}")
    return units


def _split_sentences(text: str) -> list[str]:
    parts = re.findall(r".+?(?:[。！？!?；;：:]|$)", text, flags=re.S)
    return [part.strip() for part in parts if part.strip()]


def _hard_split_token_units(text: str) -> list[str]:
    tokens = _token_units(text)
    pieces: list[str] = []
    start = 0
    step = max(CHUNK_MAX_TOKENS - CHUNK_OVERLAP_TOKENS, 1)
    while start < len(tokens):
        prefix = "" if start == 0 else "（承接上文）"
        limit = CHUNK_MAX_TOKENS - count_tokens(prefix)
        piece = f"{prefix}{_join_token_units(tokens[start : start + limit])}".strip()
        pieces.append(piece)
        start += step
    return pieces


def _semantic_overlap_tail(text: str) -> str:
    sentences = _split_sentences(text.replace("\n\n", ""))
    if not sentences:
        return ""

    selected: list[str] = []
    for sentence in reversed(sentences):
        candidate = "".join([sentence, *selected])
        if count_tokens(candidate) > CHUNK_OVERLAP_TOKENS:
            break
        selected.insert(0, sentence)
    return "".join(selected).strip()


def _split_by_size(text: str) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= CHUNK_SIZE:
        return [cleaned]

    paragraph_pieces = _split_by_paragraph(cleaned)
    if all(len(piece) <= CHUNK_SIZE for piece in paragraph_pieces):
        return paragraph_pieces
    pieces: list[str] = []
    for piece in paragraph_pieces:
        pieces.extend(_split_long_text(piece))
    return [piece for piece in pieces if piece]


def _split_by_paragraph(text: str) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    pieces: list[str] = []
    buffer: list[str] = []

    for paragraph in paragraphs:
        candidate = "\n\n".join([*buffer, paragraph]).strip()
        if len(candidate) <= CHUNK_SIZE:
            buffer.append(paragraph)
            continue
        if buffer:
            pieces.append("\n\n".join(buffer).strip())
        buffer = [paragraph]

    if buffer:
        pieces.append("\n\n".join(buffer).strip())
    return pieces


def _split_long_text(text: str) -> list[str]:
    if len(text) <= CHUNK_SIZE:
        return [text]

    pieces = []
    start = 0
    step = max(CHUNK_SIZE - CHUNK_OVERLAP, 1)
    while start < len(text):
        pieces.append(text[start : start + CHUNK_SIZE].strip())
        start += step
    return [piece for piece in pieces if piece]
