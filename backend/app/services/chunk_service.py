from dataclasses import dataclass

from backend.app.core.config import CHUNK_SIZE


@dataclass
class ChunkDraft:
    chunk_id: str
    text: str
    title: str | None
    section_title: str | None
    page_number: int | None


def build_chunks(document_id: str, text: str) -> list[ChunkDraft]:
    """Split text by Markdown-like headings first, then by fixed length."""

    sections = _split_sections(text)
    chunks: list[ChunkDraft] = []
    for section_title, section_text in sections:
        for piece in _split_by_size(section_text):
            chunks.append(
                ChunkDraft(
                    chunk_id=f"{document_id}-CHUNK-{len(chunks) + 1:04d}",
                    text=piece,
                    title=section_title,
                    section_title=section_title,
                    page_number=None,
                )
            )
    return chunks


def _split_sections(text: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, list[str]]] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") and len(stripped) <= 120:
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = stripped.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, current_lines))

    if not sections:
        return [(None, text)]
    return [(title, "\n".join(lines).strip()) for title, lines in sections if "\n".join(lines).strip()]


def _split_by_size(text: str) -> list[str]:
    cleaned = text.strip()
    if len(cleaned) <= CHUNK_SIZE:
        return [cleaned]

    pieces = []
    start = 0
    while start < len(cleaned):
        pieces.append(cleaned[start : start + CHUNK_SIZE].strip())
        start += CHUNK_SIZE
    return [piece for piece in pieces if piece]
