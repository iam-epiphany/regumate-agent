from pathlib import Path
from typing import Protocol

from backend.app.core.config import DOCUMENT_LOADER_ORDER
from backend.app.services.document_types import LoaderResult, PARSER_VERSION, ParsedBlock, ParsedDocument


class DocumentParseError(ValueError):
    pass


class DocumentLoader(Protocol):
    name: str

    def supports(self, file_path: Path) -> bool: ...

    def load(self, file_path: Path) -> LoaderResult: ...


def parse_document_text(file_path: Path) -> str:
    """Compatibility wrapper for callers that still need a plain text string."""

    return parse_document(file_path).text


def parse_document(file_path: Path, loader_name: str | None = None) -> ParsedDocument:
    """Extract structured text blocks through the configured loader adapter chain."""

    errors: list[str] = []
    for loader in _candidate_loaders(file_path, loader_name):
        try:
            result = loader.load(file_path)
            blocks = _ensure_blocks(result.blocks)
            return ParsedDocument(
                text=_join_blocks(blocks),
                blocks=blocks,
                metadata={
                    "source_format": file_path.suffix.lower().lstrip("."),
                    "parser_version": PARSER_VERSION,
                    "loader_name": result.loader_name,
                    "block_count": len(blocks),
                },
            )
        except DocumentParseError as exc:
            errors.append(f"{loader.name}: {exc}")
            if loader_name is not None:
                break

    if errors:
        raise DocumentParseError("；".join(errors))
    raise DocumentParseError("暂不支持该文档格式")


def parse_document_with_loader(file_path: Path, loader_name: str) -> ParsedDocument:
    """Run a specific loader by name for loader evaluation."""

    return parse_document(file_path=file_path, loader_name=loader_name)


def available_loader_names(file_path: Path) -> list[str]:
    return [loader.name for loader in _candidate_loaders(file_path)]


class TextDocumentLoader:
    name = "text"

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".txt"

    def load(self, file_path: Path) -> LoaderResult:
        text = _read_utf8_text(file_path)
        normalized = _normalize_text(text)
        blocks = [ParsedBlock(text=normalized, block_type="paragraph", order_index=1)] if normalized else []
        return LoaderResult(blocks=blocks, loader_name=self.name)


class MarkdownDocumentLoader:
    name = "markdown"

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".md"

    def load(self, file_path: Path) -> LoaderResult:
        text = _read_utf8_text(file_path)
        return LoaderResult(blocks=_markdown_to_blocks(text), loader_name=self.name)


class DocxDocumentLoader:
    name = "python-docx"

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".docx"

    def load(self, file_path: Path) -> LoaderResult:
        try:
            from docx import Document as DocxDocument
        except ImportError as exc:
            raise DocumentParseError("缺少 python-docx 依赖，无法解析 docx") from exc

        try:
            document = DocxDocument(str(file_path))
        except Exception as exc:
            raise DocumentParseError("docx 文档解析失败") from exc

        blocks: list[ParsedBlock] = []
        current_section: str | None = None

        for paragraph in document.paragraphs:
            text = _normalize_text(paragraph.text)
            if not text:
                continue

            style_name = (paragraph.style.name if paragraph.style is not None else "") or ""
            heading_level = _docx_heading_level(style_name)
            if heading_level is not None:
                current_section = text
                blocks.append(
                    ParsedBlock(
                        text=text,
                        block_type="heading",
                        order_index=len(blocks) + 1,
                        section_title=current_section,
                        level=heading_level,
                    )
                )
            else:
                blocks.append(
                    ParsedBlock(
                        text=text,
                        block_type="paragraph",
                        order_index=len(blocks) + 1,
                        section_title=current_section,
                    )
                )

        for table in document.tables:
            table_text = _docx_table_to_text(table)
            if table_text:
                blocks.append(
                    ParsedBlock(
                        text=table_text,
                        block_type="table",
                        order_index=len(blocks) + 1,
                        section_title=current_section,
                    )
                )
        return LoaderResult(blocks=blocks, loader_name=self.name)


class PyMuPDF4LLMLoader:
    name = "pymupdf4llm"

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf"

    def load(self, file_path: Path) -> LoaderResult:
        try:
            import pymupdf4llm
        except ImportError as exc:
            raise DocumentParseError("缺少 pymupdf4llm 依赖，无法使用版式感知 PDF loader") from exc

        try:
            pages = pymupdf4llm.to_markdown(
                str(file_path),
                page_chunks=True,
                use_ocr=False,
                show_progress=False,
                write_images=False,
                embed_images=False,
            )
        except Exception as exc:
            raise DocumentParseError("PyMuPDF4LLM 解析失败") from exc

        blocks: list[ParsedBlock] = []
        if isinstance(pages, str):
            blocks.extend(_markdown_to_blocks(pages))
        else:
            for index, page in enumerate(pages, start=1):
                metadata = page.get("metadata") or {}
                page_number = metadata.get("page_number") or index
                text = page.get("text") or ""
                blocks.extend(_markdown_to_blocks(text, page_number=int(page_number)))
        return LoaderResult(blocks=blocks, loader_name=self.name)


class DoclingDocumentLoader:
    name = "docling"

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in {".pdf", ".docx"}

    def load(self, file_path: Path) -> LoaderResult:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:
            raise DocumentParseError("缺少 docling 依赖，无法使用 Docling loader") from exc

        try:
            result = DocumentConverter().convert(str(file_path))
            markdown = result.document.export_to_markdown()
        except Exception as exc:
            raise DocumentParseError("Docling 解析失败") from exc

        return LoaderResult(blocks=_markdown_to_blocks(markdown), loader_name=self.name)


class UnstructuredDocumentLoader:
    name = "unstructured"

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in {".txt", ".md", ".docx", ".pdf"}

    def load(self, file_path: Path) -> LoaderResult:
        try:
            from unstructured.partition.auto import partition
        except ImportError as exc:
            raise DocumentParseError("缺少 unstructured 依赖，无法使用 Unstructured loader") from exc

        try:
            elements = partition(filename=str(file_path), strategy="fast")
        except Exception as exc:
            raise DocumentParseError("Unstructured 解析失败") from exc

        blocks: list[ParsedBlock] = []
        current_section: str | None = None
        for element in elements:
            text = _normalize_text(str(element))
            if not text:
                continue

            element_type = getattr(element, "category", None) or element.__class__.__name__
            metadata = getattr(element, "metadata", None)
            page_number = getattr(metadata, "page_number", None) if metadata is not None else None
            block_type = _unstructured_block_type(element_type)
            if block_type is None:
                continue
            if block_type == "heading":
                current_section = text

            blocks.append(
                ParsedBlock(
                    text=text,
                    block_type=block_type,
                    order_index=len(blocks) + 1,
                    page_number=page_number,
                    section_title=text if block_type == "heading" else current_section,
                    level=1 if block_type == "heading" else None,
                )
            )
        return LoaderResult(blocks=blocks, loader_name=self.name)


class PypdfDocumentLoader:
    name = "pypdf"

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf"

    def load(self, file_path: Path) -> LoaderResult:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise DocumentParseError("缺少 pypdf 依赖，无法解析 pdf") from exc

        try:
            reader = PdfReader(str(file_path))
        except Exception as exc:
            raise DocumentParseError("pdf 文档解析失败") from exc

        blocks: list[ParsedBlock] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = _normalize_text(page.extract_text() or "")
            if text:
                blocks.append(
                    ParsedBlock(
                        text=text,
                        block_type="page",
                        order_index=len(blocks) + 1,
                        page_number=page_number,
                    )
                )
        return LoaderResult(blocks=blocks, loader_name=self.name)


def _candidate_loaders(file_path: Path, loader_name: str | None = None) -> list[DocumentLoader]:
    suffix = file_path.suffix.lower()
    registry: dict[str, DocumentLoader] = {
        "text": TextDocumentLoader(),
        "markdown": MarkdownDocumentLoader(),
        "python-docx": DocxDocumentLoader(),
        "pymupdf4llm": PyMuPDF4LLMLoader(),
        "docling": DoclingDocumentLoader(),
        "unstructured": UnstructuredDocumentLoader(),
        "pypdf": PypdfDocumentLoader(),
    }
    loaders = [
        registry[name]
        for name in DOCUMENT_LOADER_ORDER.get(suffix, [])
        if name in registry and registry[name].supports(file_path)
    ]

    if loader_name is None:
        return loaders
    return [loader for loader in loaders if loader.name == loader_name]


def _read_utf8_text(file_path: Path) -> str:
    try:
        return file_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentParseError("文档编码不是 UTF-8") from exc


def _markdown_to_blocks(text: str, page_number: int | None = None) -> list[ParsedBlock]:
    blocks: list[ParsedBlock] = []
    current_section: str | None = None
    paragraph_lines: list[str] = []
    table_lines: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        paragraph = _normalize_text("\n".join(paragraph_lines))
        if paragraph:
            blocks.append(
                ParsedBlock(
                    text=paragraph,
                    block_type="paragraph",
                    order_index=len(blocks) + 1,
                    page_number=page_number,
                    section_title=current_section,
                )
            )
        paragraph_lines = []

    def flush_table() -> None:
        nonlocal table_lines
        table = _normalize_text("\n".join(table_lines))
        if table:
            blocks.append(
                ParsedBlock(
                    text=table,
                    block_type="table",
                    order_index=len(blocks) + 1,
                    page_number=page_number,
                    section_title=current_section,
                )
            )
        table_lines = []

    for line in text.splitlines():
        stripped = line.strip()
        heading_level = _markdown_heading_level(stripped)
        if heading_level is not None:
            flush_table()
            flush_paragraph()
            title = stripped.lstrip("#").strip()
            current_section = title
            blocks.append(
                ParsedBlock(
                    text=title,
                    block_type="heading",
                    order_index=len(blocks) + 1,
                    page_number=page_number,
                    section_title=title,
                    level=heading_level,
                )
            )
            continue

        if _looks_like_markdown_table_line(stripped):
            flush_paragraph()
            table_lines.append(line)
            continue

        if not stripped:
            flush_table()
            flush_paragraph()
            continue

        flush_table()
        paragraph_lines.append(line)

    flush_table()
    flush_paragraph()
    return blocks


def _markdown_heading_level(stripped_line: str) -> int | None:
    if not stripped_line.startswith("#") or len(stripped_line) > 120:
        return None
    marker = stripped_line.split(" ", maxsplit=1)[0]
    if not marker or any(char != "#" for char in marker) or len(marker) > 6:
        return None
    if len(stripped_line) == len(marker):
        return None
    return len(marker)


def _looks_like_markdown_table_line(stripped_line: str) -> bool:
    if "|" not in stripped_line:
        return False
    cells = [cell.strip() for cell in stripped_line.strip("|").split("|")]
    return len(cells) >= 2


def _unstructured_block_type(element_type: str) -> str | None:
    if element_type in {"Title", "Header"}:
        return "heading"
    if element_type == "Table":
        return "table"
    if element_type in {"Footer", "PageBreak"}:
        return None
    return "paragraph"


def _docx_heading_level(style_name: str) -> int | None:
    normalized = style_name.lower()
    if normalized.startswith("heading"):
        parts = normalized.split()
        if len(parts) > 1 and parts[-1].isdigit():
            return int(parts[-1])
        return 1
    return None


def _docx_table_to_text(table) -> str:
    rows: list[list[str]] = []
    for row in table.rows:
        cells = [_normalize_text(cell.text) for cell in row.cells]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""

    width = max(len(row) for row in rows)
    padded_rows = [row + [""] * (width - len(row)) for row in rows]
    header = padded_rows[0]
    separator = ["---"] * width
    body = padded_rows[1:]
    markdown_rows = [_markdown_table_row(header), _markdown_table_row(separator)]
    markdown_rows.extend(_markdown_table_row(row) for row in body)
    return "\n".join(markdown_rows).strip()


def _markdown_table_row(cells: list[str]) -> str:
    escaped = [cell.replace("|", "\\|").strip() for cell in cells]
    return "| " + " | ".join(escaped) + " |"


def _normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    compact_lines: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank:
                compact_lines.append("")
            previous_blank = True
            continue
        compact_lines.append(" ".join(line.split()))
        previous_blank = False
    return "\n".join(compact_lines).strip()


def _ensure_blocks(blocks: list[ParsedBlock]) -> list[ParsedBlock]:
    cleaned = [block for block in blocks if block.text.strip()]
    if not cleaned:
        raise DocumentParseError("文档没有可解析文本")
    for index, block in enumerate(cleaned, start=1):
        block.order_index = index
    return cleaned


def _join_blocks(blocks: list[ParsedBlock]) -> str:
    return "\n\n".join(block.text for block in blocks if block.text.strip()).strip()
