from pathlib import Path
import re
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


def parse_document(
    file_path: Path,
    loader_name: str | None = None,
    source_name: str | None = None,
) -> ParsedDocument:
    """Extract structured text blocks through the configured loader adapter chain."""

    errors: list[str] = []
    for loader in _candidate_loaders(file_path, loader_name):
        try:
            result = loader.load(file_path)
            blocks = _ensure_blocks(result.blocks)
            if source_name:
                _apply_source_name(blocks, result, source_name)
            return ParsedDocument(
                text=_join_blocks(blocks),
                blocks=blocks,
                metadata={
                    **result.metadata,
                    "source_format": file_path.suffix.lower().lstrip("."),
                    "parser_version": PARSER_VERSION,
                    "loader_name": result.loader_name,
                    "block_count": len(blocks),
                    "source_filename": source_name or file_path.name,
                },
            )
        except DocumentParseError as exc:
            errors.append(f"{loader.name}: {exc}")
            if loader_name is not None:
                break

    if errors:
        raise DocumentParseError("；".join(errors))
    raise DocumentParseError("暂不支持该文档格式")


def _apply_source_name(blocks: list[ParsedBlock], result: LoaderResult, source_name: str) -> None:
    source_path = Path(source_name)
    stem = source_path.stem
    source_title = stem.split("_", maxsplit=1)[-1] if "_" in stem else stem
    result.metadata["source_title"] = source_title
    result.metadata["source_filename"] = source_name
    year_match = re.search(r"(20\d{2})年", source_name)
    month_match = re.search(r"(?:20\d{2}年)?(\d{1,2})月", source_name)
    quarter_match = re.search(r"([一二三四1-4])季度|([1-4])季", source_name)
    quarter_value = None
    if quarter_match:
        raw = quarter_match.group(1) or quarter_match.group(2)
        quarter_value = {"一": 1, "二": 2, "三": 3, "四": 4}.get(raw, int(raw) if raw.isdigit() else None)
    for block in blocks:
        if block.metadata.get("spreadsheet_table"):
            block.metadata["source_title"] = source_title
            block.metadata["source_filename"] = source_name
            block.metadata["inferred_year"] = int(year_match.group(1)) if year_match else None
            block.metadata["inferred_month"] = int(month_match.group(1)) if month_match else None
            block.metadata["inferred_quarter"] = quarter_value
            period = block.metadata.get("period") if isinstance(block.metadata.get("period"), dict) else {}
            if year_match:
                period["year"] = int(year_match.group(1))
            if month_match:
                period["month"] = int(month_match.group(1))
            if quarter_value:
                period["quarter"] = quarter_value
            block.metadata["period"] = period


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
            from docx.table import Table
            from docx.text.paragraph import Paragraph
        except ImportError as exc:
            raise DocumentParseError("缺少 python-docx 依赖，无法解析 docx") from exc

        try:
            document = DocxDocument(str(file_path))
        except Exception as exc:
            raise DocumentParseError("docx 文档解析失败") from exc

        blocks: list[ParsedBlock] = []
        current_section: str | None = None

        table_index = 0
        for child in document.element.body.iterchildren():
            if child.tag.endswith("}p"):
                paragraph = Paragraph(child, document)
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
                continue

            if child.tag.endswith("}tbl"):
                table = Table(child, document)
                table_text = _docx_table_to_text(table)
                if table_text:
                    table_index += 1
                    blocks.append(
                        ParsedBlock(
                            text=table_text,
                            block_type="table",
                            order_index=len(blocks) + 1,
                            section_title=current_section,
                            metadata=_table_metadata(table_text, table_index),
                        )
                    )
        return LoaderResult(blocks=blocks, loader_name=self.name)


class LegacyDocDocumentLoader:
    name = "libreoffice-doc"

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".doc"

    def load(self, file_path: Path) -> LoaderResult:
        from backend.app.services.office_conversion import (
            OfficeConversionError,
            cleanup_conversion_output,
            convert_with_libreoffice_detailed,
        )

        try:
            conversion = convert_with_libreoffice_detailed(file_path, "docx")
        except OfficeConversionError as exc:
            raise DocumentParseError(str(exc)) from exc

        try:
            result = DocxDocumentLoader().load(conversion.path)
        finally:
            cleanup_conversion_output(conversion.path)
        return LoaderResult(
            blocks=result.blocks,
            loader_name=self.name,
            metadata={
                "converted_from": "doc",
                "conversion_loader": result.loader_name,
                "parser_backend": "libreoffice",
                "degraded": False,
                "degradation_reason": None,
                "conversion_elapsed_ms": conversion.elapsed_ms,
            },
        )


class AntiwordDocDocumentLoader:
    name = "antiword-doc"

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".doc"

    def load(self, file_path: Path) -> LoaderResult:
        import shutil
        import subprocess
        from time import perf_counter

        executable = shutil.which("antiword")
        if not executable:
            raise DocumentParseError("antiword executable not found")
        started = perf_counter()
        try:
            completed = subprocess.run(
                [executable, "-m", "UTF-8.txt", str(file_path)],
                check=False,
                capture_output=True,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DocumentParseError(f"antiword extraction failed: {exc}") from exc
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise DocumentParseError(f"antiword extraction failed: {detail or completed.returncode}")
        text = completed.stdout.decode("utf-8", errors="replace")
        normalized = _normalize_text(text)
        if not normalized:
            raise DocumentParseError("antiword did not produce text")
        blocks = [
            ParsedBlock(text=paragraph, block_type="paragraph", order_index=index)
            for index, paragraph in enumerate(normalized.split("\n\n"), start=1)
            if paragraph.strip()
        ]
        return LoaderResult(
            blocks=blocks,
            loader_name=self.name,
            metadata={
                "parser_backend": "antiword",
                "degraded": True,
                "degradation_reason": "LibreOffice conversion unavailable or failed; text-only extraction used",
                "conversion_elapsed_ms": round((perf_counter() - started) * 1000, 2),
            },
        )


class SpreadsheetXlsxLoader:
    name = "spreadsheet-xlsx"

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".xlsx"

    def load(self, file_path: Path) -> LoaderResult:
        from backend.app.services.spreadsheet_parser import load_xlsx_workbook

        return load_xlsx_workbook(file_path, loader_name=self.name)


class SpreadsheetXlsLoader:
    name = "spreadsheet-xls"

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".xls"

    def load(self, file_path: Path) -> LoaderResult:
        from backend.app.services.spreadsheet_parser import load_xls_workbook

        return load_xls_workbook(file_path)


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
        "libreoffice-doc": LegacyDocDocumentLoader(),
        "antiword-doc": AntiwordDocDocumentLoader(),
        "python-docx": DocxDocumentLoader(),
        "spreadsheet-xlsx": SpreadsheetXlsxLoader(),
        "spreadsheet-xls": SpreadsheetXlsLoader(),
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
    table_index = 0

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
        nonlocal table_lines, table_index
        table = _normalize_text("\n".join(table_lines))
        if table:
            table_index += 1
            blocks.append(
                ParsedBlock(
                    text=table,
                    block_type="table",
                    order_index=len(blocks) + 1,
                    page_number=page_number,
                    section_title=current_section,
                    metadata=_table_metadata(table, table_index),
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


def _table_metadata(table_text: str, table_index: int | None = None) -> dict[str, object]:
    rows = _parse_markdown_table(table_text)
    metadata: dict[str, object] = {
        "raw_table_text": table_text,
    }
    if table_index is not None:
        metadata["table_index"] = table_index
    if not rows:
        return metadata

    headers = [_normalize_header(cell, index) for index, cell in enumerate(rows[0], start=1)]
    data_rows = rows[1:]
    metadata["headers"] = headers
    metadata["rows"] = [
        {
            "row_index": row_index,
            "cells": _row_cells(headers, row),
        }
        for row_index, row in enumerate(data_rows, start=1)
    ]
    return metadata


def _parse_markdown_table(table_text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in table_text.splitlines():
        stripped = line.strip()
        if not _looks_like_markdown_table_line(stripped) or _is_markdown_separator_line(stripped):
            continue
        rows.append(_markdown_table_cells(stripped))
    return rows


def _markdown_table_cells(line: str) -> list[str]:
    return [_normalize_text(cell.replace("\\|", "|")) for cell in line.strip("|").split("|")]


def _is_markdown_separator_line(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip("|").split("|")]
    return bool(cells) and all(cell and set(cell) <= {"-", ":"} for cell in cells)


def _normalize_header(header: str, index: int) -> str:
    cleaned = _normalize_text(header)
    return cleaned or f"列{index}"


def _row_cells(headers: list[str], row: list[str]) -> dict[str, str]:
    width = max(len(headers), len(row))
    padded_headers = headers + [f"列{index}" for index in range(len(headers) + 1, width + 1)]
    padded_row = row + [""] * (width - len(row))
    return {
        header: cell
        for header, cell in zip(padded_headers, padded_row, strict=True)
        if cell.strip()
    }


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
