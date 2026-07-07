from pathlib import Path


class DocumentParseError(ValueError):
    pass


def parse_document_text(file_path: Path) -> str:
    """Extract plain text from supported knowledge-base document formats."""

    suffix = file_path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return _parse_text_file(file_path)
    if suffix == ".docx":
        return _parse_docx_file(file_path)
    if suffix == ".pdf":
        return _parse_pdf_file(file_path)
    raise DocumentParseError("暂不支持该文档格式")


def _parse_text_file(file_path: Path) -> str:
    try:
        text = file_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentParseError("文档编码不是 UTF-8") from exc
    return _ensure_text(text)


def _parse_docx_file(file_path: Path) -> str:
    try:
        from docx import Document as DocxDocument
    except ImportError as exc:
        raise DocumentParseError("缺少 python-docx 依赖，无法解析 docx") from exc

    document = DocxDocument(str(file_path))
    text = "\n".join(paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip())
    return _ensure_text(text)


def _parse_pdf_file(file_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise DocumentParseError("缺少 pypdf 依赖，无法解析 pdf") from exc

    reader = PdfReader(str(file_path))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"第 {index} 页\n{text.strip()}")
    return _ensure_text("\n\n".join(pages))


def _ensure_text(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        raise DocumentParseError("文档没有可解析文本")
    return cleaned
