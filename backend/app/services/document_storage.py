from datetime import datetime
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import DOCUMENT_DIR, SUPPORTED_DOCUMENT_EXTENSIONS, SUPPORTED_DOCUMENT_MIME_TYPES
from backend.app.models.document import Document


class UnsupportedDocumentTypeError(ValueError):
    pass


class EmptyDocumentError(ValueError):
    pass


def next_document_id(db: Session) -> str:
    today = datetime.now().strftime("%Y%m%d")
    prefix = f"DOC-{today}-"
    statement = select(Document).where(Document.document_id.like(f"{prefix}%"))
    count = len(list(db.scalars(statement)))
    return f"{prefix}{count + 1:04d}"


def save_original_document(document_id: str, filename: str, content: bytes, content_type: str | None = None) -> Path:
    if not content:
        raise EmptyDocumentError("上传文档不能为空")

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise UnsupportedDocumentTypeError("仅支持 .txt、.md、.docx、.pdf 文档")
    _validate_mime_type(suffix, content_type)
    _validate_file_content(suffix, content)

    DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
    storage_path = DOCUMENT_DIR / f"{document_id}{suffix}"
    storage_path.write_bytes(content)
    return storage_path


def _validate_mime_type(suffix: str, content_type: str | None) -> None:
    if not content_type:
        return

    normalized = content_type.split(";", maxsplit=1)[0].strip().lower()
    allowed = SUPPORTED_DOCUMENT_MIME_TYPES[suffix]
    if normalized not in allowed:
        raise UnsupportedDocumentTypeError(f"文件 MIME 类型与扩展名不匹配：{content_type}")


def _validate_file_content(suffix: str, content: bytes) -> None:
    if suffix == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise UnsupportedDocumentTypeError("PDF 文件内容校验失败")
        return

    if suffix == ".docx":
        if not content.startswith(b"PK"):
            raise UnsupportedDocumentTypeError("DOCX 文件内容校验失败")
        try:
            with ZipFile(BytesIO(content)) as archive:
                names = set(archive.namelist())
        except BadZipFile as exc:
            raise UnsupportedDocumentTypeError("DOCX 文件内容校验失败") from exc
        if "[Content_Types].xml" not in names or "word/document.xml" not in names:
            raise UnsupportedDocumentTypeError("DOCX 文件内容校验失败")
        return

    if suffix in {".txt", ".md"}:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise UnsupportedDocumentTypeError("文本文件必须使用 UTF-8 编码") from exc
        if "\x00" in text:
            raise UnsupportedDocumentTypeError("文本文件内容校验失败")
