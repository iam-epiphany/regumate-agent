from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import DOCUMENT_DIR, SUPPORTED_DOCUMENT_EXTENSIONS
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


def save_original_document(document_id: str, filename: str, content: bytes) -> Path:
    if not content:
        raise EmptyDocumentError("上传文档不能为空")

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise UnsupportedDocumentTypeError("仅支持 .txt、.md、.docx、.pdf 文档")

    DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
    storage_path = DOCUMENT_DIR / f"{document_id}{suffix}"
    storage_path.write_bytes(content)
    return storage_path
