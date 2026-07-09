from pathlib import Path
import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import DOCUMENT_DIR, INDEX_VERSION, SUPPORTED_DOCUMENT_EXTENSIONS
from backend.app.models.document import Document, DocumentChunk
from backend.app.services.chunk_service import build_chunks_from_parsed
from backend.app.services.document_parser import DocumentParseError, parse_document
from backend.app.services.vector_store_service import VectorStoreError, delete_document_vectors
from backend.app.services.vector_store_service import count_document_vectors


class DocumentDeletionError(RuntimeError):
    pass


def original_file_exists(document: Document) -> bool:
    return resolve_original_path(document).exists()


def resolve_original_path(document: Document) -> Path:
    storage_path = Path(document.storage_path)
    if storage_path.exists():
        return storage_path
    fallback = DOCUMENT_DIR / storage_path.name
    return fallback


def delete_document_record(db: Session, document: Document) -> None:
    """Delete Qdrant vectors first, then original file and SQLite rows.

    Qdrant is not transactional with SQLite. If vector cleanup fails, keep the
    document visible as delete_failed so the user can retry instead of leaving
    hidden orphan vectors behind.
    """

    chunks = db.scalars(select(DocumentChunk).where(DocumentChunk.document_id == document.document_id)).all()
    document.status = "deleting"
    document.index_error = None
    for chunk in chunks:
        chunk.index_status = "deleting"
    db.commit()

    try:
        delete_document_vectors(document.document_id, chunk_ids=[chunk.chunk_id for chunk in chunks])
    except VectorStoreError as exc:
        _mark_delete_failed(db, document, chunks, str(exc))
        raise DocumentDeletionError(str(exc)) from exc

    storage_path = Path(document.storage_path)
    if storage_path.exists():
        storage_path.unlink()

    db.delete(document)
    db.commit()


def remove_documents_with_missing_originals(db: Session) -> list[tuple[str, str]]:
    removed: list[tuple[str, str]] = []
    documents = db.scalars(select(Document).order_by(Document.uploaded_at.asc())).all()
    for document in documents:
        if original_file_exists(document):
            continue
        document_id = document.document_id
        document.status = "source_missing"
        document.index_error = "原始文件缺失，已暂停该文档参与知识库问答；请恢复文件或显式删除文档。"
        for chunk in db.scalars(select(DocumentChunk).where(DocumentChunk.document_id == document.document_id)).all():
            chunk.index_status = "source_missing"
        removed.append((document_id, "marked_source_missing"))
    if removed:
        db.commit()
    return removed


def recover_documents_from_originals(db: Session) -> list[tuple[str, str]]:
    recovered: list[tuple[str, str]] = []
    if not DOCUMENT_DIR.exists():
        return recovered

    existing_ids = set(db.scalars(select(Document.document_id)).all())
    for path in sorted(DOCUMENT_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_DOCUMENT_EXTENSIONS:
            continue
        document_id = path.stem
        if not document_id.startswith("DOC-") or document_id in existing_ids:
            continue
        try:
            parsed = parse_document(path)
            chunks = build_chunks_from_parsed(document_id=document_id, parsed=parsed, source_file=path.name)
        except DocumentParseError as exc:
            recovered.append((document_id, f"parse_failed: {exc}"))
            continue
        if not chunks:
            recovered.append((document_id, "empty"))
            continue

        vector_count = 0
        vector_error: str | None = None
        try:
            vector_count = count_document_vectors(document_id)
        except VectorStoreError as exc:
            vector_error = str(exc)
        indexed = vector_count >= len(chunks)
        document = Document(
            document_id=document_id,
            filename=path.name,
            content_type=None,
            file_type=path.suffix.lower().lstrip("."),
            size=path.stat().st_size,
            storage_path=str(path),
            status="indexed" if indexed else "uploaded",
            index_version=INDEX_VERSION,
            index_error=None
            if indexed
            else f"从原始文件恢复 SQLite 记录，向量索引需重建。{vector_error or ''}".strip(),
            chunk_count=len(chunks),
        )
        db.add(document)
        for chunk in chunks:
            db.add(
                DocumentChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=document_id,
                    text=chunk.text,
                    embedding_text=chunk.embedding_text,
                    chunk_metadata=json.dumps(chunk.metadata or {}, ensure_ascii=False),
                    token_count=chunk.token_count,
                    index_status="indexed" if indexed else "uploaded",
                    index_version=INDEX_VERSION,
                    title=chunk.title,
                    section_title=chunk.section_title,
                    page_number=chunk.page_number,
                    source_file=path.name,
                )
            )
        existing_ids.add(document_id)
        recovered.append((document_id, "indexed" if indexed else "uploaded"))
    if recovered:
        db.commit()
    return recovered


def _mark_delete_failed(db: Session, document: Document, chunks: list[DocumentChunk], error: str) -> None:
    document.status = "delete_failed"
    document.index_error = f"Qdrant 向量删除失败：{error}"[:1000]
    for chunk in chunks:
        chunk.index_status = "delete_failed"
    db.commit()
