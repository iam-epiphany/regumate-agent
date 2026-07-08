from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.document import Document, DocumentChunk
from backend.app.services.vector_store_service import VectorStoreError, delete_document_vectors


class DocumentDeletionError(RuntimeError):
    pass


def original_file_exists(document: Document) -> bool:
    return Path(document.storage_path).exists()


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
        try:
            delete_document_record(db, document)
        except DocumentDeletionError as exc:
            removed.append((document_id, f"delete_failed: {exc}"))
            continue
        removed.append((document_id, "deleted"))
    return removed


def _mark_delete_failed(db: Session, document: Document, chunks: list[DocumentChunk], error: str) -> None:
    document.status = "delete_failed"
    document.index_error = f"Qdrant 向量删除失败：{error}"[:1000]
    for chunk in chunks:
        chunk.index_status = "delete_failed"
    db.commit()
