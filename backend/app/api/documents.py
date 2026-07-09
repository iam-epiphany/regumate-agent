from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import INDEX_VERSION
from backend.app.core.database import SessionLocal, get_db
from backend.app.models.document import Document, DocumentChunk
from backend.app.schemas.documents import (
    ChunkSummary,
    DocumentDeleteResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentSummary,
    DocumentUploadResponse,
)
from backend.app.services.audit_service import log_action
from backend.app.services.chunk_service import build_chunks_from_parsed
from backend.app.services.document_parser import DocumentParseError, parse_document
from backend.app.services.document_storage import (
    EmptyDocumentError,
    UnsupportedDocumentTypeError,
    next_document_id,
    save_original_document,
)
from backend.app.services.document_indexing_service import DocumentIndexingError, index_document
from backend.app.services.document_lifecycle_service import (
    DocumentDeletionError,
    delete_document_record,
    original_file_exists,
    recover_documents_from_originals,
    remove_documents_with_missing_originals,
)


router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> DocumentUploadResponse:
    filename = file.filename or "unknown"
    content = await file.read()
    document_id = next_document_id(db)

    try:
        storage_path = save_original_document(
            document_id=document_id,
            filename=filename,
            content=content,
            content_type=file.content_type,
        )
        parsed = parse_document(storage_path)
        chunks = build_chunks_from_parsed(document_id=document_id, parsed=parsed, source_file=filename)
    except UnsupportedDocumentTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DocumentParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not chunks:
        raise HTTPException(status_code=400, detail="文档没有可入库的文本片段")

    #创建文档元信息
    document = Document(
        document_id=document_id,
        filename=filename,
        content_type=file.content_type,
        file_type=Path(filename).suffix.lower().lstrip("."),
        size=len(content),
        storage_path=str(storage_path),
        status="uploaded",
        index_version=INDEX_VERSION,
        index_error=None,
        chunk_count=len(chunks),
    )
    db.add(document)
    chunk_models: list[DocumentChunk] = []
    for chunk in chunks:
        chunk_model = DocumentChunk(
            chunk_id=chunk.chunk_id,
            document_id=document_id,
            text=chunk.text,
            embedding_text=chunk.embedding_text,
            token_count=chunk.token_count,
            index_status="uploaded",
            index_version=INDEX_VERSION,
            title=chunk.title,
            section_title=chunk.section_title,
            page_number=chunk.page_number,
            source_file=filename,
        )
        chunk_models.append(chunk_model)
        db.add(chunk_model)
    db.commit()
    db.refresh(document)
    log_action(db, "document_uploaded", "document", document_id, filename)
    background_tasks.add_task(_index_document_background, document_id)

    return DocumentUploadResponse(
        document_id=document.document_id,
        filename=document.filename,
        content_type=document.content_type,
        size=document.size,
        chunk_count=document.chunk_count,
        uploaded_at=document.uploaded_at.isoformat(),
    )


@router.post("/{document_id}/index", response_model=DocumentDetailResponse)
def build_document_index(document_id: str, db: Session = Depends(get_db)) -> DocumentDetailResponse:
    document = db.scalar(select(Document).where(Document.document_id == document_id))
    if document is None:
        raise HTTPException(status_code=404, detail="未找到指定文档")

    try:
        index_document(db, document)
    except DocumentIndexingError as exc:
        log_action(db, "document_index_failed", "document", document_id, str(exc)[:500])
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    log_action(db, "document_indexed", "document", document_id, f"chunks={document.chunk_count}")
    db.refresh(document)
    return _to_detail(document, db)


@router.get("", response_model=DocumentListResponse)
def list_documents(db: Session = Depends(get_db)) -> DocumentListResponse:
    recovered_documents = recover_documents_from_originals(db)
    for document_id, result in recovered_documents:
        detail = f"original file recovered into SQLite; result: {result}"
        log_action(db, "document_recovered_from_original", "document", document_id, detail[:500])

    removed_documents = remove_documents_with_missing_originals(db)
    for document_id, result in removed_documents:
        detail = f"original file missing; result: {result}"
        log_action(db, "document_marked_source_missing", "document", document_id, detail[:500])

    statement = select(Document).order_by(Document.uploaded_at.desc())
    documents = db.scalars(statement).all()
    return DocumentListResponse(documents=[_to_summary(document) for document in documents])


@router.get("/{document_id}", response_model=DocumentDetailResponse)
def get_document(document_id: str, db: Session = Depends(get_db)) -> DocumentDetailResponse:
    document = db.scalar(select(Document).where(Document.document_id == document_id))
    if document is None:
        raise HTTPException(status_code=404, detail="未找到指定文档")
    if not original_file_exists(document):
        document.status = "source_missing"
        document.index_error = "原始文件缺失，已暂停该文档参与知识库问答；请恢复文件或显式删除文档。"
        chunks = db.scalars(select(DocumentChunk).where(DocumentChunk.document_id == document.document_id)).all()
        for chunk in chunks:
            chunk.index_status = "source_missing"
        db.commit()
        log_action(db, "document_marked_source_missing", "document", document_id, "original file missing")
        db.refresh(document)

    return _to_detail(document, db)


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
def delete_document(document_id: str, db: Session = Depends(get_db)) -> DocumentDeleteResponse:
    document = db.scalar(select(Document).where(Document.document_id == document_id))
    if document is None:
        raise HTTPException(status_code=404, detail="未找到指定文档")
    if document.status == "indexing":
        raise HTTPException(status_code=409, detail="文档正在构建向量索引，请等待索引完成后再删除")
    if document.status == "deleting":
        raise HTTPException(status_code=409, detail="文档正在删除，请稍后刷新列表")

    try:
        delete_document_record(db, document)
    except DocumentDeletionError as exc:
        log_action(db, "document_delete_failed", "document", document_id, str(exc)[:500])
        raise HTTPException(
            status_code=503,
            detail=f"Qdrant 向量删除失败，文档已保留为 delete_failed，可稍后重试删除：{exc}",
        ) from exc

    log_action(db, "document_deleted", "document", document_id, "document deleted")
    return DocumentDeleteResponse(document_id=document_id, deleted=True, vector_warning=None)


def _to_detail(document: Document, db: Session) -> DocumentDetailResponse:
    chunks = db.scalars(
        select(DocumentChunk).where(DocumentChunk.document_id == document.document_id).order_by(DocumentChunk.id.asc())
    ).all()
    return DocumentDetailResponse(
        document_id=document.document_id,
        filename=document.filename,
        file_type=document.file_type,
        size=document.size,
        chunk_count=document.chunk_count,
        uploaded_at=document.uploaded_at.isoformat(),
        status=document.status,
        index_version=document.index_version,
        index_error=document.index_error,
        chunks=[
            ChunkSummary(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                text_preview=chunk.text[:120],
                chunk_type=_chunk_type(chunk.text),
                is_truncated=len(chunk.text) > 120,
                section_title=chunk.section_title,
                page_number=chunk.page_number,
                token_count=chunk.token_count,
                index_status=chunk.index_status,
                index_version=chunk.index_version,
                created_at=chunk.created_at.isoformat(),
            )
            for chunk in chunks
        ],
    )


def _chunk_type(text: str) -> str:
    return "table" if text.lstrip().startswith(("表格：", "表格行证据：")) or "\n|" in text or "表格行证据：" in text else "paragraph"


def _to_summary(document: Document) -> DocumentSummary:
    return DocumentSummary(
        document_id=document.document_id,
        filename=document.filename,
        file_type=document.file_type,
        size=document.size,
        chunk_count=document.chunk_count,
        uploaded_at=document.uploaded_at.isoformat(),
        status=document.status,
        index_version=document.index_version,
        index_error=document.index_error,
    )


def _index_document_background(document_id: str) -> None:
    with SessionLocal() as db:
        document = db.scalar(select(Document).where(Document.document_id == document_id))
        if document is None:
            return
        try:
            index_document(db, document)
        except DocumentIndexingError as exc:
            log_action(db, "document_index_failed", "document", document_id, str(exc)[:500])
            return
        log_action(db, "document_indexed", "document", document_id, f"chunks={document.chunk_count}")
