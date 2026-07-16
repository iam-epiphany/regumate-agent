from pathlib import Path
import json
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.core.config import INDEX_TASK_MAX_RETRIES, INDEX_VERSION, MAX_UPLOAD_BYTES
from backend.app.core.database import get_db
from backend.app.models.document import Document, DocumentChunk, DocumentIndexTask
from backend.app.schemas.documents import (
    ChunkSummary,
    DocumentDeleteResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentProcessingResponse,
    DocumentSummary,
    DocumentUploadResponse,
)
from backend.app.schemas.qa import ApiError
from backend.app.services.audit_service import log_action
from backend.app.services.document_storage import (
    EmptyDocumentError,
    DocumentTooLargeError,
    UnsupportedDocumentTypeError,
    next_document_id,
    save_original_document_stream,
)
from backend.app.services.index_task_service import enqueue_document_index
from backend.app.services.document_lifecycle_service import (
    DocumentDeletionError,
    delete_document_record,
    original_file_exists,
    recover_documents_from_originals,
    remove_documents_with_missing_originals,
    restore_documents_with_available_originals,
)


router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_document(
    file: UploadFile = File(...),
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    db: Session = Depends(get_db),
) -> DocumentUploadResponse:
    filename = file.filename or "unknown"
    request_id = (idempotency_key or uuid4().hex).strip()[:128]
    existing = db.scalar(
        select(Document).where(Document.client_request_id == request_id)
    )
    if existing is not None:
        task = db.scalar(
            select(DocumentIndexTask).where(
                DocumentIndexTask.document_id == existing.document_id
            )
        )
        if task is None:
            raise HTTPException(status_code=409, detail="重复上传记录缺少处理任务，请重试索引")
        return _to_upload_response(existing, task)

    document_id = next_document_id(db)
    storage_path: Path | None = None
    try:
        storage_path, file_size = await save_original_document_stream(
            document_id=document_id,
            filename=filename,
            stream=file,
            content_type=file.content_type,
            max_bytes=MAX_UPLOAD_BYTES,
        )
    except DocumentTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except (UnsupportedDocumentTypeError, EmptyDocumentError) as exc:
        if storage_path is not None:
            storage_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    document = Document(
        document_id=document_id,
        client_request_id=request_id,
        filename=filename,
        content_type=file.content_type,
        file_type=Path(filename).suffix.lower().lstrip("."),
        size=file_size,
        storage_path=str(storage_path),
        document_metadata=json.dumps({}, ensure_ascii=False),
        status="uploaded",
        index_version=INDEX_VERSION,
        index_error=None,
        chunk_count=0,
    )
    task = DocumentIndexTask(
        task_id=uuid4().hex,
        document_id=document_id,
        status="queued",
        stage="queued",
    )
    db.add(document)
    db.add(task)
    try:
        db.commit()
    except Exception:
        db.rollback()
        storage_path.unlink(missing_ok=True)
        raise
    db.refresh(document)
    db.refresh(task)
    log_action(db, "document_uploaded", "document", document_id, filename)
    enqueue_document_index(document_id)
    db.refresh(task)
    return _to_upload_response(document, task)


@router.post("/{document_id}/index", response_model=DocumentDetailResponse)
def build_document_index(document_id: str, db: Session = Depends(get_db)) -> DocumentDetailResponse:
    document = db.scalar(select(Document).where(Document.document_id == document_id))
    if document is None:
        raise HTTPException(status_code=404, detail="未找到指定文档")

    enqueue_document_index(document_id)
    log_action(db, "document_index_queued", "document", document_id, f"chunks={document.chunk_count}")
    db.refresh(document)
    return _to_detail(document, db)


@router.get("/{document_id}/processing", response_model=DocumentProcessingResponse)
def get_document_processing(
    document_id: str,
    db: Session = Depends(get_db),
) -> DocumentProcessingResponse:
    document = db.scalar(select(Document).where(Document.document_id == document_id))
    if document is None:
        raise HTTPException(status_code=404, detail="未找到指定文档")
    task = db.scalar(
        select(DocumentIndexTask).where(DocumentIndexTask.document_id == document_id)
    )
    if task is None:
        raise HTTPException(status_code=404, detail="未找到文档处理任务")
    return _to_processing_response(document, task)


@router.get("", response_model=DocumentListResponse)
def list_documents(
    repair_sources: bool = Query(False, description="Scan original files and repair missing-source states."),
    db: Session = Depends(get_db),
) -> DocumentListResponse:
    restored_documents = restore_documents_with_available_originals(db)
    for document_id, result in restored_documents:
        detail = f"original file is available again; result: {result}"
        log_action(db, "document_source_restored", "document", document_id, detail[:500])

    if repair_sources:
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
def get_document(
    document_id: str,
    chunk_offset: int = Query(0, ge=0),
    chunk_limit: int = Query(50, ge=1, le=200),
    validate_source: bool = Query(False, description="Mark the document source_missing if the original file is gone."),
    db: Session = Depends(get_db),
) -> DocumentDetailResponse:
    document = db.scalar(select(Document).where(Document.document_id == document_id))
    if document is None:
        raise HTTPException(status_code=404, detail="未找到指定文档")

    if document.status == "source_missing" and original_file_exists(document):
        for restored_document_id, result in restore_documents_with_available_originals(db):
            log_action(
                db,
                "document_source_restored",
                "document",
                restored_document_id,
                f"original file is available again; result: {result}"[:500],
            )
        db.refresh(document)

    if validate_source and not original_file_exists(document):
        document.status = "source_missing"
        document.index_error = "原始文件缺失，已暂停该文档参与知识库问答；请恢复文件或显式删除文档。"
        chunks = db.scalars(select(DocumentChunk).where(DocumentChunk.document_id == document.document_id)).all()
        for chunk in chunks:
            chunk.index_status = "source_missing"
        db.commit()
        log_action(db, "document_marked_source_missing", "document", document_id, "original file missing")
        db.refresh(document)

    return _to_detail(document, db, chunk_offset=chunk_offset, chunk_limit=chunk_limit)


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
def delete_document(document_id: str, db: Session = Depends(get_db)) -> DocumentDeleteResponse:
    document = db.scalar(select(Document).where(Document.document_id == document_id))
    if document is None:
        raise HTTPException(status_code=404, detail="未找到指定文档")
    if document.status in {"uploaded", "index_queued", "indexing"}:
        raise HTTPException(status_code=409, detail="文档正在解析或构建索引，请等待处理完成后再删除")
    if document.status == "deleting":
        raise HTTPException(status_code=409, detail="文档正在删除，请稍后刷新列表")

    try:
        delete_document_record(db, document)
    except DocumentDeletionError as exc:
        log_action(db, "document_delete_failed", "document", document_id, str(exc)[:500])
        raise HTTPException(
            status_code=503,
            detail=f"文档删除未完成，记录已保留为 delete_failed，可稍后重试：{exc}",
        ) from exc

    log_action(db, "document_deleted", "document", document_id, "document deleted")
    return DocumentDeleteResponse(document_id=document_id, deleted=True, vector_warning=None)


def _to_detail(
    document: Document,
    db: Session,
    *,
    chunk_offset: int = 0,
    chunk_limit: int = 50,
) -> DocumentDetailResponse:
    chunk_total = db.scalar(
        select(func.count()).select_from(DocumentChunk).where(DocumentChunk.document_id == document.document_id)
    ) or 0
    chunks = db.scalars(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document.document_id)
        .order_by(DocumentChunk.id.asc())
        .offset(chunk_offset)
        .limit(chunk_limit)
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
        metadata=_document_metadata(document),
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
                metadata=_chunk_metadata(chunk),
                created_at=chunk.created_at.isoformat(),
            )
            for chunk in chunks
        ],
        chunk_total=chunk_total,
        chunk_offset=chunk_offset,
        chunk_limit=chunk_limit,
    )


def _chunk_type(text: str) -> str:
    stripped = text.lstrip()
    table_prefixes = ("表格：", "表格摘要：", "表格行证据：", "琛ㄦ牸锛?", "琛ㄦ牸琛岃瘉鎹細")
    return "table" if stripped.startswith(table_prefixes) or "\n|" in text else "paragraph"


def _chunk_metadata(chunk: DocumentChunk) -> dict:
    if not chunk.chunk_metadata:
        return {}
    try:
        value = json.loads(chunk.chunk_metadata)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


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
        metadata=_document_metadata(document),
    )


def _document_metadata(document: Document) -> dict:
    if not document.document_metadata:
        return {}
    try:
        value = json.loads(document.document_metadata)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _to_upload_response(
    document: Document,
    task: DocumentIndexTask,
) -> DocumentUploadResponse:
    return DocumentUploadResponse(
        document_id=document.document_id,
        task_id=task.task_id or str(task.id),
        status=task.status,
        stage=task.stage,
        filename=document.filename,
        content_type=document.content_type,
        size=document.size,
        chunk_count=document.chunk_count,
        uploaded_at=document.uploaded_at.isoformat(),
        metadata=_document_metadata(document),
    )


def _to_processing_response(
    document: Document,
    task: DocumentIndexTask,
) -> DocumentProcessingResponse:
    error = None
    if task.error_code:
        error = ApiError(
            code=task.error_code,
            message=task.last_error or "文档处理失败",
            stage=task.stage or "failed",
            retryable=task.retry_count <= INDEX_TASK_MAX_RETRIES,
            request_id=task.task_id,
        )
    return DocumentProcessingResponse(
        document_id=document.document_id,
        task_id=task.task_id or str(task.id),
        status=task.status,
        stage=task.stage or "queued",
        completed_units=task.completed_units,
        total_units=task.total_units,
        error_code=task.error_code,
        error_message=task.last_error,
        error=error,
        retry_count=task.retry_count,
        updated_at=task.updated_at.isoformat(),
    )
