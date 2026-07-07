from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.models.document import Document, DocumentChunk
from backend.app.schemas.documents import (
    ChunkSummary,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentSummary,
    DocumentUploadResponse,
)
from backend.app.services.audit_service import log_action
from backend.app.services.chunk_service import build_chunks
from backend.app.services.document_parser import DocumentParseError, parse_document_text
from backend.app.services.document_storage import (
    EmptyDocumentError,
    UnsupportedDocumentTypeError,
    next_document_id,
    save_original_document,
)


router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)) -> DocumentUploadResponse:
    filename = file.filename or "unknown"
    content = await file.read()
    document_id = next_document_id(db)

    try:
        storage_path = save_original_document(document_id=document_id, filename=filename, content=content)
        text = parse_document_text(storage_path)
        chunks = build_chunks(document_id=document_id, text=text)
    except UnsupportedDocumentTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DocumentParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not chunks:
        raise HTTPException(status_code=400, detail="文档没有可入库的文本片段")

    document = Document(
        document_id=document_id,
        filename=filename,
        content_type=file.content_type,
        file_type=Path(filename).suffix.lower().lstrip("."),
        size=len(content),
        storage_path=str(storage_path),
        status="indexed",
        chunk_count=len(chunks),
    )
    db.add(document)
    for chunk in chunks:
        db.add(
            DocumentChunk(
                chunk_id=chunk.chunk_id,
                document_id=document_id,
                text=chunk.text,
                title=chunk.title,
                section_title=chunk.section_title,
                page_number=chunk.page_number,
                source_file=filename,
            )
        )
    db.commit()
    db.refresh(document)
    log_action(db, "document_uploaded", "document", document_id, filename)
    log_action(db, "document_indexed", "document", document_id, f"chunks={len(chunks)}")

    return DocumentUploadResponse(
        document_id=document.document_id,
        filename=document.filename,
        content_type=document.content_type,
        size=document.size,
        chunk_count=document.chunk_count,
        uploaded_at=document.uploaded_at.isoformat(),
    )


@router.get("", response_model=DocumentListResponse)
def list_documents(db: Session = Depends(get_db)) -> DocumentListResponse:
    statement = select(Document).order_by(Document.uploaded_at.desc())
    documents = db.scalars(statement).all()
    return DocumentListResponse(documents=[_to_summary(document) for document in documents])


@router.get("/{document_id}", response_model=DocumentDetailResponse)
def get_document(document_id: str, db: Session = Depends(get_db)) -> DocumentDetailResponse:
    document = db.scalar(select(Document).where(Document.document_id == document_id))
    if document is None:
        raise HTTPException(status_code=404, detail="未找到指定文档")

    chunks = db.scalars(
        select(DocumentChunk).where(DocumentChunk.document_id == document_id).order_by(DocumentChunk.id.asc())
    ).all()
    return DocumentDetailResponse(
        document_id=document.document_id,
        filename=document.filename,
        file_type=document.file_type,
        size=document.size,
        chunk_count=document.chunk_count,
        uploaded_at=document.uploaded_at.isoformat(),
        status=document.status,
        chunks=[
            ChunkSummary(
                chunk_id=chunk.chunk_id,
                text_preview=chunk.text[:120],
                section_title=chunk.section_title,
                page_number=chunk.page_number,
                created_at=chunk.created_at.isoformat(),
            )
            for chunk in chunks
        ],
    )


def _to_summary(document: Document) -> DocumentSummary:
    return DocumentSummary(
        document_id=document.document_id,
        filename=document.filename,
        file_type=document.file_type,
        size=document.size,
        chunk_count=document.chunk_count,
        uploaded_at=document.uploaded_at.isoformat(),
        status=document.status,
    )
