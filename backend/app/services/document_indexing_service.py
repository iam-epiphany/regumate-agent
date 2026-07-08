from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import INDEX_VERSION
from backend.app.models.document import Document, DocumentChunk
from backend.app.services.chunk_service import ChunkDraft, count_tokens
from backend.app.services.embedding_service import EmbeddingServiceError, embed_texts
from backend.app.services.vector_store_service import VectorStoreError, upsert_chunk_embeddings


class DocumentIndexingError(RuntimeError):
    pass


def index_document(db: Session, document: Document) -> None:
    if document.status in {"deleting", "delete_failed"}:
        raise DocumentIndexingError("文档正在删除或删除失败待重试，不能构建向量索引")

    chunks = db.scalars(
        select(DocumentChunk).where(DocumentChunk.document_id == document.document_id).order_by(DocumentChunk.id.asc())
    ).all()
    if not chunks:
        raise DocumentIndexingError("文档没有可索引的 chunk")

    document.status = "indexing"
    document.index_version = INDEX_VERSION
    document.index_error = None
    for chunk in chunks:
        chunk.embedding_text = chunk.embedding_text or _embedding_text(chunk)
        chunk.token_count = chunk.token_count or count_tokens(chunk.text)
        chunk.index_status = "indexing"
        chunk.index_version = INDEX_VERSION
    db.commit()

    drafts = [_to_chunk_draft(chunk) for chunk in chunks]
    try:
        embeddings = embed_texts([chunk.embedding_text for chunk in drafts])
        upsert_chunk_embeddings(chunks=drafts, embeddings=embeddings, filename=document.filename)
    except (EmbeddingServiceError, VectorStoreError) as exc:
        _mark_index_failed(db, document, chunks, str(exc))
        raise DocumentIndexingError(str(exc)) from exc

    document.status = "indexed"
    document.index_version = INDEX_VERSION
    document.index_error = None
    for chunk in chunks:
        chunk.index_status = "indexed"
        chunk.index_version = INDEX_VERSION
    db.commit()


def _to_chunk_draft(chunk: DocumentChunk) -> ChunkDraft:
    return ChunkDraft(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        text=chunk.text,
        embedding_text=chunk.embedding_text or _embedding_text(chunk),
        token_count=chunk.token_count or count_tokens(chunk.text),
        title=chunk.title,
        section_title=chunk.section_title,
        page_number=chunk.page_number,
        chunk_type=_chunk_type(chunk.text),
    )


def _embedding_text(chunk: DocumentChunk) -> str:
    chunk_type = _chunk_type(chunk.text)
    labels: list[str] = []
    if chunk.section_title:
        labels.append(f"章节：{chunk.section_title}")
    if chunk_type == "table":
        labels.append("内容类型：表格")
    label_text = "\n".join(labels)
    return f"{label_text}\n\n{chunk.text}" if labels else chunk.text


def _chunk_type(text: str) -> str:
    return "table" if text.lstrip().startswith(("表格：", "表格行证据：")) or "\n|" in text or "表格行证据：" in text else "paragraph"


def _mark_index_failed(db: Session, document: Document, chunks: list[DocumentChunk], error: str) -> None:
    document.status = "index_failed"
    document.index_error = error[:1000]
    document.index_version = INDEX_VERSION
    for chunk in chunks:
        chunk.index_status = "index_failed"
        chunk.index_version = INDEX_VERSION
    db.commit()
