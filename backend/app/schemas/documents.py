from typing import Any

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    content_type: str | None = None
    size: int
    chunk_count: int
    uploaded_at: str


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    file_type: str
    size: int
    chunk_count: int
    uploaded_at: str
    status: str
    index_version: str | None = None
    index_error: str | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]


class DocumentDeleteResponse(BaseModel):
    document_id: str
    deleted: bool
    vector_warning: str | None = None


class ChunkSummary(BaseModel):
    chunk_id: str
    text: str
    text_preview: str
    chunk_type: str = "paragraph"
    is_truncated: bool = False
    section_title: str | None = None
    page_number: int | None = None
    token_count: int = 0
    index_status: str
    index_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class DocumentDetailResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    size: int
    chunk_count: int
    uploaded_at: str
    status: str
    index_version: str | None = None
    index_error: str | None = None
    chunks: list[ChunkSummary]
