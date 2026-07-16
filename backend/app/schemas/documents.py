from typing import Any, Literal

from pydantic import BaseModel, Field
from backend.app.schemas.qa import ApiError


DocumentStatus = Literal[
    "uploaded",
    "index_queued",
    "indexing",
    "indexed",
    "index_failed",
    "deleting",
    "delete_failed",
    "source_missing",
]
DocumentTaskStatus = Literal["queued", "running", "completed", "failed"]
DocumentStage = Literal[
    "queued",
    "parsing",
    "chunking",
    "metadata_indexing",
    "embedding",
    "vector_upsert",
    "verifying",
    "completed",
    "failed",
]


class DocumentUploadResponse(BaseModel):
    document_id: str
    task_id: str
    status: DocumentTaskStatus
    stage: DocumentStage
    filename: str
    content_type: str | None = None
    size: int
    chunk_count: int
    uploaded_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentProcessingResponse(BaseModel):
    document_id: str
    task_id: str
    status: DocumentTaskStatus
    stage: DocumentStage
    completed_units: int | None = None
    total_units: int | None = None
    error_code: str | None = None
    error_message: str | None = None
    error: ApiError | None = None
    retry_count: int = 0
    updated_at: str


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    file_type: str
    size: int
    chunk_count: int
    uploaded_at: str
    status: DocumentStatus
    index_version: str | None = None
    index_error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    status: DocumentStatus
    index_version: str | None = None
    index_error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunks: list[ChunkSummary]
    chunk_total: int = 0
    chunk_offset: int = 0
    chunk_limit: int = 50
