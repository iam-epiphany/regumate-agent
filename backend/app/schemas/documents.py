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


class DocumentConflictExistingDocument(BaseModel):
    document_id: str
    filename: str
    size: int
    file_sha256: str | None = None
    status: DocumentStatus
    uploaded_at: str
    chunk_count: int


class DocumentUploadPreflightItemRequest(BaseModel):
    client_file_id: str
    filename: str
    size: int = Field(ge=0)
    file_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-fA-F]{64}$")


class DocumentUploadPreflightRequest(BaseModel):
    items: list[DocumentUploadPreflightItemRequest] = Field(min_length=1)


class DocumentUploadPreflightItem(BaseModel):
    client_file_id: str
    filename: str
    status: Literal["ready", "exact_duplicate", "name_conflict", "selection_name_conflict"]
    existing_document: DocumentConflictExistingDocument | None = None
    error_message: str | None = None


class DocumentUploadPreflightResponse(BaseModel):
    items: list[DocumentUploadPreflightItem]


class DocumentBatchUploadItem(BaseModel):
    filename: str
    status: Literal["accepted", "failed", "duplicate", "conflict"]
    document_id: str | None = None
    task_id: str | None = None
    stage: DocumentStage | None = None
    size: int | None = None
    error_message: str | None = None


class DocumentBatchUploadResponse(BaseModel):
    batch_id: str
    accepted_count: int
    failed_count: int
    items: list[DocumentBatchUploadItem]


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
