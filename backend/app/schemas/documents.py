from pydantic import BaseModel


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


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]


class ChunkSummary(BaseModel):
    chunk_id: str
    text_preview: str
    section_title: str | None = None
    page_number: int | None = None
    created_at: str


class DocumentDetailResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    size: int
    chunk_count: int
    uploaded_at: str
    status: str
    chunks: list[ChunkSummary]

