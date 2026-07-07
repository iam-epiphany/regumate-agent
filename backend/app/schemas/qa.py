from pydantic import BaseModel


class QARequest(BaseModel):
    question: str


class Citation(BaseModel):
    document_id: str
    chunk_id: str
    filename: str
    section_title: str | None = None
    page_number: int | None = None
    excerpt: str


class QAResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float
    refused: bool

