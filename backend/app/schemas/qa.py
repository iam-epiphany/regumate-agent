from typing import Any

from pydantic import BaseModel, Field


class QARequest(BaseModel):
    question: str


class Citation(BaseModel):
    document_id: str
    chunk_id: str
    filename: str
    section_title: str | None = None
    section_path: list[str] = Field(default_factory=list)
    section_number: str | None = None
    parent_section_number: str | None = None
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
    page_number: int | None = None
    excerpt: str
    score: float | None = None
    rerank_score: float | None = None
    chunk_type: str = "paragraph"
    evidence_role: str = "related_context"


class RetrievalResult(BaseModel):
    chunk_id: str
    rank: int
    score: float | None = None
    source_doc: str
    section_title: str | None = None
    section_path: list[str] = Field(default_factory=list)
    text: str
    citation_label: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMContextPackage(BaseModel):
    query: str
    mode: str = "rag_context"
    is_final_answer: bool = False
    instruction: str
    retrieval_summary: dict[str, Any]
    context_chunks: list[RetrievalResult]
    llm_prompt: str


class QAResponse(BaseModel):
    answer: str | None
    citations: list[Citation]
    confidence: float
    refused: bool
    context_package: LLMContextPackage | None = None
