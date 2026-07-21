from typing import Any, Literal

from pydantic import BaseModel, Field


class QARequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    options: list[str] = Field(default_factory=list, max_length=8)
    include_debug: bool = False


class QATaskRequest(QARequest):
    client_request_id: str = Field(min_length=8, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")


class QATaskCreateResponse(BaseModel):
    task_id: str
    client_request_id: str
    status: Literal["queued", "running", "completed", "refused", "failed", "cancelled"]


class ApiError(BaseModel):
    code: str
    message: str
    stage: str | None = None
    retryable: bool = False
    request_id: str | None = None


class RagProgressEvent(BaseModel):
    stage: Literal[
        "planning",
        "retrieval",
        "rerank",
        "context_selection",
        "prompt_build",
        "llm_generation",
        "grounding_validation",
    ]
    status: Literal["pending", "running", "completed", "skipped", "failed"]
    title: str
    detail: str
    elapsed_ms: float | None = None
    summary: dict[str, Any] | None = None
    aspect_id: str | None = None


class AnswerClaim(BaseModel):
    text: str
    citation_ids: list[str] = Field(default_factory=list)
    role: Literal[
        "conclusion",
        "regulatory_basis",
        "table_fact",
        "calculation",
        "explanation",
        "recommendation",
        "other",
    ] = "other"
    aspect_ids: list[str] = Field(default_factory=list)


class EvidenceCoverage(BaseModel):
    expected_aspect_ids: list[str] = Field(default_factory=list)
    covered_aspect_ids: list[str] = Field(default_factory=list)
    missing_aspect_ids: list[str] = Field(default_factory=list)
    complete: bool = False


class Citation(BaseModel):
    document_id: str
    chunk_id: str
    filename: str
    source_url: str | None = None
    attachment_url: str | None = None
    source_title: str | None = None
    issuing_authority: str | None = None
    publication_date: str | None = None
    document_number: str | None = None
    version_status: str | None = None
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
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    answer_type: str = "refusal"
    generation_status: str = "skipped"
    claims: list[AnswerClaim] = Field(default_factory=list)
    grounding_validation: dict[str, Any] = Field(default_factory=dict)
    refusal_reason: str | None = None
    degraded: bool = False
    evidence_coverage: EvidenceCoverage | None = None


class QAAnswerPreview(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    verified_claim_count: int = 0
    revision: int = 0


class QATaskStatusResponse(BaseModel):
    task_id: str
    client_request_id: str | None = None
    question: str
    options: list[str] = Field(default_factory=list)
    include_debug: bool = False
    status: Literal["queued", "running", "completed", "refused", "failed", "cancelled"]
    progress_events: list[RagProgressEvent] = Field(default_factory=list)
    answer_preview: QAAnswerPreview | None = None
    answer: QAResponse | None = None
    error: ApiError | None = None
    created_at: str
    updated_at: str
    completed_at: str | None = None
