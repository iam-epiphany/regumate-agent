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


class AgentTraceEntry(BaseModel):
    """One specialist-agent task recorded by the supervisor; counts and codes only."""

    agent: str
    role: str
    task: str
    status: Literal["completed", "failed", "skipped", "refused", "clarification"] = "completed"
    attempt: int = 1
    elapsed_ms: float | None = None
    model_calls: int = 0
    aspect_ids: list[str] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


class AgentDescriptor(BaseModel):
    name: str
    role: str
    capabilities: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)


class AgentRegistryResponse(BaseModel):
    supervisor: str
    agents: list[AgentDescriptor] = Field(default_factory=list)


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
    agent: str | None = None


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
    # Final display number of this citation inside the answer ("[1]", "[2]").
    # The prompt numbers every context chunk [1]..[N]; this label is the
    # renumbered position in the returned citations array so consumers can
    # match answer-text markers to array entries without positional guessing.
    citation_label: str | None = None
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
    # Deterministic structural coordinate (document -> chapter/section ->
    # article).  It is derived from the existing citation fields so that a
    # citation can be decoded and compared without another model or lookup.
    structure_locator: dict[str, Any] | None = None
    structure_label: str | None = None
    excerpt: str
    score: float | None = None
    rerank_score: float | None = None
    chunk_type: str = "paragraph"
    evidence_role: str = "related_context"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any, /) -> None:
        from backend.app.services.structure_locator_service import locator_from_citation

        if self.structure_locator is None:
            self.structure_locator = locator_from_citation(self)
        if self.structure_label is None and self.structure_locator:
            self.structure_label = str(self.structure_locator.get("display") or "") or None


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


class EvidenceRequirement(BaseModel):
    requirement_id: str
    question: str
    intent: str = "lookup"
    target_terms: list[str] = Field(default_factory=list)
    # The concrete attribute a cross-document comparison must evidence in each
    # named source (for example a definition or a division of duties).  Empty
    # keeps every existing requirement on the legacy source-only contract.
    attribute_terms: list[str] = Field(default_factory=list)
    source_titles: list[str] = Field(default_factory=list)
    subject_scope: list[str] = Field(default_factory=list)
    constraints: dict[str, Any] = Field(default_factory=dict)
    selectors: list[dict[str, Any]] = Field(default_factory=list)
    parent_id: str | None = None
    selector_index: int | None = None
    # Planner expansions may be useful for recall but are not user-requested
    # obligations.  The default keeps every existing caller compatible.
    mandatory: bool = True


class EvidenceDecision(BaseModel):
    chunk_id: str
    requirement_id: str
    stage: Literal["filtered", "not_selected", "selected", "missing"]
    reason: str


class EvidenceBundle(BaseModel):
    requirements: list[EvidenceRequirement] = Field(default_factory=list)
    evidence: list[RetrievalResult] = Field(default_factory=list)
    coverage: dict[str, list[str]] = Field(default_factory=dict)
    missing_requirements: list[str] = Field(default_factory=list)
    missing_optional_requirements: list[str] = Field(default_factory=list)
    decisions: list[EvidenceDecision] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return bool(self.requirements) and not self.missing_requirements


class ValidationReport(BaseModel):
    status: Literal["passed", "failed", "not_run", "error"] = "not_run"
    task_complete: bool = False
    citation_integrity: bool = False
    facts_supported: bool = False
    missing_requirements: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    method: str = "none"

    @property
    def passed(self) -> bool:
        return self.status == "passed" and self.task_complete and self.citation_integrity and self.facts_supported


class ExplanationRequirement(BaseModel):
    """One source/evidence obligation projected from EvidenceRequirement."""

    requirement_id: str
    question: str
    source_titles: list[str] = Field(default_factory=list)
    mandatory: bool = True
    status: Literal["covered", "missing", "optional_missing"] = "missing"
    evidence_ids: list[str] = Field(default_factory=list)
    missing_reason: str | None = None


class ExplanationClaimLink(BaseModel):
    """A final-answer claim and the evidence/requirements it points to."""

    claim_index: int
    text: str
    role: str = "other"
    evidence_ids: list[str] = Field(default_factory=list)
    requirement_ids: list[str] = Field(default_factory=list)
    aspect_ids: list[str] = Field(default_factory=list)


class ExplanationSource(BaseModel):
    """A cited evidence unit; the full text remains in QAResponse.citations."""

    evidence_id: str
    chunk_id: str
    document_id: str = ""
    source: str = ""
    source_title: str | None = None
    section_title: str | None = None
    section_path: list[str] = Field(default_factory=list)
    structure_label: str | None = None
    page_number: int | None = None
    sheet_name: str | None = None
    cell: str | None = None
    excerpt: str = ""
    requirement_ids: list[str] = Field(default_factory=list)
    claim_indices: list[int] = Field(default_factory=list)
    citation_located: bool = False


class ExplanationSourceGroup(BaseModel):
    """Source-isolated evidence view used by the side-by-side comparison layout."""

    source_title: str
    source_name: str = ""
    requirement_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    claim_indices: list[int] = Field(default_factory=list)
    status: Literal["covered", "missing", "partial"] = "missing"


class ExplanationOperand(BaseModel):
    label: str
    value: str | None = None
    unit: str | None = None
    sheet_name: str | None = None
    cell: str | None = None
    source: str | None = None
    evidence_id: str | None = None


class ExplanationCalculation(BaseModel):
    operation: str | None = None
    formula: str | None = None
    display_formula: str | None = None
    result: str | None = None
    unit: str | None = None
    operands: list[ExplanationOperand] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    source: str | None = None
    verified: bool | None = None
    verification_note: str | None = None


class ExplanationValidation(BaseModel):
    """The full ValidationReport projection used by every UI decision."""

    status: Literal["passed", "failed", "not_run", "error", "unknown"] = "unknown"
    passed: bool = False
    citation_integrity: bool = False
    facts_supported: bool = False
    task_complete: bool = False
    partial_answer: bool = False
    method: str = "none"
    reasons: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)


class ExplanationBudget(BaseModel):
    category: Literal["standard", "complex", "unknown"] = "unknown"
    total_seconds: float | None = None
    elapsed_ms: float = 0.0
    model_calls: int = 0
    model_call_limit: int = 5
    first_feedback_ms: float | None = None
    upgraded: bool = False


class ExplanationStage(BaseModel):
    stage: str
    status: str
    title: str = ""
    elapsed_ms: float | None = None
    agent: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)


class AnswerExplanation(BaseModel):
    """Display data projected from the existing plan, evidence, claims and validation."""

    question: str
    outcome: str = "answered"
    cross_document: bool = False
    requirements: list[ExplanationRequirement] = Field(default_factory=list)
    claims: list[ExplanationClaimLink] = Field(default_factory=list)
    sources: list[ExplanationSource] = Field(default_factory=list)
    source_groups: list[ExplanationSourceGroup] = Field(default_factory=list)
    calculations: list[ExplanationCalculation] = Field(default_factory=list)
    validation: ExplanationValidation = Field(default_factory=ExplanationValidation)
    sufficiency: dict[str, Any] | None = None
    budget: ExplanationBudget = Field(default_factory=ExplanationBudget)
    stages: list[ExplanationStage] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    elapsed_ms: float = 0.0
    first_feedback_ms: float | None = None


class LLMContextPackage(BaseModel):
    query: str
    mode: str = "rag_context"
    is_final_answer: bool = False
    instruction: str
    retrieval_summary: dict[str, Any]
    context_chunks: list[RetrievalResult]
    llm_prompt: str
    agent_trace: list[AgentTraceEntry] = Field(default_factory=list)


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
    refusal_code: str | None = None
    missing_variables: list[str] = Field(default_factory=list)
    ambiguous_variables: list[str] = Field(default_factory=list)
    unsupported_formula: str | None = None
    degraded: bool = False
    evidence_coverage: EvidenceCoverage | None = None
    agent_trace: list[AgentTraceEntry] = Field(default_factory=list)
    explanation: AnswerExplanation | None = None


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
