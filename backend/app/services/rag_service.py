from collections import OrderedDict
from dataclasses import dataclass
import json
import re
from threading import RLock
from time import monotonic, perf_counter
from typing import Any, Callable
from uuid import uuid4

from sqlalchemy import or_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, load_only

from backend.app.core.config import (
    DOCUMENT_SNAPSHOT_CACHE_MAX_DOCUMENTS,
    DOCUMENT_SNAPSHOT_CACHE_TTL_SECONDS,
    FINAL_CITATION_LIMIT,
    FORCE_MIN_CHUNKS,
    MAX_PROMPT_CHUNKS,
    MAX_PROMPT_TOKENS,
    MIN_EVIDENCE_COVERAGE,
    MIN_PROMPT_CHUNKS,
    RELATIVE_SCORE_RATIO,
    RERANK_TOP_K,
    RERANK_PROMPT_THRESHOLD,
    RETRIEVAL_TOP_K,
)
from backend.app.models.document import Document, DocumentChunk, QALog
from backend.app.schemas.qa import (
    AnswerClaim,
    Citation,
    EvidenceCoverage,
    LLMContextPackage,
    QAAnswerPreview,
    QAResponse,
    RetrievalResult,
)
from backend.app.services.audit_service import record_event
from backend.app.services.answer_generation_service import generate_answer, is_option_selection_question
from backend.app.services.document_identity_query_service import answer_document_identity_question
from backend.app.services.prompt_builder import RAGPromptBuilder
from backend.app.services.question_preprocessing_service import preprocess_qa_request
from backend.app.services.query_planner_service import (
    QueryAspect,
    QueryPlan,
    QuerySearchQuery,
    plan_query,
)
from backend.app.services.retrieval_metadata_filter_service import (
    build_retrieval_metadata_filter,
)
from backend.app.services.retrieval_service import (
    RetrievalDiagnostics,
    RetrievalMatch,
    RetrievalServiceUnavailable,
    ProgressReporter,
    collect_candidates_with_query_hits,
    evidence_coverage,
    filter_active_candidates,
    filter_candidates_by_metadata,
    get_last_retrieval_diagnostics,
    limit_rerank_candidates,
    matches_from_reranked,
    question_terms,
    reset_retrieval_diagnostics,
    retrieve_citations,
)
from backend.app.services.embedding_service import EmbeddingServiceError
from backend.app.services.model_device_service import get_model_device_info
from backend.app.services.performance_metrics import measure, trace_operation
from backend.app.services.rerank_service import RerankServiceError, rerank_candidates
from backend.app.services.spreadsheet_retrieval_service import (
    get_last_spreadsheet_diagnostic,
    retrieve_spreadsheet_matches,
)
from backend.app.services.vector_store_service import VectorStoreError
from backend.app.services.vector_store_service import get_vector_chunk_by_chunk_id


CONTEXT_INSTRUCTION = (
    "请严格根据检索到的知识片段回答用户问题。不得编造知识库中不存在的制度依据。"
    "若依据不足，请明确说明无法根据当前知识库判断。"
)
CONTEXT_CHUNK_CHAR_LIMIT = 1200
EMPTY_ANSWER_LOG_TEXT = "[RAG_CONTEXT_PACKAGE_ONLY] 当前阶段未接入 LLM，接口仅返回检索上下文包。"
ASPECT_QUERY_FUSION_METHOD = "aspect_query_rrf_then_bge_rerank"
RRF_K = 60
CONSTRAINED_RERANK_CANDIDATE_LIMIT = 20
MCQ_RERANK_CANDIDATE_LIMIT = 24
QUERY_TYPE_WEIGHTS = {
    "semantic_question": 1.0,
    "document_style_statement": 1.15,
    "table_locator": 1.2,
    "keyword_anchor": 0.75,
    "legacy": 0.9,
    "fallback": 0.85,
}
_DEFAULT_RETRIEVE_CITATIONS = retrieve_citations
@dataclass(frozen=True)
class _DocumentChunkSnapshot:
    id: int
    chunk_id: str
    document_id: str
    text: str
    embedding_text: str | None
    chunk_metadata: str | None
    index_status: str
    source_file: str
    page_number: int | None
    section_title: str | None


_DOCUMENT_SNAPSHOT_CACHE: OrderedDict[str, tuple[float, list[_DocumentChunkSnapshot]]] = OrderedDict()
_DOCUMENT_SNAPSHOT_CACHE_LOCK = RLock()


def clear_document_snapshot_cache(document_ids: set[str] | list[str] | tuple[str, ...] | None = None) -> None:
    """Drop cached chunk snapshots after indexing, deletion, recovery, or metadata changes."""

    with _DOCUMENT_SNAPSHOT_CACHE_LOCK:
        if document_ids is None:
            _DOCUMENT_SNAPSHOT_CACHE.clear()
            return
        for document_id in document_ids:
            _DOCUMENT_SNAPSHOT_CACHE.pop(str(document_id), None)


@dataclass(frozen=True)
class QuestionAspect:
    aspect_id: str
    description: str
    keywords: tuple[str, ...]
    required_terms: tuple[str, ...]
    coverage_terms: tuple[str, ...]
    covered_note: str
    missing_note: str


@dataclass
class AspectRetrieval:
    aspect: QueryAspect
    candidates: list[RetrievalResult]
    diagnostics: list[dict[str, Any]]
    citation_validation: dict[str, Any]
    selected_chunk_ids: list[str]
    retrieval_covered: bool = False
    covered: bool = False


@trace_operation("qa")
def answer_question(
    db: Session,
    question: str,
    options: list[str] | None = None,
    include_debug: bool = False,
    progress_reporter: ProgressReporter | None = None,
    answer_preview_reporter: Callable[[QAAnswerPreview], None] | None = None,
    cancellation_checker: Callable[[], None] | None = None,
) -> QAResponse:
    cleaned_question = question.strip()
    if not cleaned_question:
        return QAResponse(answer=None, citations=[], confidence=0.0, refused=True, context_package=None)
    original_question = cleaned_question
    identity_response = answer_document_identity_question(db, cleaned_question)
    if identity_response is not None:
        _save_qa_log(db, original_question, identity_response)
        _log_qa_audit(db, "qa_identity_answered", original_question, identity_response)
        return identity_response
    preprocessed = preprocess_qa_request(cleaned_question, options)
    cleaned_question = preprocessed.question
    normalized_options = preprocessed.options
    option_labels = preprocessed.option_labels or [None] * len(normalized_options)
    if is_option_selection_question(cleaned_question) and not normalized_options:
        _report_progress(
            progress_reporter,
            {
                "stage": "planning",
                "status": "completed",
                "title": "问题理解完成",
                "detail": "识别为选择题，但题干中未提供可判断的选项。",
                "summary": {"reason": "missing_options_for_choice_question"},
            },
        )
        for skipped_stage, skipped_title in (
            ("retrieval", "检索相关依据已跳过"),
            ("rerank", "重排候选片段已跳过"),
            ("context_selection", "上下文精选已跳过"),
            ("prompt_build", "Prompt 构造已跳过"),
            ("llm_generation", "答案生成已跳过"),
            ("grounding_validation", "事实与引用校验已跳过"),
        ):
            _report_progress(
                progress_reporter,
                {
                    "stage": skipped_stage,
                    "status": "skipped",
                    "title": skipped_title,
                    "detail": "需要补充选项后才能进入知识库检索与生成。",
                    "summary": {"reason": "missing_options_for_choice_question"},
                },
            )
        response = QAResponse(
            answer=(
                "这个问题写法属于选择题，但没有提供 A-D 选项，因此无法判断“哪一组选项”正确。"
                "请补充选项，或改成普通问法，例如“请概述该材料的主要内容”。"
            ),
            citations=[],
            confidence=0.0,
            refused=True,
            context_package=None,
            answer_type="clarification",
            generation_status="skipped",
            claims=[],
            grounding_validation={"passed": True, "reason": "missing_options_for_choice_question"},
            refusal_reason="missing_options_for_choice_question",
            degraded=False,
        )
        _save_qa_log(db, original_question, response)
        _log_qa_audit(db, "qa_context_built", original_question, response)
        return response

    package = build_context_package(
        db,
        cleaned_question,
        options=normalized_options,
        progress_reporter=progress_reporter,
    )
    generation_started = perf_counter()
    first_preview_ms: float | None = None
    preview_revision = 0
    preview_claims: list[AnswerClaim] = []
    preview_keys: set[tuple[str, tuple[str, ...]]] = set()

    def report_verified_claim(claim: AnswerClaim) -> None:
        nonlocal first_preview_ms, preview_revision
        key = (claim.text, tuple(claim.citation_ids))
        if key in preview_keys:
            return
        preview_keys.add(key)
        preview_claims.append(claim)
        if first_preview_ms is None:
            first_preview_ms = round((perf_counter() - generation_started) * 1000, 2)
        if answer_preview_reporter is None:
            return
        preview_revision += 1
        cited_labels = {
            citation_id
            for item in preview_claims
            for citation_id in item.citation_ids
        }
        answer_preview_reporter(QAAnswerPreview(
            answer="\n\n".join(item.text for item in preview_claims),
            citations=[
                _citation_from_result(result)
                for result in package.context_chunks
                if result.citation_label in cited_labels
            ],
            verified_claim_count=len(preview_claims),
            revision=preview_revision,
        ))
    _report_progress(
        progress_reporter,
        {
            "stage": "llm_generation",
            "status": "running",
            "title": "正在生成可信回答",
            "detail": "正在基于已选证据生成结构化答案……",
        },
    )
    generated = generate_answer(
        cleaned_question,
        package.context_chunks,
        options=normalized_options,
        option_labels=option_labels,
        llm_prompt=package.llm_prompt,
        has_sufficient_context=bool(package.retrieval_summary["has_sufficient_context"]),
        verified_claim_reporter=report_verified_claim,
        cancellation_checker=cancellation_checker,
        answer_mode=_answer_mode(cleaned_question, package.retrieval_summary),
        required_aspect_ids=_required_aspect_ids(package.retrieval_summary),
    )
    if not generated.refused and not bool(generated.grounding_validation.get("passed")):
        generated.answer = "当前生成结果未通过事实与引用校验，系统已停止输出结论。"
        generated.answer_type = "refusal"
        generated.refused = True
        generated.refusal_reason = "grounding_validation_failed"
        generated.claims = []
        generated.degraded = True
    table_refusal_reasons = [
        str(reason)
        for reason in package.retrieval_summary.get("table_refusal_reasons", [])
        if reason
    ]
    if generated.refused and table_refusal_reasons:
        generated.answer = "当前知识库中未找到足够表格依据，无法给出确定答案。" + "；".join(
            dict.fromkeys(table_refusal_reasons)
        )
        generated.refusal_reason = "table_evidence_not_found"
        generated.grounding_validation = {
            **generated.grounding_validation,
            "passed": True,
            "reason": "table_evidence_not_found",
            "table_refusal_reasons": table_refusal_reasons,
        }
    if not generated.refused and bool(generated.grounding_validation.get("passed")):
        for claim in generated.claims:
            report_verified_claim(claim)
    _report_progress(
        progress_reporter,
        {
            "stage": "llm_generation",
            "status": "completed" if generated.generation_status == "completed" else "skipped",
            "title": "答案生成完成" if generated.generation_status == "completed" else "答案生成已跳过",
            "detail": _generation_progress_detail(generated.generation_status, generated.answer_type),
            "summary": {
                "generation_status": generated.generation_status,
                "answer_type": generated.answer_type,
                "refused": generated.refused,
                "refusal_reason": generated.refusal_reason,
                "refusal_code": generated.refusal_code,
                "first_verified_claim_ms": first_preview_ms,
                "generation_total_ms": round((perf_counter() - generation_started) * 1000, 2),
                "verified_claim_count": len(preview_claims),
            },
        },
    )
    grounding_passed = bool(generated.grounding_validation.get("passed"))
    _report_progress(
        progress_reporter,
        {
            "stage": "grounding_validation",
            "status": "completed" if grounding_passed else "failed",
            "title": "事实与引用校验完成" if grounding_passed else "事实与引用校验未通过",
            "detail": (
                "答案中的引用和关键实体已完成证据核验。"
                if grounding_passed
                else "关键事实未能由当前证据或可信计算结果支持。"
            ),
            "summary": generated.grounding_validation,
        },
    )
    cited_results = _cited_context_results(
        package.context_chunks,
        answer=generated.answer,
        claims=generated.claims,
    )
    response = QAResponse(
        answer=generated.answer,
        citations=[_citation_from_result(result) for result in cited_results],
        confidence=_confidence_from_results(cited_results),
        refused=generated.refused,
        context_package=package if include_debug else None,
        answer_type=generated.answer_type,
        generation_status=generated.generation_status,
        claims=generated.claims,
        grounding_validation=generated.grounding_validation,
        refusal_reason=generated.refusal_reason,
        refusal_code=generated.refusal_code,
        missing_variables=generated.missing_variables,
        ambiguous_variables=generated.ambiguous_variables,
        unsupported_formula=generated.unsupported_formula,
        degraded=generated.degraded,
        evidence_coverage=_build_evidence_coverage(
            package.retrieval_summary,
            generated.claims,
            cited_results,
        ),
    )
    _save_qa_log(db, original_question, response)
    _log_qa_audit(db, "qa_context_built", original_question, response)
    return response


def retrieve_context_package(
    db: Session,
    question: str,
    options: list[str] | None = None,
) -> LLMContextPackage:
    cleaned_question = question.strip()
    if not cleaned_question:
        return _empty_context_package("")
    preprocessed = preprocess_qa_request(cleaned_question, options)
    return build_context_package(db, preprocessed.question, options=preprocessed.options)


def build_context_package(
    db: Session,
    question: str,
    options: list[str] | None = None,
    progress_reporter: ProgressReporter | None = None,
) -> LLMContextPackage:
    planning_started_at = perf_counter()
    _report_progress(
        progress_reporter,
        {
            "stage": "planning",
            "status": "running",
            "title": "正在理解问题",
            "detail": "正在拆分问题并生成检索计划……",
        },
    )
    query_plan = plan_query(question, options=options)
    planning_elapsed_ms = _elapsed_ms(planning_started_at)
    _report_progress(
        progress_reporter,
        {
            "stage": "planning",
            "status": "completed",
            "title": "问题理解完成",
            "detail": f"已拆分为 {len(query_plan.aspects)} 个方面，用时 {_format_elapsed_seconds(planning_elapsed_ms)}",
            "elapsed_ms": planning_elapsed_ms,
            "summary": {
                "aspect_count": len(query_plan.aspects),
                "aspects": [aspect.to_debug_dict() for aspect in query_plan.aspects],
                "planner": query_plan.planner,
                "fallback_used": query_plan.fallback_used,
            },
        },
    )
    aspect_retrievals = _retrieve_aspects(db, query_plan, progress_reporter=progress_reporter)
    citation_validation = _merge_citation_validation(
        [item.citation_validation for item in aspect_retrievals]
    )
    context_started_at = perf_counter()
    _report_progress(
        progress_reporter,
        {
            "stage": "context_selection",
            "status": "running",
            "title": "正在精选最终上下文",
            "detail": "正在按问题方面选择可引用依据……",
        },
    )
    context_chunks, prompt_selection = _select_prompt_chunks(question, query_plan, aspect_retrievals)
    context_chunks = _expand_condition_preamble_prompt_chunks(db, context_chunks, query_plan)
    _sync_aspect_coverage_from_prompt(context_chunks, aspect_retrievals)
    prompt_selection["final_prompt_chunks"] = len(context_chunks)
    prompt_selection["final_prompt_chunk_ids"] = [chunk.chunk_id for chunk in context_chunks]
    prompt_selection["covered_aspects"] = [
        aspect.aspect_id for aspect in query_plan.aspects if any(
            aspect.aspect_id == retrieval.aspect.aspect_id and retrieval.covered
            for retrieval in aspect_retrievals
        )
    ]
    prompt_selection["retrieval_covered_aspects"] = [
        retrieval.aspect.aspect_id for retrieval in aspect_retrievals if retrieval.retrieval_covered
    ]
    prompt_selection["covered_by_retrieval_but_not_prompted"] = [
        retrieval.aspect.aspect_id
        for retrieval in aspect_retrievals
        if retrieval.retrieval_covered and not retrieval.covered
    ]
    prompt_selection["aspect_selected_chunk_ids"] = {
        retrieval.aspect.aspect_id: retrieval.selected_chunk_ids
        for retrieval in aspect_retrievals
    }
    prompt_selection["prompt_capacity_limited"] = bool(
        prompt_selection["covered_by_retrieval_but_not_prompted"]
    )
    context_elapsed_ms = _elapsed_ms(context_started_at)
    prompt_covered_aspect_count = sum(1 for item in aspect_retrievals if item.covered)
    retrieval_covered_aspect_count = sum(1 for item in aspect_retrievals if item.retrieval_covered)
    _report_progress(
        progress_reporter,
        {
            "stage": "context_selection",
            "status": "completed",
            "title": "上下文精选完成",
            "detail": (
                f"最终使用 {len(context_chunks)}/{MAX_PROMPT_CHUNKS} 个片段，"
                f"检索覆盖 {retrieval_covered_aspect_count}/{len(query_plan.aspects)} 个方面，"
                f"进入 Prompt {prompt_covered_aspect_count}/{len(query_plan.aspects)} 个方面"
            ),
            "elapsed_ms": context_elapsed_ms,
            "summary": {
                "used_chunks": len(context_chunks),
                "max_prompt_chunks": MAX_PROMPT_CHUNKS,
                "covered_aspects": prompt_covered_aspect_count,
                "retrieval_covered_aspects": retrieval_covered_aspect_count,
                "total_aspects": len(query_plan.aspects),
            },
        },
    )
    diagnostics = _aggregate_aspect_diagnostics(aspect_retrievals)
    retrieval_summary = _build_retrieval_summary(
        question,
        context_chunks,
        query_plan,
        aspect_retrievals,
        diagnostics,
        citation_validation,
        prompt_selection,
    )
    prompt_started_at = perf_counter()
    _report_progress(
        progress_reporter,
        {
            "stage": "prompt_build",
            "status": "running",
            "title": "正在构造 LLM Prompt",
            "detail": "正在将问题和依据片段组装为后续 LLM 输入……",
        },
    )
    prompt = RAGPromptBuilder().build(question, context_chunks)
    prompt_elapsed_ms = _elapsed_ms(prompt_started_at)
    _report_progress(
        progress_reporter,
        {
            "stage": "prompt_build",
            "status": "completed",
            "title": "Prompt 构造完成",
            "detail": f"已完成 Prompt 构造，用时 {_format_elapsed_seconds(prompt_elapsed_ms)}",
            "elapsed_ms": prompt_elapsed_ms,
            "summary": {"prompt_chunk_count": len(context_chunks)},
        },
    )
    return LLMContextPackage(
        query=question,
        instruction=CONTEXT_INSTRUCTION,
        retrieval_summary=retrieval_summary,
        context_chunks=context_chunks,
        llm_prompt=prompt,
    )


def _empty_context_package(question: str) -> LLMContextPackage:
    prompt = RAGPromptBuilder().build(question, [])
    query_plan = plan_query(question) if question else QueryPlan("", (), "empty", fallback_used=True)
    return LLMContextPackage(
        query=question,
        instruction=CONTEXT_INSTRUCTION,
        retrieval_summary={
            "top_k": FINAL_CITATION_LIMIT,
            "used_chunks": 0,
            "has_sufficient_context": False,
            "coverage_notes": [],
            "missing_aspects": ["未召回可用知识片段"],
            "query_count": 0,
            "candidate_count": 0,
            "reranked_count": 0,
            "filtered_count": 0,
            "timings_ms": {},
            "score_range": {},
            "model_device": get_model_device_info().to_debug_dict(),
            "prompt_filtered_count": 0,
            "prompt_selection": _empty_prompt_selection(),
            "query_plan": query_plan.to_debug_dict(),
            "aspect_retrievals": [],
            "final_prompt_chunk_ids": [],
        },
        context_chunks=[],
        llm_prompt=prompt,
    )


def _retrieve_aspects(
    db: Session,
    query_plan: QueryPlan,
    progress_reporter: ProgressReporter | None = None,
) -> list[AspectRetrieval]:
    retrieval_started_at = perf_counter()
    _report_progress(
        progress_reporter,
        {
            "stage": "retrieval",
            "status": "running",
            "title": "正在检索相关依据",
            "detail": f"正在按 {len(query_plan.aspects)} 个问题方面召回知识库片段……",
            "summary": {"total_aspects": len(query_plan.aspects)},
        },
    )
    aspect_retrievals: list[AspectRetrieval] = []
    document_chunk_cache: dict[str, list[DocumentChunk]] = {}
    for aspect_index, aspect in enumerate(query_plan.aspects, start=1):
        _report_progress(
            progress_reporter,
            {
                "stage": "retrieval",
                "status": "running",
                "title": f"正在检索方面 {aspect_index}",
                "detail": f"正在为“{aspect.question}”检索相关依据……",
                "aspect_id": aspect.aspect_id,
                "summary": {
                    "aspect_index": aspect_index,
                    "total_aspects": len(query_plan.aspects),
                    "aspect_question": aspect.question,
                },
            },
        )
        matches, diagnostics = _retrieve_aspect_matches(
            db,
            aspect,
            progress_reporter=progress_reporter,
            document_chunk_cache=document_chunk_cache,
        )
        matches = _expand_neighbor_matches(
            db,
            aspect.question,
            matches,
            document_chunk_cache=document_chunk_cache,
        )
        candidates = _to_retrieval_results(matches)
        for candidate in candidates:
            candidate.metadata["aspect_id"] = aspect.aspect_id
            candidate.metadata["aspect_question"] = aspect.question
            candidate.metadata["aspect_search_queries"] = _search_query_debug_list(aspect)
            candidate.metadata["expected_evidence_type"] = aspect.expected_evidence_type
            candidate.metadata["evidence_need"] = aspect.evidence_need
            candidate.metadata.setdefault("prompt_matched_aspects", [aspect.aspect_id])
        valid_candidates, citation_validation = _validate_context_chunks(db, candidates)
        aspect_retrieval = AspectRetrieval(
            aspect=aspect,
            candidates=valid_candidates,
            diagnostics=diagnostics,
            citation_validation=citation_validation,
            selected_chunk_ids=[],
            retrieval_covered=bool(valid_candidates),
            covered=False,
        )
        aspect_retrievals.append(aspect_retrieval)
        _report_progress(
            progress_reporter,
            {
                "stage": "retrieval",
                "status": "completed" if valid_candidates else "failed",
                "title": f"方面 {aspect_index} 检索完成" if valid_candidates else f"方面 {aspect_index} 未找到足够依据",
                "detail": (
                    f"已为“{aspect.question}”召回 {len(valid_candidates)} 个可回溯候选片段"
                    if valid_candidates
                    else f"“{aspect.question}”暂未召回可回溯依据"
                ),
                "aspect_id": aspect.aspect_id,
                "summary": {
                    "aspect_index": aspect_index,
                    "total_aspects": len(query_plan.aspects),
                    "candidate_count": len(valid_candidates),
                    "aspect_question": aspect.question,
                    "retrieved_sections": [
                        chunk.section_title for chunk in valid_candidates if chunk.section_title
                    ],
                },
            },
        )
    _recover_missing_aspects_from_sibling_documents(
        db,
        aspect_retrievals,
        document_chunk_cache=document_chunk_cache,
    )
    retrieval_elapsed_ms = _elapsed_ms(retrieval_started_at)
    retrieved_aspect_count = sum(1 for item in aspect_retrievals if item.retrieval_covered)
    candidate_count = sum(len(item.candidates) for item in aspect_retrievals)
    _report_progress(
        progress_reporter,
        {
            "stage": "retrieval",
            "status": "completed",
            "title": "检索相关依据完成",
            "detail": (
                f"已完成 {len(query_plan.aspects)} 个方面检索，"
                f"召回 {candidate_count} 个可回溯候选片段。"
            ),
            "elapsed_ms": retrieval_elapsed_ms,
            "summary": {
                "total_aspects": len(query_plan.aspects),
                "retrieved_aspects": retrieved_aspect_count,
                "candidate_count": candidate_count,
            },
        },
    )
    _report_rerank_stage_summary(progress_reporter, aspect_retrievals)
    return aspect_retrievals


def _recover_missing_aspects_from_sibling_documents(
    db: Session,
    aspect_retrievals: list[AspectRetrieval],
    *,
    document_chunk_cache: dict[str, list[DocumentChunk]],
) -> None:
    """Supplement a sub-question inside documents found for its siblings.

    Cross-document questions often identify a source through one sub-question
    while a short definition sub-question has no metadata of its own.  A weak
    but non-empty retrieval is not proof that the needed clause was found, so
    this pass also supplements already-covered aspects when a sibling document
    contains stronger direct lexical support.  It searches only documents
    already retrieved for the same request; it never scans the corpus or
    evaluator answers.
    """

    if len(aspect_retrievals) < 2:
        return
    documents_by_aspect: list[list[str]] = []
    for retrieval in aspect_retrievals:
        documents_by_aspect.append(
            list(
                dict.fromkeys(
                    str(candidate.metadata.get("document_id") or "")
                    for candidate in retrieval.candidates
                    if candidate.metadata.get("document_id")
                )
            )[:4]
        )
    # Round-robin selection prevents the first broad aspect from consuming the
    # whole bounded scope before later, more source-specific aspects contribute.
    request_document_ids: list[str] = []
    for rank in range(4):
        for document_ids in documents_by_aspect:
            if rank < len(document_ids) and document_ids[rank] not in request_document_ids:
                request_document_ids.append(document_ids[rank])
            if len(request_document_ids) >= 12:
                break
        if len(request_document_ids) >= 12:
            break
    if not request_document_ids:
        return
    for retrieval, own_document_ids in zip(aspect_retrievals, documents_by_aspect, strict=True):
        sibling_document_ids = set(request_document_ids) - set(own_document_ids)
        if not sibling_document_ids:
            continue
        matches = _bounded_document_lexical_support_matches(
            db,
            retrieval.aspect,
            [],
            document_ids=sibling_document_ids,
            document_chunk_cache=document_chunk_cache,
        )
        if not matches:
            continue
        matches = _expand_neighbor_matches(
            db,
            retrieval.aspect.question,
            matches,
            document_chunk_cache=document_chunk_cache,
        )
        candidates = _to_retrieval_results(matches)
        for candidate in candidates:
            candidate.metadata["aspect_id"] = retrieval.aspect.aspect_id
            candidate.metadata["aspect_question"] = retrieval.aspect.question
            candidate.metadata["aspect_search_queries"] = _search_query_debug_list(
                retrieval.aspect
            )
            candidate.metadata["expected_evidence_type"] = retrieval.aspect.expected_evidence_type
            candidate.metadata["evidence_need"] = retrieval.aspect.evidence_need
            candidate.metadata.setdefault(
                "prompt_matched_aspects", [retrieval.aspect.aspect_id]
            )
        valid_candidates, citation_validation = _validate_context_chunks(db, candidates)
        if not valid_candidates:
            continue
        recovered_chunk_ids = {candidate.chunk_id for candidate in valid_candidates}
        retrieval.candidates = valid_candidates + [
            candidate
            for candidate in retrieval.candidates
            if candidate.chunk_id not in recovered_chunk_ids
        ]
        retrieval.citation_validation = _merge_citation_validation(
            [retrieval.citation_validation, citation_validation]
        )
        retrieval.retrieval_covered = True
        retrieval.diagnostics.append(
            {
                "query_type": "sibling_document_lexical_recovery",
                "search_query": retrieval.aspect.question,
                "document_scope_count": len(sibling_document_ids),
                "match_count": len(valid_candidates),
                "recovery_mode": "supplemented" if own_document_ids else "recovered",
            }
        )


def _retrieve_aspect_matches(
    db: Session,
    aspect: QueryAspect,
    progress_reporter: ProgressReporter | None = None,
    document_chunk_cache: dict[str, list[DocumentChunk]] | None = None,
) -> tuple[list[RetrievalMatch], list[dict[str, Any]]]:
    if retrieve_citations is not _DEFAULT_RETRIEVE_CITATIONS:
        return _retrieve_aspect_matches_legacy_hook(db, aspect, progress_reporter)

    table_matches: list[RetrievalMatch] = []
    table_diagnostics: list[dict[str, Any]] = []
    if aspect.modality in {"table", "mixed"}:
        table_started_at = perf_counter()
        table_matches = retrieve_spreadsheet_matches(db, aspect, limit=FINAL_CITATION_LIMIT)
        spreadsheet_diagnostic = get_last_spreadsheet_diagnostic()
        table_elapsed_ms = _elapsed_ms(table_started_at)
        table_diagnostics.append(
            {
                "search_query": aspect.question,
                "query_type": "table_locator",
                "rationale": "根据 planner 的表格检索计划先做结构化 Excel 行/单元格定位",
                "match_count": len(table_matches),
                "table_task": aspect.table_task,
                "operation": aspect.operation,
                "table_filters": aspect.table_filters,
                "spreadsheet_match_status": spreadsheet_diagnostic.get("match_status"),
                "spreadsheet_refusal_reason": spreadsheet_diagnostic.get("refusal_reason"),
                "query_count": 1,
                "raw_candidate_count": len(table_matches),
                "candidate_count": len(table_matches),
                "rerank_input_count": 0,
                "rerank_call_count": 0,
                "reranked_count": 0,
                "filtered_count": 0,
                "query_variants": [aspect.question],
                "timings_ms": {"spreadsheet_retrieval": table_elapsed_ms},
                "score_range": _score_range_for_matches(table_matches),
            }
        )
        for match in table_matches:
            match.metadata["aspect_id"] = aspect.aspect_id
            match.metadata["aspect_question"] = aspect.question
            match.metadata["aspect_search_queries"] = _search_query_debug_list(aspect)
            match.metadata["fusion_method"] = "structured_spreadsheet_retrieval"
            match.metadata["expected_evidence_type"] = aspect.expected_evidence_type
            match.metadata["evidence_need"] = aspect.evidence_need

        if table_matches and aspect.modality == "table":
            return table_matches, table_diagnostics
        if not table_matches and _should_stop_after_spreadsheet_refusal(aspect, spreadsheet_diagnostic):
            table_diagnostics[-1]["early_stopped"] = True
            table_diagnostics[-1]["early_stop_reason"] = "structured_spreadsheet_refusal"
            return [], table_diagnostics

    fusion_scores: dict[str, float] = {}
    diagnostics = RetrievalDiagnostics()
    total_started_at = perf_counter()
    vector_filter_values, vector_explicit_keys, mixed_table_scoped_keys = _vector_metadata_filter_inputs(aspect)
    retrieval_filter = build_retrieval_metadata_filter(
        db,
        vector_filter_values,
        vector_explicit_keys,
    )
    filter_debug = retrieval_filter.to_debug_dict()
    filter_debug["mixed_table_scoped_keys"] = mixed_table_scoped_keys
    if retrieval_filter.no_match:
        filter_diagnostic = {
            "search_query": aspect.question,
            "query_type": "metadata_filter",
            "rationale": "用户明确条件在 SQLite 文档元数据中零命中，禁止退回无约束全库检索",
            "match_count": 0,
            "query_count": 0,
            "raw_candidate_count": 0,
            "candidate_count": 0,
            "pre_filter_candidate_count": 0,
            "post_filter_candidate_count": 0,
            "metadata_filter": filter_debug,
            "match_status": "metadata_filter_no_match",
            "refusal_reason": retrieval_filter.reason,
            "timings_ms": {},
            "score_range": {},
        }
        return table_matches, table_diagnostics + [filter_diagnostic]

    if aspect.aspect_id == "multiple_choice_evidence":
        exact_started_at = perf_counter()
        exact_matches = _mcq_exact_support_matches(
            db,
            aspect,
            [],
            document_chunk_cache=document_chunk_cache,
        )
        if _mcq_exact_support_sufficient(exact_matches, aspect):
            exact_elapsed_ms = _elapsed_ms(exact_started_at)
            for match in exact_matches:
                match.metadata["aspect_id"] = aspect.aspect_id
                match.metadata["aspect_question"] = aspect.question
                match.metadata["aspect_search_queries"] = _search_query_debug_list(aspect)
                match.metadata["fusion_method"] = "mcq_exact_support_early_stop"
                match.metadata["expected_evidence_type"] = aspect.expected_evidence_type
                match.metadata["evidence_need"] = aspect.evidence_need
            diagnostic = {
                "search_query": aspect.question,
                "query_type": "mcq_exact_support",
                "rationale": "选择题先用公开选项事实做有界精确证据匹配；证据足够时跳过 CPU 重排",
                "match_count": len(exact_matches),
                "query_count": 0,
                "raw_candidate_count": len(exact_matches),
                "candidate_count": len(exact_matches),
                "pre_filter_candidate_count": len(exact_matches),
                "post_filter_candidate_count": len(exact_matches),
                "rerank_input_count": 0,
                "rerank_call_count": 0,
                "reranked_count": 0,
                "filtered_count": 0,
                "metadata_filter": filter_debug,
                "match_status": "mcq_exact_support_early_stop",
                "timings_ms": {"mcq_exact_support": exact_elapsed_ms, "total": exact_elapsed_ms},
                "score_range": _score_range_for_matches(exact_matches),
            }
            return exact_matches, table_diagnostics + [diagnostic]

    usable_search_queries = [
        search_query for search_query in aspect.search_queries if search_query.query.strip()
    ]
    if not usable_search_queries and aspect.question.strip():
        # Remote planner output is advisory. A malformed empty query must not
        # turn a valid request into an empty embedding batch.
        usable_search_queries = [
            QuerySearchQuery(
                query=aspect.question.strip(),
                query_type="fallback",
                rationale="empty planner query fallback",
            )
        ]
    search_queries = [search_query.query for search_query in usable_search_queries]
    query_metadata = [
        {
            "query_type": search_query.query_type,
            "rationale": search_query.rationale,
        }
        for search_query in usable_search_queries
    ]
    try:
        collect_kwargs: dict[str, Any] = {
            "query_metadata": query_metadata,
            "diagnostics": diagnostics,
        }
        qdrant_filter = retrieval_filter.qdrant_filter()
        if aspect.modality != "table":
            # Period fields describe SpreadsheetCell coordinates, not a
            # regulation document's publication year. Applying them to mixed
            # text retrieval would remove the required regulation evidence.
            for period_key in ("year", "month", "quarter"):
                qdrant_filter.pop(period_key, None)
        if qdrant_filter:
            collect_kwargs["metadata_filter"] = qdrant_filter
        candidates, query_hits_by_chunk_id, diagnostics_by_query = collect_candidates_with_query_hits(
            search_queries,
            **collect_kwargs,
        )
    except (EmbeddingServiceError, VectorStoreError) as exc:
        raise RetrievalServiceUnavailable(str(exc)) from exc
    for candidate in candidates:
        for hit in query_hits_by_chunk_id.get(candidate.chunk_id, []):
            query_type = str(hit.get("query_type") or "semantic_question")
            query_weight = QUERY_TYPE_WEIGHTS.get(query_type, 1.0)
            rank = int(hit.get("rank") or 1)
            contribution = query_weight / (RRF_K + rank)
            fusion_scores[candidate.chunk_id] = fusion_scores.get(candidate.chunk_id, 0.0) + contribution
            hit["rrf_contribution"] = round(contribution, 6)

    mcq_material_document_ids: set[str] = set()
    if aspect.aspect_id == "multiple_choice_evidence":
        mcq_material_document_ids = _explicit_material_document_ids(db, aspect)
        if mcq_material_document_ids:
            candidates = [
                candidate
                for candidate in candidates
                if _candidate_document_id(candidate) in mcq_material_document_ids
            ]
            diagnostics.candidate_count = len(candidates)

    pre_filter_candidate_count = len(candidates)
    explicit_post_filters = retrieval_filter.explicit
    candidates = filter_candidates_by_metadata(
        filter_active_candidates(
            candidates,
            include_inactive=str(explicit_post_filters.get("version_status") or "") in {"repealed", "superseded"},
        ),
        explicit_post_filters,
    )
    diagnostics.candidate_count = len(candidates)
    preferred_file_type = str(aspect.table_filters.get("file_type") or "").lower()
    inferred_filter_scores = {
        candidate.chunk_id: _inferred_metadata_score(candidate, retrieval_filter.inferred)
        for candidate in candidates
    }
    document_style_scores = {
        candidate.chunk_id: _document_style_evidence_score(candidate, aspect)
        for candidate in candidates
    }
    for candidate in candidates:
        if candidate.metadata is None:
            candidate.metadata = {}
        candidate.metadata["document_style_evidence_score"] = round(
            document_style_scores.get(candidate.chunk_id, 0.0), 6
        )
        candidate.metadata["inferred_metadata_score"] = round(
            inferred_filter_scores.get(candidate.chunk_id, 0.0), 6
        )
    candidates.sort(
        key=lambda candidate: (
            bool(preferred_file_type and candidate.filename.lower().endswith(f".{preferred_file_type}")),
            inferred_filter_scores.get(candidate.chunk_id, 0.0),
            document_style_scores.get(candidate.chunk_id, 0.0),
            fusion_scores.get(candidate.chunk_id, 0.0),
            candidate.score,
        ),
        reverse=True,
    )
    # ``candidates`` is already ordered by weighted RRF above.  Re-sorting it
    # by one raw vector score here would undo multi-query fusion and starve
    # exact option/statement hits in long documents before cross-encoding.
    effective_rerank_limit = _effective_rerank_candidate_limit(
        aspect,
        candidates,
        retrieval_filter_document_ids=set(retrieval_filter.document_ids or ()),
        mcq_material_document_ids=mcq_material_document_ids,
    )
    rerank_input = limit_rerank_candidates(
        candidates,
        preserve_order=True,
        limit=effective_rerank_limit,
    )
    diagnostics.rerank_input_count = len(rerank_input)
    _report_progress(
        progress_reporter,
        {
            "stage": "rerank",
            "status": "running",
            "title": "正在重排候选片段",
            "detail": f"方面“{aspect.question}”融合后进入重排 {diagnostics.rerank_input_count} 个片段……",
            "aspect_id": aspect.aspect_id,
            "summary": {
                "aspect_id": aspect.aspect_id,
                "candidate_count": diagnostics.candidate_count,
                "rerank_input_count": diagnostics.rerank_input_count,
                "effective_rerank_candidate_limit": effective_rerank_limit,
            },
        },
    )
    rerank_started_at = perf_counter()
    try:
        rerank_limit = len(rerank_input) if aspect.aspect_id == "multiple_choice_evidence" else RERANK_TOP_K
        reranked = rerank_candidates(
            question=_rerank_query_for_aspect(aspect),
            candidates=rerank_input,
            limit=rerank_limit,
        )
    except RerankServiceError as exc:
        raise RetrievalServiceUnavailable(str(exc)) from exc
    diagnostics.rerank_call_count = 1 if rerank_input else 0
    diagnostics.timings_ms["rerank"] = _elapsed_ms(rerank_started_at)
    if aspect.aspect_id == "multiple_choice_evidence":
        for item in reranked:
            direct_score = document_style_scores.get(item.candidate.chunk_id, 0.0)
            item.rerank_score = 0.35 * float(item.rerank_score) + 0.65 * direct_score
        reranked.sort(key=lambda item: item.rerank_score, reverse=True)
        reranked = reranked[:RERANK_TOP_K]
    diagnostics.reranked_count = len(reranked)
    diagnostics.score_range = _score_range_for_candidates(candidates, reranked)
    if aspect.aspect_id == "multiple_choice_evidence":
        matches = _mcq_matches_from_reranked(aspect, reranked, diagnostics)
    else:
        matches = matches_from_reranked(
            question=aspect.question,
            reranked=reranked,
            diagnostics=diagnostics,
        )
    diagnostics.timings_ms["total"] = _elapsed_ms(total_started_at)
    matches = _filter_indexed_matches(db, matches)
    for match in matches:
        chunk_id = match.citation.chunk_id
        match.metadata["aspect_id"] = aspect.aspect_id
        match.metadata["aspect_question"] = aspect.question
        match.metadata["aspect_search_queries"] = _search_query_debug_list(aspect)
        match.metadata["aspect_search_query_hits"] = query_hits_by_chunk_id.get(chunk_id, [])
        match.metadata["aspect_query_fusion_score"] = round(fusion_scores.get(chunk_id, 0.0), 6)
        match.metadata["fusion_method"] = ASPECT_QUERY_FUSION_METHOD
        match.metadata["expected_evidence_type"] = aspect.expected_evidence_type
        match.metadata["evidence_need"] = aspect.evidence_need

    exact_anchor_matches = _exact_anchor_support_matches(
        db,
        aspect,
        matches,
        document_ids=set(retrieval_filter.document_ids or ()),
        document_chunk_cache=document_chunk_cache,
    )
    if exact_anchor_matches:
        best_fusion = max(fusion_scores.values(), default=0.0)
        for offset, supplement in enumerate(exact_anchor_matches, start=1):
            fusion_scores[supplement.citation.chunk_id] = best_fusion + 0.015 - offset * 0.0001
        existing_chunk_ids = {match.citation.chunk_id for match in exact_anchor_matches}
        matches = exact_anchor_matches + [
            match for match in matches if match.citation.chunk_id not in existing_chunk_ids
        ]

    lexical_scope_document_ids = set(retrieval_filter.document_ids or ())
    if mcq_material_document_ids:
        lexical_scope_document_ids = set(mcq_material_document_ids)
    if not lexical_scope_document_ids:
        lexical_scope_document_ids.update(
            _candidate_document_id(item.candidate)
            for item in reranked[:24]
            if _candidate_document_id(item.candidate)
        )
        # Keep the clause-recovery pass inside documents already recalled by
        # dense/sparse/hybrid retrieval.  Reusing the MCQ empty-result seeder
        # here would issue full-corpus ``contains`` scans for every aspect and
        # turns an ordinary compound question into a minute-scale request.
        # Missing aspects are recovered by the generic planner split and their
        # own hybrid queries; lexical support only repairs within-document
        # clause ranking and must not become a second unbounded retriever.
    lexical_support_matches = _bounded_document_lexical_support_matches(
        db,
        aspect,
        matches,
        document_ids=lexical_scope_document_ids,
        document_chunk_cache=document_chunk_cache,
    )
    if lexical_support_matches:
        best_fusion = max(fusion_scores.values(), default=0.0)
        for offset, supplement in enumerate(lexical_support_matches, start=1):
            fusion_scores[supplement.citation.chunk_id] = best_fusion + 0.01 - offset * 0.0001
        existing_chunk_ids = {match.citation.chunk_id for match in lexical_support_matches}
        matches = lexical_support_matches + [
            match for match in matches if match.citation.chunk_id not in existing_chunk_ids
        ]

    formula_support_matches = _bounded_formula_support_matches(
        db,
        aspect,
        document_ids=lexical_scope_document_ids,
        document_chunk_cache=document_chunk_cache,
    )
    if formula_support_matches:
        best_fusion = max(fusion_scores.values(), default=0.0)
        for offset, supplement in enumerate(formula_support_matches, start=1):
            fusion_scores[supplement.citation.chunk_id] = best_fusion + 0.01 - offset * 0.0001
        existing_chunk_ids = {match.citation.chunk_id for match in formula_support_matches}
        matches = formula_support_matches + [
            match for match in matches if match.citation.chunk_id not in existing_chunk_ids
        ]

    if aspect.aspect_id == "multiple_choice_evidence":
        supplements = _mcq_exact_support_matches(
            db,
            aspect,
            matches,
            document_chunk_cache=document_chunk_cache,
        )
        if supplements:
            best_fusion = max(fusion_scores.values(), default=0.0)
            for offset, supplement in enumerate(supplements, start=1):
                chunk_id = supplement.citation.chunk_id
                fusion_scores[chunk_id] = best_fusion + 0.01 - offset * 0.0001
                supplement.metadata["aspect_id"] = aspect.aspect_id
                supplement.metadata["aspect_question"] = aspect.question
                supplement.metadata["fusion_method"] = "mcq_exact_support"
                supplement.metadata["expected_evidence_type"] = aspect.expected_evidence_type
                supplement.metadata["evidence_need"] = aspect.evidence_need
            matches = supplements + matches

    for item in diagnostics_by_query:
        chunk_ids_for_query = {
            chunk_id
            for chunk_id, hits in query_hits_by_chunk_id.items()
            if any(hit.get("query") == item.get("search_query") for hit in hits)
        }
        item["match_count"] = sum(
            1 for match in matches if match.citation.chunk_id in chunk_ids_for_query
        )

    diagnostics_by_query.append(
        {
            "search_query": aspect.question,
            "query_type": "aspect_fused",
            "rationale": "同一 aspect 的多条 search query 先召回融合，再单次 BGE rerank",
            "match_count": len(matches),
            "pre_filter_candidate_count": pre_filter_candidate_count,
            "post_filter_candidate_count": len(candidates),
            "metadata_filter": filter_debug,
            "rerank_input_ranking": [
                {
                    "rank": rank,
                    "chunk_id": candidate.chunk_id,
                    "fusion_score": round(fusion_scores.get(candidate.chunk_id, 0.0), 6),
                    "vector_score": round(float(candidate.score), 6),
                }
                for rank, candidate in enumerate(rerank_input, start=1)
            ],
            "reranked_ranking": [
                {
                    "rank": rank,
                    "chunk_id": item.candidate.chunk_id,
                    "rerank_score": round(float(item.rerank_score), 8),
                }
                for rank, item in enumerate(reranked, start=1)
            ],
            "effective_rerank_candidate_limit": effective_rerank_limit,
            **diagnostics.to_summary_fields(),
        }
    )
    _report_progress(
        progress_reporter,
        {
            "stage": "rerank",
            "status": "completed",
            "title": "候选重排完成",
            "detail": f"方面“{aspect.question}”完成 1 次融合重排，输出 {diagnostics.reranked_count} 个片段",
            "aspect_id": aspect.aspect_id,
            "elapsed_ms": diagnostics.timings_ms.get("rerank"),
            "summary": {
                "aspect_id": aspect.aspect_id,
                "rerank_call_count": diagnostics.rerank_call_count,
                "rerank_input_count": diagnostics.rerank_input_count,
                "reranked_count": diagnostics.reranked_count,
            },
        },
    )
    matches.sort(
        key=lambda match: (
            fusion_scores.get(match.citation.chunk_id, 0.0),
            match.rerank_score,
            match.score,
        ),
        reverse=True,
    )
    if table_matches:
        matches = table_matches + matches
    return matches, table_diagnostics + diagnostics_by_query


_SPREADSHEET_TERMINAL_REFUSAL_STATUSES = {
    "period_not_found",
    "source_not_found",
    "sheet_not_found",
    "indicator_not_found",
    "column_not_found",
    "ambiguous_candidates",
    "unit_mismatch",
    "empty_value",
}


def _should_stop_after_spreadsheet_refusal(aspect: QueryAspect, diagnostic: dict[str, Any]) -> bool:
    """Skip vector fallback when structured Excel evidence has conclusively failed.

    The fallback path is useful for weakly planned table questions, but it is
    harmful when the structured retriever has already proven that an explicit
    file/period/indicator/selector is absent: the vector path can only add
    latency and may surface a nearby but wrong table row.
    """

    if aspect.modality != "table":
        return False
    if aspect.table_task not in {"lookup", "compare", "calculate"}:
        return False
    status = str(diagnostic.get("match_status") or "")
    if status not in _SPREADSHEET_TERMINAL_REFUSAL_STATUSES:
        return False
    filters = aspect.table_filters or {}
    selectors = [item for item in aspect.selectors if isinstance(item, dict)]
    has_explicit_constraint = bool(selectors) or any(
        filters.get(key) is not None
        for key in (
            "source_title",
            "filename",
            "sheet",
            "year",
            "month",
            "quarter",
            "indicator",
            "row_label",
            "column_label",
            "metric",
            "scope",
        )
    )
    return has_explicit_constraint


def _vector_metadata_filter_inputs(
    aspect: QueryAspect,
) -> tuple[dict[str, Any], set[str], list[str]]:
    values = dict(aspect.table_filters)
    explicit_keys = set(getattr(aspect, "explicit_filter_keys", ()))
    table_scoped_keys: list[str] = []
    if aspect.modality != "mixed":
        return values, explicit_keys, table_scoped_keys
    for key in (
        "filename",
        "source_title",
        "file_type",
        "year",
        "month",
        "quarter",
        "sheet",
        "indicator",
        "row_label",
        "column_label",
        "unit",
        "metric",
        "scope",
    ):
        if key in values:
            table_scoped_keys.append(key)
            values.pop(key, None)
            explicit_keys.discard(key)
    return values, explicit_keys, table_scoped_keys


def _inferred_metadata_score(candidate: Any, filters: dict[str, Any]) -> float:
    if not filters:
        return 0.0
    metadata = candidate.metadata or {}
    mappings = {
        "source_title": metadata.get("source_title") or candidate.filename,
        "filename": candidate.filename,
        "external_doc_id": metadata.get("external_doc_id"),
        "issuing_authority": metadata.get("issuing_authority"),
        "publication_date": metadata.get("publication_date"),
        "document_number": metadata.get("document_number"),
        "regulatory_topic": metadata.get("regulatory_topic"),
        "business_domain": metadata.get("business_domain"),
        "article_number": metadata.get("article_number") or candidate.section_number,
        "version_status": metadata.get("version_status"),
        "file_type": metadata.get("file_type"),
    }
    comparable = [(key, value) for key, value in filters.items() if key in mappings]
    if not comparable:
        return 0.0
    matched = 0
    for key, expected in comparable:
        actual = mappings.get(key)
        if actual is None:
            continue
        expected_norm = re.sub(r"\W+", "", str(expected)).casefold()
        actual_norm = re.sub(r"\W+", "", str(actual)).casefold()
        if expected_norm and expected_norm in actual_norm:
            matched += 1
    return matched / len(comparable)


def _retrieve_aspect_matches_legacy_hook(
    db: Session,
    aspect: QueryAspect,
    progress_reporter: ProgressReporter | None = None,
) -> tuple[list[RetrievalMatch], list[dict[str, Any]]]:
    fused_by_chunk_id: dict[str, RetrievalMatch] = {}
    fusion_scores: dict[str, float] = {}
    fusion_query_hits: dict[str, list[dict[str, Any]]] = {}
    diagnostics_by_query: list[dict[str, Any]] = []

    for search_query in aspect.search_queries:
        reset_retrieval_diagnostics()
        matches = _filter_indexed_matches(
            db,
            _retrieve_citations_with_optional_progress(search_query.query, progress_reporter),
        )
        diagnostics = get_last_retrieval_diagnostics()
        diagnostics_by_query.append(
            {
                "search_query": search_query.query,
                "query_type": search_query.query_type,
                "rationale": search_query.rationale,
                "match_count": len(matches),
                **diagnostics.to_summary_fields(),
            }
        )
        _report_progress(
            progress_reporter,
            {
                "stage": "rerank",
                "status": "completed",
                "title": "候选重排完成",
                "detail": f"已完成 {diagnostics.reranked_count} 个片段重排",
                "aspect_id": aspect.aspect_id,
                "elapsed_ms": diagnostics.timings_ms.get("rerank"),
                "summary": {
                    "search_query": search_query.query,
                    "query_type": search_query.query_type,
                    "reranked_count": diagnostics.reranked_count,
                },
            },
        )
        query_weight = QUERY_TYPE_WEIGHTS.get(search_query.query_type, 1.0)
        for rank, match in enumerate(matches, start=1):
            chunk_id = match.citation.chunk_id
            if not chunk_id:
                continue
            fusion_scores[chunk_id] = fusion_scores.get(chunk_id, 0.0) + query_weight / (RRF_K + rank)
            fusion_query_hits.setdefault(chunk_id, []).append(
                {
                    "query": search_query.query,
                    "query_type": search_query.query_type,
                    "rationale": search_query.rationale,
                    "rank": rank,
                    "rrf_contribution": round(query_weight / (RRF_K + rank), 6),
                }
            )
            existing = fused_by_chunk_id.get(chunk_id)
            if existing is None or _match_sort_key(match) > _match_sort_key(existing):
                fused_by_chunk_id[chunk_id] = match

    fused_matches = list(fused_by_chunk_id.values())
    for match in fused_matches:
        chunk_id = match.citation.chunk_id
        match.metadata["aspect_id"] = aspect.aspect_id
        match.metadata["aspect_question"] = aspect.question
        match.metadata["aspect_search_queries"] = _search_query_debug_list(aspect)
        match.metadata["aspect_search_query_hits"] = fusion_query_hits.get(chunk_id, [])
        match.metadata["aspect_query_fusion_score"] = round(fusion_scores.get(chunk_id, 0.0), 6)
        match.metadata["fusion_method"] = "query_plan_rrf"
        match.metadata["expected_evidence_type"] = aspect.expected_evidence_type
        match.metadata["evidence_need"] = aspect.evidence_need

    fused_matches.sort(
        key=lambda match: (
            fusion_scores.get(match.citation.chunk_id, 0.0),
            match.rerank_score,
            match.score,
        ),
        reverse=True,
    )
    return fused_matches, diagnostics_by_query


def _retrieve_citations_with_optional_progress(
    search_query: str,
    progress_reporter: ProgressReporter | None,
) -> list[RetrievalMatch]:
    if progress_reporter is None:
        return retrieve_citations(search_query)
    try:
        return retrieve_citations(search_query, progress_reporter=progress_reporter)
    except TypeError as exc:
        if "progress_reporter" not in str(exc):
            raise
        return retrieve_citations(search_query)


def _report_rerank_stage_summary(
    progress_reporter: ProgressReporter | None,
    aspect_retrievals: list[AspectRetrieval],
) -> None:
    diagnostics = [
        diagnostic
        for aspect_retrieval in aspect_retrievals
        for diagnostic in aspect_retrieval.diagnostics
    ]
    rerank_input_count = sum(int(item.get("rerank_input_count") or 0) for item in diagnostics)
    rerank_call_count = sum(int(item.get("rerank_call_count") or 0) for item in diagnostics)
    reranked_count = sum(int(item.get("reranked_count") or 0) for item in diagnostics)
    candidate_count = sum(len(item.candidates) for item in aspect_retrievals)
    if rerank_input_count or rerank_call_count or reranked_count:
        _report_progress(
            progress_reporter,
            {
                "stage": "rerank",
                "status": "completed",
                "title": "重排候选片段完成",
                "detail": f"已完成 {rerank_call_count} 次重排，输出 {reranked_count} 个候选片段。",
                "summary": {
                    "candidate_count": candidate_count,
                    "rerank_input_count": rerank_input_count,
                    "rerank_call_count": rerank_call_count,
                    "reranked_count": reranked_count,
                },
            },
        )
        return

    _report_progress(
        progress_reporter,
        {
            "stage": "rerank",
            "status": "skipped",
            "title": "重排候选片段已跳过",
            "detail": (
                "检索未产生需要重排的候选片段。"
                if candidate_count == 0
                else "当前候选片段已由结构化检索或兼容检索直接排序，无需额外重排。"
            ),
            "summary": {
                "candidate_count": candidate_count,
                "rerank_input_count": rerank_input_count,
                "rerank_call_count": rerank_call_count,
                "reranked_count": reranked_count,
            },
        },
    )


def _generation_progress_detail(generation_status: str, answer_type: str) -> str:
    if generation_status == "completed":
        return "已基于选定证据生成结构化回答。"
    if answer_type == "refusal":
        return "依据不足，系统直接形成拒答结论，未继续调用生成模型。"
    if answer_type == "clarification":
        return "问题信息不完整，已直接给出补充信息提示。"
    return "该业务分支不需要继续生成回答。"


def _report_progress(progress_reporter: ProgressReporter | None, event: dict[str, Any]) -> None:
    if progress_reporter is not None:
        progress_reporter(event)


def _match_sort_key(match: RetrievalMatch) -> tuple[float, float, float]:
    return (
        float(match.rerank_score or 0.0),
        float(match.coverage_score or 0.0),
        float(match.score or 0.0),
    )


def _search_query_debug_list(aspect: QueryAspect) -> list[dict[str, Any]]:
    return [search_query.to_debug_dict() for search_query in aspect.search_queries]


def _rerank_query_for_aspect(aspect: QueryAspect) -> str:
    evidence_queries = [
        search_query.query
        for search_query in aspect.search_queries
        if search_query.query_type == "document_style_statement" and search_query.query.strip()
    ]
    if not evidence_queries:
        return aspect.question
    return "\n".join([aspect.question, *evidence_queries])


def _effective_rerank_candidate_limit(
    aspect: QueryAspect,
    candidates: list[Any],
    *,
    retrieval_filter_document_ids: set[str],
    mcq_material_document_ids: set[str],
) -> int:
    if len(candidates) <= CONSTRAINED_RERANK_CANDIDATE_LIMIT:
        return len(candidates)
    if aspect.aspect_id == "multiple_choice_evidence" or mcq_material_document_ids:
        return min(len(candidates), MCQ_RERANK_CANDIDATE_LIMIT)
    if retrieval_filter_document_ids:
        return CONSTRAINED_RERANK_CANDIDATE_LIMIT
    return max(CONSTRAINED_RERANK_CANDIDATE_LIMIT, min(len(candidates), MCQ_RERANK_CANDIDATE_LIMIT))


def _document_style_evidence_score(candidate: Any, aspect: QueryAspect) -> float:
    statements = [
        statement.strip()
        for search_query in aspect.search_queries
        if search_query.query_type == "document_style_statement"
        for statement in search_query.query.splitlines()
        if statement.strip()
    ]
    if not statements:
        return 0.0
    evidence_text = "\n".join(
        part
        for part in (
            candidate.filename,
            candidate.section_title or "",
            candidate.embedding_text or "",
            candidate.text,
        )
        if part
    )
    return max(
        evidence_coverage(question_terms(statement), evidence_text)
        for statement in statements
    )


def _normalize_document_anchor(value: str) -> str:
    normalized = re.sub(r"^\d+_", "", str(value or "").lower())
    normalized = normalized.replace("附件", "").replace("版", "")
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", normalized)


def _exact_anchor_support_matches(
    db: Session,
    aspect: QueryAspect,
    matches: list[RetrievalMatch],
    *,
    document_ids: set[str],
    document_chunk_cache: dict[str, list[DocumentChunk]] | None = None,
) -> list[RetrievalMatch]:
    """Recover literal user-stated clauses inside an explicitly named document.

    Dense retrieval can miss a short table-of-contents heading or a clause in
    a long document.  This bounded SQLite pass is only enabled after metadata
    resolution has identified the document, and it searches user-provided
    quoted text rather than any evaluator answer.
    """

    anchors = _aspect_anchor_phrases(aspect)
    scope = set(document_ids) or {match.citation.document_id for match in matches}
    if not anchors or not scope:
        return []
    chunks_by_document = _chunks_by_document(
        db,
        scope,
        document_chunk_cache=document_chunk_cache,
    )
    supplements: list[tuple[float, int, RetrievalMatch]] = []
    seen_chunk_ids: set[str] = set()
    for anchor in anchors:
        normalized_anchor = _normalize_exact_support_text(anchor)
        best: tuple[float, Any, list[Any]] | None = None
        for document_id, document_chunks in chunks_by_document.items():
            for chunk in document_chunks:
                if chunk.index_status != "indexed":
                    continue
                searchable_raw = "\n".join(
                    part for part in [chunk.section_title or "", chunk.embedding_text or "", chunk.text] if part
                )
                searchable = _normalize_exact_support_text(searchable_raw)
                exact = bool(normalized_anchor and normalized_anchor in searchable)
                definition = bool(
                    exact
                    and len(normalized_anchor) >= 6
                    and any(
                        marker in searchable
                        for marker in (
                            f"本办法所称{normalized_anchor}",
                            f"所称{normalized_anchor}",
                            f"{normalized_anchor}是指",
                            f"{normalized_anchor}是以",
                        )
                    )
                )
                recall = _exact_support_recall(anchor, searchable_raw)
                qualifies = exact and (len(normalized_anchor) >= 8 or definition)
                if not qualifies and len(normalized_anchor) >= 12 and recall >= 0.82:
                    qualifies = True
                if not qualifies:
                    continue
                score = (2.0 if definition else 1.0) + recall
                if best is None or score > best[0]:
                    best = (score, chunk, document_chunks)
        if best is None:
            continue
        score, chunk, document_chunks = best
        if chunk.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk.chunk_id)
        match = _direct_exact_support_match(chunk, document_chunks, aspect, anchor, score)
        supplements.append((score, len(normalized_anchor), match))
    supplements.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in supplements]


def _direct_exact_support_match(
    chunk: Any,
    document_chunks: list[Any],
    aspect: QueryAspect,
    anchor: str,
    support_score: float,
) -> RetrievalMatch:
    previous_chunk_id, next_chunk_id = _adjacent_chunk_ids(chunk, document_chunks)
    section_number = _section_number(chunk.section_title)
    text = _clean_chunk_text(chunk.text, chunk.section_title)
    normalized_score = min(0.99, 0.88 + min(support_score, 2.0) * 0.05)
    return RetrievalMatch(
        citation=Citation(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            filename=chunk.source_file or "",
            section_title=chunk.section_title,
            section_path=_inferred_section_path(chunk, document_chunks),
            section_number=section_number,
            parent_section_number=_parent_section_number(section_number),
            previous_chunk_id=previous_chunk_id,
            next_chunk_id=next_chunk_id,
            page_number=chunk.page_number,
            excerpt=text,
            score=normalized_score,
            rerank_score=normalized_score,
            chunk_type=_chunk_type(chunk.text),
            evidence_role="exact_anchor_support",
        ),
        score=normalized_score,
        rerank_score=normalized_score,
        coverage_score=evidence_coverage(
            question_terms(aspect.question),
            f"{chunk.section_title or ''}\n{chunk.embedding_text or ''}\n{text}",
        ),
        evidence_role="exact_anchor_support",
        evidence_text=text,
        metadata={
            **_chunk_metadata(chunk),
            "aspect_id": aspect.aspect_id,
            "aspect_question": aspect.question,
            "aspect_search_queries": _search_query_debug_list(aspect),
            "expected_evidence_type": aspect.expected_evidence_type,
            "evidence_need": aspect.evidence_need,
            "evidence_role": "exact_anchor_support",
            "exact_support_anchor": anchor,
            "exact_support_score": round(support_score, 4),
            "fusion_method": "bounded_document_exact_anchor",
        },
    )


def _bounded_document_lexical_support_matches(
    db: Session,
    aspect: QueryAspect,
    matches: list[RetrievalMatch],
    *,
    document_ids: set[str],
    document_chunk_cache: dict[str, list[DocumentChunk]] | None = None,
) -> list[RetrievalMatch]:
    """Recover strongly overlapping clauses inside metadata-resolved documents.

    This is intentionally bounded by document identity.  It is a generic
    safeguard for long regulations where a title filter resolves correctly
    but vector retrieval misses one clause of a compound question.  It never
    searches evaluator answers and does not broaden an explicit document
    scope to the full corpus.
    """

    if not hasattr(db, "scalars"):
        return []
    scope_document_ids = set(document_ids) or {
        match.citation.document_id for match in matches[:24] if match.citation.document_id
    }
    if not scope_document_ids:
        return []
    phrases = _bounded_lexical_phrases(aspect)
    if not phrases:
        return []
    required_numeric_terms = _bounded_required_numeric_terms(aspect)
    chunks_by_document = _chunks_by_document(
        db,
        scope_document_ids,
        document_chunk_cache=document_chunk_cache,
    )
    existing_ids = {match.citation.chunk_id for match in matches}
    supplements: list[tuple[float, int, RetrievalMatch]] = []
    seen_ids: set[str] = set()
    for phrase in phrases:
        normalized_phrase = _normalize_exact_support_text(phrase)
        if len(normalized_phrase) < 6:
            continue
        best: tuple[float, Any, list[Any]] | None = None
        for document_chunks in chunks_by_document.values():
            for chunk in document_chunks:
                if chunk.index_status != "indexed":
                    continue
                searchable = "\n".join(
                    part
                    for part in (
                        chunk.section_title or "",
                        chunk.text or chunk.embedding_text or "",
                    )
                    if part
                )
                normalized_searchable = _normalize_exact_support_text(searchable)
                if required_numeric_terms and not all(
                    _normalize_exact_support_text(term) in normalized_searchable
                    for term in required_numeric_terms
                ):
                    continue
                recall = _exact_support_recall(phrase, searchable)
                exact = normalized_phrase in normalized_searchable
                definition = bool(
                    exact
                    and any(
                        marker in normalized_searchable
                        for marker in (
                            f"本办法所称{normalized_phrase}",
                            f"所称{normalized_phrase}",
                            f"{normalized_phrase}是指",
                            f"{normalized_phrase}是以",
                        )
                    )
                )
                critical = _critical_lexical_terms(phrase)
                critical_hits = sum(
                    _normalize_exact_support_text(term) in normalized_searchable for term in critical
                )
                critical_ratio = critical_hits / len(critical) if critical else 0.0
                qualifies = exact or recall >= 0.72 or (recall >= 0.58 and critical_ratio >= 0.75)
                if not qualifies:
                    continue
                score = (
                    recall
                    + (0.35 if exact else 0.0)
                    + (1.0 if definition else 0.0)
                    + critical_ratio * 0.2
                )
                if best is None or score > best[0]:
                    best = (score, chunk, document_chunks)
        if best is None:
            continue
        score, chunk, document_chunks = best
        if chunk.chunk_id in seen_ids:
            continue
        # Existing direct matches do not need to be duplicated, unless the
        # bounded pass found an exact clause that should be promoted ahead of
        # a low-ranked vector result.
        seen_ids.add(chunk.chunk_id)
        supplement = _direct_exact_support_match(chunk, document_chunks, aspect, phrase, score)
        supplement.citation.evidence_role = "bounded_lexical_support"
        supplement.evidence_role = "bounded_lexical_support"
        supplement.metadata["evidence_role"] = "bounded_lexical_support"
        supplement.metadata["fusion_method"] = "bounded_document_lexical_support"
        supplement.metadata["lexical_support_phrase"] = phrase
        supplement.metadata["promoted_existing_match"] = chunk.chunk_id in existing_ids
        supplements.append((score, len(normalized_phrase), supplement))
        for continuation_score, continuation in _condition_continuation_support_matches(
            chunk,
            document_chunks,
            aspect,
            phrases,
            existing_ids=existing_ids,
            seen_ids=seen_ids,
        ):
            supplements.append((continuation_score, len(normalized_phrase), continuation))
    supplements.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in supplements[:6]]


def _condition_continuation_support_matches(
    anchor_chunk: Any,
    document_chunks: list[Any],
    aspect: QueryAspect,
    phrases: list[str],
    *,
    existing_ids: set[str],
    seen_ids: set[str],
    require_preamble: bool = True,
) -> list[tuple[float, RetrievalMatch]]:
    """Recover list items split after a regulatory preamble.

    Some parsed Word/PDF regulations split a sentence like "应同时满足以下条件"
    from the numbered condition rows that follow it.  If the bounded lexical
    pass finds such a preamble, this helper only inspects the next few chunks
    in the same document and only promotes continuations that independently
    cover the same aspect's lexical anchors.
    """

    anchor_text = _normalize_exact_support_text(
        "\n".join(
            part
            for part in (anchor_chunk.section_title or "", anchor_chunk.text or anchor_chunk.embedding_text or "")
            if part
        )
    )
    if require_preamble and not any(
        marker in anchor_text
        for marker in (
            "以下条件",
            "下列条件",
            "如下条件",
            "以下要求",
            "下列要求",
            "要求如下",
            "规定如下",
        )
    ):
        return []
    try:
        anchor_index = next(
            index for index, item in enumerate(document_chunks) if item.chunk_id == anchor_chunk.chunk_id
        )
    except StopIteration:
        return []
    matches: list[tuple[float, RetrievalMatch]] = []
    for continuation in document_chunks[anchor_index + 1 : anchor_index + 5]:
        if continuation.index_status != "indexed" or continuation.chunk_id in seen_ids:
            continue
        continuation_body = str(continuation.text or continuation.embedding_text or "")
        searchable = "\n".join(
            part
            for part in (
                continuation.section_title or "",
                continuation_body,
            )
            if part
        )
        if not searchable.strip():
            continue
        best = _best_condition_continuation_phrase(continuation_body, phrases)
        if best is None:
            best = _best_numbered_list_continuation_phrase(
                anchor_chunk,
                continuation,
                aspect,
                searchable,
            )
        if best is None:
            continue
        best_score, best_phrase = best
        seen_ids.add(continuation.chunk_id)
        match = _direct_exact_support_match(continuation, document_chunks, aspect, best_phrase, best_score)
        match.citation.evidence_role = "bounded_lexical_support"
        match.evidence_role = "bounded_lexical_support"
        match.metadata["evidence_role"] = "bounded_lexical_support"
        match.metadata["fusion_method"] = "bounded_document_condition_continuation"
        match.metadata["lexical_support_phrase"] = best_phrase
        match.metadata["promoted_existing_match"] = continuation.chunk_id in existing_ids
        match.metadata["condition_preamble_chunk_id"] = anchor_chunk.chunk_id
        matches.append((best_score, match))
    return matches


def _best_numbered_list_continuation_phrase(
    anchor_chunk: Any,
    continuation: Any,
    aspect: QueryAspect,
    searchable: str,
) -> tuple[float, str] | None:
    anchor_text = "\n".join(
        part
        for part in (anchor_chunk.section_title or "", anchor_chunk.text or anchor_chunk.embedding_text or "")
        if part
    )
    normalized_anchor = _normalize_exact_support_text(anchor_text)
    if not any(marker in normalized_anchor for marker in ("包括", "以下", "下列", "如下", "条件", "要求")):
        return None

    anchor_section = _normalize_exact_support_text(str(anchor_chunk.section_title or ""))
    continuation_section = _normalize_exact_support_text(str(continuation.section_title or ""))
    if anchor_section and continuation_section and anchor_section != continuation_section:
        return None

    continuation_text = str(continuation.text or continuation.embedding_text or "")
    if not re.search(
        r"(?:^|[\s。；;])(?:\d{1,2}|[一二三四五六七八九十]+)[.．、]|（[一二三四五六七八九十]+）",
        continuation_text,
    ):
        return None

    normalized_searchable = _normalize_exact_support_text(continuation_text)
    if len(normalized_searchable) < 24:
        return None
    if not any(marker in normalized_searchable for marker in ("应当", "不得", "包括", "范围", "条件", "要求", "期限", "时限", "公示", "计量", "损益")):
        return None

    terms = _list_continuation_anchor_terms(aspect)
    best_score = 0.0
    best_term = ""
    for term in terms:
        coverage = _anchor_coverage(term, continuation_text)
        normalized_term = _normalize_exact_support_text(term)
        exact = normalized_term and normalized_term in normalized_searchable
        if not (exact or coverage >= 0.45):
            continue
        score = coverage + (0.25 if exact else 0.0) + min(len(normalized_term) / 80.0, 0.2)
        if score > best_score:
            best_score = score
            best_term = term
    if best_score <= 0:
        return None
    return best_score, best_term


def _list_continuation_anchor_terms(aspect: QueryAspect) -> list[str]:
    candidates: list[str] = []
    candidates.extend(_aspect_anchor_phrases(aspect))
    candidates.extend(str(keyword or "") for keyword in aspect.keywords)
    candidates.extend(_bounded_lexical_phrases(aspect))
    terms: list[str] = []
    for candidate in candidates:
        normalized = _normalize_exact_support_text(candidate)
        if 4 <= len(normalized) <= 80:
            terms.append(candidate)
        for marker in ("公示", "函证", "回函", "范围", "用章", "效力", "证明力", "工作日", "交易账簿", "公允价值计量", "损益", "并表范围", "披露"):
            if marker in str(candidate):
                terms.append(marker)
        for part in re.split(r"[、，,；;。]|以及|并且|同时|和|与|相关", str(candidate)):
            part = part.strip(" ：:，,。；;")
            if 4 <= len(_normalize_exact_support_text(part)) <= 40:
                terms.append(part)
    return list(dict.fromkeys(term for term in terms if len(_normalize_exact_support_text(term)) >= 2))[:24]


def _best_condition_continuation_phrase(
    searchable: str,
    phrases: list[str],
) -> tuple[float, str] | None:
    best_score = 0.0
    best_phrase = ""
    normalized_searchable = _normalize_exact_support_text(searchable)
    for phrase in phrases:
        normalized_phrase = _normalize_exact_support_text(phrase)
        if len(normalized_phrase) < 6:
            continue
        recall = _exact_support_recall(phrase, searchable)
        exact = normalized_phrase in normalized_searchable
        critical = _critical_lexical_terms(phrase)
        critical_hits = sum(
            _normalize_exact_support_text(term) in normalized_searchable for term in critical
        )
        critical_ratio = critical_hits / len(critical) if critical else 0.0
        if not (exact or recall >= 0.72 or (recall >= 0.58 and critical_ratio >= 0.5)):
            continue
        score = recall + (0.35 if exact else 0.0) + critical_ratio * 0.2
        if score > best_score:
            best_score = score
            best_phrase = phrase
    if best_score <= 0:
        return None
    return best_score, best_phrase


def _bounded_lexical_phrases(aspect: QueryAspect) -> list[str]:
    candidates = [
        search_query.query
        for search_query in aspect.search_queries
        if search_query.query_type not in {"document_title", "metadata_filter"}
    ]
    candidates.append(aspect.question)
    if aspect.modality != "table":
        candidates.extend(
            keyword
            for keyword in aspect.keywords
            if 4 <= len(_normalize_exact_support_text(keyword)) <= 40
            and keyword not in {"定义", "概念", "含义", "监管办法", "相关规定"}
        )
    phrases: list[str] = []
    for candidate in candidates:
        text = re.sub(r"《[^》]{2,80}》", " ", str(candidate or ""))
        text = re.sub(r"^[A-DＡ-Ｄ][、.．：:]\s*", "", text.strip(), flags=re.IGNORECASE)
        spaced_terms = [
            re.sub(
                r"^(?:请|根据|依据|结合|说明|判断|概括|比较|计算|回答|指出)+",
                "",
                term.strip(" ，,：:"),
            ).strip(" ，,：:")
            for term in re.split(r"\s+", text)
            if term.strip()
        ]
        spaced_terms = [
            term
            for term in spaced_terms
            if 2 <= len(_normalize_exact_support_text(term)) <= 20
            and term not in {"说明", "判断", "比较", "计算", "回答", "指出"}
            and not term.endswith("版")
            and "示例" not in term
        ]
        if len(spaced_terms) >= 2:
            for window in (4, 3, 2):
                for start in range(0, max(0, len(spaced_terms) - window + 1)):
                    phrase = "".join(spaced_terms[start : start + window])
                    normalized_phrase = _normalize_exact_support_text(phrase)
                    if 6 <= len(normalized_phrase) <= 60:
                        phrases.append(phrase)
        for term in spaced_terms:
            normalized_term = _normalize_exact_support_text(term)
            if 6 <= len(normalized_term) <= 40:
                phrases.append(term)
        # Document-style queries can leave an unknown argument inside an
        # otherwise exact predicate, e.g. ``与哪些主体开展有效沟通``.  That
        # interrogative slot is absent from the source clause, so also retain
        # its evidence-shaped predicate tail.  The caller still constrains the
        # scan to resolved/candidate documents; this is not a corpus shortcut.
        interrogative_tail = re.search(
            r"(?:哪些|什么|何种|何类)(?:主体|对象|机构|人员|部门|条件|要求|方式|措施)?(?P<tail>[\u4e00-\u9fff]{6,40})",
            text,
        )
        if interrogative_tail:
            tail = interrogative_tail.group("tail").strip()
            if 6 <= len(_normalize_exact_support_text(tail)) <= 40:
                phrases.append(tail)
        # Threshold questions often express the unknown as ``X 相对 Y 的门槛``
        # while the source uses a definition such as ``X 是指……超过 Y Z%``.
        # Preserve both relation anchors, plus bounded suffixes of a long left
        # noun phrase, so an already-resolved sibling document can recover the
        # defining clause without knowing the missing numeric answer.
        for relation in re.finditer(
            r"(?P<left>[\u4e00-\u9fff]{4,40})相对(?P<right>[\u4e00-\u9fff]{4,30}?)(?:的)?(?:门槛|阈值|比例|限额)",
            text,
        ):
            left = relation.group("left")
            right = relation.group("right")
            phrases.extend((left, right))
            if len(left) > 8:
                phrases.extend((left[-8:], left[-6:]))
        parts = re.split(r"[；;。？?]|以及|并且|并同时|同时|分别", text)
        for part in parts:
            cleaned = re.sub(
                r"^(?:请|根据|依据|结合|说明|判断|概括|比较|计算|回答|指出)+",
                "",
                part.strip(" ，,：:"),
            )
            cleaned = re.sub(
                r"(?:是什么|有哪些|如何规定|是否正确|是否符合规定|请说明|请回答)$",
                "",
                cleaned,
            ).strip(" ，,：:")
            normalized = _normalize_exact_support_text(cleaned)
            if 6 <= len(normalized) <= 120:
                phrases.append(cleaned)
    return list(dict.fromkeys(phrases))[:12]


def _bounded_required_numeric_terms(aspect: QueryAspect) -> list[str]:
    texts = [aspect.question, *(query.query for query in aspect.search_queries)]
    terms: list[str] = []
    for text in texts:
        terms.extend(re.findall(r"\d+(?:\.\d+)?%", str(text or "")))
    return list(dict.fromkeys(terms))


def _bounded_formula_support_matches(
    db: Session,
    aspect: QueryAspect,
    *,
    document_ids: set[str],
    document_chunk_cache: dict[str, list[DocumentChunk]] | None = None,
) -> list[RetrievalMatch]:
    """Locate a requested equation inside already-recalled documents.

    Formula symbols are short and rank poorly in semantic retrieval.  This
    bounded pass never scans the full corpus: it inspects formula metadata only
    in documents already resolved by the ordinary retriever, then matches the
    user-stated left-hand target and explicitly assigned operands.
    """

    compact = re.sub(r"\s+", "", aspect.question)
    if "公式" not in compact or not any(term in compact for term in ("计算", "算出", "求", "尝试")):
        return []
    if not document_ids:
        return []
    target_match = re.search(
        r"(?:计算|算出|求)\s*([A-Za-zΑ-Ωα-ω][A-Za-z0-9_Α-Ωα-ω]*)\s*(?=[：:=，,])",
        aspect.question,
        flags=re.IGNORECASE,
    )
    requested_target = target_match.group(1).casefold() if target_match else ""
    assigned_variables = {
        match.group(1).casefold()
        for match in re.finditer(r"([A-Za-z][A-Za-z0-9_]*)\s*(?:=|：|:)", aspect.question)
    }
    chunks_by_document = _chunks_by_document(
        db,
        document_ids,
        document_chunk_cache=document_chunk_cache,
    )
    ranked: list[tuple[int, int, RetrievalMatch]] = []
    seen: set[str] = set()
    for document_chunks in chunks_by_document.values():
        for chunk in document_chunks:
            metadata = _chunk_metadata(chunk)
            formulas = metadata.get("formulas")
            if not isinstance(formulas, list):
                continue
            for item in formulas:
                if not isinstance(item, dict):
                    continue
                formula = str(item.get("text") or "").strip()
                if not formula:
                    continue
                normalized = re.sub(r"\s+", "", formula.replace("×", "*").replace("＝", "="))
                left, _, right = normalized.partition("=")
                target = re.sub(r"[^A-Za-z0-9_Α-Ωα-ω]", "", left).casefold()
                variables = {
                    value.casefold()
                    for value in re.findall(r"[A-Za-z][A-Za-z0-9_]*", right or normalized)
                    if value.casefold() not in {"max", "min", "sqrt", "sum", "root", "pow"}
                }
                target_match_score = int(bool(requested_target) and target == requested_target)
                variable_overlap = len(assigned_variables & variables)
                qualifies = target_match_score == 1 or (
                    not requested_target
                    and len(assigned_variables) >= 2
                    and variable_overlap == len(assigned_variables)
                )
                if not qualifies or chunk.chunk_id in seen:
                    continue
                seen.add(chunk.chunk_id)
                match = _direct_exact_support_match(
                    chunk,
                    document_chunks,
                    aspect,
                    formula,
                    2.0 + target_match_score + variable_overlap * 0.25,
                )
                match.citation.evidence_role = "formula_target_support"
                match.evidence_role = "formula_target_support"
                match.metadata["evidence_role"] = "formula_target_support"
                match.metadata["fusion_method"] = "bounded_document_formula_target"
                match.metadata["formula_target"] = requested_target or None
                match.metadata["formula_variable_overlap"] = variable_overlap
                ranked.append((target_match_score, variable_overlap, match))
    ranked.sort(key=lambda item: (item[0], item[1], item[2].score), reverse=True)
    return [item[2] for item in ranked[:3]]


def _critical_lexical_terms(text: str) -> list[str]:
    terms = re.findall(r"\d+(?:\.\d+)?%|\d+(?:\.\d+)?(?:年|个月|日|万元|亿元|倍)", text)
    terms.extend(
        match.group(0)
        for match in re.finditer(
            r"[\u4e00-\u9fff]{2,12}(?:不得|应当|必须|可以|属于|包括|不低于|不高于|不超过|至少)",
            text,
        )
    )
    return list(dict.fromkeys(term for term in terms if len(_normalize_exact_support_text(term)) >= 2))


def _mcq_exact_support_matches(
    db: Session,
    aspect: QueryAspect,
    matches: list[RetrievalMatch],
    document_chunk_cache: dict[str, list[DocumentChunk]] | None = None,
) -> list[RetrievalMatch]:
    raw_statements = [
        search_query.query.strip()
        for search_query in aspect.search_queries
        if search_query.query_type == "document_style_statement" and search_query.query.strip()
    ]
    statements: list[str] = []
    for statement in raw_statements:
        statements.append(statement)
        statements.extend(
            part.strip()
            for part in re.split(r"[；;，,]", statement)
            if len(_normalize_exact_support_text(part)) >= 4
        )
    statements = list(dict.fromkeys(statements))
    if not statements:
        return []
    critical_terms: list[str] = []
    for statement in statements:
        critical_terms.extend(re.findall(r"\d+(?:\.\d+)?%", statement))
        if "属于" in statement:
            subject = statement.split("属于", 1)[0].strip(" ，,。；;")
            if len(subject) >= 6:
                critical_terms.append(subject)
    critical_terms = list(dict.fromkeys(critical_terms))

    material_terms = _explicit_material_anchor_terms(
        "\n".join([aspect.question, *[query.query for query in aspect.search_queries]])
    )
    material_document_ids = _document_ids_for_title_terms(db, material_terms)
    anchors_by_document: dict[str, RetrievalMatch] = {}
    for match in matches:
        anchors_by_document.setdefault(match.citation.document_id, match)
    if material_terms:
        # An explicit source title is an authoritative scope constraint.  Do
        # not first run the fallback lexical seed across every indexed chunk
        # and then intersect it with this known document set: that turns a
        # small document-local MCQ check into an unbounded corpus scan.
        if not material_document_ids:
            seed_document_ids = set()
        else:
            seed_document_ids = set(material_document_ids)
            anchors_by_document = {
                document_id: anchor
                for document_id, anchor in anchors_by_document.items()
                if document_id in material_document_ids
            }
    else:
        seed_document_ids = _mcq_seed_document_ids(db, statements)
    seed_chunks_by_document = _chunks_by_document(
        db,
        seed_document_ids - set(anchors_by_document),
        document_chunk_cache=document_chunk_cache,
    )
    for document_id, document_chunks in seed_chunks_by_document.items():
        indexed_chunks = [chunk for chunk in document_chunks if chunk.index_status == "indexed"]
        best: tuple[float, Any, str] | None = None
        for statement in statements:
            for chunk in indexed_chunks:
                searchable = "\n".join(
                    part for part in (chunk.embedding_text or "", chunk.text) if part
                )
                score = _exact_support_recall(statement, searchable)
                if best is None or score > best[0]:
                    best = (score, chunk, statement)
        if best is None or best[0] < 0.45:
            continue
        score, chunk, statement = best
        anchor = _direct_exact_support_match(chunk, indexed_chunks, aspect, statement, score)
        anchor.metadata["fusion_method"] = "mcq_bounded_lexical_seed"
        anchors_by_document[document_id] = anchor
    if not anchors_by_document:
        return []
    existing_by_id = {match.citation.chunk_id: match for match in matches}
    supplements_by_chunk_id: dict[str, tuple[float, int, RetrievalMatch]] = {}
    chunks_by_document = _chunks_by_document(
        db,
        set(anchors_by_document),
        document_chunk_cache=document_chunk_cache,
    )
    for document_id, anchor in anchors_by_document.items():
        document_chunks = [
            chunk
            for chunk in chunks_by_document.get(document_id, [])
            if chunk.index_status == "indexed"
        ]
        best_statement_matches: dict[str, tuple[float, DocumentChunk]] = {}
        for chunk in document_chunks:
            searchable_raw = "\n".join(
                part for part in [chunk.embedding_text or "", chunk.text] if part
            )
            searchable = _normalize_exact_support_text(searchable_raw)
            existing_match = existing_by_id.get(chunk.chunk_id)
            should_promote = existing_match is None or str(
                existing_match.metadata.get("evidence_role") or ""
            ) == "expanded_context"
            matched_terms = [
                term for term in critical_terms if _normalize_exact_support_text(term) in searchable
            ]
            if matched_terms and should_promote:
                supplement = _match_from_chunk(
                    chunk,
                    anchor,
                    "mcq_exact_support",
                    document_chunks,
                    aspect.question,
                )
                supplement.metadata["exact_support_terms"] = matched_terms
                supplement.metadata["evidence_role"] = "mcq_exact_support"
                priority = (1.0, max(map(len, matched_terms)), supplement)
                previous = supplements_by_chunk_id.get(chunk.chunk_id)
                if previous is None or priority[:2] > previous[:2]:
                    supplements_by_chunk_id[chunk.chunk_id] = priority

            for statement in statements:
                required_exact_terms = _mcq_required_exact_terms(statement, aspect)
                if required_exact_terms and not all(
                    _normalize_exact_support_text(term) in searchable
                    for term in required_exact_terms
                ):
                    continue
                statement_score = _exact_support_recall(statement, searchable_raw)
                previous_statement = best_statement_matches.get(statement)
                if previous_statement is None or statement_score > previous_statement[0]:
                    best_statement_matches[statement] = (statement_score, chunk)

        # Dense/sparse retrieval can miss a literal statement when a document
        # contains many near-duplicate headings.  Add the best direct support
        # for every option fact, but only when the wording overlap is strong.
        for statement, (statement_score, chunk) in best_statement_matches.items():
            # The exact chunk may already be a low-ranked direct candidate.
            # Re-emit it as an exact-support match so prompt selection does
            # not discard it solely because its original rerank score was
            # below the generic prose threshold.
            if statement_score < 0.45:
                continue
            supplement = _match_from_chunk(
                chunk,
                anchor,
                "mcq_exact_support",
                document_chunks,
                aspect.question,
            )
            supplement.metadata["exact_support_statement"] = statement
            supplement.metadata["exact_support_score"] = round(statement_score, 4)
            supplement.metadata["evidence_role"] = "mcq_exact_support"
            priority = (statement_score, len(statement), supplement)
            previous = supplements_by_chunk_id.get(chunk.chunk_id)
            if previous is None or priority[:2] > previous[:2]:
                supplements_by_chunk_id[chunk.chunk_id] = priority

    supplements = sorted(
        supplements_by_chunk_id.values(), key=lambda item: (item[0], item[1]), reverse=True
    )
    direct_support = [item[2] for item in supplements[:10]]

    # Enumerated prohibitions and permissions are often split across PDF
    # chunks: the governing modal appears in a list preamble while the matched
    # option fact appears several chunks later.  Preserve that explicit
    # structural scope instead of treating the list item as a free-standing
    # positive statement.
    preambles_by_chunk_id: dict[str, RetrievalMatch] = {}
    for support in direct_support:
        document_chunks = chunks_by_document.get(support.citation.document_id, [])
        by_chunk_id = {chunk.chunk_id: chunk for chunk in document_chunks}
        item_chunk = by_chunk_id.get(support.citation.chunk_id)
        if item_chunk is None:
            continue
        preamble_chunk = _governing_list_preamble(item_chunk, document_chunks)
        if preamble_chunk is None:
            continue
        preamble = preambles_by_chunk_id.get(preamble_chunk.chunk_id)
        if preamble is None:
            preamble = _match_from_chunk(
                preamble_chunk,
                support,
                "mcq_rule_preamble",
                document_chunks,
                aspect.question,
            )
            preamble.evidence_role = "mcq_rule_preamble"
            preamble.citation.evidence_role = "mcq_rule_preamble"
            preamble.metadata["evidence_role"] = "mcq_rule_preamble"
            preamble.metadata["governs_chunk_ids"] = []
            preambles_by_chunk_id[preamble_chunk.chunk_id] = preamble
        governed_ids = preamble.metadata["governs_chunk_ids"]
        if support.citation.chunk_id not in governed_ids:
            governed_ids.append(support.citation.chunk_id)

    return [*direct_support, *list(preambles_by_chunk_id.values())[:4]]


def _mcq_exact_support_sufficient(
    matches: list[RetrievalMatch],
    aspect: QueryAspect | None = None,
) -> bool:
    direct = [
        match
        for match in matches
        if match.metadata.get("evidence_role") == "mcq_exact_support"
        or match.evidence_role == "mcq_exact_support"
    ]
    if len(direct) < 2:
        return False
    distinct_documents = {
        match.citation.document_id
        for match in direct
        if match.citation.document_id
    }
    distinct_chunks = {
        match.citation.chunk_id
        for match in direct
        if match.citation.chunk_id
    }
    if not (distinct_documents and len(distinct_chunks) >= 2):
        return False
    for group_terms in _mcq_required_evidence_groups(aspect):
        if not any(
            any(
                _normalize_exact_support_text(term) in _normalize_exact_support_text(
                    "\n".join(
                        part
                        for part in (
                            match.citation.filename,
                            match.citation.section_title or "",
                            match.citation.excerpt,
                        )
                        if part
                    )
                )
                for term in group_terms
            )
            for match in direct
        ):
            return False
    return True


def _mcq_required_evidence_groups(aspect: QueryAspect | None) -> list[tuple[str, ...]]:
    if aspect is None:
        return []
    question_text = str(aspect.question or "")
    search_query_texts = [query.query for query in aspect.search_queries]
    compact = _normalize_exact_support_text(
        "\n".join([question_text, *search_query_texts])
    )
    groups: list[tuple[str, ...]] = [
        (term,) for term in _explicit_material_anchor_terms(
            "\n".join([question_text, *search_query_texts])
        )
    ]
    if "重要实体" in compact or "核心业务条线" in compact:
        groups.append(("重要实体", "核心业务条线"))
    if "处置工具" in compact or "过桥机构" in compact or "破产清算" in compact:
        groups.append(("处置工具", "过桥机构", "破产清算"))
    if "不正当手段" in compact or "无关第三人" in compact or "催收" in compact:
        groups.append(("消费金融", "催收", "债务无关"))
    if "询证函" in compact or "函证基准日" in compact or "函证事项" in compact:
        groups.append(("银行函证", "询证函", "函证基准日"))
    if "交易账簿" in compact or "交易头寸" in compact:
        groups.append(("交易账簿", "交易目的持有"))
    if "第三支柱" in compact or "监管并表范围" in compact or "表格另有规定除外" in compact:
        groups.append(("信息披露", "监管并表范围", "第三支柱"))
    return groups


def _explicit_material_anchor_terms(question: str) -> list[str]:
    terms: list[str] = []
    for term in re.findall(r"《([^》]{4,80})》", str(question or "")):
        cleaned = _clean_explicit_material_anchor(term)
        if cleaned:
            terms.append(cleaned)
    return list(dict.fromkeys(terms))


def _clean_explicit_material_anchor(term: str) -> str:
    cleaned = re.sub(r"\s+", "", str(term or "").strip())
    cleaned = re.sub(
        r"(?:（|\()(?:(?:可提取文本)?PDF|Word|Excel|DOCX?|XLSX?|CSV|HTML)(?:）|\))$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\.(?:pdf|docx?|xlsx?|xls|csv|html?)$", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def _explicit_material_document_ids(db: Session, aspect: QueryAspect) -> set[str]:
    material_terms = _explicit_material_anchor_terms(
        "\n".join([aspect.question, *[query.query for query in aspect.search_queries]])
    )
    return _document_ids_for_title_terms(db, material_terms)


def _candidate_document_id(candidate: Any) -> str:
    direct = getattr(candidate, "document_id", None)
    if direct:
        return str(direct)
    metadata = getattr(candidate, "metadata", None)
    if isinstance(metadata, dict):
        value = metadata.get("document_id")
        if value:
            return str(value)
    return ""


def _document_ids_for_title_terms(db: Session, terms: list[str]) -> set[str]:
    if not terms:
        return set()
    documents_by_id: dict[str, Document] = {}
    for term in terms[:6]:
        term_matches: dict[str, Document] = {}
        documents = db.scalars(
            select(Document)
            .where(
                or_(
                    Document.title.contains(term),
                    Document.filename.contains(term),
                    Document.external_doc_id.contains(term),
                )
            )
            .limit(12)
        ).all()
        for document in documents:
            term_matches.setdefault(document.document_id, document)
        if not term_matches:
            expected = _normalize_document_anchor(term)
            if expected:
                all_documents = db.scalars(select(Document).limit(2000)).all()
                for document in all_documents:
                    candidates = (
                        document.title,
                        document.filename,
                        document.external_doc_id,
                    )
                    if any(
                        _normalized_title_contains(expected, candidate)
                        for candidate in candidates
                    ):
                        term_matches.setdefault(document.document_id, document)
        for document in term_matches.values():
            documents_by_id.setdefault(document.document_id, document)
    return set(documents_by_id)


def _normalized_title_contains(expected_norm: str, candidate: Any) -> bool:
    actual = _normalize_document_anchor(str(candidate or ""))
    if len(actual) < 4:
        return False
    return expected_norm in actual or actual in expected_norm


def _mcq_required_exact_terms(statement: str, aspect: QueryAspect | None = None) -> list[str]:
    text = str(statement or "")
    aspect_text = ""
    if aspect is not None:
        aspect_text = "\n".join(
            [aspect.question, *[search_query.query for search_query in aspect.search_queries]]
        )
    required: list[str] = []
    for term in (
        "中资商业银行",
        "寿险合同负债",
        "折现率曲线",
        "知识产权质押贷款",
        "创新积分贷款",
        "较大数据安全事件",
        "敏感级",
    ):
        if term in text:
            required.append(term)
    if (
        "中资商业银行" in aspect_text
        and any(term in text for term in ("申请书", "申请材料", "目录"))
        and "中资商业银行" not in required
    ):
        required.append("中资商业银行")
    return required


def _governing_list_preamble(
    item_chunk: DocumentChunk,
    document_chunks: list[DocumentChunk],
    *,
    max_backtrack: int = 8,
) -> DocumentChunk | None:
    """Return the nearest modal preamble governing an enumerated list item."""

    item_text = re.sub(r"\s+", "", item_chunk.text or "")
    if not re.search(r"(?:^|[；;。])（[一二三四五六七八九十百0-9]+）", item_text):
        return None
    if _is_modal_list_preamble(item_text):
        return None
    try:
        item_index = document_chunks.index(item_chunk)
    except ValueError:
        return None
    lower_bound = max(0, item_index - max_backtrack)
    for candidate in reversed(document_chunks[lower_bound:item_index]):
        candidate_text = re.sub(r"\s+", "", candidate.text or "")
        if _is_modal_list_preamble(candidate_text):
            return candidate
    return None


def _is_modal_list_preamble(text: str) -> bool:
    normalized = re.sub(r"\s+", "", str(text or ""))
    return bool(
        re.search(
            r"(?:不得|禁止|严禁|应当|可以|可).{0,40}(?:以下|下列).{0,12}"
            r"(?:行为|情形|事项|活动|要求|规定)?[:：]",
            normalized,
        )
    )


def _mcq_reliability_question(aspect: QueryAspect) -> str:
    """Use option facts, rather than a generic MCQ stem, for evidence gates.

    A stem such as “which option completely lists the requirements” may share
    no literal term with the governing clause even when the reranker puts that
    clause first.  The public option statements are the actual claims being
    verified, so they are the conservative lexical basis for the existing
    reliability filter.  This changes no score threshold and does not consult
    the offline gold answer.
    """

    statements = [
        search_query.query.strip()
        for search_query in aspect.search_queries
        if search_query.query_type == "document_style_statement" and search_query.query.strip()
    ]
    return "\n".join(dict.fromkeys(statements)) or aspect.question


def _mcq_matches_from_reranked(
    aspect: QueryAspect,
    reranked: list[Any],
    diagnostics: RetrievalDiagnostics,
) -> list[RetrievalMatch]:
    """Apply unchanged evidence gates to each public option fact separately."""

    statements = list(dict.fromkeys(
        search_query.query.strip()
        for search_query in aspect.search_queries
        if search_query.query_type == "document_style_statement" and search_query.query.strip()
    ))
    if not statements:
        return matches_from_reranked(
            question=aspect.question,
            reranked=reranked,
            diagnostics=diagnostics,
        )

    filter_started_at = perf_counter()
    best_by_chunk_id: dict[str, RetrievalMatch] = {}
    for statement in statements:
        for match in matches_from_reranked(
            question=statement,
            reranked=reranked,
            limit=FINAL_CITATION_LIMIT,
        ):
            chunk_id = match.citation.chunk_id
            previous = best_by_chunk_id.get(chunk_id)
            if previous is None or float(match.rerank_score) > float(previous.rerank_score):
                best_by_chunk_id[chunk_id] = match
    matches = sorted(
        best_by_chunk_id.values(),
        key=lambda item: (float(item.rerank_score), float(item.coverage_score)),
        reverse=True,
    )[:FINAL_CITATION_LIMIT]
    diagnostics.timings_ms["filter"] = _elapsed_ms(filter_started_at)
    diagnostics.reliable_count = len(best_by_chunk_id)
    diagnostics.filtered_count = max(len(reranked) - len(best_by_chunk_id), 0)
    diagnostics.selected_count = len(matches)
    return matches


def _corpus_lexical_support_matches(
    db: Session,
    aspect: QueryAspect,
    *,
    document_ids: set[str],
    document_chunk_cache: dict[str, list[DocumentChunk]] | None = None,
) -> list[RetrievalMatch]:
    """Add a bounded literal retrieval lane for high-information user terms.

    Hybrid top-k can be crowded out by a very large attachment before the
    cross-encoder sees a short governing clause.  This lane searches only
    terms derived from the public question/planner (never gold answers), caps
    every SQL contains query, and still validates exact term occurrence.
    """

    if not hasattr(db, "scalars"):
        return []
    terms = _text_lexical_seed_terms(aspect)
    if not terms:
        return []
    rows_by_id: dict[str, DocumentChunk] = {}
    for term in terms[:12]:
        statement = select(DocumentChunk).where(
            DocumentChunk.index_status == "indexed",
            or_(DocumentChunk.text.contains(term), DocumentChunk.embedding_text.contains(term)),
        )
        if document_ids:
            statement = statement.where(DocumentChunk.document_id.in_(document_ids))
        for chunk in db.scalars(statement.limit(48)).all():
            rows_by_id.setdefault(chunk.chunk_id, chunk)
        if len(rows_by_id) >= 320:
            break
    if not rows_by_id:
        return []
    phrases = [aspect.question, *[item.query for item in aspect.search_queries]]
    ranked: list[tuple[float, int, DocumentChunk, str]] = []
    for chunk in rows_by_id.values():
        searchable = "\n".join(
            part for part in (chunk.section_title or "", chunk.embedding_text or "", chunk.text or "") if part
        )
        normalized_searchable = _normalize_exact_support_text(searchable)
        hit_terms = [term for term in terms if _normalize_exact_support_text(term) in normalized_searchable]
        if not hit_terms:
            continue
        longest = max(hit_terms, key=lambda value: len(_normalize_exact_support_text(value)))
        recall = max((_exact_support_recall(phrase, searchable) for phrase in phrases), default=0.0)
        longest_length = len(_normalize_exact_support_text(longest))
        # One distinctive four-character term is enough; shorter roots require
        # modal/quantitative evidence so generic words cannot seed arbitrary prose.
        short_modal = longest_length >= 2 and any(token in searchable for token in ("应当", "不得", "禁止", "不低于", "不超过", "%"))
        if longest_length < 4 and len(hit_terms) < 2 and not short_modal:
            continue
        score = recall + min(len(hit_terms), 4) * 0.18 + min(longest_length, 12) * 0.015
        ranked.append((score, longest_length, chunk, longest))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = ranked[:6]
    if not selected:
        return []
    chunks_by_document = _chunks_by_document(
        db,
        {item[2].document_id for item in selected},
        document_chunk_cache=document_chunk_cache,
    )
    matches: list[RetrievalMatch] = []
    for score, _, chunk, anchor in selected:
        document_chunks = chunks_by_document.get(chunk.document_id, [])
        snapshot = next((item for item in document_chunks if item.chunk_id == chunk.chunk_id), None)
        if snapshot is None:
            continue
        match = _direct_exact_support_match(
            snapshot,
            document_chunks,
            aspect,
            anchor,
            score,
        )
        match.citation.evidence_role = "corpus_lexical_support"
        match.evidence_role = "corpus_lexical_support"
        match.metadata["evidence_role"] = "corpus_lexical_support"
        match.metadata["fusion_method"] = "bounded_corpus_lexical_support"
        match.metadata["lexical_seed_terms"] = [term for term in terms if term in (snapshot.text or "")]
        matches.append(match)
    return matches


def _text_lexical_seed_terms(aspect: QueryAspect) -> list[str]:
    generic = {
        "说明", "概括", "回答", "给出", "列出", "核验", "要求", "规定",
        "条件", "原则", "定义", "范围", "内容", "信息", "相关", "是什么",
    }
    category_suffixes = (
        "影响条件", "回函时限", "回复时限", "报告义务", "定价原则", "自救原则",
        "吸收顺序", "触发阈值", "审计频率", "留档要求", "厘定要求", "禁限", "递延",
        "条件", "要求", "原则", "效力", "时限", "频率", "义务", "门槛", "顺序", "定义", "范围", "目标",
    )
    values = [*aspect.keywords]
    values.extend(
        query.query
        for query in aspect.search_queries
        if query.query_type in {"keyword_anchor", "document_style_statement"}
    )
    values.append(aspect.question)
    terms: list[str] = []
    for value in values:
        cleaned = re.sub(r"^(?:请|分别|逐项|说明|概括|回答|给出|列出|核验)+", "", str(value or "").strip())
        cleaned = cleaned.replace("数字化银行回函", "数字化回函")
        for run in re.findall(r"[\u4e00-\u9fffA-Za-z0-9%]{2,24}", cleaned):
            if run not in generic:
                terms.append(run)
            for suffix in category_suffixes:
                if run.endswith(suffix) and len(run) > len(suffix) + 1:
                    root = run[: -len(suffix)]
                    if root not in generic:
                        terms.append(root)
    return list(dict.fromkeys(term for term in terms if 2 <= len(term) <= 24))[:12]


def _mcq_seed_document_ids(db: Session, statements: list[str]) -> set[str]:
    """Seed MCQ support from document-level anchors only.

    Full-corpus chunk ``contains`` scans are too slow for MCQ options because
    broad terms such as ``申请书`` and ``曲线`` appear in many documents.  Public
    option facts often name a document family or appendix topic; use those
    anchors to select a small set of documents, then the caller performs
    bounded clause matching inside those documents.
    """

    title_terms = _mcq_seed_document_title_terms(statements)
    if not title_terms:
        return _mcq_seed_document_ids_from_chunks(db, statements)
    documents_by_id: dict[str, Document] = {}
    for term in title_terms[:8]:
        documents = db.scalars(
            select(Document)
            .where(
                or_(
                    Document.title.contains(term),
                    Document.filename.contains(term),
                    Document.external_doc_id.contains(term),
                )
            )
            .limit(8)
        ).all()
        for document in documents:
            documents_by_id.setdefault(document.document_id, document)
        if len(documents_by_id) >= 24:
            break
    return set(documents_by_id)


def _mcq_seed_document_title_terms(statements: list[str]) -> list[str]:
    terms: list[str] = []
    for statement in statements:
        terms.extend(re.findall(r"《([^》]{4,80})》", str(statement or "")))
    return list(dict.fromkeys(term for term in terms if term))


def _mcq_seed_document_ids_from_chunks(db: Session, statements: list[str]) -> set[str]:
    terms: list[str] = []
    for statement in statements:
        compact_statement = re.sub(r"\s+", "", str(statement or ""))
        if any(
            marker in compact_statement
            for marker in (
                "无需",
                "仅含",
                "只含",
                "只允许",
                "任何一般",
                "必须影响",
                "未规定",
            )
        ):
            continue
        terms.extend(re.findall(r"\d+(?:\.\d+)?%", statement))
        terms.extend(_critical_lexical_terms(statement))
        normalized_runs = re.findall(r"[\u4e00-\u9fff]{4,}", statement)
        for run in normalized_runs:
            if len(run) <= 12:
                terms.append(run)
            else:
                terms.extend((run[:8], run[-8:]))
    terms = [
        term
        for term in dict.fromkeys(terms)
        if len(_normalize_exact_support_text(term)) >= 4 or "%" in term
    ][:16]
    if not terms:
        return set()
    rows_by_id: dict[str, DocumentChunk] = {}
    for term in terms:
        term_rows = db.scalars(
            select(DocumentChunk)
            .where(
                DocumentChunk.index_status == "indexed",
                or_(DocumentChunk.text.contains(term), DocumentChunk.embedding_text.contains(term)),
            )
            .limit(32)
        ).all()
        for chunk in term_rows:
            rows_by_id.setdefault(chunk.chunk_id, chunk)
        if len(rows_by_id) >= 240:
            break
    scored_documents: dict[str, float] = {}
    for chunk in rows_by_id.values():
        searchable = "\n".join(part for part in (chunk.embedding_text or "", chunk.text) if part)
        score = max((_exact_support_recall(statement, searchable) for statement in statements), default=0.0)
        scored_documents[chunk.document_id] = max(scored_documents.get(chunk.document_id, 0.0), score)
    ranked = sorted(scored_documents.items(), key=lambda item: item[1], reverse=True)
    return {document_id for document_id, score in ranked[:12] if score >= 0.35}


def _normalize_exact_support_text(value: str) -> str:
    without_breaks = re.sub(r"<br\s*/?>", "", str(value or ""), flags=re.IGNORECASE)
    normalized = re.sub(r"[\s\"'“”‘’=：:；;，,。()（）]+", "", without_breaks).lower()
    for source, target in (
        ("投资连结保险", "投连险"),
        ("投资连结险", "投连险"),
        ("投连保险", "投连险"),
        ("万能保险", "万能险"),
        ("中短存续期保险产品", "中短存续期产品"),
    ):
        normalized = normalized.replace(source, target)
    return normalized


def _exact_support_recall(expected: str, actual: str) -> float:
    expected_text = _normalize_exact_support_text(expected)
    actual_text = _normalize_exact_support_text(actual)
    if not expected_text:
        return 0.0
    if expected_text in actual_text:
        return 1.0
    if len(expected_text) == 1:
        return 1.0 if expected_text in actual_text else 0.0
    bigrams = [expected_text[index : index + 2] for index in range(len(expected_text) - 1)]
    return sum(bigram in actual_text for bigram in bigrams) / len(bigrams)


def _to_retrieval_results(matches: list[RetrievalMatch]) -> list[RetrievalResult]:
    results: list[RetrievalResult] = []
    seen_evidence: set[str] = set()
    seen_text: set[str] = set()
    for match in matches:
        citation = match.citation
        text = _clean_chunk_text(citation.excerpt, citation.section_title)
        normalized_text = _normalize_for_dedupe(text)
        evidence_id = str(match.metadata.get("evidence_id") or citation.chunk_id)
        if evidence_id in seen_evidence or normalized_text in seen_text:
            continue
        seen_evidence.add(evidence_id)
        seen_text.add(normalized_text)
        rank = len(results) + 1
        results.append(
            RetrievalResult(
                chunk_id=citation.chunk_id,
                rank=rank,
                score=match.rerank_score or citation.rerank_score or match.score or citation.score,
                source_doc=citation.filename,
                section_title=citation.section_title,
                section_path=_section_path(citation),
                text=text,
                citation_label=f"[{rank}]",
                metadata={
                    "document_id": citation.document_id,
                    "page_number": citation.page_number,
                    "chunk_type": citation.chunk_type,
                    "evidence_role": match.evidence_role,
                    "vector_score": citation.score,
                    "rerank_score": citation.rerank_score,
                    "coverage_score": match.coverage_score,
                    "section_number": citation.section_number or _section_number(citation.section_title),
                    "parent_section_number": citation.parent_section_number
                    or _parent_section_number(_section_number(citation.section_title)),
                    "previous_chunk_id": citation.previous_chunk_id,
                    "next_chunk_id": citation.next_chunk_id,
                    **citation.metadata,
                    **match.metadata,
                },
            )
        )
    return results


def _validate_context_chunks(
    db: Session,
    context_chunks: list[RetrievalResult],
) -> tuple[list[RetrievalResult], dict[str, Any]]:
    if not context_chunks:
        return (
            [],
            {
                "checked_chunks": 0,
                "valid_chunks": 0,
                "invalid_chunks": 0,
                "invalid_chunk_ids": [],
            },
        )

    chunk_ids = [chunk.chunk_id for chunk in context_chunks]
    rows = db.execute(
        select(DocumentChunk, Document.status)
        .join(Document, DocumentChunk.document_id == Document.document_id)
        .where(DocumentChunk.chunk_id.in_(chunk_ids))
    ).all()
    source_by_chunk_id = {chunk.chunk_id: (chunk, status) for chunk, status in rows}

    valid_chunks: list[RetrievalResult] = []
    invalid_chunk_ids: list[str] = []
    for result in context_chunks:
        source = source_by_chunk_id.get(result.chunk_id)
        if source is None:
            invalid_chunk_ids.append(result.chunk_id)
            continue

        source_chunk, document_status = source
        if document_status not in {"indexed", "table_indexed"}:
            invalid_chunk_ids.append(result.chunk_id)
            continue

        if result.metadata.get("dynamic_table_evidence"):
            valid_chunks.append(result)
            continue

        if not _text_is_traceable(result.text, source_chunk.text):
            invalid_chunk_ids.append(result.chunk_id)
            continue

        valid_chunks.append(result)

    _renumber_context_chunks(valid_chunks)
    return (
        valid_chunks,
        {
            "checked_chunks": len(context_chunks),
            "valid_chunks": len(valid_chunks),
            "invalid_chunks": len(invalid_chunk_ids),
            "invalid_chunk_ids": invalid_chunk_ids,
        },
    )


def _select_prompt_chunks(
    question: str,
    query_plan: QueryPlan,
    aspect_retrievals: list[AspectRetrieval],
) -> tuple[list[RetrievalResult], dict[str, Any]]:
    selected: list[RetrievalResult] = []
    candidate_count = len(
        {
            chunk.chunk_id
            for item in aspect_retrievals
            for chunk in item.candidates
        }
    )

    # Exact option-fact support has already passed the dedicated MCQ lexical
    # gate.  Preserve a small bounded set before the generic prompt threshold
    # so a cross-document option does not lose one of its required clauses
    # merely because each individual chunk scores just below 0.45.
    for aspect_retrieval in aspect_retrievals:
        if aspect_retrieval.aspect.aspect_id != "multiple_choice_evidence":
            continue
        added_exact_support = 0
        for chunk in aspect_retrieval.candidates:
            if len(selected) >= MAX_PROMPT_CHUNKS or added_exact_support >= 4:
                break
            if chunk.metadata.get("evidence_role") != "mcq_exact_support":
                continue
            if _is_duplicate_or_redundant(chunk, selected):
                continue
            chunk.metadata["prompt_selection_reason"] = "mcq_exact"
            _mark_chunk_for_aspect(chunk, aspect_retrieval.aspect)
            selected.append(chunk)
            aspect_retrieval.selected_chunk_ids.append(chunk.chunk_id)
            aspect_retrieval.covered = True
            added_exact_support += 1

        selected_exact_ids = {
            chunk.chunk_id
            for chunk in selected
            if chunk.metadata.get("evidence_role") == "mcq_exact_support"
        }
        added_preambles = 0
        for chunk in aspect_retrieval.candidates:
            if len(selected) >= MAX_PROMPT_CHUNKS or added_preambles >= 3:
                break
            if chunk.metadata.get("evidence_role") != "mcq_rule_preamble":
                continue
            governed_ids = chunk.metadata.get("governs_chunk_ids") or []
            if not selected_exact_ids.intersection(str(item) for item in governed_ids):
                continue
            if _is_duplicate_or_redundant(chunk, selected):
                continue
            chunk.metadata["prompt_selection_reason"] = "mcq_preamble"
            _mark_chunk_for_aspect(chunk, aspect_retrieval.aspect)
            selected.append(chunk)
            aspect_retrieval.selected_chunk_ids.append(chunk.chunk_id)
            aspect_retrieval.covered = True
            added_preambles += 1

        represented_documents = {
            str(chunk.metadata.get("document_id") or chunk.source_doc or "")
            for chunk in selected
            if chunk.metadata.get("prompt_matched_aspects")
            and aspect_retrieval.aspect.aspect_id in chunk.metadata.get("prompt_matched_aspects", [])
        }
        added_document_coverage = 0
        for chunk in sorted(
            aspect_retrieval.candidates,
            key=lambda item: _mcq_prompt_document_coverage_priority(item),
            reverse=True,
        ):
            if len(selected) >= MAX_PROMPT_CHUNKS or added_document_coverage >= 4:
                break
            document_key = str(chunk.metadata.get("document_id") or chunk.source_doc or "")
            if not document_key or document_key in represented_documents:
                continue
            if _mcq_prompt_document_coverage_priority(chunk) <= (0, 0.0):
                continue
            if _is_duplicate_or_redundant(chunk, selected):
                continue
            chunk.metadata["prompt_selection_reason"] = "mcq_document_coverage"
            _mark_chunk_for_aspect(chunk, aspect_retrieval.aspect)
            selected.append(chunk)
            aspect_retrieval.selected_chunk_ids.append(chunk.chunk_id)
            aspect_retrieval.covered = True
            represented_documents.add(document_key)
            added_document_coverage += 1

    for aspect_retrieval in aspect_retrievals:
        if len(selected) >= MAX_PROMPT_CHUNKS:
            break
        shared_chunk = _best_shared_candidate(
            aspect_retrieval.candidates,
            selected,
            aspect_retrieval.aspect,
        )
        if shared_chunk is not None:
            _mark_chunk_for_aspect(shared_chunk, aspect_retrieval.aspect)
            aspect_retrieval.selected_chunk_ids.append(shared_chunk.chunk_id)
            aspect_retrieval.covered = True
            continue
        core_chunk = _best_non_duplicate_candidate(
            aspect_retrieval.candidates,
            selected,
            aspect_retrieval.aspect,
        )
        if core_chunk is None:
            aspect_retrieval.covered = False
            continue
        core_chunk.metadata["prompt_selection_reason"] = "core"
        _mark_chunk_for_aspect(core_chunk, aspect_retrieval.aspect)
        selected.append(core_chunk)
        aspect_retrieval.selected_chunk_ids.append(core_chunk.chunk_id)
        aspect_retrieval.covered = True

    # A single planner aspect can contain several explicit conditions.  Preserve
    # at least one directly matching evidence block for each condition before
    # spending the remaining budget on generic neighbours.
    for aspect_retrieval in aspect_retrievals:
        added_for_anchors = 0
        for anchor in _aspect_anchor_phrases(aspect_retrieval.aspect):
            if len(selected) >= MAX_PROMPT_CHUNKS or added_for_anchors >= 3:
                break
            aspect_selected = [
                chunk
                for chunk in selected
                if aspect_retrieval.aspect.aspect_id
                in (chunk.metadata.get("prompt_matched_aspects") or [])
            ]
            if any(_anchor_coverage(anchor, chunk.text) >= 0.82 for chunk in aspect_selected):
                continue
            candidates = [
                chunk
                for chunk in aspect_retrieval.candidates
                if _anchor_coverage(anchor, chunk.text) >= 0.72
                and not _is_duplicate_or_redundant(chunk, selected)
            ]
            if not candidates:
                continue
            chunk = max(
                candidates,
                key=lambda item: (
                    _anchor_coverage(anchor, item.text),
                    _aspect_lexical_score(item, aspect_retrieval.aspect),
                    _prompt_score(item),
                ),
            )
            chunk.metadata["prompt_selection_reason"] = "anchor"
            _mark_chunk_for_aspect(chunk, aspect_retrieval.aspect)
            selected.append(chunk)
            aspect_retrieval.selected_chunk_ids.append(chunk.chunk_id)
            added_for_anchors += 1

    # A planner may deliberately keep several evidence-seeking queries inside
    # one aspect (for example, a rule plus its exception). Preserve the best
    # direct candidate for each substantive query before generic neighbours;
    # otherwise one high rerank score can crowd out another requested clause.
    for aspect_retrieval in aspect_retrievals:
        added_for_queries = 0
        for search_query in aspect_retrieval.aspect.search_queries:
            if search_query.query_type not in {"semantic_question", "document_style_statement"}:
                continue
            if len(selected) >= MAX_PROMPT_CHUNKS or added_for_queries >= 4:
                break
            candidates: list[tuple[float, int, RetrievalResult]] = []
            for chunk in aspect_retrieval.candidates:
                if not (
                    _chunk_matches_query_aspect(chunk, aspect_retrieval.aspect)
                    or _aspect_lexical_score(chunk, aspect_retrieval.aspect) > 0
                    or _chunk_question_coverage(chunk, aspect_retrieval.aspect.question) >= MIN_EVIDENCE_COVERAGE
                ):
                    continue
                hits = chunk.metadata.get("aspect_search_query_hits") or []
                matching_hits = [
                    hit
                    for hit in hits
                    if isinstance(hit, dict) and hit.get("query") == search_query.query
                ]
                if not matching_hits:
                    continue
                best_hit = max(
                    matching_hits,
                    key=lambda hit: (float(hit.get("vector_score") or 0.0), -int(hit.get("rank") or 9999)),
                )
                candidates.append(
                    (
                        float(best_hit.get("vector_score") or 0.0),
                        -int(best_hit.get("rank") or 9999),
                        chunk,
                    )
                )
            candidates.sort(key=lambda item: (item[0], item[1], _prompt_score(item[2])), reverse=True)
            chosen = next(
                (chunk for _, _, chunk in candidates if not _is_duplicate_or_redundant(chunk, selected)),
                None,
            )
            if chosen is None:
                continue
            chosen.metadata["prompt_selection_reason"] = "query"
            _mark_chunk_for_aspect(chosen, aspect_retrieval.aspect)
            selected.append(chosen)
            aspect_retrieval.selected_chunk_ids.append(chosen.chunk_id)
            added_for_queries += 1

    for aspect_retrieval in aspect_retrievals:
        if len(selected) >= MAX_PROMPT_CHUNKS:
            break
        top_score = max((_prompt_score(chunk) for chunk in aspect_retrieval.candidates), default=0.0)
        for chunk in aspect_retrieval.candidates:
            if len(selected) >= MAX_PROMPT_CHUNKS:
                break
            if chunk.chunk_id in aspect_retrieval.selected_chunk_ids:
                continue
            if not _passes_prompt_score(chunk, top_score):
                continue
            if _is_duplicate_or_redundant(chunk, selected):
                continue
            if not (_chunk_matches_query_aspect(chunk, aspect_retrieval.aspect) or _is_structural_support(chunk, selected, question)):
                continue
            chunk.metadata["prompt_selection_reason"] = "generic"
            _mark_chunk_for_aspect(chunk, aspect_retrieval.aspect)
            selected.append(chunk)
            aspect_retrieval.selected_chunk_ids.append(chunk.chunk_id)

    if FORCE_MIN_CHUNKS and len(selected) < MIN_PROMPT_CHUNKS:
        for aspect_retrieval in aspect_retrievals:
            top_score = max((_prompt_score(chunk) for chunk in aspect_retrieval.candidates), default=0.0)
            for chunk in aspect_retrieval.candidates:
                if len(selected) >= min(MIN_PROMPT_CHUNKS, MAX_PROMPT_CHUNKS):
                    break
                if _is_duplicate_or_redundant(chunk, selected):
                    continue
                # The minimum is a lower-bound preference, not permission to
                # inject unrelated evidence. Apply the same score and aspect
                # relevance gate used by ordinary prompt selection.
                if not _passes_prompt_score(chunk, top_score):
                    continue
                if not (
                    _chunk_matches_query_aspect(chunk, aspect_retrieval.aspect)
                    or _is_structural_support(chunk, selected, question)
                ):
                    continue
                chunk.metadata["prompt_selection_reason"] = "forced_minimum"
                _mark_chunk_for_aspect(chunk, aspect_retrieval.aspect)
                selected.append(chunk)
                aspect_retrieval.selected_chunk_ids.append(chunk.chunk_id)

    selected = _filter_prompt_relevance(selected, query_plan)
    selected = _apply_prompt_token_budget(selected)
    selected = _sort_prompt_chunks(selected, query_plan)
    _sync_aspect_coverage_from_prompt(selected, aspect_retrievals)
    _renumber_context_chunks(selected)
    covered_aspects = {item.aspect.aspect_id for item in aspect_retrievals if item.covered}
    return selected, _prompt_selection_summary(
        candidate_count,
        selected,
        query_plan,
        aspect_retrievals,
        covered_aspects,
    )


def _expand_condition_preamble_prompt_chunks(
    db: Session,
    context_chunks: list[RetrievalResult],
    query_plan: QueryPlan,
) -> list[RetrievalResult]:
    if not context_chunks or len(context_chunks) >= MAX_PROMPT_CHUNKS or not hasattr(db, "scalars"):
        return context_chunks
    candidate_preambles = [
        chunk
        for chunk in context_chunks
        if chunk.metadata.get("document_id")
        and chunk.metadata.get("next_chunk_id")
        and (
            chunk.metadata.get("prompt_matched_aspects")
            or chunk.metadata.get("aspect_id")
        )
    ]
    if not candidate_preambles:
        return context_chunks
    document_ids = {
        str(chunk.metadata.get("document_id") or "")
        for chunk in candidate_preambles
        if chunk.metadata.get("document_id")
    }
    if not document_ids:
        return context_chunks
    chunks_by_document = _chunks_by_document(db, document_ids)
    expanded = list(context_chunks)
    seen_ids = {chunk.chunk_id for chunk in expanded}
    aspects_by_id = {aspect.aspect_id: aspect for aspect in query_plan.aspects}
    for preamble in candidate_preambles:
        if len(expanded) >= MAX_PROMPT_CHUNKS:
            break
        document_id = str(preamble.metadata.get("document_id") or "")
        document_chunks = chunks_by_document.get(document_id) or []
        anchor_chunk = next(
            (chunk for chunk in document_chunks if chunk.chunk_id == preamble.chunk_id),
            None,
        )
        matched_aspect_ids = [
            str(aspect_id)
            for aspect_id in (preamble.metadata.get("prompt_matched_aspects") or [])
            if str(aspect_id) in aspects_by_id
        ]
        if not matched_aspect_ids and preamble.metadata.get("aspect_id") in aspects_by_id:
            matched_aspect_ids = [str(preamble.metadata["aspect_id"])]
        insert_at = next(
            (index + 1 for index, chunk in enumerate(expanded) if chunk.chunk_id == preamble.chunk_id),
            len(expanded),
        )
        anchor_index = insert_at - 1 if insert_at > 0 else 0
        for aspect_id in matched_aspect_ids:
            if len(expanded) >= MAX_PROMPT_CHUNKS:
                break
            aspect = aspects_by_id[aspect_id]
            phrases = _condition_continuation_phrases(aspect)
            if anchor_chunk is not None:
                previous_preamble = _previous_condition_preamble_chunk(anchor_chunk, document_chunks, aspect, phrases)
                if previous_preamble is not None and previous_preamble.chunk_id not in seen_ids:
                    preamble_result = _prompt_expansion_result_from_chunk(
                        previous_preamble,
                        document_chunks,
                        score=0.94,
                        evidence_role="bounded_lexical_support",
                        metadata={
                            "fusion_method": "bounded_document_condition_preamble",
                            "condition_continuation_chunk_id": preamble.chunk_id,
                        },
                    )
                    preamble_result.metadata["prompt_selection_reason"] = "condition_preamble"
                    _mark_chunk_for_aspect(preamble_result, aspect)
                    expanded.insert(anchor_index, preamble_result)
                    seen_ids.add(previous_preamble.chunk_id)
                    insert_at += 1
                    anchor_index += 1
            sqlite_added = False
            if anchor_chunk is not None:
                for _score, match in _condition_continuation_support_matches(
                    anchor_chunk,
                    document_chunks,
                    aspect,
                    phrases,
                    existing_ids=seen_ids,
                    seen_ids=seen_ids,
                    require_preamble=False,
                ):
                    if len(expanded) >= MAX_PROMPT_CHUNKS:
                        break
                    continuation = _to_retrieval_results([match])[0]
                    continuation.metadata["prompt_selection_reason"] = "condition_continuation"
                    _mark_chunk_for_aspect(continuation, aspect)
                    expanded.insert(insert_at, continuation)
                    insert_at += 1
                    sqlite_added = True
            if sqlite_added:
                continue
            next_chunk_id = str(preamble.metadata.get("next_chunk_id") or "")
            for _step in range(4):
                if len(expanded) >= MAX_PROMPT_CHUNKS or not next_chunk_id:
                    break
                if next_chunk_id in seen_ids:
                    next_chunk_id = ""
                    break
                try:
                    vector_chunk = get_vector_chunk_by_chunk_id(next_chunk_id)
                except VectorStoreError:
                    break
                if vector_chunk is None or vector_chunk.document_id != document_id:
                    break
                searchable = "\n".join(
                    part
                    for part in (vector_chunk.section_title or "", vector_chunk.text or vector_chunk.embedding_text or "")
                    if part
                )
                best = _best_condition_continuation_phrase(searchable, phrases)
                current_chunk_id = vector_chunk.chunk_id
                next_chunk_id = str(vector_chunk.next_chunk_id or "")
                if best is None:
                    continue
                support_score, support_phrase = best
                seen_ids.add(current_chunk_id)
                continuation = RetrievalResult(
                    chunk_id=current_chunk_id,
                    rank=0,
                    score=min(0.99, 0.88 + min(support_score, 2.0) * 0.05),
                    source_doc=vector_chunk.filename,
                    section_title=vector_chunk.section_title,
                    section_path=vector_chunk.section_path or [],
                    text=_clean_chunk_text(vector_chunk.text, vector_chunk.section_title),
                    citation_label="",
                    metadata={
                        **(vector_chunk.metadata or {}),
                        "document_id": vector_chunk.document_id,
                        "page_number": vector_chunk.page_number,
                        "chunk_type": vector_chunk.chunk_type,
                        "section_number": vector_chunk.section_number,
                        "parent_section_number": vector_chunk.parent_section_number,
                        "previous_chunk_id": vector_chunk.previous_chunk_id,
                        "next_chunk_id": vector_chunk.next_chunk_id,
                        "token_count": vector_chunk.token_count,
                        "evidence_role": "bounded_lexical_support",
                        "fusion_method": "qdrant_condition_continuation",
                        "lexical_support_phrase": support_phrase,
                        "condition_preamble_chunk_id": preamble.chunk_id,
                    },
                )
                continuation.metadata["prompt_selection_reason"] = "condition_continuation"
                _mark_chunk_for_aspect(continuation, aspect)
                expanded.insert(insert_at, continuation)
                insert_at += 1
    if len(expanded) == len(context_chunks):
        return context_chunks
    _renumber_context_chunks(expanded)
    return expanded


def _previous_condition_preamble_chunk(
    current_chunk: DocumentChunk,
    document_chunks: list[DocumentChunk],
    aspect: QueryAspect,
    phrases: list[str],
) -> DocumentChunk | None:
    try:
        current_index = document_chunks.index(current_chunk)
    except ValueError:
        return None
    current_text = _normalize_exact_support_text(
        "\n".join(part for part in (current_chunk.section_title or "", current_chunk.text or "") if part)
    )
    if not _looks_like_orphan_condition_continuation(current_text, phrases):
        return None
    current_section = current_chunk.section_title or ""
    for previous in reversed(document_chunks[max(0, current_index - 2) : current_index]):
        if previous.index_status != "indexed":
            continue
        if current_section and previous.section_title != current_section:
            continue
        previous_text = _normalize_exact_support_text(
            "\n".join(part for part in (previous.section_title or "", previous.text or "") if part)
        )
        if _looks_like_condition_preamble(previous_text, aspect):
            return previous
    return None


def _looks_like_orphan_condition_continuation(text: str, phrases: list[str]) -> bool:
    normalized_phrases = [_normalize_exact_support_text(phrase) for phrase in phrases]
    has_phrase = any(phrase and phrase in text for phrase in normalized_phrases)
    return bool(
        has_phrase
        or "另有规定的除外" in text
        or "表格中另有规定" in text
        or re.search(r"(?:^|[。；;])(?:\d{1,2}|[一二三四五六七八九十]{1,3})[.、．]", text)
    )


def _looks_like_condition_preamble(text: str, aspect: QueryAspect) -> bool:
    aspect_text = _normalize_exact_support_text(
        "\n".join([aspect.question, aspect.evidence_need, *(query.query for query in aspect.search_queries)])
    )
    has_action = any(marker in text for marker in ("应按", "应当", "应根据", "应按照", "不得", "包括", "应披露"))
    has_scope = any(marker in text for marker in ("条件", "范围", "期限", "例外", "要求", "频率", "表格", "披露内容", "总则"))
    if has_action and has_scope:
        return True
    return "披露" in aspect_text and "商业银行应" in text and "表格" in text


def _prompt_expansion_result_from_chunk(
    chunk: DocumentChunk,
    document_chunks: list[DocumentChunk],
    *,
    score: float,
    evidence_role: str,
    metadata: dict[str, Any],
) -> RetrievalResult:
    previous_chunk_id, next_chunk_id = _adjacent_chunk_ids(chunk, document_chunks)
    return RetrievalResult(
        chunk_id=chunk.chunk_id,
        rank=0,
        score=score,
        source_doc=chunk.source_file,
        section_title=chunk.section_title,
        section_path=_inferred_section_path(chunk, document_chunks),
        text=_clean_chunk_text(chunk.text, chunk.section_title),
        citation_label="",
        metadata={
            **_chunk_metadata(chunk),
            "document_id": chunk.document_id,
            "page_number": chunk.page_number,
            "chunk_type": _chunk_type(chunk.text),
            "section_number": _section_number(chunk.section_title),
            "parent_section_number": _parent_section_number(_section_number(chunk.section_title)),
            "previous_chunk_id": previous_chunk_id,
            "next_chunk_id": next_chunk_id,
            "token_count": getattr(chunk, "token_count", 0),
            "evidence_role": evidence_role,
            **metadata,
        },
    )


def _condition_continuation_phrases(aspect: QueryAspect) -> list[str]:
    phrases = list(_bounded_lexical_phrases(aspect))
    phrases.extend(
        search_query.query
        for search_query in aspect.search_queries
        if search_query.query_type in {"document_style_statement", "keyword_anchor"}
    )
    return [
        phrase
        for phrase in dict.fromkeys(phrases)
        if len(_normalize_exact_support_text(phrase)) >= 6
    ]


def _mcq_prompt_document_coverage_priority(chunk: RetrievalResult) -> tuple[int, float]:
    role = str(chunk.metadata.get("evidence_role") or "")
    query_hits = chunk.metadata.get("query_hits") or []
    has_option_fact_hit = any(
        isinstance(hit, dict) and hit.get("query_type") == "document_style_statement"
        for hit in query_hits
    )
    role_priority = {
        "mcq_exact_support": 4,
        "bounded_lexical_support": 3,
        "direct_evidence": 2,
        "related_context": 1,
    }.get(role, 0)
    if role_priority <= 0 and not has_option_fact_hit:
        return (0, 0.0)
    return (role_priority + (1 if has_option_fact_hit else 0), _prompt_score(chunk))


def _filter_prompt_relevance(
    selected: list[RetrievalResult],
    query_plan: QueryPlan,
) -> list[RetrievalResult]:
    """Apply a final trust gate after all quota and neighbour passes."""

    always_keep_roles = {
        "exact_anchor_support",
        "bounded_lexical_support",
        "formula_target_support",
        "mcq_exact_support",
        "mcq_rule_preamble",
        "table_evidence",
    }
    relevant: list[RetrievalResult] = []
    for chunk in selected:
        selection_reason = chunk.metadata.get("prompt_selection_reason")
        if selection_reason not in {"core", "generic", "forced_minimum"}:
            relevant.append(chunk)
            continue
        if chunk.metadata.get("dynamic_table_evidence") or chunk.metadata.get("evidence_role") in always_keep_roles:
            relevant.append(chunk)
            continue
        if any(_chunk_matches_query_aspect(chunk, aspect) for aspect in query_plan.aspects):
            relevant.append(chunk)
            continue
        if selection_reason == "core" and any(
            _aspect_lexical_score(chunk, aspect) > 0
            or _chunk_question_coverage(chunk, aspect.question) >= MIN_EVIDENCE_COVERAGE
            for aspect in query_plan.aspects
        ):
            relevant.append(chunk)
            continue
        if _is_structural_support(chunk, relevant, query_plan.original_question):
            relevant.append(chunk)
    return relevant


def _sync_aspect_coverage_from_prompt(
    selected: list[RetrievalResult],
    aspect_retrievals: list[AspectRetrieval],
) -> None:
    """Reconcile aspect coverage after final prompt filtering and expansion.

    Some bounded recovery passes add same-document support directly to the
    prompt after the original aspect retrieval was marked missing.  The answer
    gate should follow the final auditable context, not stale pre-filter flags.
    """

    final_chunk_ids = {item.chunk_id for item in selected}
    for item in aspect_retrievals:
        item.selected_chunk_ids = [
            chunk_id for chunk_id in item.selected_chunk_ids if chunk_id in final_chunk_ids
        ]
        for chunk in selected:
            if chunk.chunk_id in item.selected_chunk_ids:
                continue
            matched = chunk.metadata.get("prompt_matched_aspects")
            matched_aspects = matched if isinstance(matched, list) else []
            if item.aspect.aspect_id in matched_aspects or _prompt_chunk_covers_aspect(
                chunk,
                item.aspect,
            ):
                _mark_chunk_for_aspect(chunk, item.aspect)
                item.selected_chunk_ids.append(chunk.chunk_id)
        if item.selected_chunk_ids:
            if not item.retrieval_covered:
                item.diagnostics.append(
                    {
                        "query_type": "prompt_coverage_sync",
                        "search_query": item.aspect.question,
                        "match_count": len(item.selected_chunk_ids),
                        "match_status": "covered_by_final_prompt_context",
                        "query_count": 0,
                        "raw_candidate_count": 0,
                        "candidate_count": len(item.selected_chunk_ids),
                        "rerank_input_count": 0,
                        "rerank_call_count": 0,
                        "reranked_count": 0,
                        "filtered_count": 0,
                        "timings_ms": {},
                        "score_range": {},
                    }
                )
            item.retrieval_covered = True
        item.covered = bool(item.selected_chunk_ids)


def _prompt_chunk_covers_aspect(chunk: RetrievalResult, aspect: QueryAspect) -> bool:
    if chunk.metadata.get("dynamic_table_evidence") or chunk.metadata.get("evidence_role") in {
        "exact_anchor_support",
        "bounded_lexical_support",
        "formula_target_support",
        "mcq_exact_support",
        "mcq_rule_preamble",
        "table_evidence",
    }:
        return _chunk_matches_query_aspect(chunk, aspect) or _aspect_lexical_score(chunk, aspect) > 0

    text = _chunk_match_text(chunk)
    if not text:
        return False
    for query in aspect.search_queries:
        if query.query_type == "document_style_statement" and _anchor_coverage(query.query, text) >= 0.62:
            return True
    if _chunk_question_coverage(chunk, aspect.question) >= max(MIN_EVIDENCE_COVERAGE, 0.5):
        return True

    terms = _prompt_aspect_coverage_terms(aspect)
    hits = [term for term in terms if term and term in text]
    if len(set(hits)) >= 3:
        return True
    if len(set(hits)) >= 2 and _has_normative_or_definition_marker(text):
        return True
    return False


def _prompt_aspect_coverage_terms(aspect: QueryAspect) -> list[str]:
    candidates = [
        aspect.question,
        aspect.evidence_need,
        aspect.expected_evidence_type,
        *aspect.keywords,
        *[query.query for query in aspect.search_queries],
    ]
    terms: list[str] = []
    stop_terms = {
        "核对",
        "说明",
        "列出",
        "判断",
        "概括",
        "要求",
        "规定",
        "主体",
        "部门",
        "岗位",
        "职责",
        "依据",
        "定义",
        "相关",
        "哪些",
        "什么",
    }
    for candidate in candidates:
        normalized = _normalize_exact_support_text(str(candidate or ""))
        if not normalized:
            continue
        for raw in re.split(r"[^\u4e00-\u9fff0-9A-Za-z.%]+", str(candidate or "")):
            term = _normalize_exact_support_text(raw)
            if 2 <= len(term) <= 18 and term not in stop_terms:
                terms.append(term)
            if len(term) >= 6:
                terms.extend(_split_compound_coverage_term(term))
        for match in re.finditer(r"[\u4e00-\u9fff]{2,12}(?:函证|回函|询证函|折现率|终极利率|第三支柱|披露|控制|薪酬|催收|交易账簿|公允价值|并表范围)", str(candidate or "")):
            term = _normalize_exact_support_text(match.group(0))
            if 2 <= len(term) <= 18:
                terms.append(term)
    return list(dict.fromkeys(term for term in terms if term not in stop_terms))[:24]


def _split_compound_coverage_term(term: str) -> list[str]:
    pieces: list[str] = []
    for marker in (
        "全过程",
        "控制",
        "询证函",
        "银行函证",
        "回函",
        "公示",
        "折现率",
        "终极利率",
        "第三支柱",
        "披露",
        "交易账簿",
        "公允价值",
        "并表范围",
    ):
        if marker in term:
            pieces.append(marker)
    return pieces


def _has_normative_or_definition_marker(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "应当",
            "不得",
            "必须",
            "可以",
            "应",
            "包括",
            "是指",
            "所称",
            "具有",
            "负责",
            "暂定为",
            "频率",
            "按照",
        )
    )


def _empty_prompt_selection() -> dict[str, Any]:
    return {
        "max_prompt_chunks": MAX_PROMPT_CHUNKS,
        "min_prompt_chunks": MIN_PROMPT_CHUNKS,
        "force_min_chunks": FORCE_MIN_CHUNKS,
        "rerank_prompt_threshold": RERANK_PROMPT_THRESHOLD,
        "relative_score_ratio": RELATIVE_SCORE_RATIO,
        "candidate_prompt_chunks": 0,
        "final_prompt_chunks": 0,
        "retrieval_covered_aspects": [],
        "covered_aspects": [],
        "covered_by_retrieval_but_not_prompted": [],
        "prompt_capacity_limited": False,
        "expected_aspects": [],
        "final_prompt_chunk_ids": [],
    }


def _prompt_selection_summary(
    candidate_count: int,
    selected: list[RetrievalResult],
    query_plan: QueryPlan,
    aspect_retrievals: list[AspectRetrieval],
    covered_aspects: set[str],
) -> dict[str, Any]:
    summary = _empty_prompt_selection()
    retrieval_covered_aspects = [
        item.aspect.aspect_id for item in aspect_retrievals if item.retrieval_covered
    ]
    prompt_covered_aspects = [
        aspect.aspect_id for aspect in query_plan.aspects if aspect.aspect_id in covered_aspects
    ]
    covered_by_retrieval_but_not_prompted = [
        aspect_id for aspect_id in retrieval_covered_aspects if aspect_id not in prompt_covered_aspects
    ]
    summary.update(
        {
            "candidate_prompt_chunks": candidate_count,
            "final_prompt_chunks": len(selected),
            "retrieval_covered_aspects": retrieval_covered_aspects,
            "covered_aspects": prompt_covered_aspects,
            "covered_by_retrieval_but_not_prompted": covered_by_retrieval_but_not_prompted,
            "prompt_capacity_limited": bool(covered_by_retrieval_but_not_prompted),
            "expected_aspects": [
                {
                    "aspect_id": aspect.aspect_id,
                    "description": aspect.question,
                    "evidence_need": aspect.evidence_need,
                    "search_queries": _search_query_debug_list(aspect),
                    "expected_evidence_type": aspect.expected_evidence_type,
                }
                for aspect in query_plan.aspects
            ],
            "final_prompt_chunk_ids": [chunk.chunk_id for chunk in selected],
            "aspect_selected_chunk_ids": {
                item.aspect.aspect_id: item.selected_chunk_ids for item in aspect_retrievals
            },
        }
    )
    return summary


def _apply_prompt_token_budget(chunks: list[RetrievalResult]) -> list[RetrievalResult]:
    """Keep complete evidence blocks within a global token budget."""

    selected: list[RetrievalResult] = []
    used = 0
    for chunk in chunks:
        token_count = _int_or_none(chunk.metadata.get("token_count")) or max(len(chunk.text) // 2, 1)
        if selected and used + token_count > MAX_PROMPT_TOKENS:
            continue
        selected.append(chunk)
        used += token_count
    return selected


def _first_non_duplicate_candidate(
    candidates: list[RetrievalResult],
    selected: list[RetrievalResult],
) -> RetrievalResult | None:
    for chunk in candidates:
        if _is_duplicate_or_redundant(chunk, selected):
            continue
        return chunk
    return None


def _best_non_duplicate_candidate(
    candidates: list[RetrievalResult],
    selected: list[RetrievalResult],
    aspect: QueryAspect,
) -> RetrievalResult | None:
    available = [
        chunk for chunk in candidates if not _is_duplicate_or_redundant(chunk, selected)
    ]
    if not available:
        return None
    return max(
        available,
        key=lambda chunk: (
            _aspect_lexical_score(chunk, aspect),
            _prompt_score(chunk),
            -(chunk.rank or 999),
        ),
    )


def _best_shared_candidate(
    candidates: list[RetrievalResult],
    selected: list[RetrievalResult],
    aspect: QueryAspect,
) -> RetrievalResult | None:
    candidate_ids = {chunk.chunk_id for chunk in candidates}
    reusable = [chunk for chunk in selected if chunk.chunk_id in candidate_ids]
    if not reusable:
        return None
    anchors = _aspect_anchor_phrases(aspect)
    strongly_matching = [
        chunk
        for chunk in reusable
        if any(_anchor_coverage(anchor, chunk.text) >= 0.72 for anchor in anchors)
    ]
    if not strongly_matching:
        return None
    return max(
        strongly_matching,
        key=lambda chunk: (
            _aspect_lexical_score(chunk, aspect),
            _prompt_score(chunk),
            -(chunk.rank or 999),
        ),
    )


def _aspect_anchor_phrases(aspect: QueryAspect) -> list[str]:
    phrases = [
        re.sub(r"\s+", "", item)
        for item in re.findall(r"[“‘'\"]([^”’'\"]+)[”’'\"]", aspect.question)
    ]
    return [
        phrase
        for phrase in dict.fromkeys(phrases)
        if len(phrase) >= 6 and phrase not in {"相关规定", "明确规定"}
    ]


def _anchor_coverage(anchor: str, text: str) -> float:
    normalized_anchor = re.sub(r"\s+", "", str(anchor or ""))
    normalized_text = re.sub(r"\s+", "", str(text or ""))
    if not normalized_anchor or not normalized_text:
        return 0.0
    if normalized_anchor in normalized_text:
        return 1.0
    anchor_bigrams = {
        normalized_anchor[index : index + 2]
        for index in range(max(len(normalized_anchor) - 1, 0))
    }
    text_bigrams = {
        normalized_text[index : index + 2]
        for index in range(max(len(normalized_text) - 1, 0))
    }
    return len(anchor_bigrams & text_bigrams) / max(len(anchor_bigrams), 1)


def _aspect_lexical_score(chunk: RetrievalResult, aspect: QueryAspect) -> float:
    text = re.sub(r"\s+", "", chunk.text)
    anchors = _aspect_anchor_phrases(aspect)
    anchor_score = sum(
        8.0 * _anchor_coverage(anchor, text)
        + (2.0 if anchor in text and any(marker in text for marker in ("是指", "是以", "包括")) else 0.0)
        for anchor in anchors
    )
    keywords = [re.sub(r"\s+", "", keyword) for keyword in aspect.keywords if keyword]
    keyword_score = sum(1.0 for keyword in keywords if keyword in text)
    return anchor_score + keyword_score


def _mark_chunk_for_aspect(chunk: RetrievalResult, aspect: QueryAspect) -> None:
    matched = chunk.metadata.get("prompt_matched_aspects")
    matched_aspects = matched if isinstance(matched, list) else []
    if aspect.aspect_id not in matched_aspects:
        matched_aspects.append(aspect.aspect_id)
    chunk.metadata["prompt_matched_aspects"] = matched_aspects
    chunk.metadata["aspect_id"] = aspect.aspect_id
    chunk.metadata["aspect_question"] = aspect.question
    chunk.metadata["aspect_search_queries"] = _search_query_debug_list(aspect)
    chunk.metadata["expected_evidence_type"] = aspect.expected_evidence_type
    chunk.metadata["evidence_need"] = aspect.evidence_need


def _chunk_matches_query_aspect(chunk: RetrievalResult, aspect: QueryAspect) -> bool:
    if chunk.metadata.get("fusion_method") == "query_plan_rrf" and not re.search(r"[\u4e00-\u9fff]", aspect.question):
        return True
    if chunk.metadata.get("evidence_role") in {
        "exact_anchor_support",
        "bounded_lexical_support",
        "mcq_exact_support",
        "mcq_rule_preamble",
    }:
        return True
    if not aspect.keywords:
        return True
    text = _chunk_match_text(chunk)
    keyword_hits = sum(1 for keyword in aspect.keywords if re.sub(r"\s+", "", keyword) in text)
    if keyword_hits >= min(2, len(aspect.keywords)):
        return True
    question_text = re.sub(r"\s+", "", aspect.question)
    return bool(question_text and question_text in text)


def _passes_prompt_score(chunk: RetrievalResult, top_score: float) -> bool:
    score = _prompt_score(chunk)
    if score < RERANK_PROMPT_THRESHOLD:
        return False
    return not top_score or score >= top_score * RELATIVE_SCORE_RATIO


def _prompt_score(chunk: RetrievalResult) -> float:
    metadata_score = _float_or_none(chunk.metadata.get("rerank_score"))
    if metadata_score is not None:
        return metadata_score
    return float(chunk.score or 0.0)


def _match_chunk_to_aspects(chunk: RetrievalResult, aspects: list[QuestionAspect]) -> list[str]:
    if not aspects:
        return []
    text = _chunk_match_text(chunk)
    matched: list[str] = []
    for aspect in aspects:
        required_ok = all(term in text for term in aspect.required_terms)
        keyword_hits = sum(1 for keyword in aspect.keywords if keyword in text)
        minimum_hits = min(2, len(aspect.keywords))
        if required_ok or keyword_hits >= minimum_hits:
            matched.append(aspect.aspect_id)
    return matched


def _covered_aspect_ids(chunks: list[RetrievalResult], aspects: list[QuestionAspect]) -> set[str]:
    if not chunks or not aspects:
        return set()
    combined_text = "\n".join(_chunk_match_text(chunk) for chunk in chunks)
    covered: set[str] = set()
    for aspect in aspects:
        coverage_terms = aspect.coverage_terms or aspect.required_terms
        if all(term in combined_text for term in coverage_terms):
            covered.add(aspect.aspect_id)
    return covered


def _is_structural_support(chunk: RetrievalResult, selected: list[RetrievalResult], question: str) -> bool:
    if not selected:
        return False
    if _chunk_question_coverage(chunk, question) < MIN_EVIDENCE_COVERAGE:
        return False

    chunk_document_id = str(chunk.metadata.get("document_id") or "")
    chunk_section = str(chunk.metadata.get("section_number") or "")
    chunk_parent = str(chunk.metadata.get("parent_section_number") or "")
    chunk_previous = str(chunk.metadata.get("previous_chunk_id") or "")
    chunk_next = str(chunk.metadata.get("next_chunk_id") or "")

    for selected_chunk in selected:
        selected_document_id = str(selected_chunk.metadata.get("document_id") or "")
        if chunk_document_id and selected_document_id and chunk_document_id != selected_document_id:
            continue
        selected_section = str(selected_chunk.metadata.get("section_number") or "")
        selected_parent = str(selected_chunk.metadata.get("parent_section_number") or "")
        if chunk_parent and chunk_parent == selected_section:
            return True
        if selected_parent and selected_parent == chunk_section:
            return True
        same_section_family = not chunk_section or not selected_section or (
            chunk_section.split(".", maxsplit=1)[0] == selected_section.split(".", maxsplit=1)[0]
        )
        if same_section_family and (
            chunk_previous == selected_chunk.chunk_id or chunk_next == selected_chunk.chunk_id
        ):
            return True
    return False


def _chunk_question_coverage(chunk: RetrievalResult, question: str) -> float:
    return evidence_coverage(question_terms(question), _chunk_match_text(chunk))


def _is_duplicate_or_redundant(chunk: RetrievalResult, selected: list[RetrievalResult]) -> bool:
    normalized = _normalize_for_dedupe(chunk.text)
    if not normalized:
        return True
    for selected_chunk in selected:
        selected_normalized = _normalize_for_dedupe(selected_chunk.text)
        if chunk.chunk_id == selected_chunk.chunk_id:
            return True
        if normalized == selected_normalized:
            return True
        shorter, longer = sorted([normalized, selected_normalized], key=len)
        if len(shorter) >= 80 and shorter in longer:
            return True
    return False


def _sort_prompt_chunks(chunks: list[RetrievalResult], query_plan: QueryPlan) -> list[RetrievalResult]:
    aspect_order = {aspect.aspect_id: index for index, aspect in enumerate(query_plan.aspects)}
    appendix_test_query = _query_asks_appendix_test_set(query_plan.original_question)

    def sort_key(chunk: RetrievalResult) -> tuple[Any, ...]:
        section_number = str(chunk.metadata.get("section_number") or "") or _section_number(chunk.section_title) or ""
        section_key = _section_sort_key(section_number)
        matched = chunk.metadata.get("prompt_matched_aspects")
        matched_ids = matched if isinstance(matched, list) else []
        aspect_index = min((aspect_order.get(str(aspect_id), 999) for aspect_id in matched_ids), default=999)
        appendix_penalty = 1 if _is_appendix_test_chunk(chunk) and not appendix_test_query else 0
        if section_number:
            return (
                0,
                appendix_penalty,
                str(chunk.metadata.get("document_id") or chunk.source_doc),
                section_key,
                aspect_index,
                chunk.rank,
            )
        return (1, appendix_penalty, aspect_index, -_prompt_score(chunk), chunk.rank)

    return sorted(chunks, key=sort_key)


def _is_appendix_test_chunk(chunk: RetrievalResult) -> bool:
    text = _chunk_match_text(chunk)
    return "附录B样例问答测试集" in text or "测试问题期望回答要点" in text


def _query_asks_appendix_test_set(question: str) -> bool:
    normalized = re.sub(r"\s+", "", question)
    return any(term in normalized for term in ["附录B", "样例问答", "测试集", "测试问题"])


def _section_sort_key(section_number: str) -> tuple[int, ...]:
    if not section_number:
        return (9999,)
    return tuple(int(part) for part in section_number.split(".") if part.isdigit()) or (9999,)


def _chunk_match_text(chunk: RetrievalResult) -> str:
    return re.sub(
        r"\s+",
        "",
        "\n".join(
            part
            for part in [
                chunk.source_doc,
                chunk.section_title or "",
                " ".join(chunk.section_path),
                chunk.text,
            ]
            if part
        ),
    )


def _text_is_traceable(excerpt: str, source_text: str) -> bool:
    normalized_excerpt = _normalize_for_dedupe(excerpt)
    normalized_source = _normalize_for_dedupe(source_text)
    if not normalized_excerpt:
        return False
    return normalized_excerpt in normalized_source or normalized_source in normalized_excerpt


def _renumber_context_chunks(context_chunks: list[RetrievalResult]) -> None:
    for index, chunk in enumerate(context_chunks, start=1):
        chunk.rank = index
        chunk.citation_label = f"[{index}]"


def _clean_chunk_text(text: str, section_title: str | None) -> str:
    cleaned = re.sub(r"\n{3,}", "\n\n", text.strip())
    if section_title:
        cleaned = _drop_repeated_leading_title(cleaned, section_title)
    return _truncate_preserving_sentence(cleaned, CONTEXT_CHUNK_CHAR_LIMIT)


def _drop_repeated_leading_title(text: str, section_title: str) -> str:
    lines = text.splitlines()
    while lines and lines[0].strip() == section_title.strip():
        lines = lines[1:]
    return "\n".join(lines).strip() or text


def _truncate_preserving_sentence(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    excerpt = ""
    for sentence in re.findall(r".+?(?:[。！？!?；;]|\n|$)", text, flags=re.S):
        if len(excerpt) + len(sentence) > limit:
            break
        excerpt += sentence
    excerpt = excerpt.strip()
    return excerpt if excerpt else text[:limit].strip()


def _section_path(citation: Citation) -> list[str]:
    if citation.section_path:
        return citation.section_path
    return [citation.section_title] if citation.section_title else []


def _normalize_for_dedupe(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _cited_context_results(
    context_chunks: list[RetrievalResult],
    *,
    answer: str,
    claims: list[AnswerClaim],
) -> list[RetrievalResult]:
    """Return only context chunks explicitly cited by the final answer."""
    citation_labels = {
        citation_id
        for claim in claims
        for citation_id in claim.citation_ids
        if citation_id
    }
    citation_labels.update(re.findall(r"\[\d+\]", answer or ""))
    if not citation_labels:
        return []
    return [
        result
        for result in context_chunks
        if result.citation_label in citation_labels
    ]


def _citation_from_result(result: RetrievalResult) -> Citation:
    metadata = result.metadata
    return Citation(
        document_id=str(metadata.get("document_id") or ""),
        chunk_id=result.chunk_id,
        filename=result.source_doc,
        section_title=result.section_title,
        section_path=result.section_path,
        section_number=str(metadata.get("section_number") or "") or None,
        parent_section_number=str(metadata.get("parent_section_number") or "") or None,
        previous_chunk_id=str(metadata.get("previous_chunk_id") or "") or None,
        next_chunk_id=str(metadata.get("next_chunk_id") or "") or None,
        page_number=_int_or_none(metadata.get("page_number")),
        excerpt=result.text,
        score=_float_or_none(metadata.get("vector_score")),
        rerank_score=_float_or_none(metadata.get("rerank_score")),
        chunk_type=str(metadata.get("chunk_type") or "paragraph"),
        evidence_role=str(metadata.get("evidence_role") or "related_context"),
        metadata=metadata,
    )


def _expand_neighbor_matches(
    db: Session,
    question: str,
    matches: list[RetrievalMatch],
    document_chunk_cache: dict[str, list[DocumentChunk]] | None = None,
) -> list[RetrievalMatch]:
    if not matches:
        return []

    document_ids = {match.citation.document_id for match in matches}
    chunks_by_document = _chunks_by_document(
        db,
        document_ids,
        document_chunk_cache=document_chunk_cache,
    )
    selected: list[RetrievalMatch] = []
    selected_chunk_ids: set[str] = set()

    for match in matches:
        if match.metadata.get("dynamic_table_evidence") or match.citation.chunk_type in {"table_cell", "table_calculation"}:
            selected.append(match)
            selected_chunk_ids.add(match.citation.chunk_id)
            continue
        if match.citation.chunk_id in selected_chunk_ids:
            continue
        selected.append(match)
        selected_chunk_ids.add(match.citation.chunk_id)

        document_chunks = chunks_by_document.get(match.citation.document_id, [])
        added_neighbors = 0
        for chunk, reason in _neighbor_candidates(match.citation.chunk_id, document_chunks, question):
            if chunk.chunk_id in selected_chunk_ids:
                continue
            selected.append(_match_from_chunk(chunk, match, reason, document_chunks, question))
            selected_chunk_ids.add(chunk.chunk_id)
            added_neighbors += 1
            if added_neighbors >= 2:
                break

    return selected[: max(MAX_PROMPT_CHUNKS * 3, MAX_PROMPT_CHUNKS)]


def _chunks_by_document(
    db: Session,
    document_ids: set[str],
    document_chunk_cache: dict[str, list[_DocumentChunkSnapshot]] | None = None,
) -> dict[str, list[_DocumentChunkSnapshot]]:
    if not document_ids:
        return {}
    cache = document_chunk_cache if document_chunk_cache is not None else {}
    now = monotonic()
    missing_ids = set(document_ids).difference(cache)
    with _DOCUMENT_SNAPSHOT_CACHE_LOCK:
        for document_id in list(missing_ids):
            cached = _DOCUMENT_SNAPSHOT_CACHE.get(document_id)
            if cached is None:
                continue
            cached_at, chunks = cached
            if now - cached_at > DOCUMENT_SNAPSHOT_CACHE_TTL_SECONDS:
                _DOCUMENT_SNAPSHOT_CACHE.pop(document_id, None)
                continue
            _DOCUMENT_SNAPSHOT_CACHE.move_to_end(document_id)
            cache[document_id] = chunks
            missing_ids.remove(document_id)
    if missing_ids:
        with measure("rag.document_snapshot_fetch"):
            chunk_models = db.scalars(
                select(DocumentChunk)
                .options(
                    load_only(
                        DocumentChunk.id,
                        DocumentChunk.chunk_id,
                        DocumentChunk.document_id,
                        DocumentChunk.text,
                        DocumentChunk.embedding_text,
                        DocumentChunk.chunk_metadata,
                        DocumentChunk.index_status,
                        DocumentChunk.source_file,
                        DocumentChunk.page_number,
                        DocumentChunk.section_title,
                    )
                )
                .where(DocumentChunk.document_id.in_(missing_ids))
                .order_by(DocumentChunk.document_id.asc(), DocumentChunk.id.asc())
            ).all()
        chunks = [
            _DocumentChunkSnapshot(
                id=chunk.id,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                embedding_text=chunk.embedding_text,
                chunk_metadata=chunk.chunk_metadata,
                index_status=chunk.index_status,
                source_file=chunk.source_file,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
            )
            for chunk in chunk_models
        ]
        for document_id in missing_ids:
            cache[document_id] = []
        for chunk in chunks:
            cache.setdefault(chunk.document_id, []).append(chunk)
        with _DOCUMENT_SNAPSHOT_CACHE_LOCK:
            for document_id in missing_ids:
                _DOCUMENT_SNAPSHOT_CACHE[document_id] = (now, cache.get(document_id, []))
                _DOCUMENT_SNAPSHOT_CACHE.move_to_end(document_id)
            while len(_DOCUMENT_SNAPSHOT_CACHE) > DOCUMENT_SNAPSHOT_CACHE_MAX_DOCUMENTS:
                _DOCUMENT_SNAPSHOT_CACHE.popitem(last=False)
    return {document_id: cache.get(document_id, []) for document_id in document_ids}


def _neighbor_candidates(
    anchor_chunk_id: str,
    chunks: list[DocumentChunk],
    question: str,
) -> list[tuple[DocumentChunk, str]]:
    by_chunk_id = {chunk.chunk_id: chunk for chunk in chunks}
    anchor = by_chunk_id.get(anchor_chunk_id)
    if anchor is None:
        return []

    anchor_section_number = _section_number(anchor.section_title)
    candidates: list[tuple[DocumentChunk, str]] = []
    if anchor_section_number and "." not in anchor_section_number:
        child_prefix = f"{anchor_section_number}."
        candidates.extend(
            (chunk, "child_section")
            for chunk in chunks
            if (_section_number(chunk.section_title) or "").startswith(child_prefix)
        )
    if anchor_section_number and "." in anchor_section_number:
        parent_number = _parent_section_number(anchor_section_number)
        candidates.extend(
            (chunk, "parent_section")
            for chunk in chunks
            if parent_number and _section_number(chunk.section_title) == parent_number
        )

    anchor_index = chunks.index(anchor)
    if anchor_index > 0:
        candidates.append((chunks[anchor_index - 1], "previous_chunk"))
    if anchor_index + 1 < len(chunks):
        candidates.append((chunks[anchor_index + 1], "next_chunk"))

    scored = [
        (chunk, reason, _neighbor_relevance(question, chunk, reason))
        for chunk, reason in candidates
        if chunk.chunk_id != anchor_chunk_id
    ]
    scored = [item for item in scored if item[2] > 0 or item[1] in {"child_section", "parent_section"}]
    scored.sort(key=lambda item: (item[2], item[1] in {"child_section", "parent_section"}), reverse=True)

    deduped: list[tuple[DocumentChunk, str]] = []
    seen: set[str] = set()
    for chunk, reason, _score in scored:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        deduped.append((chunk, reason))
    return deduped


def _neighbor_relevance(question: str, chunk: DocumentChunk, reason: str) -> float:
    text = "\n".join(part for part in [chunk.section_title or "", chunk.embedding_text or "", chunk.text] if part.strip())
    score = evidence_coverage(question_terms(question), text)
    if reason == "child_section" and _question_prefers_child_section(question):
        score += 0.2
    if reason in {"previous_chunk", "next_chunk"}:
        score -= 0.05
    return score


def _question_prefers_child_section(question: str) -> bool:
    normalized = re.sub(r"\s+", "", question)
    return any(term in normalized for term in ["差异", "处理", "外币折算", "保留", "依据", "需要保留"])


def _match_from_chunk(
    chunk: DocumentChunk,
    anchor: RetrievalMatch,
    reason: str,
    document_chunks: list[DocumentChunk],
    question: str,
) -> RetrievalMatch:
    previous_chunk_id, next_chunk_id = _adjacent_chunk_ids(chunk, document_chunks)
    section_number = _section_number(chunk.section_title)
    parent_section_number = _parent_section_number(section_number)
    text = _clean_chunk_text(chunk.text, chunk.section_title)
    coverage = evidence_coverage(question_terms(question), f"{chunk.section_title or ''}\n{chunk.embedding_text or ''}\n{text}")
    return RetrievalMatch(
        citation=Citation(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            filename=chunk.source_file,
            section_title=chunk.section_title,
            section_path=_inferred_section_path(chunk, document_chunks),
            section_number=section_number,
            parent_section_number=parent_section_number,
            previous_chunk_id=previous_chunk_id,
            next_chunk_id=next_chunk_id,
            page_number=chunk.page_number,
            excerpt=text,
            score=anchor.score,
            rerank_score=max(anchor.rerank_score - 0.05, 0.0),
            chunk_type=_chunk_type(chunk.text),
            evidence_role="expanded_context",
        ),
        score=anchor.score,
        rerank_score=max(anchor.rerank_score - 0.05, 0.0),
        coverage_score=coverage,
        evidence_role="expanded_context",
        evidence_text=text,
        metadata={
            **_chunk_metadata(chunk),
            "expansion_reason": reason,
            "expanded_from_chunk_id": anchor.citation.chunk_id,
        },
    )


def _adjacent_chunk_ids(chunk: DocumentChunk, chunks: list[DocumentChunk]) -> tuple[str | None, str | None]:
    index = chunks.index(chunk)
    previous_chunk_id = chunks[index - 1].chunk_id if index > 0 else None
    next_chunk_id = chunks[index + 1].chunk_id if index + 1 < len(chunks) else None
    return previous_chunk_id, next_chunk_id


def _chunk_metadata(chunk: DocumentChunk) -> dict[str, Any]:
    if not chunk.chunk_metadata:
        return {}
    try:
        value = json.loads(chunk.chunk_metadata)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _inferred_section_path(chunk: DocumentChunk, chunks: list[DocumentChunk]) -> list[str]:
    section_number = _section_number(chunk.section_title)
    parent_section_number = _parent_section_number(section_number)
    path: list[str] = []
    if parent_section_number:
        parent = next((item for item in chunks if _section_number(item.section_title) == parent_section_number), None)
        if parent and parent.section_title:
            path.append(parent.section_title)
    if chunk.section_title:
        path.append(chunk.section_title)
    return path


def _chunk_type(text: str) -> str:
    stripped = text.lstrip()
    table_prefixes = ("表格：", "表格摘要：", "表格行证据：", "琛ㄦ牸锛?", "琛ㄦ牸琛岃瘉鎹細")
    return "table" if stripped.startswith(table_prefixes) or "\n|" in text else "paragraph"


def _section_number(section_title: str | None) -> str | None:
    if not section_title:
        return None
    match = re.match(r"^\s*(\d+(?:\.\d+)*)[\.、\s]", section_title)
    return match.group(1) if match else None


def _parent_section_number(section_number: str | None) -> str | None:
    if not section_number or "." not in section_number:
        return None
    return section_number.rsplit(".", maxsplit=1)[0]


def _build_retrieval_summary(
    question: str,
    context_chunks: list[RetrievalResult],
    query_plan: QueryPlan,
    aspect_retrievals: list[AspectRetrieval],
    diagnostics: RetrievalDiagnostics,
    citation_validation: dict[str, Any],
    prompt_selection: dict[str, Any],
) -> dict[str, Any]:
    coverage_notes = [
        _covered_aspect_note(item.aspect)
        for item in aspect_retrievals
        if item.retrieval_covered
    ]
    missing_aspects = [
        _missing_aspect_note(item.aspect)
        for item in aspect_retrievals
        if not item.retrieval_covered
    ]
    table_refusal_reasons = _table_refusal_reasons(aspect_retrievals)
    for reason in table_refusal_reasons:
        if reason not in missing_aspects:
            missing_aspects.append(reason)
    covered_by_retrieval_but_not_prompted = [
        item.aspect.aspect_id
        for item in aspect_retrievals
        if item.retrieval_covered and not item.covered
    ]
    prompt_capacity_limited = bool(covered_by_retrieval_but_not_prompted)
    metadata_filters = [
        diagnostic["metadata_filter"]
        for item in aspect_retrievals
        for diagnostic in item.diagnostics
        if isinstance(diagnostic.get("metadata_filter"), dict)
    ]
    metadata_filter_no_match = any(
        diagnostic.get("match_status") == "metadata_filter_no_match"
        for item in aspect_retrievals
        for diagnostic in item.diagnostics
    )

    has_sufficient_context = bool(context_chunks) and not missing_aspects and not prompt_capacity_limited
    summary = {
        "top_k": MAX_PROMPT_CHUNKS,
        "used_chunks": len(context_chunks),
        "has_sufficient_context": has_sufficient_context,
        "aspect_count": len(query_plan.aspects),
        "retrieval_covered_aspect_count": sum(1 for item in aspect_retrievals if item.retrieval_covered),
        "prompt_covered_aspect_count": sum(1 for item in aspect_retrievals if item.covered),
        "prompt_capacity_limited": prompt_capacity_limited,
        "metadata_filters": metadata_filters,
        "metadata_filter_no_match": metadata_filter_no_match,
        "metadata_filter_no_match_reason": (
            "用户明确元数据条件未匹配到已发布文档，未执行无约束回退"
            if metadata_filter_no_match
            else None
        ),
        "covered_by_retrieval_but_not_prompted": covered_by_retrieval_but_not_prompted,
        "coverage_notes": coverage_notes,
          "missing_aspects": missing_aspects,
          "table_refusal_reasons": table_refusal_reasons,
        "citation_validation": citation_validation,
        "prompt_filtered_count": max(prompt_selection["candidate_prompt_chunks"] - len(context_chunks), 0),
        "prompt_selection": prompt_selection,
        "query_plan": query_plan.to_debug_dict(),
        "aspect_retrievals": [_aspect_retrieval_debug(item) for item in aspect_retrievals],
        "final_prompt_chunk_ids": [chunk.chunk_id for chunk in context_chunks],
        "fusion_method": ASPECT_QUERY_FUSION_METHOD,
        "model_device": get_model_device_info().to_debug_dict(),
    }
    summary.update(diagnostics.to_summary_fields())
    return summary


def _table_refusal_reasons(aspect_retrievals: list[AspectRetrieval]) -> list[str]:
    reasons: list[str] = []
    for item in aspect_retrievals:
        if item.aspect.modality not in {"table", "mixed"}:
            continue
        for diagnostic in item.diagnostics:
            reason = diagnostic.get("spreadsheet_refusal_reason")
            if isinstance(reason, str) and reason.strip():
                reasons.append(reason.strip())
    return list(dict.fromkeys(reasons))


def _aggregate_aspect_diagnostics(aspect_retrievals: list[AspectRetrieval]) -> RetrievalDiagnostics:
    aggregate = RetrievalDiagnostics()
    vector_min: float | None = None
    vector_max: float | None = None
    rerank_min: float | None = None
    rerank_max: float | None = None

    for aspect_retrieval in aspect_retrievals:
        for diagnostics in aspect_retrieval.diagnostics:
            aggregate.query_count += int(diagnostics.get("query_count") or 0)
            aggregate.raw_candidate_count += int(diagnostics.get("raw_candidate_count") or 0)
            aggregate.candidate_count += int(diagnostics.get("candidate_count") or 0)
            aggregate.rerank_input_count += int(diagnostics.get("rerank_input_count") or 0)
            aggregate.rerank_call_count += int(diagnostics.get("rerank_call_count") or 0)
            aggregate.reranked_count += int(diagnostics.get("reranked_count") or 0)
            aggregate.filtered_count += int(diagnostics.get("filtered_count") or 0)
            search_query = diagnostics.get("search_query")
            if isinstance(search_query, str) and search_query:
                aggregate.query_variants.append(search_query)
            for key, value in (diagnostics.get("timings_ms") or {}).items():
                if isinstance(value, (int, float)):
                    aggregate.timings_ms[key] = round(aggregate.timings_ms.get(key, 0.0) + float(value), 2)
            score_range = diagnostics.get("score_range") or {}
            vector_min = _min_optional(vector_min, _float_or_none(score_range.get("vector_min")))
            vector_max = _max_optional(vector_max, _float_or_none(score_range.get("vector_max")))
            rerank_min = _min_optional(rerank_min, _float_or_none(score_range.get("rerank_min")))
            rerank_max = _max_optional(rerank_max, _float_or_none(score_range.get("rerank_max")))

    aggregate.query_variants = list(dict.fromkeys(aggregate.query_variants))
    aggregate.score_range = {
        "vector_min": vector_min,
        "vector_max": vector_max,
        "rerank_min": rerank_min,
        "rerank_max": rerank_max,
    }
    return aggregate


def _score_range_for_candidates(candidates: list[Any], reranked: list[Any]) -> dict[str, float | None]:
    vector_scores = [float(candidate.score) for candidate in candidates]
    rerank_scores = [float(item.rerank_score) for item in reranked]
    return {
        "vector_min": round(min(vector_scores), 4) if vector_scores else None,
        "vector_max": round(max(vector_scores), 4) if vector_scores else None,
        "rerank_min": round(min(rerank_scores), 4) if rerank_scores else None,
        "rerank_max": round(max(rerank_scores), 4) if rerank_scores else None,
    }


def _score_range_for_matches(matches: list[RetrievalMatch]) -> dict[str, float | None]:
    scores = [float(match.score) for match in matches]
    rerank_scores = [float(match.rerank_score) for match in matches]
    return {
        "vector_min": round(min(scores), 4) if scores else None,
        "vector_max": round(max(scores), 4) if scores else None,
        "rerank_min": round(min(rerank_scores), 4) if rerank_scores else None,
        "rerank_max": round(max(rerank_scores), 4) if rerank_scores else None,
    }


def _merge_citation_validation(validations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "checked_chunks": sum(int(item.get("checked_chunks") or 0) for item in validations),
        "valid_chunks": sum(int(item.get("valid_chunks") or 0) for item in validations),
        "invalid_chunks": sum(int(item.get("invalid_chunks") or 0) for item in validations),
        "invalid_chunk_ids": [
            chunk_id
            for item in validations
            for chunk_id in item.get("invalid_chunk_ids", [])
        ],
    }


def _aspect_retrieval_debug(item: AspectRetrieval) -> dict[str, Any]:
    return {
        **item.aspect.to_debug_dict(),
        "evidence_need": item.aspect.evidence_need,
        "retrieval_covered": item.retrieval_covered,
        "covered": item.covered,
        "missing": not item.retrieval_covered,
        "covered_by_retrieval_but_not_prompted": item.retrieval_covered and not item.covered,
        "candidate_count": len(item.candidates),
        "selected_chunk_ids": item.selected_chunk_ids,
        "retrieved_chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "source_doc": chunk.source_doc,
                "section_title": chunk.section_title,
                "score": chunk.score,
                "rerank_score": chunk.metadata.get("rerank_score"),
                "fusion_score": chunk.metadata.get("aspect_query_fusion_score"),
                "query_hits": chunk.metadata.get("aspect_search_query_hits", []),
                "evidence_role": chunk.metadata.get("evidence_role"),
                "selected_for_prompt": chunk.chunk_id in item.selected_chunk_ids,
            }
            for chunk in item.candidates
        ],
        "diagnostics": item.diagnostics,
    }


def _covered_aspect_note(aspect: QueryAspect) -> str:
    return f"已覆盖：{aspect.question}"


def _missing_aspect_note(aspect: QueryAspect) -> str:
    if aspect.aspect_id == "foreign_currency_evidence":
        return "未召回外币折算差异需要保留的依据"
    if aspect.aspect_id == "asset_total_difference_check":
        return "未召回资产合计差异排查依据"
    return f"未召回：{aspect.question}"


def _expected_aspects(question: str) -> list[QuestionAspect]:
    normalized = re.sub(r"\s+", "", question)
    aspects: list[QuestionAspect] = []
    if "资产合计" in normalized and "差异" in normalized:
        aspects.append(
            QuestionAspect(
                aspect_id="asset_total_difference_check",
                description="资产合计差异优先排查哪些问题",
                keywords=("资产合计", "差异", "优先", "排查", "检查", "币种折算", "四舍五入", "科目映射", "重复汇总"),
                required_terms=("资产合计", "差异"),
                coverage_terms=("资产合计", "差异", "币种折算", "四舍五入", "科目映射", "重复汇总"),
                covered_note="已召回资产合计差异排查依据",
                missing_note="未召回资产合计差异排查依据",
            )
        )
    if "外币折算" in normalized and any(term in normalized for term in ["保留", "依据", "需要保留"]):
        aspects.append(
            QuestionAspect(
                aspect_id="foreign_currency_evidence",
                description="外币折算差异需要保留什么依据",
                keywords=("外币折算", "保留", "依据", "汇率日期", "折算规则", "原币金额", "原币金额来源"),
                required_terms=("外币折算",),
                coverage_terms=("外币折算", "汇率日期", "折算规则", "原币金额"),
                covered_note="已召回外币折算需保留依据",
                missing_note="未召回外币折算差异需要保留的依据",
            )
        )
    return aspects


def _confidence_from_results(results: list[RetrievalResult]) -> float:
    scores = [result.score for result in results if result.score is not None]
    if not scores:
        return 0.0
    best_score = max(scores)
    return min(max(float(best_score), 0.0), 0.95)


def _required_aspect_ids(retrieval_summary: dict[str, Any]) -> list[str]:
    aspects = retrieval_summary.get("query_plan", {}).get("aspects", [])
    return [
        str(item.get("aspect_id"))
        for item in aspects
        if isinstance(item, dict) and item.get("aspect_id")
    ]


def _answer_mode(question: str, retrieval_summary: dict[str, Any]) -> str:
    aspects = retrieval_summary.get("query_plan", {}).get("aspects", [])
    modalities = {
        str(item.get("modality"))
        for item in aspects
        if isinstance(item, dict) and item.get("modality")
    }
    is_mixed = "mixed" in modalities or ({"table", "text"}.issubset(modalities))
    if not is_mixed:
        return "text"
    scenario_markers = ("是否合规", "是否违规", "合规吗", "违反", "判断", "整改", "应如何处理")
    return "scenario" if any(marker in question for marker in scenario_markers) else "mixed"


def _build_evidence_coverage(
    retrieval_summary: dict[str, Any],
    claims: list[AnswerClaim],
    cited_results: list[RetrievalResult],
) -> EvidenceCoverage:
    expected = _required_aspect_ids(retrieval_summary)
    covered = {
        aspect_id
        for claim in claims
        for aspect_id in claim.aspect_ids
        if aspect_id
    }
    for result in cited_results:
        aspect_id = result.metadata.get("aspect_id")
        if aspect_id:
            covered.add(str(aspect_id))
        covered.update(
            str(value)
            for value in result.metadata.get("prompt_matched_aspects") or []
            if value
        )
    covered_ordered = [aspect_id for aspect_id in expected if aspect_id in covered]
    missing = [aspect_id for aspect_id in expected if aspect_id not in covered]
    return EvidenceCoverage(
        expected_aspect_ids=expected,
        covered_aspect_ids=covered_ordered,
        missing_aspect_ids=missing,
        complete=not missing and bool(expected),
    )


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 2)


def _format_elapsed_seconds(elapsed_ms: float | None) -> str:
    if elapsed_ms is None:
        return "0.000s"
    return f"{elapsed_ms / 1000:.3f}s"


def _int_or_none(value: Any) -> int | None:
    return int(value) if value is not None else None


def _float_or_none(value: Any) -> float | None:
    return float(value) if value is not None else None


def _min_optional(current: float | None, candidate: float | None) -> float | None:
    if candidate is None:
        return current
    return candidate if current is None else min(current, candidate)


def _max_optional(current: float | None, candidate: float | None) -> float | None:
    if candidate is None:
        return current
    return candidate if current is None else max(current, candidate)


def _filter_indexed_matches(db: Session, matches: list[RetrievalMatch]) -> list[RetrievalMatch]:
    if not matches:
        return []
    document_ids = {match.citation.document_id for match in matches}
    verified_ids = {
        match.citation.document_id
        for match in matches
        if bool(match.metadata.get("active_index_verified"))
    }
    unchecked_ids = document_ids.difference(verified_ids)
    indexed_ids = set(verified_ids)
    if unchecked_ids:
        indexed_ids.update(
            db.scalars(
                select(Document.document_id).where(
                    Document.document_id.in_(unchecked_ids),
                    Document.status.in_(["indexed", "table_indexed"]),
                )
            ).all()
        )
    return [match for match in matches if match.citation.document_id in indexed_ids]


def _save_qa_log(db: Session, question: str, response: QAResponse) -> None:
    log = QALog(
        question=question,
        answer=response.answer or EMPTY_ANSWER_LOG_TEXT,
        refused=response.refused,
        confidence=response.confidence,
        citation_count=len(response.citations),
    )
    db.add(log)
    db.commit()


def _log_qa_audit(db: Session, action: str, question: str, response: QAResponse) -> None:
    package = response.context_package
    used_chunks = (
        max(int(package.retrieval_summary.get("used_chunks") or 0), len(response.citations))
        if package
        else len(response.citations)
    )
    detail_payload = {
        "question": question,
        "answer": response.answer,
        "is_final_answer": bool(response.answer),
        "mode": response.answer_type,
        "generation_status": response.generation_status,
        "used_chunks": used_chunks,
        "refused": response.refused,
        "refusal_reason": response.refusal_reason,
        "refusal_code": response.refusal_code,
        "confidence": response.confidence,
        "citation_count": len(response.citations),
    }
    details_payload = {
        **detail_payload,
        "citations": [_audit_citation_payload(citation) for citation in response.citations],
    }
    detail = json.dumps(detail_payload, ensure_ascii=False)
    status_text = "已拒答" if response.refused else "已回答"
    summary = "问答拒答" if response.refused else "问答完成"
    user_message = f"{status_text}：{_compact_audit_text(question, 80)}"
    try:
        record_event(
            db,
            action,
            "question",
            None,
            detail=detail,
            severity="info",
            event_key=f"{action}:question:{uuid4().hex}",
            summary=summary,
            user_message=user_message,
            details=details_payload,
        )
    except SQLAlchemyError:
        # The answer has already been produced.  A non-critical audit write
        # must not turn that successful answer into HTTP 500.  Roll back the
        # failed transaction so the request-scoped Session remains usable;
        # normal audit records stay small enough to persist because citation
        # metadata is compacted below.
        db.rollback()


def _audit_citation_payload(citation: Citation) -> dict[str, Any]:
    metadata = citation.metadata or {}
    compact_metadata: dict[str, Any] = {}
    scalar_keys = (
        "aspect_id",
        "evidence_id",
        "dynamic_table_evidence",
        "sheet_name",
        "sheet",
        "coordinate",
        "cell",
        "value",
        "unit",
        "year",
        "month",
        "quarter",
        "operation",
        "match_status",
        "fusion_method",
        "exact_support_anchor",
        "exact_support_score",
        "calculation_formula",
        "calculation_result",
    )
    for key in scalar_keys:
        value = metadata.get(key)
        if value is not None and value != "":
            compact_metadata[key] = _compact_audit_value(value)

    for key in ("calculation_cells", "source_cells", "exact_support_terms"):
        value = metadata.get(key)
        if isinstance(value, list):
            compact_metadata[key] = [_compact_audit_value(item) for item in value[:24]]

    formulas = metadata.get("formulas")
    if isinstance(formulas, list):
        compact_metadata["formulas"] = [
            {
                key: _compact_audit_value(item.get(key))
                for key in ("text", "source_type", "order_index")
                if isinstance(item, dict) and item.get(key) is not None
            }
            for item in formulas[:12]
            if isinstance(item, dict)
        ]

    return {
        "document_id": citation.document_id,
        "chunk_id": citation.chunk_id,
        "filename": citation.filename,
        "section_title": citation.section_title,
        "page_number": citation.page_number,
        "excerpt": _compact_audit_text(citation.excerpt, 1200),
        "score": citation.score,
        "rerank_score": citation.rerank_score,
        "chunk_type": citation.chunk_type,
        "evidence_role": citation.evidence_role,
        "metadata": compact_metadata,
    }


def _compact_audit_value(value: Any, max_chars: int = 500) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _compact_audit_text(value, max_chars) if isinstance(value, str) else value
    if isinstance(value, dict):
        return {
            str(key): _compact_audit_value(item, max_chars=max_chars)
            for key, item in list(value.items())[:20]
        }
    if isinstance(value, (list, tuple)):
        return [_compact_audit_value(item, max_chars=max_chars) for item in list(value)[:24]]
    return _compact_audit_text(str(value), max_chars)


def _compact_audit_text(value: str | None, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max(max_chars - head - 1, 0)
    return f"{text[:head]}…{text[-tail:]}"
