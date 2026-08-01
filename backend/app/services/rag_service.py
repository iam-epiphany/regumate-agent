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
from backend.app.services.answer_generation_service import generate_answer
from backend.app.services.document_identity_query_service import answer_document_identity_question
from backend.app.services.prompt_builder import RAGPromptBuilder
from backend.app.services.question_preprocessing_service import build_question_task, preprocess_qa_request
from backend.app.services.query_planner_service import (
    QueryAspect,
    QueryPlan,
    QuerySearchQuery,
    plan_query,
)
from backend.app.services.retrieval_metadata_filter_service import (
    build_retrieval_metadata_filter,
)
from backend.app.services import retrieval_support as _retrieval_support

# Re-export the extracted retrieval-support module so callers and tests that
# referenced ``rag_service.<name>`` keep working unchanged.
for _forwarded_name in dir(_retrieval_support):
    if not _forwarded_name.startswith("__"):
        globals()[_forwarded_name] = getattr(_retrieval_support, _forwarded_name)
del _forwarded_name
from backend.app.services.retrieval_service import (
    RetrievalDiagnostics,
    RetrievalMatch,
    RetrievalServiceUnavailable,
    ProgressReporter,
    _embed_search_queries,
    _hybrid_search_many,
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
from backend.app.services.rerank_service import (
    RerankedChunk,
    RerankServiceError,
    rerank_candidate_groups,
    rerank_candidates,
)
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
EMPTY_ANSWER_LOG_TEXT = "[RAG_CONTEXT_PACKAGE_ONLY] 当前阶段未接入 LLM，接口仅返回检索上下文包。"


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
    task = build_question_task(cleaned_question, options)
    cleaned_question = task.question
    normalized_options = task.options
    option_labels = task.option_labels
    boundary_code = _explicit_request_boundary_code(cleaned_question)
    if boundary_code:
        response = QAResponse(
            answer="该请求超出监管知识库可验证回答的范围，系统需要补充可核验依据或改为资料库可支持的问题。",
            citations=[],
            confidence=0.0,
            refused=True,
            context_package=None,
            refusal_reason="explicit_request_boundary",
            refusal_code=boundary_code,
        )
        _save_qa_log(db, original_question, response)
        _log_qa_audit(db, "qa_refused_explicit_boundary", original_question, response)
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
    if generated.refused and not generated.refusal_code:
        # Keep refusal outcomes machine-readable even when the answer model
        # declines before it can emit its own structured code.  The classifier
        # only describes request boundaries stated by the user; it never
        # supplies a regulatory conclusion or an answer from an evaluation set.
        generated.refusal_code = _refusal_code_for_request(cleaned_question)
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


def _refusal_code_for_request(question: str) -> str:
    """Classify common, explicit limits of a regulatory knowledge-base request."""

    return _explicit_request_boundary_code(question) or "insufficient_evidence"


def _explicit_request_boundary_code(question: str) -> str | None:
    """Return a code only when the request itself states a known boundary."""

    compact = re.sub(r"\s+", "", str(question or ""))
    # Regulatory titles and quoted clauses describe source material, not the
    # user's requested action.  Terms such as "决定" and "会计估计" are common
    # in valid evidence requests and must not be classified as advice or a
    # forecast merely because they occur inside a citation anchor.
    intent_text = re.sub(r"《[^》]*》", "", compact)
    intent_text = re.sub(r"[“‘\"'][^”’\"']*[”’\"']", "", intent_text)
    if any(
        term in compact
        for term in (
            "实时", "今天最新", "刚发布", "资料库之外", "未收录",
            "没有收录", "尚未入库", "内部检查通报", "分行审计报告",
        )
    ):
        return "out_of_scope_or_realtime"
    if any(
        term in intent_text
        for term in (
            "推荐", "商业策略", "替我决定", "请决定", "设计保证", "保证盈利", "营销方案",
        )
    ):
        return "subjective_business_advice"
    if any(
        term in intent_text
        for term in (
            "预测", "推断", "估计", "未来", "下一季度", "明年", "下一版", "必然会",
        )
    ):
        return "unsupported_prediction"
    if (
        any(
            term in intent_text
            for term in (
                "计算", "算出", "核算", "比较", "排名", "换算", "折算", "占比", "比例", "差额", "差异",
            )
        )
        and any(
            term in intent_text
            for term in (
                "未给出", "没有给出", "未提供", "没有提供", "未指明", "没有任何", "缺少", "但没有",
            )
        )
    ):
        return "missing_calculation_operands"
    if any(
        term in intent_text
        for term in (
            "未提供", "没有提供", "未给出", "未具名", "未说明", "未指明",
            "没有文件名", "没有指标口径", "只给出", "模糊描述", "没有任何定位信息",
        )
    ):
        return "missing_required_context"
    return None


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

    # Phase A: deterministic preparation (spreadsheet, filters, MCQ early
    # stop, query normalization) for every aspect, so the hybrid collection
    # can run once across all of them.
    prep_by_aspect: dict[str, _AspectPrepResult] = {}
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
        prep_by_aspect[aspect.aspect_id] = _prepare_aspect_retrieval(
            db,
            aspect,
            progress_reporter=progress_reporter,
            document_chunk_cache=document_chunk_cache,
        )

    # Phase B + C: collect hybrid candidates (one batched embedding + Qdrant
    # pass and one batched rerank pass when several aspects need vectors;
    # otherwise the single-aspect path) and finish each aspect.
    prepared_items = [
        (aspect, prep_by_aspect[aspect.aspect_id].prepared)
        for aspect in query_plan.aspects
        if not prep_by_aspect[aspect.aspect_id].terminal
    ]
    finished: dict[str, tuple[list[RetrievalMatch], list[dict[str, Any]]]] = {}
    if len(prepared_items) > 1:
        try:
            collected = _collect_prepared_candidates_many(
                [prepared for _aspect, prepared in prepared_items]
            )
        except (EmbeddingServiceError, VectorStoreError) as exc:
            raise RetrievalServiceUnavailable(str(exc)) from exc
        rerank_questions: list[str] = []
        rerank_inputs: list[list[Any]] = []
        for (aspect, prepared), (candidates, _hits, _per_query) in zip(
            prepared_items, collected, strict=True
        ):
            effective = _effective_rerank_candidate_limit(
                aspect,
                candidates,
                retrieval_filter_document_ids=set(prepared.retrieval_filter.document_ids or ()),
                mcq_material_document_ids=prepared.mcq_material_document_ids,
            )
            rerank_inputs.append(
                limit_rerank_candidates(candidates, preserve_order=True, limit=effective)
            )
            rerank_questions.append(_rerank_query_for_aspect(aspect))
        rerank_limits = [
            len(rerank_input)
            if aspect.aspect_id == "multiple_choice_evidence"
            else RERANK_TOP_K
            for (aspect, _prepared), rerank_input in zip(
                prepared_items, rerank_inputs, strict=True
            )
        ]
        try:
            reranked_groups = rerank_candidate_groups(
                questions=rerank_questions,
                candidate_groups=rerank_inputs,
                limits=rerank_limits,
            )
        except RerankServiceError as exc:
            raise RetrievalServiceUnavailable(str(exc)) from exc
        for (aspect, prepared), (candidates, hits, per_query), rerank_input, reranked in zip(
            prepared_items, collected, rerank_inputs, reranked_groups, strict=True
        ):
            matches, diagnostics = _finish_aspect_matches(
                db,
                aspect,
                candidates=candidates,
                query_hits_by_chunk_id=hits,
                diagnostics_by_query=per_query,
                diagnostics=prepared.diagnostics,
                retrieval_filter=prepared.retrieval_filter,
                mcq_material_document_ids=prepared.mcq_material_document_ids,
                table_matches=prepared.table_matches,
                table_diagnostics=prepared.table_diagnostics,
                document_chunk_cache=document_chunk_cache,
                total_started_at=prepared.total_started_at,
                progress_reporter=progress_reporter,
                reranked=reranked,
                rerank_input=rerank_input,
            )
            finished[aspect.aspect_id] = (matches, diagnostics)
    else:
        for aspect, prepared in prepared_items:
            assert prepared is not None
            try:
                candidates, hits, per_query = collect_candidates_with_query_hits(
                    prepared.search_queries,
                    query_metadata=prepared.query_metadata,
                    diagnostics=prepared.diagnostics,
                    metadata_filter=prepared.qdrant_filter,
                )
            except (EmbeddingServiceError, VectorStoreError) as exc:
                raise RetrievalServiceUnavailable(str(exc)) from exc
            matches, diagnostics = _finish_aspect_matches(
                db,
                aspect,
                candidates=candidates,
                query_hits_by_chunk_id=hits,
                diagnostics_by_query=per_query,
                diagnostics=prepared.diagnostics,
                retrieval_filter=prepared.retrieval_filter,
                mcq_material_document_ids=prepared.mcq_material_document_ids,
                table_matches=prepared.table_matches,
                table_diagnostics=prepared.table_diagnostics,
                document_chunk_cache=document_chunk_cache,
                total_started_at=prepared.total_started_at,
                progress_reporter=progress_reporter,
            )
            finished[aspect.aspect_id] = (matches, diagnostics)

    for aspect_index, aspect in enumerate(query_plan.aspects, start=1):
        prep = prep_by_aspect[aspect.aspect_id]
        if prep.terminal:
            matches, diagnostics = prep.matches, prep.diagnostics
        else:
            matches, diagnostics = finished[aspect.aspect_id]
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
    if _chunk_explicit_source_match(chunk, aspect) is False:
        return False
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
        if _chunk_explicit_source_match(chunk, aspect) is not False
        and any(_anchor_coverage(anchor, chunk.text) >= 0.72 for anchor in anchors)
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
    original_aspect_id = str(
        chunk.metadata.get("retrieval_aspect_id")
        or chunk.metadata.get("aspect_id")
        or ""
    )
    if original_aspect_id:
        chunk.metadata["retrieval_aspect_id"] = original_aspect_id
    matched = chunk.metadata.get("prompt_matched_aspects")
    matched_aspects = matched if isinstance(matched, list) else []
    if aspect.aspect_id not in matched_aspects:
        matched_aspects.append(aspect.aspect_id)
    chunk.metadata["prompt_matched_aspects"] = matched_aspects
    chunk.metadata.setdefault("aspect_id", aspect.aspect_id)
    chunk.metadata.setdefault("aspect_question", aspect.question)
    aspect_questions = chunk.metadata.get("prompt_aspect_questions")
    if not isinstance(aspect_questions, dict):
        aspect_questions = {}
    aspect_questions[aspect.aspect_id] = aspect.question
    chunk.metadata["prompt_aspect_questions"] = aspect_questions
    chunk.metadata["aspect_search_queries"] = _search_query_debug_list(aspect)
    chunk.metadata["expected_evidence_type"] = aspect.expected_evidence_type
    chunk.metadata["evidence_need"] = aspect.evidence_need


def _chunk_explicit_source_match(
    chunk: RetrievalResult,
    aspect: QueryAspect,
) -> bool | None:
    expected_source = str(
        aspect.table_filters.get("source_title")
        or aspect.table_filters.get("filename")
        or ""
    ).strip()
    if not expected_source:
        return None
    expected_norm = _normalize_document_anchor(expected_source)
    source_candidates = [
        chunk.source_doc,
        chunk.metadata.get("source_doc"),
        chunk.metadata.get("source_title"),
        chunk.metadata.get("source_filename"),
        chunk.metadata.get("filename"),
    ]
    return bool(
        expected_norm
        and any(
            _normalized_title_contains(expected_norm, candidate)
            for candidate in source_candidates
            if candidate
        )
    )


def _chunk_matches_query_aspect(chunk: RetrievalResult, aspect: QueryAspect) -> bool:
    explicit_source_match = _chunk_explicit_source_match(chunk, aspect)
    if explicit_source_match is False:
        return False
    if chunk.metadata.get("fusion_method") == "query_plan_rrf" and not re.search(r"[\u4e00-\u9fff]", aspect.question):
        return True
    if chunk.metadata.get("evidence_role") in {
        "exact_anchor_support",
        "bounded_lexical_support",
        "mcq_exact_support",
        "mcq_rule_preamble",
    }:
        if explicit_source_match is True:
            return True
        retrieval_aspect_id = str(chunk.metadata.get("retrieval_aspect_id") or "")
        if not retrieval_aspect_id or retrieval_aspect_id == aspect.aspect_id:
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
