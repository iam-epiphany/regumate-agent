from dataclasses import dataclass
import json
import re
from time import perf_counter
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import (
    FINAL_CITATION_LIMIT,
    FORCE_MIN_CHUNKS,
    MAX_PROMPT_CHUNKS,
    MIN_EVIDENCE_COVERAGE,
    MIN_PROMPT_CHUNKS,
    RELATIVE_SCORE_RATIO,
    RERANK_PROMPT_THRESHOLD,
)
from backend.app.models.document import Document, DocumentChunk, QALog
from backend.app.schemas.qa import Citation, LLMContextPackage, QAResponse, RetrievalResult
from backend.app.services.audit_service import log_action
from backend.app.services.prompt_builder import RAGPromptBuilder
from backend.app.services.query_planner_service import QueryAspect, QueryPlan, plan_query
from backend.app.services.retrieval_service import (
    RetrievalDiagnostics,
    RetrievalMatch,
    ProgressReporter,
    evidence_coverage,
    get_last_retrieval_diagnostics,
    question_terms,
    reset_retrieval_diagnostics,
    retrieve_citations,
)


CONTEXT_INSTRUCTION = (
    "请严格根据检索到的知识片段回答用户问题。不得编造知识库中不存在的制度依据。"
    "若依据不足，请明确说明无法根据当前知识库判断。"
)
CONTEXT_CHUNK_CHAR_LIMIT = 1200
EMPTY_ANSWER_LOG_TEXT = "[RAG_CONTEXT_PACKAGE_ONLY] 当前阶段未接入 LLM，接口仅返回检索上下文包。"
ASPECT_QUERY_FUSION_METHOD = "aspect_query_rrf_then_bge_rerank"
RRF_K = 60
QUERY_TYPE_WEIGHTS = {
    "semantic_question": 1.0,
    "document_style_statement": 1.15,
    "keyword_anchor": 0.75,
    "legacy": 0.9,
    "fallback": 0.85,
}


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
    covered: bool = False


def answer_question(
    db: Session,
    question: str,
    progress_reporter: ProgressReporter | None = None,
) -> QAResponse:
    cleaned_question = question.strip()
    if not cleaned_question:
        return QAResponse(answer=None, citations=[], confidence=0.0, refused=True, context_package=None)

    package = build_context_package(db, cleaned_question, progress_reporter=progress_reporter)
    response = QAResponse(
        answer=None,
        citations=[_citation_from_result(result) for result in package.context_chunks],
        confidence=_confidence_from_results(package.context_chunks),
        refused=not package.retrieval_summary["has_sufficient_context"],
        context_package=package,
    )
    _save_qa_log(db, cleaned_question, response)
    _log_qa_audit(db, "qa_context_built", cleaned_question, response)
    return response


def retrieve_context_package(db: Session, question: str) -> LLMContextPackage:
    cleaned_question = question.strip()
    if not cleaned_question:
        return _empty_context_package("")
    return build_context_package(db, cleaned_question)


def build_context_package(
    db: Session,
    question: str,
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
    query_plan = plan_query(question)
    planning_elapsed_ms = _elapsed_ms(planning_started_at)
    _report_progress(
        progress_reporter,
        {
            "stage": "planning",
            "status": "completed",
            "title": "问题理解完成",
            "detail": f"已拆分为 {len(query_plan.aspects)} 个方面，用时 {planning_elapsed_ms:.0f}ms",
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
    context_elapsed_ms = _elapsed_ms(context_started_at)
    covered_aspect_count = sum(1 for item in aspect_retrievals if item.covered)
    _report_progress(
        progress_reporter,
        {
            "stage": "context_selection",
            "status": "completed",
            "title": "上下文精选完成",
            "detail": (
                f"最终使用 {len(context_chunks)}/{MAX_PROMPT_CHUNKS} 个片段，"
                f"覆盖 {covered_aspect_count}/{len(query_plan.aspects)} 个方面"
            ),
            "elapsed_ms": context_elapsed_ms,
            "summary": {
                "used_chunks": len(context_chunks),
                "max_prompt_chunks": MAX_PROMPT_CHUNKS,
                "covered_aspects": covered_aspect_count,
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
            "detail": f"已完成 Prompt 构造，用时 {prompt_elapsed_ms:.0f}ms",
            "elapsed_ms": prompt_elapsed_ms,
            "summary": {"prompt_chunk_count": len(context_chunks)},
        },
    )
    _report_progress(
        progress_reporter,
        {
            "stage": "llm_generation",
            "status": "skipped",
            "title": "LLM 生成暂未接入",
            "detail": "当前阶段未接入最终 LLM 生成，接口仅返回可用于生成的上下文包。",
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
    aspect_retrievals: list[AspectRetrieval] = []
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
        )
        matches = _expand_neighbor_matches(db, aspect.question, matches)
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
    return aspect_retrievals


def _retrieve_aspect_matches(
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
        match.metadata["fusion_method"] = ASPECT_QUERY_FUSION_METHOD
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


def _to_retrieval_results(matches: list[RetrievalMatch]) -> list[RetrievalResult]:
    results: list[RetrievalResult] = []
    seen_chunks: set[str] = set()
    seen_text: set[str] = set()
    for match in matches:
        citation = match.citation
        text = _clean_chunk_text(citation.excerpt, citation.section_title)
        normalized_text = _normalize_for_dedupe(text)
        if citation.chunk_id in seen_chunks or normalized_text in seen_text:
            continue
        seen_chunks.add(citation.chunk_id)
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
        if document_status != "indexed":
            invalid_chunk_ids.append(result.chunk_id)
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

    for aspect_retrieval in aspect_retrievals:
        if len(selected) >= MAX_PROMPT_CHUNKS:
            break
        core_chunk = _first_non_duplicate_candidate(aspect_retrieval.candidates, selected)
        if core_chunk is None:
            aspect_retrieval.covered = False
            continue
        _mark_chunk_for_aspect(core_chunk, aspect_retrieval.aspect)
        selected.append(core_chunk)
        aspect_retrieval.selected_chunk_ids.append(core_chunk.chunk_id)
        aspect_retrieval.covered = True

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
            _mark_chunk_for_aspect(chunk, aspect_retrieval.aspect)
            selected.append(chunk)
            aspect_retrieval.selected_chunk_ids.append(chunk.chunk_id)

    if FORCE_MIN_CHUNKS and len(selected) < MIN_PROMPT_CHUNKS:
        for aspect_retrieval in aspect_retrievals:
            for chunk in aspect_retrieval.candidates:
                if len(selected) >= min(MIN_PROMPT_CHUNKS, MAX_PROMPT_CHUNKS):
                    break
                if _is_duplicate_or_redundant(chunk, selected):
                    continue
                _mark_chunk_for_aspect(chunk, aspect_retrieval.aspect)
                selected.append(chunk)
                aspect_retrieval.selected_chunk_ids.append(chunk.chunk_id)

    selected = _sort_prompt_chunks(selected, query_plan)
    _renumber_context_chunks(selected)
    covered_aspects = {item.aspect.aspect_id for item in aspect_retrievals if item.covered}
    return selected, _prompt_selection_summary(
        candidate_count,
        selected,
        query_plan,
        aspect_retrievals,
        covered_aspects,
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
        "covered_aspects": [],
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
    summary.update(
        {
            "candidate_prompt_chunks": candidate_count,
            "final_prompt_chunks": len(selected),
            "covered_aspects": [aspect.aspect_id for aspect in query_plan.aspects if aspect.aspect_id in covered_aspects],
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


def _first_non_duplicate_candidate(
    candidates: list[RetrievalResult],
    selected: list[RetrievalResult],
) -> RetrievalResult | None:
    for chunk in candidates:
        if _is_duplicate_or_redundant(chunk, selected):
            continue
        return chunk
    return None


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
        if chunk_previous == selected_chunk.chunk_id or chunk_next == selected_chunk.chunk_id:
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

    def sort_key(chunk: RetrievalResult) -> tuple[Any, ...]:
        section_number = str(chunk.metadata.get("section_number") or "") or _section_number(chunk.section_title) or ""
        section_key = _section_sort_key(section_number)
        matched = chunk.metadata.get("prompt_matched_aspects")
        matched_ids = matched if isinstance(matched, list) else []
        aspect_index = min((aspect_order.get(str(aspect_id), 999) for aspect_id in matched_ids), default=999)
        if section_number:
            return (0, str(chunk.metadata.get("document_id") or chunk.source_doc), section_key, aspect_index, chunk.rank)
        return (1, aspect_index, -_prompt_score(chunk), chunk.rank)

    return sorted(chunks, key=sort_key)


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
    )


def _expand_neighbor_matches(db: Session, question: str, matches: list[RetrievalMatch]) -> list[RetrievalMatch]:
    if not matches:
        return []

    document_ids = {match.citation.document_id for match in matches}
    chunks_by_document = _chunks_by_document(db, document_ids)
    selected: list[RetrievalMatch] = []
    selected_chunk_ids: set[str] = set()

    for match in matches:
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


def _chunks_by_document(db: Session, document_ids: set[str]) -> dict[str, list[DocumentChunk]]:
    if not document_ids:
        return {}
    chunks = db.scalars(
        select(DocumentChunk)
        .where(DocumentChunk.document_id.in_(document_ids))
        .order_by(DocumentChunk.document_id.asc(), DocumentChunk.id.asc())
    ).all()
    grouped: dict[str, list[DocumentChunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.document_id, []).append(chunk)
    return grouped


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
        metadata={"expansion_reason": reason, "expanded_from_chunk_id": anchor.citation.chunk_id},
    )


def _adjacent_chunk_ids(chunk: DocumentChunk, chunks: list[DocumentChunk]) -> tuple[str | None, str | None]:
    index = chunks.index(chunk)
    previous_chunk_id = chunks[index - 1].chunk_id if index > 0 else None
    next_chunk_id = chunks[index + 1].chunk_id if index + 1 < len(chunks) else None
    return previous_chunk_id, next_chunk_id


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
    return "table" if text.lstrip().startswith(("表格：", "表格行证据：")) or "\n|" in text or "表格行证据：" in text else "paragraph"


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
        if item.covered
    ]
    missing_aspects = [
        _missing_aspect_note(item.aspect)
        for item in aspect_retrievals
        if not item.covered
    ]

    has_sufficient_context = bool(context_chunks) and not missing_aspects
    summary = {
        "top_k": MAX_PROMPT_CHUNKS,
        "used_chunks": len(context_chunks),
        "has_sufficient_context": has_sufficient_context,
        "coverage_notes": coverage_notes,
        "missing_aspects": missing_aspects,
        "citation_validation": citation_validation,
        "prompt_filtered_count": max(prompt_selection["candidate_prompt_chunks"] - len(context_chunks), 0),
        "prompt_selection": prompt_selection,
        "query_plan": query_plan.to_debug_dict(),
        "aspect_retrievals": [_aspect_retrieval_debug(item) for item in aspect_retrievals],
        "final_prompt_chunk_ids": [chunk.chunk_id for chunk in context_chunks],
        "fusion_method": ASPECT_QUERY_FUSION_METHOD,
    }
    summary.update(diagnostics.to_summary_fields())
    return summary


def _aggregate_aspect_diagnostics(aspect_retrievals: list[AspectRetrieval]) -> RetrievalDiagnostics:
    aggregate = RetrievalDiagnostics()
    vector_min: float | None = None
    vector_max: float | None = None
    rerank_min: float | None = None
    rerank_max: float | None = None

    for aspect_retrieval in aspect_retrievals:
        for diagnostics in aspect_retrieval.diagnostics:
            aggregate.query_count += int(diagnostics.get("query_count") or 0)
            aggregate.candidate_count += int(diagnostics.get("candidate_count") or 0)
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


def _aspect_retrieval_debug(item: AspectRetrieval) -> dict[str, Any]:
    return {
        **item.aspect.to_debug_dict(),
        "evidence_need": item.aspect.evidence_need,
        "covered": item.covered,
        "missing": not item.covered,
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


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 2)


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
    indexed_ids = set(
        db.scalars(
            select(Document.document_id).where(
                Document.document_id.in_(document_ids),
                Document.status == "indexed",
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
    detail = json.dumps(
        {
            "question": question,
            "answer": response.answer,
            "is_final_answer": package.is_final_answer if package else False,
            "mode": package.mode if package else "rag_context",
            "used_chunks": package.retrieval_summary["used_chunks"] if package else 0,
            "refused": response.refused,
            "confidence": response.confidence,
        },
        ensure_ascii=False,
    )
    log_action(db, action, "question", None, detail[:4000])
