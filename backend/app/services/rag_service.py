from backend.app.core.rag_features import features, capture_question, event
import json
import re
from contextlib import nullcontext
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.app.core.config import (
    FINAL_CITATION_LIMIT,
    FORCE_MIN_CHUNKS,
    MAX_PROMPT_CHUNKS,
    MAX_PROMPT_TOKENS,
    MIN_PROMPT_CHUNKS,
    RELATIVE_SCORE_RATIO,
    RERANK_TOP_K,
    RERANK_PROMPT_THRESHOLD,
)
from backend.app.models.document import Document, DocumentChunk, QALog
from backend.app.schemas.qa import (
    AnswerClaim,
    Citation,
    EvidenceCoverage,
    EvidenceBundle,
    EvidenceDecision,
    EvidenceRequirement,
    LLMContextPackage,
    QAAnswerPreview,
    QAResponse,
    RetrievalResult,
)
from backend.app.services.audit_service import record_event
from backend.app.services.agents.contracts import Blackboard, board_context, board_for
from backend.app.services.agents.supervisor import Supervisor
from backend.app.services.answer_generation_service import generate_answer
from backend.app.services.document_identity_query_service import answer_document_identity_question
from backend.app.services.prompt_builder import RAGPromptBuilder
from backend.app.services.question_preprocessing_service import build_question_task, preprocess_qa_request
from backend.app.services.query_planner_service import (
    QueryAspect,
    QueryPlan,
    plan_query,
)
from backend.app.services.retrieval_support import (
    AspectRetrieval,
    QUERY_TYPE_WEIGHTS,
    RRF_K,
    _AspectPrepResult,
    _DOCUMENT_SNAPSHOT_CACHE,
    _DOCUMENT_SNAPSHOT_CACHE_LOCK,
    _build_retrieval_summary,
    _collect_prepared_candidates_many,
    _effective_rerank_candidate_limit,
    _elapsed_ms,
    _expand_neighbor_matches,
    _finish_aspect_matches,
    _normalize_for_dedupe,
    _prepare_aspect_retrieval,
    _report_progress,
    _rerank_query_for_aspect,
    _search_query_debug_list,
    _to_retrieval_results,
)
from backend.app.services.retrieval_service import (
    RetrievalDiagnostics,
    RetrievalMatch,
    RetrievalServiceUnavailable,
    ProgressReporter,
    collect_candidates_with_query_hits,
    rerank_input_with_document_coverage,
)
from backend.app.services.embedding_service import EmbeddingServiceError
from backend.app.services.model_device_service import get_model_device_info
from backend.app.services.performance_metrics import trace_operation, qa_execution
from backend.app.services.rerank_service import (
    RerankServiceError,
    rerank_candidate_groups,
)
from backend.app.services.vector_store_service import VectorStoreError


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


@qa_execution
@capture_question
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
    board = Blackboard(
        db=db,
        original_question=cleaned_question,
        question=cleaned_question,
        options=list(options or []),
        include_debug=include_debug,
        progress_reporter=progress_reporter,
        answer_preview_reporter=answer_preview_reporter,
        cancellation_checker=cancellation_checker,
    )
    board.notes["raw_options"] = options
    with board_context(board):
        return Supervisor().run(board)


def _refusal_code_for_request(question: str) -> str:
    """Classify common, explicit limits of a regulatory knowledge-base request."""

    return _explicit_request_boundary_code(question) or "insufficient_evidence"


def _explicit_request_boundary_code(question: str) -> str | None:
    # Only explicit requests, never bank identities or regulatory concepts.
    text = re.sub(r'《[^》]*》|[“\"‘\'][^”\"’\']*[”\"’\']', '', question)
    compact = re.sub(r'\s+', '', text)
    if re.search(r'请预测|请预计|预测未来|(?:请|帮我|能否|能不能).{0,16}预测|请断言.*必然|保证预测正确', compact):
        return 'unsupported_prediction'
    if re.search(r'替我决定|请决定|保证盈利|要不要(?:买入|投资|收购)', compact):
        return 'subjective_business_advice'
    if re.search(r'(?:尚未|没有|未)(?:上传|收录|入库)|(?:请查询|请获取|请提供).*(?:此刻|实时|当前最新)', compact):
        return 'out_of_scope_or_realtime'
    if re.search(r'(?:未给出|没有给出|没有提供|未提供|缺少|没有).*(?:操作数|数值|余额|净额|变量|期间|金额|币种|汇率)', compact):
        return 'missing_calculation_operands'
    if re.search(r'未指明|没有指定|没有任何定位信息|没有文件名|未提供任何', compact):
        return 'missing_required_context'
    return None


@qa_execution
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


@qa_execution
def retrieve_evidence(db: Session, query_plan: QueryPlan,
                      progress_reporter: ProgressReporter | None = None) -> EvidenceBundle:
    """Replayable retrieval stage: accepts one existing plan, never replans it."""
    board = board_for(db, query_plan.original_question, progress_reporter=progress_reporter)
    with (board_context(board) if board.standalone else nullcontext()):
        bundle = Supervisor().gather_evidence(board, query_plan)
        if board.standalone and isinstance(bundle.diagnostics, dict):
            bundle.diagnostics["agent_trace"] = [entry.model_dump() for entry in board.trace]
        return bundle


def build_context_package(
    db: Session,
    question: str,
    options: list[str] | None = None,
    progress_reporter: ProgressReporter | None = None,
) -> LLMContextPackage:
    board = board_for(db, question, options=options, progress_reporter=progress_reporter)
    with (board_context(board) if board.standalone else nullcontext()):
        package = Supervisor().build_context(board, question, options)
        if board.standalone:
            package.agent_trace = list(board.trace)
        return package


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
    board = board_for(db, query_plan.original_question, progress_reporter=progress_reporter)
    with (board_context(board) if board.standalone else nullcontext()):
        return Supervisor().gather_aspects(board, query_plan)


# Chunk text often cites another regulation by bracketed title; such titles
# seed the bounded sibling-document scope below.  Capped so one request adds
# at most two cited documents beyond the already-retrieved scope.


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
        # Keep whole indexed evidence units; prefix excerpts lose definitions,
        # conditions and exceptions at the end of a chunk.
        result.text = source_chunk.text
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


def _select_prompt_chunks(question: str, query_plan: QueryPlan, aspect_retrievals: list[AspectRetrieval]):
    from backend.app.services.evidence_contract_service import source_matches
    selected = []
    requirements = {r.requirement_id: r for r in query_plan.requirements}
    groups = []
    for item in aspect_retrievals:
        requirement = requirements[item.aspect.aspect_id]
        source_candidates = [c for c in item.candidates if source_matches(requirement.source_titles, c)]
        best = max((_prompt_score(c) for c in source_candidates), default=0.0)
        group = [c for c in source_candidates if (c.metadata.get('dynamic_table_evidence') or
                      c.metadata.get('evidence_role') in {'mcq_exact_support', 'definition_clause', 'exact_anchor_support'} or
                      _prompt_score(c) >= max(RERANK_PROMPT_THRESHOLD, best * RELATIVE_SCORE_RATIO))]
        # Deterministic cells and explicitly located clauses precede semantic background.
        group.sort(key=lambda c: (
            bool(c.metadata.get('dynamic_table_evidence')),
            bool(c.metadata.get('source_section_scope')),
            c.metadata.get('evidence_role') in {'mcq_exact_support', 'definition_clause', 'exact_anchor_support'},
            _prompt_score(c)), reverse=True)
        groups.append((item, group))
    # Round-robin reserves space for every source before adding more from one.
    for index in range(max((len(g) for _, g in groups), default=0)):
        for item, group in groups:
            if index >= len(group) or len(selected) >= MAX_PROMPT_CHUNKS:
                continue
            chunk = group[index]
            if any(c.chunk_id == chunk.chunk_id for c in selected):
                existing = next(c for c in selected if c.chunk_id == chunk.chunk_id)
                _mark_chunk_for_aspect(existing, item.aspect)
                continue
            duplicate = next((c for c in selected if c.metadata.get('document_id') == chunk.metadata.get('document_id') and
                              _normalize_for_dedupe(c.text) == _normalize_for_dedupe(chunk.text)), None)
            if duplicate is not None:
                _mark_chunk_for_aspect(duplicate, item.aspect)
                continue
            _mark_chunk_for_aspect(chunk, item.aspect)
            chunk.metadata['prompt_selection_reason'] = 'requirement_round_robin'
            selected.append(chunk)
    # Atomic calculation blocks are never split by token pruning.
    atomic = [c for c in selected if c.metadata.get('dynamic_table_evidence')]
    selected = atomic + _apply_prompt_token_budget([c for c in selected if c not in atomic])
    selected = selected[:MAX_PROMPT_CHUNKS]
    _renumber_context_chunks(selected)
    for item in aspect_retrievals:
        item.selected_chunk_ids = [c.chunk_id for c in selected if item.aspect.aspect_id in
                                   c.metadata.get('prompt_matched_aspects', [])]
        item.covered = bool(item.selected_chunk_ids)
    return selected, _prompt_selection_summary(sum(len(x.candidates) for x in aspect_retrievals), selected,
        query_plan, aspect_retrievals, {x.aspect.aspect_id for x in aspect_retrievals if x.covered})


def _retrieval_support_structural_context(db: Session, selected: list[RetrievalResult]) -> list[RetrievalResult]:
    """Select complete source units, reserving one unit for every requirement."""
    from backend.app.services.retrieval_support import _chunks_by_document, _chunk_metadata, _table_row_source_unit
    if not selected or not hasattr(db, 'scalars'):
        return selected
    ids = {str(c.metadata.get('document_id')) for c in selected if c.metadata.get('document_id')
           and not c.metadata.get('dynamic_table_evidence')}
    snapshots = _chunks_by_document(db, ids) if ids else {}
    enumeration = r'^\s*(?:[（(][一二三四五六七八九十百\d]+[）)]|[一二三四五六七八九十]+[、.]|\d+[.、\s])'
    heading = r'^\s*第[\d一二三四五六七八九十百]+[条章节]'
    units = []
    for anchor in selected:
        source = snapshots.get(str(anchor.metadata.get('document_id')), [])
        position = next((i for i, c in enumerate(source) if c.chunk_id == anchor.chunk_id), None)
        indices = [position] if position is not None else []
        incomplete = False
        table_row = position is not None and _chunk_metadata(source[position]).get('table_chunk_role') == 'row'
        if table_row:
            unit_ids = {c.chunk_id for c in _table_row_source_unit(source[position], source)}
            indices = [i for i, c in enumerate(source) if c.chunk_id in unit_ids]
        if position is not None and anchor.metadata.get('source_section_scope'):
            major = r'(?:^|\n)\s*([一二三四五六七八九十百]+[、.][^\n]{1,80}|第[一二三四五六七八九十百\d]+章[^\n]{0,80})'
            own_heading = re.search(major, source[position].text)
            own_title = own_heading[1].strip() if own_heading else ''
            indices = [position]
            for i in range(position+1, len(source)):
                next_heading = re.search(major, source[i].text)
                if next_heading and next_heading[1].strip() != own_title:
                    break
                if source[i].index_status != 'indexed':
                    incomplete = True
                    break
                indices.append(i)
        if (position is not None and not table_row and not anchor.metadata.get('source_section_scope')
                and not anchor.metadata.get('source_heading_fact') and not anchor.metadata.get('source_definition_fact')):
            current = source[position].text.strip()
            # An orphan numbered item or exception needs its governing preamble.
            if re.match(enumeration, current) or re.match(r'^(?:但|除|.*?另有规定.*?除外)', current):
                for i in range(position-1, max(-1, position-9), -1):
                    prev = source[i]
                    if prev.index_status != 'indexed':
                        incomplete = True
                        break
                    indices.insert(0, i)
                    if not re.match(enumeration, prev.text):
                        break
            open_list = bool(re.search(r'[:：]\s*$|下列|以下|包括[:：]', current[-250:]) or
                             re.match(enumeration, current))
            if open_list:
                for i in range(position+1, len(source)):
                    following = source[i]
                    text = following.text.strip()
                    if re.match(heading, text):
                        break
                    if re.fullmatch(r'[\s—\-\d]+', text):
                        continue
                    if not (re.match(enumeration, text) or re.match(r'^>\s*\d', text)
                            or re.match(r'^(?:但|除)', text)):
                        break
                    if following.index_status != 'indexed':
                        incomplete = True
                        break
                    indices.append(i)
                    if len(indices) >= MAX_PROMPT_CHUNKS:
                        incomplete = i+1 < len(source) and bool(re.match(enumeration, source[i+1].text))
                        break
        # Section titles are navigation metadata, not evidence. Include the
        # actual source heading even when PDF page/footnote fragments split it
        # from the retrieved item; stay within the declared source section.
        title = re.sub(r'\s+', '', source[position].section_title or '') if position is not None else ''
        if title and not table_row:
            for i in range(position, -1, -1):
                candidate = source[i]
                if re.sub(r'\s+', '', candidate.section_title or '') != title:
                    break
                if title not in re.sub(r'\s+', '', candidate.text):
                    continue
                if candidate.index_status != 'indexed':
                    incomplete = True
                else:
                    indices.append(i)
                break
        indices = sorted(set(indices))
        unit = []
        for i in indices:
            c = source[i]
            item = anchor if c.chunk_id == anchor.chunk_id else RetrievalResult(
                chunk_id=c.chunk_id, rank=0, source_doc=c.source_file, text=c.text,
                section_title=c.section_title, citation_label='', score=anchor.score,
                metadata={**anchor.metadata, **_chunk_metadata(c), 'document_id': c.document_id,
                          'evidence_unit': anchor.chunk_id, 'prompt_selection_reason': 'source_unit_continuation'})
            unit.append(item)
        units.append((anchor, unit or [anchor], incomplete))
    # Preserve one primary unit per requirement before admitting background units.
    primary, background, reserved = [], [], set()
    for unit in units:
        tags = set(unit[0].metadata.get('prompt_matched_aspects') or [])
        tags.update([unit[0].metadata['aspect_id']] if unit[0].metadata.get('aspect_id') else [])
        (primary if tags-reserved else background).append(unit)
        reserved.update(tags)
    result, seen, tokens = [], set(), 0
    for anchor, unit, incomplete in primary+background:
        additions = [c for c in unit if c.chunk_id not in seen]
        cost = sum(_int_or_none(c.metadata.get('token_count')) or max(len(c.text)//2, 1) for c in additions)
        fits = len(result)+len(additions) <= MAX_PROMPT_CHUNKS and tokens+cost <= MAX_PROMPT_TOKENS
        if not fits or incomplete:
            if (anchor, unit, incomplete) in primary:
                anchor.metadata['unit_incomplete'] = True
                anchor.metadata['selection_exclusion_reason'] = 'required_source_unit_incomplete_or_over_budget'
                if anchor.chunk_id not in seen and len(result) < MAX_PROMPT_CHUNKS:
                    result.append(anchor); seen.add(anchor.chunk_id)
            continue
        result.extend(additions)
        seen.update(c.chunk_id for c in additions)
        tokens += cost
    _renumber_context_chunks(result)
    return result


def _sync_aspect_coverage_from_prompt(selected: list[RetrievalResult],
                                     aspect_retrievals: list[AspectRetrieval],
                                     *, bundle: EvidenceBundle | None = None) -> None:
    from backend.app.services.evidence_contract_service import build_evidence_bundle
    if bundle is None:
        plan = QueryPlan('', tuple(item.aspect for item in aspect_retrievals), 'existing-plan')
        bundle = build_evidence_bundle(list(plan.requirements), selected)
    for item in aspect_retrievals:
        item.selected_chunk_ids = bundle.coverage.get(item.aspect.aspect_id, [])
        item.covered = bool(item.selected_chunk_ids) and item.aspect.aspect_id not in bundle.missing_requirements
        item.retrieval_covered = item.retrieval_covered or bool(item.selected_chunk_ids)
        for chunk in selected:
            if chunk.chunk_id in item.selected_chunk_ids:
                _mark_chunk_for_aspect(chunk, item.aspect)


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


def _prompt_score(chunk: RetrievalResult) -> float:
    metadata_score = _float_or_none(chunk.metadata.get("rerank_score"))
    if metadata_score is not None:
        return metadata_score
    return float(chunk.score or 0.0)


# Definition-type questions (“X是如何定义的/包括哪些”) are answered by the
# clause that defines the term (“X是指…/所称X…”).  A definition clause can sit
# one or two ranks below a numerically stronger neighbour, so generic prompt
# selection gives it a small deterministic bonus instead of letting the
# neighbour crowd it out.  Subject qualifiers (“财产保险公司” vs
# “人身保险公司”, “内部评级法” vs “权重法”) get the same tie-break treatment:
# a chunk that contains the institution or method named in the question is
# more likely to answer it than an equally scored sibling.


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


def _confidence_from_results(results: list[RetrievalResult]) -> float:
    scores = [result.score for result in results if result.score is not None]
    if not scores:
        return 0.0
    best_score = max(scores)
    return min(max(float(best_score), 0.0), 0.95)


def _required_aspect_ids(retrieval_summary: dict[str, Any]) -> list[str]:
    if not features().coverage_gate:
        return []
    requirements = retrieval_summary.get("evidence_bundle", {}).get("requirements")
    if requirements is not None:
        return [r["requirement_id"] for r in requirements]
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
        covered.update(result.metadata.get("supported_requirement_ids") or [])
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
