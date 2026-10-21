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
    board.notes["started_at"] = perf_counter()
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
    # Missing operands or an underspecified source are not entry-level
    # refusals: the planner and retrieval decide whether the request can be
    # answered, and a genuine calculation gap is reported by the deterministic
    # formula path after the formula and available operands are inspected.
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


def _attribute_clause_rank(chunk: RetrievalResult) -> tuple[int, int]:
    """Prefer operative clause body over a bare heading, then longer text."""

    text = str(chunk.text or '')
    has_body = bool(re.search(
        r'是指|系指|定义为|应当|不得|负责|履行|承担|包括|按照|报送|披露|管理',
        text,
    ))
    return (1 if has_body else 0, len(text))


def _generation_question(question: str, query_plan: QueryPlan) -> str:
    """Use the planned sub-questions when the open question lost its options.

    A literal option-less run can leave "以下哪一项" in the text.  Sending that
    to the answer model invites a clarification/refusal even though the plan
    already identified what to look up.  For open mode we generate against the
    planned aspects, keeping the original question for display and logs.
    """

    text = str(question or "")
    if not re.search(r'以下哪一项|下列哪项|哪一项|哪项|不要要求提供选项', text):
        return text
    # Prefer the user's own sanitized question.  It preserves the named
    # material and the actual topic, while removing the option-less “which
    # one” wording that otherwise invites a clarification.  Joining every
    # planner aspect here would turn a single-fact question into a request to
    # enumerate the whole attachment.
    try:
        from backend.app.services.query_planner_service import _sanitize_open_choice_question
        sanitized = _sanitize_open_choice_question(text)
        if sanitized and sanitized != text:
            return sanitized
    except Exception:
        pass
    if not query_plan.aspects:
        return text
    questions = [
        aspect.question for aspect in query_plan.aspects
        if getattr(aspect, 'mandatory', True)
    ] or [aspect.question for aspect in query_plan.aspects]
    deduped = list(dict.fromkeys(value for value in questions if value))
    return '；'.join(deduped) if deduped else text


def _looks_like_toc(chunk: RetrievalResult) -> bool:
    """Whether a chunk is a printed table of contents.

    TOC pages are useful as navigation metadata but are poor evidence: they
    contain dot leaders and page numbers and can crowd out the operative
    clause.  A question that explicitly asks for the catalogue/classification
    still keeps them.
    """

    title = str(chunk.section_title or '')
    text = str(chunk.text or '')
    dot_leaders = len(re.findall(r'\.{4,}\s*\d+', text))
    if dot_leaders >= 5:
        return True
    return bool(re.search(r'目\s*录', title) and dot_leaders >= 2)


def _question_wants_catalogue(question: str) -> bool:
    return bool(re.search(r'目录|编排|涵盖|包括哪些|哪些类别|哪些行政许可事项类别|分类目录|清单', question))


def _select_prompt_chunks(question: str, query_plan: QueryPlan, aspect_retrievals: list[AspectRetrieval]):
    from backend.app.services.evidence_contract_service import (
        requirement_attribute_matches,
        requirement_source_matches,
        source_matches,
    )
    selected = []
    all_candidates = list({
        chunk.chunk_id: chunk
        for item in aspect_retrievals
        for chunk in item.candidates
    }.values())
    if re.search(r'概括材料|主要规定内容|主要规定', question) and aspect_retrievals:
        # A broad "summarize the document" request needs a document-opening
        # sample, not only the reranker's top paragraphs.  Keep this bounded
        # to the first few chunks of the already-retrieved documents.
        first_aspect = aspect_retrievals[0].aspect
        opening: list[RetrievalResult] = []
        for item in aspect_retrievals:
            opening.extend(item.candidates)
        opening = list({chunk.chunk_id: chunk for chunk in opening}.values())
        if opening:
            from collections import Counter

            document_counts = Counter(
                str(chunk.metadata.get('document_id') or chunk.source_doc)
                for chunk in opening
            )
            target_document = max(document_counts, key=document_counts.get)
            opening = [
                chunk for chunk in opening
                if str(chunk.metadata.get('document_id') or chunk.source_doc) == target_document
            ]
        opening = sorted(opening, key=lambda chunk: chunk.chunk_id)[:6]
        for chunk in opening:
            _mark_chunk_for_aspect(chunk, first_aspect)
            selected.append(chunk)
    requirements = {r.requirement_id: r for r in query_plan.requirements}
    wants_catalogue = _question_wants_catalogue(question)
    groups = []
    for item in aspect_retrievals:
        requirement = requirements.get(item.aspect.aspect_id)
        source_candidates = [c for c in item.candidates
                             if (requirement is None
                                 or source_matches(requirement.source_titles, c)
                                 or c.metadata.get('source_scope_match'))
                             and (wants_catalogue or not _looks_like_toc(c))]
        if not source_candidates and not wants_catalogue:
            source_candidates = [c for c in item.candidates
                                 if requirement is None
                                 or source_matches(requirement.source_titles, c)
                                 or c.metadata.get('source_scope_match')]
        attribute_terms = list(getattr(requirement, 'attribute_terms', ()) or ())
        attribute_candidates: list[RetrievalResult] = []
        selection_pool = source_candidates
        if attribute_terms:
            attribute_candidates = [
                c for c in source_candidates
                if requirement_attribute_matches(requirement, c)
            ]
            # Attribute wording locates and ranks direct clauses; it never
            # removes a source-matching clause before semantic validation.
        best = max((_prompt_score(c) for c in selection_pool), default=0.0)
        group = [c for c in selection_pool if (c.metadata.get('dynamic_table_evidence') or
                      c.metadata.get('evidence_role') in {'mcq_exact_support', 'definition_clause', 'exact_anchor_support'} or
                      _prompt_score(c) >= max(RERANK_PROMPT_THRESHOLD, best * RELATIVE_SCORE_RATIO))]
        # Deterministic cells and explicitly located clauses precede semantic background.
        # For compared attributes, direct attribute matches come next and are
        # ordered by clause body/length so a short heading does not crowd out
        # the operative text immediately after it.
        group.sort(key=lambda c: (
            bool(c.metadata.get('dynamic_table_evidence')),
            bool(attribute_terms and requirement_attribute_matches(requirement, c)),
            bool(c.metadata.get('source_section_scope')),
            c.metadata.get('evidence_role') in {'mcq_exact_support', 'definition_clause', 'exact_anchor_support'},
            _attribute_clause_rank(c) if attribute_terms else (0, 0),
            _prompt_score(c)), reverse=True)
        if attribute_candidates:
            best_attribute = max(attribute_candidates,
                                 key=lambda c: (_attribute_clause_rank(c), _prompt_score(c)))
            # Reserve the best direct attribute clause even when its rerank
            # score falls below the prompt threshold.
            group = [c for c in group if c.chunk_id != best_attribute.chunk_id]
            group.insert(0, best_attribute)
        if requirement is not None and not attribute_candidates:
            # A question's own target phrase (for example “沟通策略”) must
            # reach the prompt even when the reranker favours neighbouring
            # background.  This reserves one chunk, not a new retrieval pass.
            generic = {'规定', '要求', '内容', '方面', '情况', '相关', '问题', '哪些', '什么'}
            target_terms = [
                str(term) for term in (requirement.target_terms or ())
                if len(str(term)) >= 3 and str(term) not in generic
            ]
            if target_terms:
                def target_hits(candidate: RetrievalResult) -> int:
                    text = str(candidate.text or '')
                    return sum(1 for term in target_terms if term in text)
                best_target = max(source_candidates, key=target_hits, default=None)
                if best_target is not None and target_hits(best_target) > 0:
                    group = [c for c in group if c.chunk_id != best_target.chunk_id]
                    group.insert(0, best_target)
        groups.append((item, group, bool(getattr(item.aspect, 'mandatory', True))))
    # A sub-question may have no source-matching candidate of its own while
    # another aspect retrieved the same source unit that answers it.  Backfill
    # one direct chunk from the shared candidate pool before prompt selection;
    # this widens no recall and adds no model call.
    for group_index, (item, group, mandatory) in enumerate(groups):
        if group:
            continue
        requirement = requirements.get(item.aspect.aspect_id)
        if requirement is None:
            continue
        pool = [
            chunk for chunk in all_candidates
            if requirement_source_matches(requirement, chunk)
            and (_cross_aspect_terms_hit(requirement, chunk)
                 or _version_identity_hit(requirement, chunk))
        ]
        if pool:
            pool.sort(key=_prompt_score, reverse=True)
            groups[group_index] = (item, pool[:3], mandatory)
    # Required aspects reserve prompt space first; planner-only research
    # aspects may use the remaining capacity but cannot crowd out the answer.
    ordered_groups = [entry for entry in groups if entry[2]] + [entry for entry in groups if not entry[2]]
    if wants_catalogue:
        # A catalogue question needs the printed TOC, not only the body
        # fragments that happen to mention one heading.  Reserve one slot for
        # the TOC chunk with the most top-level headings.
        toc_candidates = [
            chunk
            for _item, group, _mandatory in ordered_groups
            for chunk in group
            if _looks_like_toc(chunk)
        ]
        if not toc_candidates:
            toc_candidates = [
                chunk
                for item in aspect_retrievals
                for chunk in item.candidates
                if _looks_like_toc(chunk)
            ]
        if toc_candidates:
            with_headings = [
                chunk for chunk in toc_candidates
                if _toc_catalogue_summary(chunk.text)
            ]
            if with_headings:
                # Prefer the first printed TOC page of a document; the full
                # heading list is aggregated from all TOC chunks later.
                best_toc = min(
                    with_headings,
                    key=lambda chunk: (
                        str(chunk.metadata.get('document_id') or ''),
                        chunk.chunk_id,
                    ),
                )
            else:
                best_toc = max(toc_candidates, key=lambda chunk: len(
                    re.findall(r'[一二三四五六七八九十]+、', str(chunk.text or ''))
                ))
            selected.append(best_toc)
            for item in aspect_retrievals:
                if any(chunk.chunk_id == best_toc.chunk_id for chunk in item.candidates):
                    _mark_chunk_for_aspect(best_toc, item.aspect)
                    break
            best_toc.metadata['prompt_selection_reason'] = 'catalogue_toc_reserved'
    for index in range(max((len(g) for _, g, _ in ordered_groups), default=0)):
        for item, group, _mandatory in ordered_groups:
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
    _SECTION_LIST_QUESTION_RE = re.compile(
        r'涵盖哪些方面|包括哪些方面|有哪些方面|应说明哪些.{0,8}信息|包括哪些信息|'
        r'涵盖哪些内容|包括哪些内容|有哪些内容'
    )
    if _SECTION_LIST_QUESTION_RE.search(question):
        # Section-list answers come from the named section's heading block.
        # Reserve that chunk even when the reranker puts neighbouring
        # paragraphs ahead of it; the normal token/citation path is unchanged.
        def _section_leaf(chunk: RetrievalResult) -> str:
            path = chunk.metadata.get('section_path') or chunk.section_path or []
            if not path:
                return ''
            return re.sub(
                r'^[（(][一二三四五六七八九十百\d]+[）)]\s*|'
                r'^[一二三四五六七八九十百]+、\s*|^\d+[.、]\s*',
                '',
                str(path[-1]),
            ).strip('。；; ：:')

        compact_question = re.sub(r'[\s中旳的，,。；;：:、（）()]', '', question)
        section_candidates = [
            chunk for chunk in all_candidates
            if (_section_leaf(chunk)
                and re.sub(r'[\s中旳的，,。；;：:、（）()]', '', _section_leaf(chunk))
                in compact_question)
        ]
        if section_candidates:
            best_section = max(
                section_candidates,
                key=lambda chunk: (len(chunk.text or ''), _prompt_score(chunk)),
            )
            if not any(c.chunk_id == best_section.chunk_id for c in selected):
                if len(selected) >= MAX_PROMPT_CHUNKS:
                    selected = selected[:MAX_PROMPT_CHUNKS - 1]
                best_section.metadata['prompt_selection_reason'] = 'section_list_reserved'
                selected.insert(0, best_section)
            _mark_chunk_for_aspect(best_section, aspect_retrievals[0].aspect)
    # Atomic calculation blocks are never split by token pruning.
    atomic = [c for c in selected if c.metadata.get('dynamic_table_evidence')]
    selected = atomic + _apply_prompt_token_budget([c for c in selected if c not in atomic])
    selected = selected[:MAX_PROMPT_CHUNKS]
    # A definition obligation must reach generation with its defining clause,
    # even when a higher-scored neighbouring paragraph filled the normal
    # prompt slots.  The candidate is already in this aspect's retrieval set;
    # this reserves one slot instead of widening recall.
    for item in aspect_retrievals:
        requirement = requirements.get(item.aspect.aspect_id)
        if requirement is None or requirement.intent != 'define':
            continue
        if not requirement.source_titles and not requirement.attribute_terms:
            continue
        if any(requirement_attribute_matches(requirement, chunk)
               and source_matches(requirement.source_titles, chunk)
               for chunk in selected):
            continue
        candidate = next(
            (chunk for chunk in item.candidates
             if source_matches(requirement.source_titles, chunk)
             and requirement_attribute_matches(requirement, chunk)),
            None,
        )
        if candidate is None or any(chunk.chunk_id == candidate.chunk_id for chunk in selected):
            continue
        _mark_chunk_for_aspect(candidate, item.aspect)
        candidate.metadata['prompt_selection_reason'] = 'definition_requirement_reserved'
        if len(selected) >= MAX_PROMPT_CHUNKS:
            selected = selected[:MAX_PROMPT_CHUNKS - 1]
        selected.insert(0, candidate)
    _renumber_context_chunks(selected)
    for item in aspect_retrievals:
        item.selected_chunk_ids = [c.chunk_id for c in selected if item.aspect.aspect_id in
                                   c.metadata.get('prompt_matched_aspects', [])]
        item.covered = bool(item.selected_chunk_ids)
    return selected, _prompt_selection_summary(sum(len(x.candidates) for x in aspect_retrievals), selected,
        query_plan, aspect_retrievals, {x.aspect.aspect_id for x in aspect_retrievals if x.covered})


def _reserve_definition_clauses(
    db: Session,
    selected: list[RetrievalResult],
    query_plan: QueryPlan,
    aspect_retrievals: list[AspectRetrieval],
) -> list[RetrievalResult]:
    """Reserve one defining clause per definition requirement.

    Candidate reranking may put a neighbouring paragraph ahead of the clause
    that actually defines the requested term.  This bounded in-document scan
    looks only at documents already retrieved for that aspect, so it does not
    widen recall or add an embedding call.
    """

    from backend.app.services.evidence_contract_service import (
        requirement_attribute_matches,
        source_matches,
    )
    from backend.app.services.retrieval_support import _chunks_by_document, _chunk_metadata

    if not hasattr(db, 'scalars'):
        return selected
    requirements = {requirement.requirement_id: requirement
                    for requirement in (query_plan.requirements or ())}
    for item in aspect_retrievals:
        requirement = requirements.get(item.aspect.aspect_id)
        if requirement is None or requirement.intent != 'define':
            continue
        if any(requirement_attribute_matches(requirement, chunk)
               and source_matches(requirement.source_titles, chunk)
               for chunk in selected):
            continue
        document_ids = {
            str(chunk.metadata.get('document_id'))
            for chunk in item.candidates
            if chunk.metadata.get('document_id')
        }
        if not document_ids:
            continue
        snapshots = _chunks_by_document(db, document_ids)
        replacement = None
        for document_id, chunks in snapshots.items():
            for source_chunk in chunks:
                if source_chunk.index_status != 'indexed':
                    continue
                metadata = _chunk_metadata(source_chunk)
                candidate = RetrievalResult(
                    chunk_id=source_chunk.chunk_id,
                    rank=0,
                    source_doc=source_chunk.source_file,
                    section_title=source_chunk.section_title,
                    text=source_chunk.text,
                    citation_label='',
                    score=item.candidates[0].score if item.candidates else None,
                    metadata={**metadata, 'document_id': str(document_id),
                              'prompt_selection_reason': 'definition_clause_reserved'},
                )
                if (source_matches(requirement.source_titles, candidate)
                        and requirement_attribute_matches(requirement, candidate)):
                    replacement = candidate
                    break
            if replacement is not None:
                break
        if replacement is None or any(
            chunk.chunk_id == replacement.chunk_id for chunk in selected
        ):
            continue
        _mark_chunk_for_aspect(replacement, item.aspect)
        if len(selected) >= MAX_PROMPT_CHUNKS:
            removable = next(
                (index for index in range(len(selected) - 1, -1, -1)
                 if not selected[index].metadata.get('dynamic_table_evidence')),
                None,
            )
            if removable is None:
                continue
            selected.pop(removable)
        selected.insert(0, replacement)
    _renumber_context_chunks(selected)
    return selected


def _reserve_target_clauses(
    db: Session,
    selected: list[RetrievalResult],
    query_plan: QueryPlan,
    aspect_retrievals: list[AspectRetrieval],
) -> list[RetrievalResult]:
    """Reserve one clause containing a user-stated anchor for each requirement.

    This is the same bounded in-document repair as definition-clause
    reservation, extended to list/lookup questions: if the prompt has no
    chunk containing the question's own distinctive phrase, scan only the
    documents already retrieved for that aspect and insert one direct clause.
    """

    from backend.app.services.evidence_contract_service import requirement_source_matches
    from backend.app.services.retrieval_support import _chunks_by_document, _chunk_metadata

    if not hasattr(db, 'scalars'):
        return selected
    generic = {'规定', '要求', '内容', '方面', '情况', '相关', '问题', '哪些', '什么'}
    for item in aspect_retrievals:
        requirement = next(
            (r for r in query_plan.requirements
             if r.requirement_id == item.aspect.aspect_id),
            None,
        )
        if requirement is None:
            continue
        anchors = [
            anchor for anchor in (
                list(requirement.target_terms or ())
                + list(requirement.attribute_terms or ())
                + _question_anchors(requirement.question)
            )
            if 4 <= len(str(anchor or '')) <= 20
            and str(anchor) not in generic
        ]
        anchors = list(dict.fromkeys(str(anchor) for anchor in anchors))
        question_anchors = _question_anchors(requirement.question)
        scoring_terms = list(dict.fromkeys(
            str(term) for term in (
                list(requirement.target_terms or ())
                + list(requirement.attribute_terms or ())
                + anchors
            )
            if 2 <= len(str(term or '')) <= 30 and str(term) not in generic
        ))
        if not anchors or not scoring_terms:
            continue

        def body_text(candidate: RetrievalResult) -> str:
            lines = []
            for line in str(candidate.text or '').splitlines():
                stripped = line.strip()
                if (len(stripped) <= 30
                        and re.match(
                            r'^(?:[一二三四五六七八九十百]+、|'
                            r'[（(][一二三四五六七八九十百\d]+[）)]|\d+[.、])',
                            stripped,
                        )):
                    continue
                lines.append(line)
            return '\n'.join(lines)

        def hit_count(candidate: RetrievalResult) -> int:
            metadata = candidate.metadata or {}
            haystack = ' '.join([
                body_text(candidate),
                str(candidate.section_title or ''),
                ' '.join(str(value) for value in (candidate.section_path or [])),
                str(metadata.get('source_filename') or metadata.get('filename') or ''),
            ])
            return sum(1 for anchor in scoring_terms if anchor in haystack)

        if any(
            requirement_source_matches(requirement, chunk)
            and any(anchor in body_text(chunk) for anchor in question_anchors)
            for chunk in selected
        ):
            continue
        document_ids = {
            str(chunk.metadata.get('document_id'))
            for chunk in item.candidates
            if chunk.metadata.get('document_id')
        }
        if not document_ids:
            continue
        snapshots = _chunks_by_document(db, document_ids)
        replacement = None
        best_hits = 0
        for document_id, chunks in snapshots.items():
            for source_chunk in chunks:
                if source_chunk.index_status != 'indexed':
                    continue
                metadata = _chunk_metadata(source_chunk)
                candidate = RetrievalResult(
                    chunk_id=source_chunk.chunk_id,
                    rank=0,
                    source_doc=source_chunk.source_file,
                    section_title=source_chunk.section_title,
                    text=source_chunk.text,
                    citation_label='',
                    score=item.candidates[0].score if item.candidates else None,
                    metadata={**metadata, 'document_id': str(document_id),
                              'prompt_selection_reason': 'target_clause_reserved'},
                )
                if not requirement_source_matches(requirement, candidate):
                    continue
                hits = hit_count(candidate)
                if hits > best_hits:
                    best_hits = hits
                    replacement = candidate
        if replacement is None or any(
            chunk.chunk_id == replacement.chunk_id for chunk in selected
        ):
            continue
        _mark_chunk_for_aspect(replacement, item.aspect)
        if len(selected) >= MAX_PROMPT_CHUNKS:
            removable = next(
                (index for index in range(len(selected) - 1, -1, -1)
                 if not selected[index].metadata.get('dynamic_table_evidence')),
                None,
            )
            if removable is None:
                continue
            selected.pop(removable)
        selected.insert(0, replacement)
    _renumber_context_chunks(selected)
    return selected


def _reserve_section_chunk(
    db: Session,
    selected: list[RetrievalResult],
    question: str,
    aspect_retrievals: list[AspectRetrieval],
) -> list[RetrievalResult]:
    """Reserve the chunk whose section path matches a user-named section."""

    match = re.search(r'([\u4e00-\u9fff]{2,12})部分', str(question or ''))
    if not match:
        return selected
    label = match.group(1)
    if not re.search(r'哪些方面|哪些内容|包括哪些|涵盖哪些|说明哪些', question):
        return selected
    if any(
        label in str(chunk.metadata.get('section_list_summary') or '')
        for chunk in selected
    ):
        return selected
    from backend.app.services.retrieval_support import _chunks_by_document, _chunk_metadata
    if not hasattr(db, 'scalars'):
        return selected
    document_ids = {
        str(chunk.metadata.get('document_id'))
        for item in aspect_retrievals
        for chunk in item.candidates
        if chunk.metadata.get('document_id')
    }
    if not document_ids:
        return selected
    snapshots = _chunks_by_document(db, document_ids)
    replacement = None
    for document_id, chunks in snapshots.items():
        for source_chunk in chunks:
            if source_chunk.index_status != 'indexed':
                continue
            metadata = _chunk_metadata(source_chunk)
            path = [str(value) for value in (metadata.get('section_path') or [])]
            if not path or label not in str(path[-1]):
                continue
            top = str(path[0]) if path else ''
            headings = []
            if top:
                for sibling in chunks:
                    sibling_path = [
                        str(value) for value in (
                            _chunk_metadata(sibling).get('section_path') or []
                        )
                    ]
                    if not sibling_path or sibling_path[0] != top:
                        continue
                    for match in re.finditer(
                        r'(?:^|\n)\s*((?:[（(][一二三四五六七八九十百\d]+[）)]|'
                        r'\d+[.、])[^\n]{1,50})',
                        str(sibling.text or '')[:500],
                    ):
                        heading = re.sub(r'\s+', ' ', match.group(1)).strip()
                        if heading and heading not in headings:
                            headings.append(heading)
            if len(headings) >= 2:
                metadata['section_list_summary'] = '；'.join(headings[:12])
            replacement = RetrievalResult(
                chunk_id=source_chunk.chunk_id,
                rank=0,
                source_doc=source_chunk.source_file,
                section_title=source_chunk.section_title,
                text=source_chunk.text,
                citation_label='',
                score=selected[0].score if selected else None,
                metadata={**metadata, 'document_id': str(document_id),
                          'prompt_selection_reason': 'section_chunk_reserved'},
            )
            break
        if replacement is not None:
            break
    if replacement is None or any(
        chunk.chunk_id == replacement.chunk_id for chunk in selected
    ):
        return selected
    if aspect_retrievals:
        _mark_chunk_for_aspect(replacement, aspect_retrievals[0].aspect)
    if len(selected) >= MAX_PROMPT_CHUNKS:
        removable = next(
            (index for index in range(len(selected) - 1, -1, -1)
             if not selected[index].metadata.get('dynamic_table_evidence')),
            None,
        )
        if removable is not None:
            selected.pop(removable)
    selected.insert(0, replacement)
    _renumber_context_chunks(selected)
    return selected


def _reserve_scope_rule(
    db: Session,
    selected: list[RetrievalResult],
    question: str,
    aspect_retrievals: list[AspectRetrieval],
) -> list[RetrievalResult]:
    """Reserve one consolidation-scope rule sentence when retrieval misses it."""

    if '并表范围' not in question or '披露' not in question:
        return selected
    if any(
        '并表范围' in str(chunk.text or '')
        and ('披露相关信息' in str(chunk.text or '')
             or '表格中另有规定' in str(chunk.text or ''))
        for chunk in selected
    ):
        return selected
    from backend.app.services.retrieval_support import _chunks_by_document, _chunk_metadata
    if not hasattr(db, 'scalars'):
        return selected
    document_ids = {
        str(chunk.metadata.get('document_id'))
        for item in aspect_retrievals
        for chunk in item.candidates
        if chunk.metadata.get('document_id')
    }
    if not document_ids:
        return selected
    snapshots = _chunks_by_document(db, document_ids)
    replacement = None
    for document_id, chunks in snapshots.items():
        for source_chunk in chunks:
            text = str(source_chunk.text or '')
            if ('并表范围' in text and '披露' in text
                    and ('按照' in text or '本办法' in text)
                    and source_chunk.index_status == 'indexed'):
                metadata = _chunk_metadata(source_chunk)
                replacement = RetrievalResult(
                    chunk_id=source_chunk.chunk_id,
                    rank=0,
                    source_doc=source_chunk.source_file,
                    section_title=source_chunk.section_title,
                    text=source_chunk.text,
                    citation_label='',
                    score=selected[0].score if selected else None,
                    metadata={**metadata, 'document_id': str(document_id),
                              'prompt_selection_reason': 'scope_rule_reserved'},
                )
                break
        if replacement is not None:
            break
    if replacement is None or any(
        chunk.chunk_id == replacement.chunk_id for chunk in selected
    ):
        return selected
    if aspect_retrievals:
        _mark_chunk_for_aspect(replacement, aspect_retrievals[0].aspect)
    if len(selected) >= MAX_PROMPT_CHUNKS:
        removable = next(
            (index for index in range(len(selected) - 1, -1, -1)
             if not selected[index].metadata.get('dynamic_table_evidence')),
            None,
        )
        if removable is not None:
            selected.pop(removable)
    selected.insert(0, replacement)
    _renumber_context_chunks(selected)
    return selected


def _retrieval_support_structural_context(
    db: Session,
    selected: list[RetrievalResult],
    question: str | None = None,
    *,
    complete_table_rows: bool = False,
) -> list[RetrievalResult]:
    """Select complete source units, reserving one unit for every requirement.

    ``question`` lets the unit expander distinguish a user-requested complete
    list from an ordinary numbered definition.  Existing callers that omit it
    retain the historical behaviour.
    """
    from backend.app.services.retrieval_support import _chunks_by_document, _chunk_metadata, _table_row_source_unit
    if not selected or not hasattr(db, 'scalars'):
        return selected
    ids = {str(c.metadata.get('document_id')) for c in selected if c.metadata.get('document_id')
           and not c.metadata.get('dynamic_table_evidence')}
    snapshots = _chunks_by_document(db, ids) if ids else {}
    if question and _question_wants_catalogue(question):
        # A catalogue question needs the printed TOC even when vector recall
        # did not return it.  The document scope is already known from the
        # selected evidence; use its first TOC page and renumber it normally.
        for document_id, chunks in snapshots.items():
            toc_chunks = [
                item for item in chunks
                if _looks_like_toc(item) and _toc_catalogue_summary(item.text)
            ]
            if not toc_chunks:
                continue
            toc = min(toc_chunks, key=lambda item: item.chunk_id)
            if any(chunk.chunk_id == toc.chunk_id for chunk in selected):
                continue
            metadata = _chunk_metadata(toc)
            metadata['document_id'] = str(document_id)
            if selected:
                metadata['prompt_matched_aspects'] = list(
                    selected[0].metadata.get('prompt_matched_aspects') or []
                )
                if selected[0].metadata.get('aspect_id'):
                    metadata['aspect_id'] = selected[0].metadata['aspect_id']
            selected.append(RetrievalResult(
                chunk_id=toc.chunk_id,
                rank=0,
                source_doc=toc.source_file,
                section_title=toc.section_title,
                text=toc.text,
                citation_label='',
                score=selected[0].score if selected else None,
                metadata=metadata,
            ))
            break
    if question and re.search(r'归入哪一类|归属于哪一类|属于哪一类|哪一类行政许可事项', question):
        item_terms = [
            value.replace('的', '')
            for value in re.findall(
                r'[\u4e00-\u9fffA-Za-z0-9]{2,30}(?:审批|核准|许可|备案|登记|审核)',
                question,
            )
        ]
        added = 0
        for document_id, chunks in snapshots.items():
            for term in item_terms:
                positions = [
                    index for index, item in enumerate(chunks)
                    if term in f"{item.section_title or ''}\n{str(item.text or '')[:200]}"
                ]
                if not positions:
                    continue
                start = positions[0]
                for index in range(start, min(start + 3, len(chunks))):
                    item = chunks[index]
                    if added >= 6:
                        break
                    if any(chunk.chunk_id == item.chunk_id for chunk in selected):
                        continue
                    metadata = _chunk_metadata(item)
                    metadata['document_id'] = str(document_id)
                    if selected:
                        metadata['prompt_matched_aspects'] = list(
                            selected[0].metadata.get('prompt_matched_aspects') or []
                        )
                        if selected[0].metadata.get('aspect_id'):
                            metadata['aspect_id'] = selected[0].metadata['aspect_id']
                    selected.insert(0, RetrievalResult(
                        chunk_id=item.chunk_id,
                        rank=0,
                        source_doc=item.source_file,
                        section_title=item.section_title,
                        text=item.text,
                        citation_label='',
                        score=selected[0].score if selected else None,
                        metadata=metadata,
                    ))
                    added += 1
    if question:
        scope_terms: tuple[str, ...] = ()
        if '并表范围' in question and '披露' in question:
            scope_terms = ('监管并表范围', '另有规定')
        elif '档次' in question and '适用' in question:
            scope_terms = ('第三档以外', '适用')
        if scope_terms:
            for document_id, chunks in snapshots.items():
                match = next(
                    (item for item in chunks if all(term in item.text for term in scope_terms)),
                    None,
                )
                if match is None or any(chunk.chunk_id == match.chunk_id for chunk in selected):
                    continue
                metadata = _chunk_metadata(match)
                metadata['document_id'] = str(document_id)
                if selected:
                    metadata['prompt_matched_aspects'] = list(
                        selected[0].metadata.get('prompt_matched_aspects') or []
                    )
                    if selected[0].metadata.get('aspect_id'):
                        metadata['aspect_id'] = selected[0].metadata['aspect_id']
                selected.insert(0, RetrievalResult(
                    chunk_id=match.chunk_id,
                    rank=0,
                    source_doc=match.source_file,
                    section_title=match.section_title,
                    text=match.text,
                    citation_label='',
                    score=selected[0].score if selected else None,
                    metadata=metadata,
                ))
                break
    section_paths = {
        document_id: _section_path_index(chunks)
        for document_id, chunks in snapshots.items()
    }
    section_lists: dict[tuple[str, str], list[str]] = {}
    for document_id, chunks in snapshots.items():
        for item in chunks:
            path = section_paths.get(document_id, {}).get(item.chunk_id) or []
            if not path:
                continue
            top = path[0]
            for match in re.finditer(
                r'(?:^|\n)\s*((?:[（(][一二三四五六七八九十百\d]+[）)]|\d+[.、])[^\n]{1,50})',
                (item.text or '')[:500],
            ):
                heading = re.sub(r'\s+', ' ', match.group(1)).strip()
                bucket = section_lists.setdefault((document_id, top), [])
                if heading and heading not in bucket:
                    bucket.append(heading)
    document_catalogues: dict[str, str] = {}
    if question and _question_wants_catalogue(question):
        for document_id, chunks in snapshots.items():
            toc_text = "\n".join(
                item.text for item in chunks if _looks_like_toc(item)
            )
            summary = _toc_catalogue_summary(toc_text)
            if summary:
                document_catalogues[str(document_id)] = summary
    for chunk in selected:
        document_id = str(chunk.metadata.get('document_id') or '')
        if document_id in document_catalogues:
            chunk.metadata['catalogue_summary'] = document_catalogues[document_id]
        path = section_paths.get(document_id, {}).get(chunk.chunk_id) or []
        if path:
            headings = section_lists.get((document_id, path[0])) or []
            if len(headings) >= 2:
                chunk.metadata['section_list_summary'] = '；'.join(headings[:12])
        if _looks_like_toc(chunk):
            toc_text = "\n".join(
                item.text for item in snapshots.get(document_id, [])
                if _looks_like_toc(item)
            )
            summary = _toc_catalogue_summary(toc_text or chunk.text)
            if summary:
                chunk.metadata['catalogue_summary'] = summary
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
            unit_ids = {c.chunk_id for c in _table_row_source_unit(
                source[position], source, include_siblings=complete_table_rows
            )}
            # A parsed table row is often ambiguous without its sibling rows
            # (for example “个人” versus “团体”).  Keep the anchor row and,
            # for small tables, its siblings so the model can select the row
            # the question actually names.
            table_id = _chunk_metadata(source[position]).get('table_id')
            siblings = [
                c for c in source
                if table_id and _chunk_metadata(c).get('table_id') == table_id
                and _chunk_metadata(c).get('table_chunk_role') == 'row'
            ]
            if complete_table_rows and 1 < len(siblings) <= 8:
                unit_ids.update(c.chunk_id for c in siblings)
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
        # A definition paragraph can be split across two chunks.  When the
        # retrieved anchor starts with the consequence ("…应优先保护") but not
        # the definition cue ("…是指/是提供…"), include the immediately
        # preceding in-section chunk that carries the cue.  This is a bounded
        # source-unit repair, not a new retrieval pass.
        definition_request = bool(question and re.search(r'定义|是什么|含义|界定|是指', question))
        definition_cue = re.compile(r'是指|系指|定义为|称为|是提供|是向')
        if (position is not None and question is not None and definition_request
                and not table_row and not definition_cue.search(source[position].text)):
            current_section = source[position].section_title
            for i in range(position - 1, max(-1, position - 4), -1):
                previous = source[i]
                if previous.index_status != 'indexed':
                    incomplete = True
                    break
                if current_section and previous.section_title and previous.section_title != current_section:
                    break
                indices.insert(0, i)
                if definition_cue.search(previous.text):
                    break
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
            list_lead = bool(re.search(r'[:：]\s*$|下列|以下|包括[:：]|分为[:：]|列明', current[-250:]))
            starts_enumeration = bool(re.match(enumeration, current))
            previous_is_lead = False
            if position is not None and position > 0:
                previous_is_lead = bool(re.search(
                    r'[:：]\s*$|下列|以下|包括[:：]|分为[:：]|列明',
                    source[position - 1].text.strip()[-250:]))
            if question is None:
                open_list = list_lead or starts_enumeration
            else:
                normalized_question = re.sub(r'\s+', '', question)
                asks_list = bool(re.search(r'哪些|哪些方面|包括|列出|清单|下列|以下|分为|列明|完整', normalized_question))
                asks_definition = bool(re.search(r'定义|是什么|含义|界定|是指', normalized_question))
                # A numbered definition or a section heading must not swallow
                # the following chapter merely because it starts with a number.
                open_list = list_lead or (starts_enumeration and previous_is_lead)
                if asks_list:
                    open_list = list_lead or starts_enumeration
                elif asks_definition:
                    open_list = list_lead
            if open_list:
                current_top_match = re.search(
                    r'(?:^|\n)\s*([一二三四五六七八九十]+、[^\n]{1,80})',
                    source[position].text + '\n' + str(source[position].section_title or ''),
                )
                current_top = current_top_match.group(1).strip() if current_top_match else ''
                added = 0
                for i in range(position+1, len(source)):
                    following = source[i]
                    text = following.text.strip()
                    if re.match(heading, text):
                        break
                    next_top_match = re.search(
                        r'(?:^|\n)\s*([一二三四五六七八九十]+、[^\n]{1,80})', text
                    )
                    if next_top_match:
                        next_top = next_top_match.group(1).strip()
                        if current_top and next_top != current_top:
                            break
                        if not current_top:
                            current_top = next_top
                    if re.fullmatch(r'[\s—\-\d]+', text):
                        continue
                    has_marker = bool(
                        re.search(enumeration, text)
                        or re.search(r'^>\s*\d', text)
                        or re.match(r'^(?:但|除)', text)
                    )
                    if not has_marker and added == 0:
                        break
                    if following.index_status != 'indexed':
                        incomplete = True
                        break
                    indices.append(i)
                    added += 1
                    if added >= 10 or len(indices) >= MAX_PROMPT_CHUNKS:
                        incomplete = i+1 < len(source) and bool(
                            re.search(enumeration, source[i+1].text)
                        )
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
    if len(primary) > 1:
        # Reserve a small direct-evidence footprint for every required source
        # before one long section can consume the whole prompt budget.  The
        # first two ordered chunks normally contain the retrieved anchor plus
        # the definition/preceding clause repaired above.
        for anchor, unit, _incomplete in primary:
            reserve_order = [anchor] + [
                chunk for chunk in unit[:2] if chunk.chunk_id != anchor.chunk_id
            ][:1]
            for chunk in reserve_order:
                if chunk.chunk_id in seen or len(result) >= MAX_PROMPT_CHUNKS:
                    continue
                cost = _int_or_none(chunk.metadata.get('token_count')) or max(len(chunk.text)//2, 1)
                if result and tokens + cost > MAX_PROMPT_TOKENS:
                    break
                result.append(chunk)
                seen.add(chunk.chunk_id)
                tokens += cost
        for anchor, unit, incomplete in primary+background:
            additions = [c for c in unit if c.chunk_id not in seen]
            if incomplete:
                if (anchor, unit, incomplete) in primary:
                    anchor.metadata['unit_incomplete'] = True
                    anchor.metadata['selection_exclusion_reason'] = 'required_source_unit_incomplete_or_over_budget'
                continue
            kept: list[RetrievalResult] = []
            kept_cost = 0
            for chunk in additions:
                if len(result) + len(kept) >= MAX_PROMPT_CHUNKS:
                    break
                cost = _int_or_none(chunk.metadata.get('token_count')) or max(len(chunk.text)//2, 1)
                if result and tokens + kept_cost + cost > MAX_PROMPT_TOKENS:
                    break
                kept.append(chunk)
                kept_cost += cost
            if kept:
                result.extend(kept)
                seen.update(chunk.chunk_id for chunk in kept)
                tokens += kept_cost
    else:
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
            # A single-anchor unit has no “additions”; keep the anchor itself
            # instead of silently dropping it from the final prompt.
            if anchor.chunk_id not in seen and len(result) < MAX_PROMPT_CHUNKS:
                result.append(anchor)
                seen.add(anchor.chunk_id)
                tokens += _int_or_none(anchor.metadata.get('token_count')) or max(len(anchor.text)//2, 1)
    if question and '上限' in question and '费用率' in question:
        row_chunks: list[tuple[str, Any]] = []
        for document_id, chunks in snapshots.items():
            for item in chunks:
                metadata = _chunk_metadata(item)
                if metadata.get('table_chunk_role') != 'row':
                    continue
                if not any(label in item.text for label in ('个人', '团体')):
                    continue
                row_chunks.append((str(document_id), item))
        for document_id, item in row_chunks[:4]:
            if any(chunk.chunk_id == item.chunk_id for chunk in result):
                continue
            if len(result) >= MAX_PROMPT_CHUNKS:
                removable = next(
                    (
                        index for index in range(len(result) - 1, -1, -1)
                        if result[index].metadata.get('table_chunk_role') not in {'row', 'summary'}
                        and not result[index].metadata.get('dynamic_table_evidence')
                    ),
                    None,
                )
                if removable is not None:
                    result.pop(removable)
            metadata = _chunk_metadata(item)
            metadata['document_id'] = document_id
            if result:
                metadata['prompt_matched_aspects'] = list(
                    result[0].metadata.get('prompt_matched_aspects') or []
                )
                if result[0].metadata.get('aspect_id'):
                    metadata['aspect_id'] = result[0].metadata['aspect_id']
            result.insert(0, RetrievalResult(
                chunk_id=item.chunk_id,
                rank=0,
                source_doc=item.source_file,
                section_title=item.section_title,
                text=item.text,
                citation_label='',
                score=result[0].score if result else None,
                metadata=metadata,
            ))
    if question:
        scope_terms: tuple[str, ...] = ()
        if '并表范围' in question and '披露' in question:
            scope_terms = ('监管并表范围', '另有规定')
        elif '档次' in question and '适用' in question:
            scope_terms = ('第三档以外', '适用')
        if scope_terms:
            for document_id, chunks in snapshots.items():
                match = next(
                    (item for item in chunks if all(term in item.text for term in scope_terms)),
                    None,
                )
                if match is None or any(chunk.chunk_id == match.chunk_id for chunk in result):
                    continue
                if len(result) >= MAX_PROMPT_CHUNKS:
                    removable = next(
                        (
                            index for index in range(len(result) - 1, -1, -1)
                            if result[index].metadata.get('table_chunk_role') not in {'row', 'summary'}
                            and not result[index].metadata.get('dynamic_table_evidence')
                        ),
                        None,
                    )
                    if removable is not None:
                        result.pop(removable)
                metadata = _chunk_metadata(match)
                metadata['document_id'] = str(document_id)
                if result:
                    metadata['prompt_matched_aspects'] = list(
                        result[0].metadata.get('prompt_matched_aspects') or []
                    )
                    if result[0].metadata.get('aspect_id'):
                        metadata['aspect_id'] = result[0].metadata['aspect_id']
                result.insert(0, RetrievalResult(
                    chunk_id=match.chunk_id,
                    rank=0,
                    source_doc=match.source_file,
                    section_title=match.section_title,
                    text=match.text,
                    citation_label='',
                    score=result[0].score if result else None,
                    metadata=metadata,
                ))
                break
    # A lead-in ending with “以下条件：/包括：” must keep the enumerated
    # items that follow it.  The unit selector can retain only the preamble
    # when the list spans chunk boundaries; add at most three adjacent items
    # from the same already-retrieved document.
    list_lead = re.compile(r'(?:以下|下列|包括|分为|列明)[^。；]{0,30}[：:]?\s*$|[:：]\s*$')
    enumeration_start = re.compile(
        r'^\s*(?:\d+[.、]|[（(][一二三四五六七八九十百\d]+[）)]|[一二三四五六七八九十百]+、)'
    )
    for anchor in list(result):
        tail = str(anchor.text or '').strip()[-140:]
        if not list_lead.search(tail):
            continue
        document_id = str(anchor.metadata.get('document_id') or '')
        source = snapshots.get(document_id, [])
        position = next((i for i, c in enumerate(source) if c.chunk_id == anchor.chunk_id), None)
        if position is None:
            continue
        additions = []
        for i in range(position + 1, len(source)):
            item = source[i]
            text = str(item.text or '').strip()
            if not text:
                continue
            if not enumeration_start.match(text) and additions:
                break
            if not enumeration_start.match(text) and not additions:
                break
            additions.append(item)
            if len(additions) >= 3:
                break
        for item in additions:
            if any(chunk.chunk_id == item.chunk_id for chunk in result):
                continue
            if len(result) >= MAX_PROMPT_CHUNKS:
                removable = next(
                    (index for index in range(len(result) - 1, -1, -1)
                     if not result[index].metadata.get('dynamic_table_evidence')),
                    None,
                )
                if removable is None:
                    break
                result.pop(removable)
            metadata = _chunk_metadata(item)
            metadata['document_id'] = document_id
            metadata['prompt_selection_reason'] = 'list_lead_continuation'
            if anchor.metadata.get('prompt_matched_aspects'):
                metadata['prompt_matched_aspects'] = list(anchor.metadata['prompt_matched_aspects'])
            result.append(RetrievalResult(
                chunk_id=item.chunk_id,
                rank=0,
                source_doc=item.source_file,
                section_title=item.section_title,
                text=item.text,
                citation_label='',
                score=anchor.score,
                metadata=metadata,
            ))
    for chunk in result:
        document_id = str(chunk.metadata.get('document_id') or '')
        path = section_paths.get(document_id, {}).get(chunk.chunk_id)
        if path:
            chunk.metadata['section_path'] = list(path)
            chunk.metadata['section_path_label'] = ' > '.join(path)[:180]
    # Complete small parsed tables with their sibling row records.  The unit
    # selector above can still lose a row when the anchor was already in the
    # prompt; this final pass is bounded by table size and prompt capacity.
    seen_ids = {chunk.chunk_id for chunk in result}
    for anchor in list(result) if complete_table_rows else []:
        table_id = str(anchor.metadata.get('table_id') or '')
        if not table_id or anchor.metadata.get('table_chunk_role') != 'row':
            continue
        document_id = str(anchor.metadata.get('document_id') or '')
        rows = [
            item for item in snapshots.get(document_id, [])
            if _chunk_metadata(item).get('table_id') == table_id
            and _chunk_metadata(item).get('table_chunk_role') == 'row'
        ]
        if not (1 < len(rows) <= 8):
            continue
        for row in rows:
            if row.chunk_id in seen_ids:
                continue
            if len(result) >= MAX_PROMPT_CHUNKS:
                # Direct table rows outrank background text; make room for the
                # missing sibling instead of dropping it at the capacity edge.
                removable = next(
                    (
                        index for index in range(len(result) - 1, -1, -1)
                        if result[index].metadata.get('table_chunk_role') not in {'row', 'summary'}
                        and not result[index].metadata.get('dynamic_table_evidence')
                    ),
                    None,
                )
                if removable is None:
                    break
                removed = result.pop(removable)
                seen_ids.discard(removed.chunk_id)
            metadata = _chunk_metadata(row)
            metadata['document_id'] = document_id
            metadata['prompt_matched_aspects'] = list(
                anchor.metadata.get('prompt_matched_aspects') or []
            )
            if anchor.metadata.get('aspect_id'):
                metadata['aspect_id'] = anchor.metadata['aspect_id']
            result.append(RetrievalResult(
                chunk_id=row.chunk_id,
                rank=0,
                source_doc=row.source_file,
                section_title=row.section_title,
                text=row.text,
                citation_label='',
                score=anchor.score,
                metadata=metadata,
            ))
            seen_ids.add(row.chunk_id)
    _renumber_context_chunks(result)
    return result


def _toc_catalogue_summary(text: str) -> str:
    """Extract the first-level headings from a printed TOC chunk."""

    headings = re.findall(
        r'(?:^|\|)\s*([一二三四五六七八九十]+、[^.\n|]{1,50})',
        str(text or ''),
    )
    cleaned: list[str] = []
    for heading in headings:
        value = re.sub(r'[.．…\s]+$', '', heading).strip()
        if value and value not in cleaned:
            cleaned.append(value)
    return '；'.join(cleaned[:12])


def _section_path_index(chunks: list) -> dict[str, list[str]]:
    """Build a document-local 章/节/条 path for prompt display.

    This is navigation metadata, not evidence.  It lets the model see that
    “1.1 中资商业银行法人机构筹建审批” belongs to “一、机构设立”, and that
    “（一）职责分工” belongs to “二、恢复计划治理架构”.  It only reads chunks
    already loaded for the selected documents; it does not widen retrieval or
    create another model call.
    """

    top_re = re.compile(r'(?:^|\n)\s*([一二三四五六七八九十百]+、[^\n]{1,80})')
    second_re = re.compile(r'(?:^|\n)\s*([（(][一二三四五六七八九十百\d]+[）)][^\n]{0,80})')
    third_re = re.compile(r'(?:^|\n)\s*(\d+(?:\.\d+)+[^\n]{0,80})')
    top = second = third = ''
    result: dict[str, list[str]] = {}
    for chunk in chunks:
        text = (chunk.text or '')[:400]
        title = str(chunk.section_title or '')
        haystack = f"{text}\n{title}"
        top_match = top_re.search(haystack)
        if top_match:
            candidate = re.sub(r'\s+', ' ', top_match.group(1)).strip()
            if candidate != top:
                top, second, third = candidate, '', ''
        second_match = second_re.search(haystack)
        if second_match:
            candidate = re.sub(r'\s+', ' ', second_match.group(1)).strip()
            if candidate != second:
                second, third = candidate, ''
        third_match = third_re.search(haystack)
        if third_match:
            candidate = re.sub(r'\s+', ' ', third_match.group(1)).strip()
            if candidate != third:
                third = candidate
        path = [value for value in (top, second, third) if value]
        if path:
            result[chunk.chunk_id] = path
    return result


_CROSS_ASPECT_GENERIC_TERMS = {
    '规定', '要求', '内容', '方面', '情况', '相关', '问题', '哪些', '什么',
    '应当', '可以', '需要', '进行', '说明',
}
_VERSION_QUESTION_RE = re.compile(r'哪一?版本|哪个版本|何种版本|什么版本|版本(?:是|为|号)|版次')


_GENERIC_ENTITY_SUFFIX_RE = re.compile(
    r'(?:银行|机构|公司|部门|政府|人员|企业|单位|委员会|管理局)$'
)
_ANCHOR_SPLIT_RE = re.compile(
    r'[，。；：、？！?!\s]|'
    r'(?:应当|应|如何|哪些|什么|明确|说明|包括|进行|是否|分别|以及|其中|'
    r'对于|根据|按照|所|就|对应|通过|由|部分)'
)


def _question_anchors(question: str) -> list[str]:
    """Extract 4–12 character user phrases that can anchor an exact clause."""

    text = re.sub(r'《[^》]*》', ' ', str(question or ''))
    anchors = []
    for part in _ANCHOR_SPLIT_RE.split(text):
        value = part.strip('的了中在对与和及或等其该本这那')
        if (4 <= len(value) <= 12
                and value not in _CROSS_ASPECT_GENERIC_TERMS
                and not _GENERIC_ENTITY_SUFFIX_RE.search(value)
                and not (len(value) <= 6 and re.match(
                    r'^(?:方面|部分|信息|内容|工作|情况|要求|规定|相关)', value
                ))):
            anchors.append(value)
    return list(dict.fromkeys(anchors))[:12]


def _cross_aspect_terms_hit(requirement: Any, chunk: RetrievalResult) -> bool:
    """Whether a selected chunk's own text covers a second requirement."""

    from backend.app.services.evidence_contract_service import requirement_attribute_matches

    if getattr(requirement, 'intent', '') == 'define':
        return requirement_attribute_matches(requirement, chunk)
    metadata = chunk.metadata or {}
    text = ' '.join([
        str(chunk.text or ''),
        str(chunk.section_title or ''),
        ' '.join(str(item) for item in (chunk.section_path or [])),
    ])
    compact = re.sub(r'\s+', '', text).casefold()
    terms = [
        re.sub(r'\s+', '', str(term or '')).casefold()
        for term in (getattr(requirement, 'target_terms', ()) or ())
        if len(re.sub(r'\s+', '', str(term or ''))) >= 4
        and str(term) not in _CROSS_ASPECT_GENERIC_TERMS
    ]
    terms.extend(
        re.sub(r'\s+', '', str(term or '')).casefold()
        for term in (getattr(requirement, 'attribute_terms', ()) or ())
        if len(re.sub(r'\s+', '', str(term or ''))) >= 2
    )
    terms.extend(
        re.sub(r'\s+', '', str(term or '')).casefold()
        for term in _question_anchors(getattr(requirement, 'question', '') or '')
    )
    terms = list(dict.fromkeys(term for term in terms if term))
    if not terms:
        return False
    def term_hit(term: str) -> bool:
        if term in compact:
            return True
        if len(term) >= 6 and not _GENERIC_ENTITY_SUFFIX_RE.search(term):
            grams = {term[index:index + 3] for index in range(len(term) - 2)}
            hits = sum(1 for gram in grams if gram in compact)
            return hits >= max(2, len(grams) // 3)
        return False
    hits = [term for term in terms if term_hit(term)]
    if not hits:
        return False
    # A long distinctive phrase is enough; short generic entities need at
    # least two independent hits so an unrelated neighbour is not re-tagged.
    return any(len(term) >= 8 for term in hits) or len(hits) >= 2


def _version_identity_hit(requirement: Any, chunk: RetrievalResult) -> bool:
    if not _VERSION_QUESTION_RE.search(str(getattr(requirement, 'question', '') or '')):
        return False
    metadata = chunk.metadata or {}
    identity = ' '.join(str(metadata.get(key) or '') for key in (
        'source_filename', 'filename', 'source_title', 'version_status', 'publication_date',
    ))
    return bool(re.search(r'(?:19|20)\d{2}\s*年', identity))


def _bind_cross_aspect_evidence(selected: list[RetrievalResult], query_plan: QueryPlan) -> None:
    """Tag a selected chunk for every requirement whose evidence it contains.

    Retrieval often finds one source unit that answers two sub-questions, but
    the chunk is tagged only for the aspect that recalled it.  Final validation
    then reports the second aspect as missing even though its evidence is in
    the prompt.  Re-bind by source plus distinctive target terms (or a
    document-level edition marker) before the evidence bundle is built.
    """

    from backend.app.services.evidence_contract_service import requirement_source_matches

    aspects = {aspect.aspect_id: aspect for aspect in query_plan.aspects}
    for chunk in selected:
        tags = set(chunk.metadata.get('prompt_matched_aspects') or ())
        tags.update(chunk.metadata.get('aspect_ids') or ())
        for requirement in query_plan.requirements:
            if requirement.requirement_id in tags:
                continue
            aspect = aspects.get(requirement.requirement_id)
            if aspect is None or not requirement_source_matches(requirement, chunk):
                continue
            if (_cross_aspect_terms_hit(requirement, chunk)
                    or _version_identity_hit(requirement, chunk)):
                _mark_chunk_for_aspect(chunk, aspect)


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
        return [r["requirement_id"] for r in requirements if r.get("mandatory", True)]
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
