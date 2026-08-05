"""Retrieval support for the RAG pipeline.

Extracted from rag_service (pure movement, no logic change): per-aspect
retrieval preparation, hybrid collection, reranking, lexical/anchor/formula
and MCQ support passes, neighbor expansion, and their shared text helpers.

The module is self-contained: it does not import from rag_service, and
rag_service imports its public entries from here.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import re
from threading import RLock
from time import monotonic, perf_counter
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session, load_only

from backend.app.core.config import (
    DOCUMENT_SNAPSHOT_CACHE_MAX_DOCUMENTS,
    DOCUMENT_SNAPSHOT_CACHE_TTL_SECONDS,
    FINAL_CITATION_LIMIT,
    MAX_PROMPT_CHUNKS,
    RERANK_TOP_K,
    RETRIEVAL_TOP_K,
)
from backend.app.models.document import Document, DocumentChunk
from backend.app.schemas.qa import Citation, RetrievalResult
from backend.app.services.query_planner_service import QueryAspect, QuerySearchQuery
from backend.app.services.retrieval_metadata_filter_service import build_retrieval_metadata_filter
from backend.app.services.retrieval_service import (
    RetrievalDiagnostics,
    RetrievalMatch,
    RetrievalServiceUnavailable,
    ProgressReporter,
    _elapsed_ms,
    _embed_search_queries,
    _hybrid_search_many,
    _score_range,
    collect_candidates_with_query_hits,
    evidence_coverage,
    filter_active_candidates,
    filter_candidates_by_metadata,
    get_last_retrieval_diagnostics,
    limit_rerank_candidates,
    rerank_input_with_document_coverage,
    matches_from_reranked,
    question_terms,
    reset_retrieval_diagnostics,
    retrieve_citations,
)
from backend.app.services.embedding_service import EmbeddingServiceError
from backend.app.services.model_device_service import get_model_device_info
from backend.app.services.performance_metrics import measure
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

CONTEXT_CHUNK_CHAR_LIMIT = 1200
ASPECT_QUERY_FUSION_METHOD = "aspect_query_rrf_then_bge_rerank"
RRF_K = 60
# CPU cross-encoder cost is ~1.3s per pair regardless of batch size, so the
# rerank input size directly drives retrieval latency.  Open questions are
# capped tighter than MCQ questions, whose option-fact evidence must stay
# visible to the deterministic choice recovery (official 300 hard gate).
CONSTRAINED_RERANK_CANDIDATE_LIMIT = 12
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


@dataclass
class _PreparedAspect:
    """Deterministic retrieval preparation for one aspect, shared by the
    single-aspect and batched collection paths."""

    aspect: QueryAspect
    search_queries: list[str]
    query_metadata: list[dict[str, Any]]
    qdrant_filter: dict[str, Any] | None
    table_matches: list[RetrievalMatch]
    table_diagnostics: list[dict[str, Any]]
    retrieval_filter: Any
    diagnostics: RetrievalDiagnostics
    total_started_at: float
    mcq_material_document_ids: set[str]


@dataclass
class _AspectPrepResult:
    terminal: bool
    matches: list[RetrievalMatch]
    diagnostics: list[dict[str, Any]]
    prepared: _PreparedAspect | None


def _prepare_aspect_retrieval(
    db: Session,
    aspect: QueryAspect,
    progress_reporter: ProgressReporter | None = None,
    document_chunk_cache: dict[str, list[DocumentChunk]] | None = None,
) -> _AspectPrepResult:
    """Run the deterministic pre-collection stages for one aspect.

    Covers the legacy hook, structured spreadsheet retrieval with terminal
    refusal early-stops, metadata filter construction with the no-match gate,
    MCQ exact-support early stop, and search-query normalization.  Terminal
    outcomes return matches/diagnostics directly; otherwise a prepared aspect
    is returned for collection (single or batched).
    """

    if retrieve_citations is not _DEFAULT_RETRIEVE_CITATIONS:
        matches, diagnostics = _retrieve_aspect_matches_legacy_hook(db, aspect, progress_reporter)
        return _AspectPrepResult(terminal=True, matches=matches, diagnostics=diagnostics, prepared=None)

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
            return _AspectPrepResult(True, table_matches, table_diagnostics, None)
        if not table_matches and _should_stop_after_spreadsheet_refusal(aspect, spreadsheet_diagnostic):
            table_diagnostics[-1]["early_stopped"] = True
            table_diagnostics[-1]["early_stop_reason"] = "structured_spreadsheet_refusal"
            return _AspectPrepResult(True, [], table_diagnostics, None)

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
        return _AspectPrepResult(True, table_matches, table_diagnostics + [filter_diagnostic], None)

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
            return _AspectPrepResult(True, exact_matches, table_diagnostics + [diagnostic], None)

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
    qdrant_filter = retrieval_filter.qdrant_filter()
    if aspect.modality != "table":
        # Period fields describe SpreadsheetCell coordinates, not a
        # regulation document's publication year. Applying them to mixed
        # text retrieval would remove the required regulation evidence.
        for period_key in ("year", "month", "quarter"):
            qdrant_filter.pop(period_key, None)
    mcq_material_document_ids: set[str] = set()
    if aspect.aspect_id == "multiple_choice_evidence":
        mcq_material_document_ids = _explicit_material_document_ids(db, aspect)
    return _AspectPrepResult(
        terminal=False,
        matches=[],
        diagnostics=[],
        prepared=_PreparedAspect(
            aspect=aspect,
            search_queries=search_queries,
            query_metadata=query_metadata,
            qdrant_filter=qdrant_filter or None,
            table_matches=table_matches,
            table_diagnostics=table_diagnostics,
            retrieval_filter=retrieval_filter,
            diagnostics=diagnostics,
            total_started_at=total_started_at,
            mcq_material_document_ids=mcq_material_document_ids,
        ),
    )


def _collect_prepared_candidates_many(
    prepared: list[_PreparedAspect],
) -> list[tuple[list[Any], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]]:
    """Collect hybrid candidates for many prepared aspects in one pass.

    All search queries across all aspects are embedded and searched in a
    single Qdrant batch call (each query keeps its own metadata filter), then
    the results are split back per aspect with the same fusion bookkeeping
    that the single-aspect collector performs.
    """

    if not prepared:
        return []
    all_queries: list[str] = []
    all_metadata: list[dict[str, Any]] = []
    all_filters: list[dict[str, Any] | None] = []
    for item in prepared:
        all_queries.extend(item.search_queries)
        all_metadata.extend(item.query_metadata)
        all_filters.extend([item.qdrant_filter] * len(item.search_queries))
    embedding_started_at = perf_counter()
    query_embeddings = _embed_search_queries(all_queries)
    embedding_elapsed_ms = _elapsed_ms(embedding_started_at)
    qdrant_started_at = perf_counter()
    raw_results_by_query = _hybrid_search_many(
        query_embeddings,
        limit=RETRIEVAL_TOP_K,
        metadata_filters=all_filters,
    )
    qdrant_elapsed_ms = _elapsed_ms(qdrant_started_at)
    for item in prepared:
        # The batched pass timings belong to every participating aspect so
        # per-aspect diagnostics keep the same fields as the single path.
        item.diagnostics.timings_ms["embedding"] = embedding_elapsed_ms
        item.diagnostics.timings_ms["qdrant"] = qdrant_elapsed_ms
    results: list[tuple[list[Any], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]] = []
    offset = 0
    for item in prepared:
        count = len(item.search_queries)
        collect_kwargs: dict[str, Any] = {
            "query_metadata": item.query_metadata,
            "diagnostics": item.diagnostics,
            "query_embeddings": query_embeddings[offset : offset + count],
            "raw_results_by_query": raw_results_by_query[offset : offset + count],
        }
        if item.qdrant_filter:
            collect_kwargs["metadata_filter"] = item.qdrant_filter
        candidates, query_hits_by_chunk_id, per_query_diagnostics = collect_candidates_with_query_hits(
            item.search_queries,
            **collect_kwargs,
        )
        results.append((candidates, query_hits_by_chunk_id, per_query_diagnostics))
        offset += count
    return results


def _finish_aspect_matches(
    db: Session,
    aspect: QueryAspect,
    *,
    candidates: list[Any],
    query_hits_by_chunk_id: dict[str, list[dict[str, Any]]],
    diagnostics_by_query: list[dict[str, Any]],
    diagnostics: RetrievalDiagnostics,
    retrieval_filter: Any,
    mcq_material_document_ids: set[str],
    table_matches: list[RetrievalMatch],
    table_diagnostics: list[dict[str, Any]],
    document_chunk_cache: dict[str, list[DocumentChunk]] | None,
    total_started_at: float,
    progress_reporter: ProgressReporter | None = None,
    reranked: list[RerankedChunk] | None = None,
    rerank_input: list[Any] | None = None,
) -> tuple[list[RetrievalMatch], list[dict[str, Any]]]:
    """Turn collected hybrid candidates into validated, reranked matches.

    Shared by the single-aspect path (``reranked=None``, which reranks here)
    and the batched path (``reranked`` supplied by the caller so all aspects
    share one inference pass).  ``diagnostics`` is the collector's diagnostics
    object so query-level counters survive into the fused record.  Per-aspect
    rerank progress events are only emitted on the single-aspect path; the
    batched path reports once in ``_retrieve_aspects``.
    """

    filter_debug = retrieval_filter.to_debug_dict()
    fusion_scores: dict[str, float] = {}
    for candidate in candidates:
        for hit in query_hits_by_chunk_id.get(candidate.chunk_id, []):
            query_type = str(hit.get("query_type") or "semantic_question")
            query_weight = QUERY_TYPE_WEIGHTS.get(query_type, 1.0)
            rank = int(hit.get("rank") or 1)
            contribution = query_weight / (RRF_K + rank)
            fusion_scores[candidate.chunk_id] = fusion_scores.get(candidate.chunk_id, 0.0) + contribution
            hit["rrf_contribution"] = round(contribution, 6)

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
    retrieval_fit_scores = {
        candidate.chunk_id: _retrieval_fit_score(candidate, aspect)
        for candidate in candidates
    }
    candidates.sort(
        key=lambda candidate: (
            bool(preferred_file_type and candidate.filename.lower().endswith(f".{preferred_file_type}")),
            inferred_filter_scores.get(candidate.chunk_id, 0.0),
            document_style_scores.get(candidate.chunk_id, 0.0),
            retrieval_fit_scores.get(candidate.chunk_id, 0.0),
            fusion_scores.get(candidate.chunk_id, 0.0),
            candidate.score + candidate.anchor_boost,
        ),
        reverse=True,
    )
    # ``candidates`` is already ordered by weighted RRF above.  Re-sorting it
    # by one raw vector score here would undo multi-query fusion and starve
    # exact option/statement hits in long documents before cross-encoding.
    if rerank_input is None:
        effective_rerank_limit = _effective_rerank_candidate_limit(
            aspect,
            candidates,
            retrieval_filter_document_ids=set(retrieval_filter.document_ids or ()),
            mcq_material_document_ids=mcq_material_document_ids,
        )
        rerank_input = rerank_input_with_document_coverage(
            candidates,
            limit=effective_rerank_limit,
        )
    else:
        effective_rerank_limit = len(rerank_input)
    diagnostics.rerank_input_count = len(rerank_input)
    if reranked is None:
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
    else:
        diagnostics.rerank_call_count = 1 if rerank_input else 0
    if aspect.aspect_id == "multiple_choice_evidence":
        for item in reranked:
            direct_score = document_style_scores.get(item.candidate.chunk_id, 0.0)
            # The blended score orders MCQ candidates; the raw cross-encoder
            # score is preserved for the reliability gates in _is_reliable so a
            # strong model match is never filtered by a weak lexical term.
            item.raw_rerank_score = float(item.rerank_score)
            item.rerank_score = 0.35 * float(item.rerank_score) + 0.65 * direct_score
        reranked.sort(key=lambda item: item.rerank_score, reverse=True)
        reranked = reranked[:RERANK_TOP_K]
    diagnostics.reranked_count = len(reranked)
    diagnostics.score_range = _score_range(candidates, reranked)
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

    # Real candidates' fusion scores, captured before any supplement injection:
    # later passes must not chain off previously injected synthetic scores.
    real_fusion_best = max(fusion_scores.values(), default=0.0)

    exact_anchor_matches = _exact_anchor_support_matches(
        db,
        aspect,
        matches,
        document_ids=set(retrieval_filter.document_ids or ()),
        document_chunk_cache=document_chunk_cache,
    )
    if exact_anchor_matches:
        # Supplements outrank the aspect's real matches so an anchored clause
        # is not drowned out by rerank noise, but their internal order follows
        # the real support score instead of insertion order.
        exact_anchor_matches.sort(key=lambda match: match.rerank_score, reverse=True)
        for offset, supplement in enumerate(exact_anchor_matches, start=1):
            fusion_scores[supplement.citation.chunk_id] = min(real_fusion_best + 0.015, 0.99) - offset * 0.0001
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
        lexical_support_matches.sort(key=lambda match: match.rerank_score, reverse=True)
        for offset, supplement in enumerate(lexical_support_matches, start=1):
            fusion_scores[supplement.citation.chunk_id] = min(real_fusion_best + 0.01, 0.99) - offset * 0.0001
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
        formula_support_matches.sort(key=lambda match: match.rerank_score, reverse=True)
        for offset, supplement in enumerate(formula_support_matches, start=1):
            fusion_scores[supplement.citation.chunk_id] = min(real_fusion_best + 0.01, 0.99) - offset * 0.0001
        existing_chunk_ids = {match.citation.chunk_id for match in formula_support_matches}
        matches = formula_support_matches + [
            match for match in matches if match.citation.chunk_id not in existing_chunk_ids
        ]

    # Definition-clause channel: definition questions ("X是如何定义的") are
    # answered by the clause that actually defines the quoted term ("X是指…"/
    # "所称X…").  That clause can lose to numerically stronger neighbours in
    # vector ranking even when it was recalled (e.g. "最低资本" definition sat
    # at RRF rank 2 but never reached the prompt).  A bounded lexical scan for
    # the quoted term inside already-recalled documents, with a full-corpus
    # fallback, promotes the defining clause deterministically.
    definition_clause_matches = _definition_clause_support_matches(
        db,
        aspect,
        matches,
        document_chunk_cache=document_chunk_cache,
    )
    if definition_clause_matches:
        definition_clause_matches.sort(key=lambda match: match.rerank_score, reverse=True)
        for offset, supplement in enumerate(definition_clause_matches, start=1):
            fusion_scores[supplement.citation.chunk_id] = min(real_fusion_best + 0.02, 0.99) - offset * 0.0001
        existing_chunk_ids = {match.citation.chunk_id for match in definition_clause_matches}
        matches = definition_clause_matches + [
            match for match in matches if match.citation.chunk_id not in existing_chunk_ids
        ]

    # Intent-gap supplement (CRAG-lite): when the question's own keywords or
    # quoted terms are absent from the recalled evidence, the right clause may
    # still live in the same documents (e.g. "实物结算…可以不适用初始保证金"
    # exists in the same 办法 whose other 豁免 clauses were recalled).  A
    # bounded lexical scan over the recalled documents promotes those clauses;
    # prohibition-intent questions ("禁止性规定/不得") additionally search
    # "{institution}不得" clauses that generic vector ranking may have missed.
    intent_gap_matches = _intent_gap_supplement(
        db,
        aspect,
        matches,
        document_chunk_cache=document_chunk_cache,
    )
    if intent_gap_matches:
        intent_gap_matches.sort(key=lambda match: match.rerank_score, reverse=True)
        for offset, supplement in enumerate(intent_gap_matches, start=1):
            fusion_scores[supplement.citation.chunk_id] = min(real_fusion_best + 0.015, 0.99) - offset * 0.0001
        existing_chunk_ids = {match.citation.chunk_id for match in intent_gap_matches}
        matches = intent_gap_matches + [
            match for match in matches if match.citation.chunk_id not in existing_chunk_ids
        ]

    # List-completeness supplement (CRAG-lite, list questions): “包括哪些/哪些
    # 情形” answers live in an enumeration clause (“（一）…（二）…”).  When such a
    # clause carries the question's topic terms but lost the prompt seat to
    # numerically stronger neighbours, promote it inside the recalled documents.
    list_completeness_matches = _list_completeness_supplement(
        db,
        aspect,
        matches,
        document_chunk_cache=document_chunk_cache,
    )
    if list_completeness_matches:
        list_completeness_matches.sort(key=lambda match: match.rerank_score, reverse=True)
        for offset, supplement in enumerate(list_completeness_matches[: _LIST_COMPLETENESS_LIMIT], start=1):
            fusion_scores[supplement.citation.chunk_id] = min(real_fusion_best + 0.01, 0.99) - offset * 0.0001
        existing_chunk_ids = {match.citation.chunk_id for match in list_completeness_matches}
        matches = list_completeness_matches + [
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
            supplements.sort(key=lambda match: match.rerank_score, reverse=True)
            for offset, supplement in enumerate(supplements, start=1):
                chunk_id = supplement.citation.chunk_id
                fusion_scores[chunk_id] = min(real_fusion_best + 0.01, 0.99) - offset * 0.0001
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
    if reranked is None:
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


def _retrieve_aspect_matches(
    db: Session,
    aspect: QueryAspect,
    progress_reporter: ProgressReporter | None = None,
    document_chunk_cache: dict[str, list[DocumentChunk]] | None = None,
) -> tuple[list[RetrievalMatch], list[dict[str, Any]]]:
    prepared = _prepare_aspect_retrieval(
        db,
        aspect,
        progress_reporter=progress_reporter,
        document_chunk_cache=document_chunk_cache,
    )
    if prepared.terminal:
        return prepared.matches, prepared.diagnostics
    prepared_aspect = prepared.prepared
    assert prepared_aspect is not None
    try:
        collect_kwargs: dict[str, Any] = {
            "query_metadata": prepared_aspect.query_metadata,
            "diagnostics": prepared_aspect.diagnostics,
        }
        if prepared_aspect.qdrant_filter:
            collect_kwargs["metadata_filter"] = prepared_aspect.qdrant_filter
        candidates, query_hits_by_chunk_id, diagnostics_by_query = collect_candidates_with_query_hits(
            prepared_aspect.search_queries,
            **collect_kwargs,
        )
    except (EmbeddingServiceError, VectorStoreError) as exc:
        raise RetrievalServiceUnavailable(str(exc)) from exc
    return _finish_aspect_matches(
        db,
        aspect,
        candidates=candidates,
        query_hits_by_chunk_id=query_hits_by_chunk_id,
        diagnostics_by_query=diagnostics_by_query,
        diagnostics=prepared_aspect.diagnostics,
        retrieval_filter=prepared_aspect.retrieval_filter,
        mcq_material_document_ids=prepared_aspect.mcq_material_document_ids,
        table_matches=prepared_aspect.table_matches,
        table_diagnostics=prepared_aspect.table_diagnostics,
        document_chunk_cache=document_chunk_cache,
        total_started_at=prepared_aspect.total_started_at,
        progress_reporter=progress_reporter,
    )


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
    return CONSTRAINED_RERANK_CANDIDATE_LIMIT


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


# Definition questions are answered by the clause defining the quoted term.
# The marker set mirrors rag_service._DEFINITION_QUESTION_MARKERS; kept here so
# the retrieval channel does not depend on rag_service internals.
_DEFINITION_QUESTION_MARKERS = ("定义", "如何界定", "是指什么", "指什么", "是什么", "包括哪些", "如何理解")


# Mutual-exclusion term pairs for attachment-level disambiguation: when the
# question names one method/institution class, chunks whose source document
# title carries the same term win the rerank-input seat over equally scored
# chunks from the rival attachment ("内部评级法" vs "权重法", "非寿险" vs
# "寿险", "财产险" vs "人身险").  Generic and corpus-independent.
_ATTACHMENT_TERM_PAIRS = (
    ("内部评级法", "权重法"),
    ("权重法", "内部评级法"),
    ("标准法", "内部模型法"),
    ("内部模型法", "标准法"),
    ("非寿险", "寿险"),
    ("寿险", "非寿险"),
    ("财产险", "人身险"),
    ("人身险", "财产险"),
    ("财产保险公司", "人身保险公司"),
    ("人身保险公司", "财产保险公司"),
    ("商业银行", "财务公司"),
    ("财务公司", "商业银行"),
)
_SECTION_SUBJECT_QUALIFIERS = (
    "财产保险公司", "人身保险公司", "财产险公司", "人身险公司", "再保险公司",
    "保险公司", "保险集团", "商业银行", "财务公司", "信托公司", "货币经纪公司",
    "金融租赁公司", "消费金融公司", "非寿险", "寿险", "财产险", "人身险",
)


def _retrieval_fit_score(candidate: Any, aspect: QueryAspect) -> float:
    """Attachment/section-level fit tie-break for rerank-input ordering.

    Two generic signals, both corpus-independent:
    - attachment fit: the question's method/institution term appears in the
      candidate's source title (and the rival term does not);
    - section fit: the question's subject qualifier appears in the candidate's
      section title/path (same-document paragraph disambiguation, e.g. the
      财产保险公司 pressure-scenario paragraph vs the 人身保险公司 one).
    """

    question = str(aspect.question or "")
    if not question:
        return 0.0
    filename = str(candidate.filename or "")
    score = 0.0
    for term, rival in _ATTACHMENT_TERM_PAIRS:
        if term in question:
            if term in filename:
                score += 0.3
            elif rival in filename:
                score -= 0.25
    for match in re.finditer(r"附件\s*([0-9一二三四五六七八九十]+)", question):
        if f"附件{match.group(1)}" in filename:
            score += 0.3
    section = " ".join(
        part
        for part in (
            candidate.section_title or "",
            *(candidate.section_path or []),
        )
        if part
    )
    if section:
        for qualifier in _SECTION_SUBJECT_QUALIFIERS:
            if qualifier in question and qualifier in section:
                score += 0.2
    return score


_INTENT_GAP_STOPWORDS = {
    "监管要求", "处理", "资产", "业务", "情形", "条件", "范围", "规定", "要求",
    "事项", "行为", "公司", "银行", "保险", "机构", "相关", "有关", "具体",
    "主要", "应当", "可以", "不得", "是否", "如何", "哪些", "什么", "多少",
    "规定", "规则", "办法", "通知", "内容", "情况", "问题", "信息", "数据",
}
_INTENT_GAP_PROHIBITION_MARKERS = ("禁止性", "不得", "禁止", "不允许", "负面清单")
_INTENT_GAP_PATTERN_LIMIT = 6
# “不符合X标准的资产如何处理”类问题的答案条款是监管文本中的兜底条款
# （“对不符合…标准的…，应纳入…处理/视为…/比照…”）。这类条款的关键词是
# 否定词（3 字）与处理动词（2 字），都被词长/停用词过滤排除，因此单独成通道。
_INTENT_GAP_FALLBACK_NEGATIONS = ("不符合", "不满足", "未达到", "未符合", "无法")
_INTENT_GAP_FALLBACK_ACTIONS = ("处理", "处置", "对待", "纳入", "视为", "比照")
_LIST_COMPLETENESS_MARKERS = (
    "哪些", "包括哪些", "有哪些", "哪些情形", "哪些条件", "哪些情况", "哪些主体",
    "哪些假设", "哪些原则", "哪些内容", "包括什么",
    "下列情形之一", "以下情形之一", "具备下列条件", "符合下列条件",
    "包括但不限于",
)
_LIST_COMPLETENESS_LIMIT = 6


def _intent_gap_supplement(
    db: Session,
    aspect: QueryAspect,
    matches: list[RetrievalMatch],
    *,
    document_chunk_cache: dict[str, list[DocumentChunk]] | None = None,
) -> list[RetrievalMatch]:
    """Deterministic clause recovery for keywords missing from the evidence.

    Generic and bounded by the already-recalled documents: a term the user or
    the planner named (aspect keywords, quoted phrases) that is absent from
    every recalled excerpt is searched lexically inside those same documents.
    Prohibition questions additionally scan "{institution}不得" clauses, which
    vector ranking frequently misses in favour of nearby 监管指标 paragraphs.
    """

    if not hasattr(db, "scalars") or not matches:
        return []
    question = str(aspect.question or "")
    terms: set[str] = set()
    for keyword in aspect.keywords:
        cleaned = re.sub(r"\s+", "", str(keyword or ""))
        if 3 <= len(cleaned) <= 20 and cleaned not in _INTENT_GAP_STOPWORDS:
            terms.add(cleaned)
    for quoted in re.findall(r"[“‘'\"]([^”’'\"]+)[”’'\"]", question):
        cleaned = re.sub(r"\s+", "", quoted)
        if 2 <= len(cleaned) <= 24:
            terms.add(cleaned)
    if any(marker in question for marker in _INTENT_GAP_PROHIBITION_MARKERS):
        for keyword in aspect.keywords:
            cleaned = re.sub(r"\s+", "", str(keyword or ""))
            if 4 <= len(cleaned) <= 12 and any(tail in cleaned for tail in ("公司", "银行", "机构")):
                terms.add(f"{cleaned}不得")
    fallback_terms: set[str] = set()
    if any(negation in question for negation in _INTENT_GAP_FALLBACK_NEGATIONS) and any(
        action in question for action in _INTENT_GAP_FALLBACK_ACTIONS
    ):
        # Fallback-clause channel: “不符合…如何处理” answers live in catch-all
        # clauses that mention both the negation and a treatment verb.  The
        # negation itself is the highest-signal anchor, so search for it and
        # require a treatment verb inside the same chunk during the scan.
        for negation in _INTENT_GAP_FALLBACK_NEGATIONS:
            if negation in question:
                terms.add(negation)
                fallback_terms.add(negation)
    if not terms:
        return []
    evidence_text = "\n".join(str(match.evidence_text or "") for match in matches[:24])
    normalized_evidence = _normalize_exact_support_text(evidence_text)
    missing = [
        term
        for term in sorted(terms)
        if _normalize_exact_support_text(term) not in normalized_evidence
    ]
    if not missing:
        return []
    scope_document_ids = {
        str(match.citation.document_id or "")
        for match in matches
        if match.citation.document_id
    }
    if not scope_document_ids:
        return []
    chunks_by_document = _chunks_by_document(
        db,
        scope_document_ids,
        document_chunk_cache=document_chunk_cache,
    )
    existing_ids = {match.citation.chunk_id for match in matches}
    supplements: list[RetrievalMatch] = []
    matched_patterns: set[str] = set()
    for term in missing[: _INTENT_GAP_PATTERN_LIMIT]:
        normalized_term = _normalize_exact_support_text(term)
        is_fallback_term = term in fallback_terms
        for document_chunks in chunks_by_document.values():
            for chunk in document_chunks:
                if chunk.index_status != "indexed" or chunk.chunk_id in existing_ids:
                    continue
                searchable = str(chunk.text or chunk.embedding_text or "")
                if normalized_term not in _normalize_exact_support_text(searchable):
                    continue
                if is_fallback_term and not any(
                    action in searchable for action in _INTENT_GAP_FALLBACK_ACTIONS
                ):
                    continue
                document_chunks_for_match = document_chunks
                match = _direct_exact_support_match(
                    chunk,
                    document_chunks_for_match,
                    aspect,
                    term,
                    support_score=1.2,
                )
                match.citation.evidence_role = "intent_gap_support"
                match.evidence_role = "intent_gap_support"
                match.metadata["evidence_role"] = "intent_gap_support"
                match.metadata["fusion_method"] = "intent_gap_lexical"
                match.metadata["intent_gap_term"] = term
                supplements.append(match)
                existing_ids.add(chunk.chunk_id)
                matched_patterns.add(term)
                break
    return supplements


def _list_completeness_supplement(
    db: Session,
    aspect: QueryAspect,
    matches: list[RetrievalMatch],
    *,
    document_chunk_cache: dict[str, list[DocumentChunk]] | None = None,
) -> list[RetrievalMatch]:
    """Recover list-clause chunks for enumeration questions.

    “包括哪些/有哪些/哪些情形” questions are answered by a clause that
    enumerates items (“（一）…（二）…”, “1. … 2. …”).  Hybrid ranking often
    recalls the items' paragraphs but not the clause that frames the list, or
    the framing clause itself is split across chunk boundaries.  This pass
    scans the already-recalled documents for enumeration-bearing chunks that
    mention the question's topic terms and were not recalled, then promotes
    them so the answer generator sees the whole list framing clause.
    """

    if not hasattr(db, "scalars") or not matches:
        return []
    question = str(aspect.question or "")
    if not any(marker in question for marker in _LIST_COMPLETENESS_MARKERS):
        return []
    topic_terms = []
    for keyword in aspect.keywords:
        cleaned = re.sub(r"\s+", "", str(keyword or ""))
        if 4 <= len(cleaned) <= 20 and cleaned not in _INTENT_GAP_STOPWORDS:
            topic_terms.append(cleaned)
    if not topic_terms:
        return []
    scope_document_ids = {
        str(match.citation.document_id or "")
        for match in matches
        if match.citation.document_id
    }
    if not scope_document_ids:
        return []
    chunks_by_document = _chunks_by_document(
        db,
        scope_document_ids,
        document_chunk_cache=document_chunk_cache,
    )
    existing_ids = {match.citation.chunk_id for match in matches}
    supplements: list[RetrievalMatch] = []
    for document_chunks in chunks_by_document.values():
        for chunk in document_chunks:
            if chunk.index_status != "indexed" or chunk.chunk_id in existing_ids:
                continue
            text = str(chunk.text or chunk.embedding_text or "")
            if not _looks_like_enumeration_clause(text):
                continue
            if not any(term in text for term in topic_terms):
                continue
            match = _direct_exact_support_match(
                chunk,
                document_chunks,
                aspect,
                topic_terms[0],
                support_score=1.1,
            )
            match.citation.evidence_role = "list_completeness_support"
            match.evidence_role = "list_completeness_support"
            match.metadata["evidence_role"] = "list_completeness_support"
            match.metadata["fusion_method"] = "list_completeness_lexical"
            supplements.append(match)
            existing_ids.add(chunk.chunk_id)
    return supplements


def _looks_like_enumeration_clause(text: str) -> bool:
    """True when the text carries at least three enumeration items.

    Covers the item styles used by regulation texts: （一）…（二）…, (一)…(二)…,
    1. … 2. … (with a following non-digit so 1.5% rates are not counted), and
    circled numbers ①②③.  A bare count is intentionally loose — the topic-term
    filter in the caller keeps false positives out of the prompt.
    """

    if len(text) < 24:
        return False
    chinese_items = len(re.findall(r"[（(][一二三四五六七八九十]{1,3}[）)]", text))
    if chinese_items >= 3:
        return True
    numeric_items = len(
        re.findall(r"(?<!\d)\d{1,2}[.．、](?!\d)", text)
    )
    if numeric_items >= 3:
        return True
    circled_items = len(re.findall(r"[①②③④⑤⑥⑦⑧⑨⑩]", text))
    return circled_items >= 3


def _definition_clause_support_matches(
    db: Session,
    aspect: QueryAspect,
    matches: list[RetrievalMatch],
    *,
    document_chunk_cache: dict[str, list[DocumentChunk]] | None = None,
) -> list[RetrievalMatch]:
    """Deterministically locate the clause that defines a quoted term.

    Definition questions (“X是如何定义的/如何界定”) are answered by the
    clause carrying “X是指…/所称X…”.  Such a clause is routinely recalled by
    hybrid retrieval yet loses the prompt seat to numerically stronger
    neighbours (real case: “最低资本” definition at RRF rank 2 never reached
    the prompt).  This channel scans the already-recalled documents first
    (cheap), then falls back to the full corpus (bounded by the quoted term),
    and promotes every defining clause with a high deterministic score so
    prompt selection cannot drop it.
    """

    if not hasattr(db, "scalars"):
        return []
    question = str(aspect.question or "")
    if not any(marker in question for marker in _DEFINITION_QUESTION_MARKERS):
        return []
    quoted = [
        re.sub(r"\s+", "", term)
        for term in re.findall(r"[“‘'\"]([^”’'\"]+)[”’'\"]", question)
        if term
    ]
    quoted = [term for term in quoted if 2 <= len(term) <= 24]
    if not quoted:
        return []
    patterns: list[str] = []
    for term in quoted:
        patterns.extend(
            [
                f"{term}是指",
                f"所称{term}",
                f"{term}指",
                f"{term}是由于",
            ]
        )
    patterns = list(dict.fromkeys(patterns))
    scope_document_ids = {
        str(match.citation.document_id or "")
        for match in matches
        if match.citation.document_id
    }
    if not scope_document_ids:
        return []
    found = _definition_clause_chunks(db, patterns, scope_document_ids, document_chunk_cache)
    if not found:
        found = _definition_clause_chunks(db, patterns, None, document_chunk_cache)
    if not found:
        return []
    existing_ids = {match.citation.chunk_id for match in matches}
    supplements: list[RetrievalMatch] = []
    for chunk, matched_pattern in found:
        if chunk.chunk_id in existing_ids or chunk.index_status != "indexed":
            continue
        document_chunks = (
            document_chunk_cache or {}
        ).get(str(chunk.document_id)) or _chunks_by_document(
            db,
            {str(chunk.document_id)},
            document_chunk_cache=document_chunk_cache,
        ).get(str(chunk.document_id))
        if not document_chunks:
            document_chunks = [chunk]
        elif chunk not in document_chunks:
            # Full-corpus fallback chunks are not part of the cached list;
            # keep the chunk itself so neighbour lookup can anchor on it.
            document_chunks = [chunk, *document_chunks]
        match = _direct_exact_support_match(
            chunk,
            document_chunks,
            aspect,
            matched_pattern,
            support_score=1.5,
        )
        match.citation.evidence_role = "definition_clause"
        match.evidence_role = "definition_clause"
        match.metadata["evidence_role"] = "definition_clause"
        match.metadata["fusion_method"] = "definition_clause_lexical"
        match.metadata["definition_clause_pattern"] = matched_pattern
        match.metadata["definition_clause_term"] = matched_pattern.replace("是指", "").replace("所称", "").replace("是指", "").replace("指", "").replace("是由于", "")
        supplements.append(match)
        existing_ids.add(chunk.chunk_id)
        # A PDF/Word clause can be clipped exactly at the chunk boundary
        # ("…而导致 29" where the next chunk continues "直接或间接损失的
        # 风险…").  Promote the immediate successor so the definition
        # sentence is complete for the generator.
        next_chunk_id = getattr(chunk, "next_chunk_id", None)
        if next_chunk_id:
            successor = next(
                (item for item in document_chunks if item.chunk_id == next_chunk_id),
                None,
            )
            if successor is not None and successor.index_status == "indexed" and successor.chunk_id not in existing_ids:
                successor_match = _direct_exact_support_match(
                    successor,
                    document_chunks,
                    aspect,
                    matched_pattern,
                    support_score=1.4,
                )
                successor_match.citation.evidence_role = "definition_clause"
                successor_match.evidence_role = "definition_clause"
                successor_match.metadata["evidence_role"] = "definition_clause"
                successor_match.metadata["fusion_method"] = "definition_clause_lexical"
                successor_match.metadata["definition_clause_pattern"] = matched_pattern
                successor_match.metadata["definition_clause_successor"] = True
                supplements.append(successor_match)
                existing_ids.add(successor.chunk_id)
    return supplements


def _definition_clause_chunks(
    db: Session,
    patterns: list[str],
    document_ids: set[str] | None,
    document_chunk_cache: dict[str, list[DocumentChunk]] | None = None,
) -> list[tuple[Any, str]]:
    """Return (chunk, matched pattern) for chunks containing a definition pattern."""

    if document_ids:
        chunks_by_document = _chunks_by_document(
            db,
            document_ids,
            document_chunk_cache=document_chunk_cache,
        )
        found: list[tuple[Any, str]] = []
        for chunk in (c for chunks in chunks_by_document.values() for c in chunks):
            searchable = str(chunk.text or chunk.embedding_text or "")
            normalized_searchable = _normalize_exact_support_text(searchable)
            for pattern in patterns:
                if _normalize_exact_support_text(pattern) in normalized_searchable:
                    found.append((chunk, pattern))
                    break
        return found
    # Full-corpus fallback: the term may live in a document vector retrieval
    # never recalled.  Bound it to the quoted term's definition patterns only.
    import sqlalchemy as _sa

    statement = (
        _sa.select(DocumentChunk)
        .where(
            DocumentChunk.index_status == "indexed",
            or_(
                *(
                    DocumentChunk.text.contains(pattern)
                    for pattern in patterns
                )
            ),
        )
        .limit(24)
    )
    rows = db.execute(statement).scalars().all()
    found = []
    for chunk in rows:
        searchable = _normalize_exact_support_text(str(chunk.text or chunk.embedding_text or ""))
        matched = next(
            (pattern for pattern in patterns if _normalize_exact_support_text(pattern) in searchable),
            None,
        )
        if matched:
            found.append((chunk, matched))
    return found


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


# When the same chunk carries several evidence roles (a bounded lexical hit
# plus an MCQ exact-support hit, for example), the highest-value role must
# survive deduplication: MCQ exact support is what prompt selection keys on.
_ROLE_PRIORITY = {
    "mcq_exact_support": 6,
    "list_completeness_support": 5,
    "intent_gap_support": 4,
    "definition_clause": 4,
    "mcq_rule_preamble": 3,
    "exact_anchor_support": 2,
    "bounded_lexical_support": 1,
    "direct_evidence": 0,
}


def _to_retrieval_results(matches: list[RetrievalMatch]) -> list[RetrievalResult]:
    best_by_evidence_id: dict[str, tuple[int, RetrievalMatch]] = {}
    for match in matches:
        citation = match.citation
        evidence_id = str(match.metadata.get("evidence_id") or citation.chunk_id)
        role = str(match.metadata.get("evidence_role") or match.evidence_role or "")
        priority = _ROLE_PRIORITY.get(role, 0)
        previous = best_by_evidence_id.get(evidence_id)
        if previous is None or priority > previous[0]:
            best_by_evidence_id[evidence_id] = (priority, match)
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
        chosen = best_by_evidence_id.get(evidence_id)
        if chosen is None or chosen[1] is not match:
            # A lower-value role of the same chunk appeared first; wait for
            # the highest-value version instead of emitting the weaker one.
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


def _expand_neighbor_matches(
    db: Session,
    question: str,
    matches: list[RetrievalMatch],
    document_chunk_cache: dict[str, list[DocumentChunk]] | None = None,
) -> list[RetrievalMatch]:
    if not matches:
        return []

    # Same chunk can carry several evidence roles; keep the highest-value
    # role (mcq_exact_support over bounded_lexical_support, etc.) before
    # expanding neighbours, or the weaker first version would shadow the
    # stronger one for prompt selection.
    best_by_chunk_id: dict[str, RetrievalMatch] = {}
    order: list[str] = []
    for match in matches:
        chunk_id = match.citation.chunk_id
        role = str(match.metadata.get("evidence_role") or match.evidence_role or "")
        previous = best_by_chunk_id.get(chunk_id)
        if previous is None:
            best_by_chunk_id[chunk_id] = match
            order.append(chunk_id)
        elif _ROLE_PRIORITY.get(role, 0) > _ROLE_PRIORITY.get(
            str(previous.metadata.get("evidence_role") or previous.evidence_role or ""), 0
        ):
            best_by_chunk_id[chunk_id] = match
    matches = [best_by_chunk_id[chunk_id] for chunk_id in order]

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


def _score_range_for_matches(matches: list[RetrievalMatch]) -> dict[str, float | None]:
    scores = [float(match.score) for match in matches]
    rerank_scores = [float(match.rerank_score) for match in matches]
    return {
        "vector_min": round(min(scores), 4) if scores else None,
        "vector_max": round(max(scores), 4) if scores else None,
        "rerank_min": round(min(rerank_scores), 4) if rerank_scores else None,
        "rerank_max": round(max(rerank_scores), 4) if rerank_scores else None,
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
