"""Batched multi-aspect retrieval path tests.

The single-aspect path keeps its original behaviour; these tests pin the new
batched path: one embedding+Qdrant collection pass, one rerank pass across all
aspects, and per-aspect finishing with the group's reranked results.
"""

from __future__ import annotations

import pytest

from backend.app.schemas.qa import Citation, RetrievalResult
from backend.app.services.query_planner_service import (
    QueryPlan,
    QuerySearchQuery,
)
from backend.app.services import retrieval_support
from backend.app.services.rag_service import (
    QueryAspect,
    _retrieve_aspects,
)
from backend.app.services.rerank_service import RerankedChunk
from backend.app.services.retrieval_service import RetrievalMatch
from backend.app.services.vector_store_service import VectorSearchResult


def _aspect(aspect_id: str, question: str) -> QueryAspect:
    return QueryAspect(
        aspect_id=aspect_id,
        question=question,
        search_queries=(
            QuerySearchQuery(question, "semantic_question", "贴近用户意图"),
            QuerySearchQuery(f"{question} 制度原文", "document_style_statement", "贴近制度原文"),
        ),
        evidence_need=question,
        keywords=(question,),
    )


def _plan(*aspects: QueryAspect) -> QueryPlan:
    return QueryPlan(
        original_question=" ".join(aspect.question for aspect in aspects),
        aspects=tuple(aspects),
        planner="test",
        fallback_used=False,
    )


def _candidate(chunk_index: int, aspect_id: str) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=f"DOC-TEST-CHUNK-{aspect_id}-{chunk_index:04d}",
        document_id="DOC-TEST-0001",
        filename="rules.md",
        section_title=None,
        page_number=None,
        text=f"方面 {aspect_id} 的候选证据 {chunk_index}",
        embedding_text=f"方面 {aspect_id} 的候选证据 {chunk_index}",
        token_count=20,
        score=1.0 - chunk_index * 0.01,
        chunk_type="paragraph",
        metadata={"aspect_id": aspect_id},
    )


def _match(aspect_id: str, chunk_index: int) -> RetrievalMatch:
    candidate = _candidate(chunk_index, aspect_id)
    return RetrievalMatch(
        citation=Citation(
            document_id=candidate.document_id,
            chunk_id=candidate.chunk_id,
            filename=candidate.filename,
            excerpt=candidate.text,
            score=candidate.score,
            rerank_score=0.9,
        ),
        score=candidate.score,
        rerank_score=0.9,
        evidence_role="direct_evidence",
        evidence_text=candidate.text,
        metadata={"aspect_id": aspect_id},
    )


def _prepared_result(aspect: QueryAspect):
    from backend.app.services.retrieval_support import _PreparedAspect, _AspectPrepResult
    from backend.app.services.retrieval_service import RetrievalDiagnostics

    class _FilterStub:
        explicit: dict = {}
        inferred: dict = {}
        document_ids: list[str] = []

        @staticmethod
        def qdrant_filter() -> dict:
            return {}

        @staticmethod
        def to_debug_dict() -> dict:
            return {}

    return _AspectPrepResult(
        terminal=False,
        matches=[],
        diagnostics=[],
        prepared=_PreparedAspect(
            aspect=aspect,
            search_queries=[query.query for query in aspect.search_queries],
            query_metadata=[
                {"query_type": query.query_type, "rationale": query.rationale}
                for query in aspect.search_queries
            ],
            qdrant_filter=None,
            table_matches=[],
            table_diagnostics=[],
            retrieval_filter=_FilterStub(),
            diagnostics=RetrievalDiagnostics(),
            total_started_at=0.0,
            mcq_material_document_ids=set(),
        ),
    )


def test_retrieve_aspects_batches_collection_and_rerank_across_aspects(monkeypatch) -> None:
    from backend.app.services import rag_service

    aspect_a = _aspect("aspect_a", "方面甲如何处理")
    aspect_b = _aspect("aspect_b", "方面乙如何处理")
    query_plan = _plan(aspect_a, aspect_b)

    collected_results = [
        (
            [_candidate(0, "aspect_a"), _candidate(1, "aspect_a")],
            {},
            [{"search_query": q.query} for q in aspect_a.search_queries],
        ),
        (
            [_candidate(0, "aspect_b"), _candidate(1, "aspect_b")],
            {},
            [{"search_query": q.query} for q in aspect_b.search_queries],
        ),
    ]
    reranked_groups = [
        [RerankedChunk(candidate=_candidate(0, "aspect_a"), rerank_score=0.9)],
        [RerankedChunk(candidate=_candidate(0, "aspect_b"), rerank_score=0.95)],
    ]
    finished_calls: list[str] = []

    monkeypatch.setattr(
        rag_service,
        "_prepare_aspect_retrieval",
        lambda db, aspect, progress_reporter=None, document_chunk_cache=None: (
            _prepared_result(aspect_a) if aspect.aspect_id == "aspect_a" else _prepared_result(aspect_b)
        ),
    )
    collect_calls = []

    def fake_collect(prepared):
        collect_calls.append([item.aspect.aspect_id for item in prepared])
        return collected_results

    monkeypatch.setattr(rag_service, "_collect_prepared_candidates_many", fake_collect)
    rerank_calls = []

    def fake_rerank_groups(*, questions, candidate_groups, limits):
        rerank_calls.append((questions, [len(group) for group in candidate_groups], limits))
        return reranked_groups

    monkeypatch.setattr(rag_service, "rerank_candidate_groups", fake_rerank_groups)

    def fake_finish(db, aspect, **kwargs):
        finished_calls.append(aspect.aspect_id)
        assert kwargs["reranked"] is not None
        assert kwargs["rerank_input"] is not None
        return [_match(aspect.aspect_id, 0)], []

    monkeypatch.setattr(rag_service, "_finish_aspect_matches", fake_finish)
    monkeypatch.setattr(
        rag_service, "_expand_neighbor_matches",
        lambda db, question, matches, document_chunk_cache=None: matches,
    )
    monkeypatch.setattr(
        rag_service, "_validate_context_chunks",
        lambda db, candidates: (
            [RetrievalResult(**item.model_dump(exclude={"metadata"}), metadata=item.metadata) for item in candidates],
            {"valid": True},
        ),
    )

    retrievals = _retrieve_aspects(None, query_plan)

    assert collect_calls == [["aspect_a", "aspect_b"]]
    assert len(rerank_calls) == 1
    questions, group_sizes, limits = rerank_calls[0]
    from backend.app.services.rag_service import _rerank_query_for_aspect

    assert questions == [_rerank_query_for_aspect(aspect_a), _rerank_query_for_aspect(aspect_b)]
    assert group_sizes == [2, 2]
    assert finished_calls == ["aspect_a", "aspect_b"]
    assert [item.aspect.aspect_id for item in retrievals] == ["aspect_a", "aspect_b"]
    assert retrievals[0].retrieval_covered is True
    assert retrievals[1].retrieval_covered is True


def test_retrieve_aspects_single_prepared_uses_single_path(monkeypatch) -> None:
    from backend.app.services import rag_service

    aspect = _aspect("aspect_single", "单独方面如何处理")
    query_plan = _plan(aspect)

    monkeypatch.setattr(
        rag_service,
        "_prepare_aspect_retrieval",
        lambda db, aspect, progress_reporter=None, document_chunk_cache=None: _prepared_result(aspect),
    )
    collect_calls = []

    def fake_collect(prepared):
        collect_calls.append(prepared)
        raise AssertionError("batched collector must not run for a single prepared aspect")

    monkeypatch.setattr(rag_service, "_collect_prepared_candidates_many", fake_collect)
    monkeypatch.setattr(rag_service, "rerank_candidate_groups", lambda **kwargs: (_ for _ in ()).throw(AssertionError("group rerank must not run for a single aspect")))
    finished_calls = []

    def fake_finish(db, aspect, **kwargs):
        finished_calls.append(aspect.aspect_id)
        assert kwargs.get("reranked") is None
        return [_match(aspect.aspect_id, 0)], []

    monkeypatch.setattr(rag_service, "_finish_aspect_matches", fake_finish)
    monkeypatch.setattr(
        rag_service, "_expand_neighbor_matches",
        lambda db, question, matches, document_chunk_cache=None: matches,
    )
    monkeypatch.setattr(
        rag_service, "_validate_context_chunks",
        lambda db, candidates: (
            [RetrievalResult(**item.model_dump(exclude={"metadata"}), metadata=item.metadata) for item in candidates],
            {"valid": True},
        ),
    )

    retrievals = _retrieve_aspects(None, query_plan)

    assert finished_calls == ["aspect_single"]
    assert retrievals[0].aspect.aspect_id == "aspect_single"
    assert not collect_calls


def test_retrieve_aspects_terminal_prepared_aspect_skips_collection(monkeypatch) -> None:
    from backend.app.services import rag_service

    aspect = _aspect("aspect_terminal", "直接终结的方面")
    query_plan = _plan(aspect)

    terminal = rag_service._AspectPrepResult(
        terminal=True,
        matches=[_match("aspect_terminal", 0)],
        diagnostics=[],
        prepared=None,
    )
    monkeypatch.setattr(
        rag_service,
        "_prepare_aspect_retrieval",
        lambda db, aspect, progress_reporter=None, document_chunk_cache=None: terminal,
    )
    monkeypatch.setattr(
        rag_service,
        "_collect_prepared_candidates_many",
        lambda prepared: (_ for _ in ()).throw(AssertionError("terminal aspect must not collect")),
    )
    monkeypatch.setattr(
        rag_service, "rerank_candidate_groups",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("terminal aspect must not rerank")),
    )
    monkeypatch.setattr(
        rag_service, "_expand_neighbor_matches",
        lambda db, question, matches, document_chunk_cache=None: matches,
    )
    monkeypatch.setattr(
        rag_service, "_validate_context_chunks",
        lambda db, candidates: (
            [RetrievalResult(**item.model_dump(exclude={"metadata"}), metadata=item.metadata) for item in candidates],
            {"valid": True},
        ),
    )

    retrievals = _retrieve_aspects(None, query_plan)

    assert retrievals[0].aspect.aspect_id == "aspect_terminal"
    assert retrievals[0].retrieval_covered is True
