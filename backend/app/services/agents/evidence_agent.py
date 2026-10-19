"""Metadata filtering, option exact support, hybrid retrieval, rerank, bounded supplements and neighbor/reference expansion.

Wraps the deterministic retrieval mainline unchanged; the code below is moved
verbatim from ``rag_service`` and only dispatched per task.
"""
from __future__ import annotations

from typing import Any

from backend.app.services.agents.base import SpecialistAgent
from backend.app.services.agents.contracts import AgentResult, Blackboard


class EvidenceAgent(SpecialistAgent):
    name = "evidence"
    role = "制度检索"
    tasks = ('legacy', 'prepare', 'collect', 'expand')
    capabilities = frozenset({'metadata_filter', 'option_exact_support', 'hybrid_retrieval', 'rerank', 'bounded_supplement', 'neighbor_expansion', 'cross_document_reference'})

    def run(self, board: Blackboard, task: str, **inputs: Any) -> AgentResult:
        if task == "legacy":
            from backend.app.services import retrieval_support

            aspect = inputs["aspect"]
            matches, diagnostics = retrieval_support._retrieve_aspect_matches_legacy_hook(
                board.db, aspect, board.reporter
            )
            return AgentResult(
                output=(matches, diagnostics),
                summary={"matches": len(matches)},
            )
        if task == "prepare":
            from backend.app.services import retrieval_support

            aspect = inputs["aspect"]
            result = retrieval_support._prepare_text_stage(
                board.db,
                aspect,
                inputs["table_stage"],
                progress_reporter=board.reporter,
                document_chunk_cache=board.document_chunk_cache,
            )
            terminal_reason = None
            if result.terminal:
                for diagnostic in reversed(result.diagnostics):
                    status = diagnostic.get("match_status")
                    if status:
                        terminal_reason = status
                        break
            return AgentResult(
                output=result,
                summary={
                    "terminal": result.terminal,
                    "terminal_reason": terminal_reason,
                    "search_queries": len(result.prepared.search_queries) if result.prepared else 0,
                },
            )
        if task == "collect":
            return self._collect(board, inputs["prepared_items"])
        if task == "expand":
            return self._expand(board, inputs["aspect"], inputs["matches"])
        raise NotImplementedError(f"{self.name}.{task} is not implemented yet")

    def _collect(self, board: Blackboard, prepared_items: list[tuple[Any, Any]]) -> AgentResult:
        from backend.app.services import rag_service

        finished: dict[str, tuple[list[Any], list[dict[str, Any]]]] = {}
        if len(prepared_items) > 1:
            try:
                collected = rag_service._collect_prepared_candidates_many(
                    [prepared for _aspect, prepared in prepared_items]
                )
            except (rag_service.EmbeddingServiceError, rag_service.VectorStoreError) as exc:
                raise rag_service.RetrievalServiceUnavailable(str(exc)) from exc
            rerank_questions: list[str] = []
            rerank_inputs: list[list[Any]] = []
            for (aspect, prepared), (candidates, hits, _per_query) in zip(
                prepared_items, collected, strict=True
            ):
                if not rag_service.features().rerank:
                    candidates.sort(key=lambda candidate: sum(
                        rag_service.QUERY_TYPE_WEIGHTS.get(str(hit.get("query_type") or "semantic_question"), 1.0)
                        / (rag_service.RRF_K + int(hit.get("rank") or 1))
                        for hit in hits.get(candidate.chunk_id, [])
                    ), reverse=True)
                effective = rag_service._effective_rerank_candidate_limit(
                    aspect,
                    candidates,
                    retrieval_filter_document_ids=set(prepared.retrieval_filter.document_ids or ()),
                    mcq_material_document_ids=prepared.mcq_material_document_ids,
                )
                rerank_inputs.append(
                    rag_service.rerank_input_with_document_coverage(candidates, limit=effective)
                )
                rerank_questions.append(rag_service._rerank_query_for_aspect(aspect))
            rerank_limits = [
                len(rerank_input)
                if aspect.aspect_id.startswith("multiple_choice_evidence")
                else rag_service.RERANK_TOP_K
                for (aspect, _prepared), rerank_input in zip(
                    prepared_items, rerank_inputs, strict=True
                )
            ]
            try:
                reranked_groups = rag_service.rerank_candidate_groups(
                    questions=rerank_questions,
                    candidate_groups=rerank_inputs,
                    limits=rerank_limits,
                )
            except rag_service.RerankServiceError as exc:
                raise rag_service.RetrievalServiceUnavailable(str(exc)) from exc
            for (aspect, prepared), (candidates, hits, per_query), rerank_input, reranked in zip(
                prepared_items, collected, rerank_inputs, reranked_groups, strict=True
            ):
                matches, diagnostics = rag_service._finish_aspect_matches(
                    board.db,
                    aspect,
                    candidates=candidates,
                    query_hits_by_chunk_id=hits,
                    diagnostics_by_query=per_query,
                    diagnostics=prepared.diagnostics,
                    retrieval_filter=prepared.retrieval_filter,
                    mcq_material_document_ids=prepared.mcq_material_document_ids,
                    located_matches=prepared.located_matches,
                    table_diagnostics=prepared.table_diagnostics,
                    document_chunk_cache=board.document_chunk_cache,
                    total_started_at=prepared.total_started_at,
                    progress_reporter=board.reporter,
                    reranked=reranked,
                    rerank_input=rerank_input,
                )
                finished[aspect.aspect_id] = (matches, diagnostics)
        else:
            for aspect, prepared in prepared_items:
                assert prepared is not None
                try:
                    candidates, hits, per_query = rag_service.collect_candidates_with_query_hits(
                        prepared.search_queries,
                        query_metadata=prepared.query_metadata,
                        diagnostics=prepared.diagnostics,
                        metadata_filter=prepared.qdrant_filter,
                    )
                except (rag_service.EmbeddingServiceError, rag_service.VectorStoreError) as exc:
                    raise rag_service.RetrievalServiceUnavailable(str(exc)) from exc
                matches, diagnostics = rag_service._finish_aspect_matches(
                    board.db,
                    aspect,
                    candidates=candidates,
                    query_hits_by_chunk_id=hits,
                    diagnostics_by_query=per_query,
                    diagnostics=prepared.diagnostics,
                    retrieval_filter=prepared.retrieval_filter,
                    mcq_material_document_ids=prepared.mcq_material_document_ids,
                    located_matches=prepared.located_matches,
                    table_diagnostics=prepared.table_diagnostics,
                    document_chunk_cache=board.document_chunk_cache,
                    total_started_at=prepared.total_started_at,
                    progress_reporter=board.reporter,
                )
                finished[aspect.aspect_id] = (matches, diagnostics)
        rerank_calls = sum(
            int(diagnostic.get("rerank_call_count") or 0)
            for _matches, diagnostics in finished.values()
            for diagnostic in diagnostics
        )
        return AgentResult(
            output=finished,
            summary={
                "aspects": len(prepared_items),
                "batched": len(prepared_items) > 1,
                "rerank_calls": rerank_calls,
            },
        )

    def _expand(self, board: Blackboard, aspect: Any, matches: list[Any]) -> AgentResult:
        from backend.app.services import rag_service

        matches = rag_service._expand_neighbor_matches(
            board.db,
            aspect.question,
            matches,
            document_chunk_cache=board.document_chunk_cache,
        )
        candidates = rag_service._to_retrieval_results(matches)
        for candidate in candidates:
            candidate.metadata["aspect_id"] = aspect.aspect_id
            candidate.metadata["aspect_question"] = aspect.question
            candidate.metadata["aspect_search_queries"] = rag_service._search_query_debug_list(aspect)
            candidate.metadata["expected_evidence_type"] = aspect.expected_evidence_type
            candidate.metadata["evidence_need"] = aspect.evidence_need
            candidate.metadata.setdefault("prompt_matched_aspects", [aspect.aspect_id])
        return AgentResult(output=candidates, summary={"candidates": len(candidates)})
