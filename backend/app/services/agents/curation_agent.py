"""Context selection, complete source units, EvidenceBundle assembly and prompt construction.

The code below is moved verbatim from ``rag_service``; only task dispatch and
trace summaries are added.
"""
from __future__ import annotations

from time import perf_counter
from typing import Any

from backend.app.services.agents.base import SpecialistAgent
from backend.app.services.agents.contracts import AgentResult, Blackboard


class CurationAgent(SpecialistAgent):
    name = "curation"
    role = "证据整编"
    tasks = ('curate', 'prompt')
    capabilities = frozenset({'context_selection', 'complete_source_units', 'evidence_bundle', 'prompt_build'})

    def run(self, board: Blackboard, task: str, **inputs: Any) -> AgentResult:
        if task == "curate":
            return self._curate(board, inputs["query_plan"], inputs["aspect_retrievals"])
        if task == "prompt":
            return self._prompt(board, inputs["question"], inputs["context_chunks"])
        raise NotImplementedError(f"{self.name}.{task} is not implemented yet")

    def _curate(self, board: Blackboard, query_plan: Any, aspect_retrievals: list[Any]) -> AgentResult:
        from backend.app.services import rag_service
        from backend.app.services.evidence_sufficiency_service import (
            audit_evidence_sufficiency,
        )

        question = query_plan.original_question
        citation_validation = rag_service._merge_citation_validation(
            [item.citation_validation for item in aspect_retrievals]
        )
        context_started_at = perf_counter()
        rag_service._report_progress(
            board.reporter,
            {
                "stage": "context_selection",
                "status": "running",
                "title": "正在精选最终上下文",
                "detail": "正在按问题方面选择可引用依据……",
            },
        )
        context_chunks, prompt_selection = rag_service._select_prompt_chunks(
            question, query_plan, aspect_retrievals
        )
        context_chunks = rag_service._retrieval_support_structural_context(
            board.db,
            context_chunks,
            question=question,
            complete_table_rows=True,
        )
        context_chunks = rag_service._reserve_definition_clauses(
            board.db, context_chunks, query_plan, aspect_retrievals
        )
        context_chunks = rag_service._reserve_target_clauses(
            board.db, context_chunks, query_plan, aspect_retrievals
        )
        rag_service._bind_cross_aspect_evidence(context_chunks, query_plan)
        from backend.app.services.evidence_contract_service import build_evidence_bundle
        bundle = build_evidence_bundle(list(query_plan.requirements), context_chunks,
                                      list({c.chunk_id: c for c in
                                            [c for item in aspect_retrievals for c in item.candidates] + context_chunks}.values()))
        for item in aspect_retrievals:
            for diagnostic in item.diagnostics:
                bundle.decisions.extend(rag_service.EvidenceDecision.model_validate(d) for d in diagnostic.get('candidate_decisions', []))
        rag_service.event("context", chunks=context_chunks, query_plan=query_plan.to_debug_dict())
        rag_service._sync_aspect_coverage_from_prompt(context_chunks, aspect_retrievals, bundle=bundle)
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
        context_elapsed_ms = rag_service._elapsed_ms(context_started_at)
        prompt_covered_aspect_count = sum(1 for item in aspect_retrievals if item.covered)
        retrieval_covered_aspect_count = sum(1 for item in aspect_retrievals if item.retrieval_covered)
        rag_service._report_progress(
            board.reporter,
            {
                "stage": "context_selection",
                "status": "completed",
                "title": "上下文精选完成",
                "detail": (
                    f"最终使用 {len(context_chunks)}/{rag_service.MAX_PROMPT_CHUNKS} 个片段，"
                    f"检索覆盖 {retrieval_covered_aspect_count}/{len(query_plan.aspects)} 个方面，"
                    f"进入 Prompt {prompt_covered_aspect_count}/{len(query_plan.aspects)} 个方面"
                ),
                "elapsed_ms": context_elapsed_ms,
                "summary": {
                    "used_chunks": len(context_chunks),
                    "max_prompt_chunks": rag_service.MAX_PROMPT_CHUNKS,
                    "covered_aspects": prompt_covered_aspect_count,
                    "retrieval_covered_aspects": retrieval_covered_aspect_count,
                    "total_aspects": len(query_plan.aspects),
                },
            },
        )
        diagnostics = rag_service._aggregate_aspect_diagnostics(aspect_retrievals)
        retrieval_summary = rag_service._build_retrieval_summary(
            question,
            context_chunks,
            query_plan,
            aspect_retrievals,
            diagnostics,
            citation_validation,
            prompt_selection,
        )
        retrieval_summary["evidence_bundle"] = bundle.model_dump(exclude={'evidence', 'diagnostics'})
        retrieval_summary["has_sufficient_context"] = bool(context_chunks) and bundle.complete
        retrieval_summary["missing_requirements"] = bundle.missing_requirements
        retrieval_summary["missing_optional_requirements"] = bundle.missing_optional_requirements
        sufficiency = audit_evidence_sufficiency(
            requirements=query_plan.requirements,
            context_chunks=context_chunks,
            coverage=bundle.coverage,
            missing_requirements=bundle.missing_requirements,
            aspect_retrievals=aspect_retrievals,
        )
        retrieval_summary["evidence_sufficiency"] = sufficiency
        bundle.diagnostics = retrieval_summary
        return AgentResult(
            output=bundle,
            summary={
                "selected_chunks": len(context_chunks),
                "complete": bool(bundle.complete),
                "missing_requirements": list(bundle.missing_requirements),
                "missing_optional_requirements": list(bundle.missing_optional_requirements),
                "sufficiency_status": sufficiency["status"],
                "missing_facets": len(sufficiency["missing_facets"]),
            },
        )

    def _prompt(self, board: Blackboard, question: str, context_chunks: list[Any]) -> AgentResult:
        from backend.app.services import rag_service

        prompt_started_at = perf_counter()
        rag_service._report_progress(
            board.reporter,
            {
                "stage": "prompt_build",
                "status": "running",
                "title": "正在构造 LLM Prompt",
                "detail": "正在将问题和依据片段组装为后续 LLM 输入……",
            },
        )
        prompt = rag_service.RAGPromptBuilder().build(question, context_chunks)
        prompt_elapsed_ms = rag_service._elapsed_ms(prompt_started_at)
        rag_service._report_progress(
            board.reporter,
            {
                "stage": "prompt_build",
                "status": "completed",
                "title": "Prompt 构造完成",
                "detail": f"已完成 Prompt 构造，用时 {rag_service._format_elapsed_seconds(prompt_elapsed_ms)}",
                "elapsed_ms": prompt_elapsed_ms,
                "summary": {"prompt_chunk_count": len(context_chunks)},
            },
        )
        return AgentResult(output=prompt, summary={"prompt_chunk_count": len(context_chunks)})
