"""Supervisor dispatch for the trusted QA mainline.

The supervisor owns routing and response assembly; retrieval, generation and
verification stay in their existing deterministic services and are reached
through the same module-attribute seams the mainline and its tests use.  The
supervisor itself never calls a model.
"""
from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import TYPE_CHECKING, Any, Iterator

from backend.app.services.agents.contracts import AgentResult, Blackboard
from backend.app.services.agents.registry import AgentRegistry, default_registry

if TYPE_CHECKING:
    from backend.app.schemas.qa import EvidenceBundle, LLMContextPackage, QAResponse
    from backend.app.services.retrieval_support import AspectRetrieval


def _prep_terminal_reason(result: Any) -> str | None:
    if not result.terminal:
        return None
    for diagnostic in reversed(result.diagnostics):
        status = diagnostic.get("match_status")
        if status:
            return status
    return None


class Supervisor:
    name = "supervisor"
    role = "调度"

    def __init__(self, registry: AgentRegistry | None = None):
        self.registry = registry if registry is not None else default_registry()

    @contextmanager
    def _active(self, board: Blackboard) -> Iterator[None]:
        previous = board.active_agent
        board.active_agent = self.name
        try:
            yield
        finally:
            board.active_agent = previous

    def gather_aspects(self, board: Blackboard, query_plan: Any) -> list[AspectRetrieval]:
        from backend.app.services import rag_service, retrieval_support
        from backend.app.services.retrieval_support import (
            _AspectPrepResult,
            _DEFAULT_PREPARE_ASPECT_RETRIEVAL,
            _TableStageResult,
            legacy_retrieval_override_active,
        )

        with self._active(board):
            retrieval_started_at = perf_counter()
            rag_service._report_progress(
                board.reporter,
                {
                    "stage": "retrieval",
                    "status": "running",
                    "title": "正在检索相关依据",
                    "detail": f"正在按 {len(query_plan.aspects)} 个问题方面召回知识库片段……",
                    "summary": {"total_aspects": len(query_plan.aspects)},
                },
            )
            aspect_retrievals: list[Any] = []

            # Phase A: deterministic preparation (spreadsheet, filters, MCQ
            # early stop, query normalization) for every aspect, so the hybrid
            # collection can run once across all of them.
            prep_by_aspect: dict[str, Any] = {}
            aspect_routes: list[dict[str, Any]] = []
            for aspect_index, aspect in enumerate(query_plan.aspects, start=1):
                rag_service._report_progress(
                    board.reporter,
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
                composite = None
                if rag_service._prepare_aspect_retrieval is not _DEFAULT_PREPARE_ASPECT_RETRIEVAL:
                    composite = rag_service._prepare_aspect_retrieval
                elif retrieval_support._prepare_aspect_retrieval is not _DEFAULT_PREPARE_ASPECT_RETRIEVAL:
                    composite = retrieval_support._prepare_aspect_retrieval
                if composite is not None:
                    # Compatibility seam for callers that replace the composite
                    # preparation entry on either module (batching tests pin it).
                    prep = composite(
                        board.db,
                        aspect,
                        progress_reporter=board.reporter,
                        document_chunk_cache=board.document_chunk_cache,
                    )
                    route_agent = "evidence"
                    terminal_reason = _prep_terminal_reason(prep)
                elif legacy_retrieval_override_active():
                    legacy = self.registry.get("evidence").execute(
                        board,
                        "legacy",
                        aspect=aspect,
                        aspect_ids=[aspect.aspect_id],
                    )
                    matches, diagnostics = legacy.output
                    prep = _AspectPrepResult(True, matches, diagnostics, None)
                    route_agent = "evidence"
                    terminal_reason = "legacy_retrieval_override"
                else:
                    if aspect.modality in {"table", "mixed"}:
                        located = self.registry.get("table").execute(
                            board,
                            "locate",
                            aspect=aspect,
                            aspect_ids=[aspect.aspect_id],
                        )
                        table_stage = located.output
                    else:
                        table_stage = _TableStageResult([], [], False, None)
                    if table_stage.terminal:
                        prep = _AspectPrepResult(
                            True,
                            table_stage.located_matches,
                            table_stage.table_diagnostics,
                            None,
                        )
                        route_agent = "table"
                        terminal_reason = table_stage.terminal_reason
                    else:
                        prepared = self.registry.get("evidence").execute(
                            board,
                            "prepare",
                            aspect=aspect,
                            table_stage=table_stage,
                            aspect_ids=[aspect.aspect_id],
                        )
                        prep = prepared.output
                        route_agent = "evidence"
                        terminal_reason = _prep_terminal_reason(prep)
                prep_by_aspect[aspect.aspect_id] = prep
                aspect_routes.append(
                    {
                        "aspect_id": aspect.aspect_id,
                        "agent": route_agent,
                        "terminal": bool(prep.terminal),
                        "terminal_reason": terminal_reason,
                    }
                )
            board.notes["aspect_routes"] = aspect_routes

            # Phase B + C: collect hybrid candidates (one batched embedding +
            # Qdrant pass and one batched rerank pass when several aspects need
            # vectors; otherwise the single-aspect path) and finish each aspect.
            prepared_items = [
                (aspect, prep_by_aspect[aspect.aspect_id].prepared)
                for aspect in query_plan.aspects
                if not prep_by_aspect[aspect.aspect_id].terminal
            ]
            finished: dict[str, tuple[list[Any], list[dict[str, Any]]]] = {}
            if prepared_items:
                collected = self.registry.get("evidence").execute(
                    board,
                    "collect",
                    prepared_items=prepared_items,
                    aspect_ids=[aspect.aspect_id for aspect, _prepared in prepared_items],
                )
                finished = collected.output

            for aspect_index, aspect in enumerate(query_plan.aspects, start=1):
                prep = prep_by_aspect[aspect.aspect_id]
                if prep.terminal:
                    matches, diagnostics = prep.matches, prep.diagnostics
                else:
                    matches, diagnostics = finished[aspect.aspect_id]
                expanded = self.registry.get("evidence").execute(
                    board,
                    "expand",
                    aspect=aspect,
                    matches=matches,
                    aspect_ids=[aspect.aspect_id],
                )
                candidates = expanded.output
                traced = self.registry.get("verifier").execute(
                    board,
                    "trace_evidence",
                    candidates=candidates,
                    aspect_ids=[aspect.aspect_id],
                )
                valid_candidates, citation_validation = traced.output
                aspect_retrieval = rag_service.AspectRetrieval(
                    aspect=aspect,
                    candidates=valid_candidates,
                    diagnostics=diagnostics,
                    citation_validation=citation_validation,
                    selected_chunk_ids=[],
                    retrieval_covered=bool(valid_candidates),
                    covered=False,
                )
                aspect_retrievals.append(aspect_retrieval)
                rag_service._report_progress(
                    board.reporter,
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
            retrieval_elapsed_ms = rag_service._elapsed_ms(retrieval_started_at)
            retrieved_aspect_count = sum(1 for item in aspect_retrievals if item.retrieval_covered)
            candidate_count = sum(len(item.candidates) for item in aspect_retrievals)
            rag_service._report_progress(
                board.reporter,
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
            rag_service._report_rerank_stage_summary(board.reporter, aspect_retrievals)
            board.aspect_retrievals = aspect_retrievals
            return aspect_retrievals

    def gather_evidence(self, board: Blackboard, query_plan: Any) -> EvidenceBundle:
        from backend.app.services import rag_service

        with self._active(board):
            aspect_retrievals = rag_service._retrieve_aspects(
                board.db,
                query_plan,
                progress_reporter=board.reporter,
            )
            board.aspect_retrievals = aspect_retrievals
            curated = self.registry.get("curation").execute(
                board,
                "curate",
                query_plan=query_plan,
                aspect_retrievals=aspect_retrievals,
                aspect_ids=[aspect.aspect_id for aspect in query_plan.aspects],
            )
            bundle = curated.output
            board.evidence_bundle = bundle
            return bundle

    def build_context(self, board: Blackboard, question: str, options: list[str] | None) -> LLMContextPackage:
        from backend.app.schemas.qa import LLMContextPackage
        from backend.app.services import rag_service

        with self._active(board):
            planned = self.registry.get("planner").execute(
                board,
                "plan",
                question=question,
                options=options,
            )
            query_plan = planned.output
            if query_plan.clarification_reason:
                package = LLMContextPackage(query=question, instruction=rag_service.CONTEXT_INSTRUCTION,
                    retrieval_summary={'has_sufficient_context': False, 'used_chunks': 0,
                                       'query_plan': query_plan.to_debug_dict(),
                                       'clarification_reason': query_plan.clarification_reason},
                    context_chunks=[], llm_prompt='')
                board.context_package = package
                return package
            bundle = rag_service.retrieve_evidence(
                board.db,
                query_plan,
                progress_reporter=board.reporter,
            )
            injected = self.registry.get("curation").execute(
                board,
                "prompt",
                question=question,
                context_chunks=bundle.evidence,
            )
            package = LLMContextPackage(
                query=question,
                instruction=rag_service.CONTEXT_INSTRUCTION,
                retrieval_summary=bundle.diagnostics,
                context_chunks=bundle.evidence,
                llm_prompt=injected.output,
            )
            board.context_package = package
            return package

    def run(self, board: Blackboard) -> QAResponse:
        """Execute the unchanged QA mainline under one supervisor trace."""
        from backend.app.schemas.qa import (
            AnswerClaim,
            EvidenceRequirement,
            QAAnswerPreview,
            QAResponse,
        )
        from backend.app.services import rag_service
        from backend.app.services.answer_generation_service import (
            GeneratedAnswer,
            refusal_answer,
            run_generation_loop,
        )
        from backend.app.services.performance_metrics import model_call_count

        with self._active(board):
            raw_options = board.notes.get("raw_options")
            routed = self.registry.get("triage").execute(board, "route", raw_options=raw_options)
            route = routed.output["route"]

            if route == "identity":
                response = self.registry.get("identity").execute(board, "answer").output
                self._record_trace(board, response)
                return response

            if route == "boundary":
                boundary_code = board.boundary_code or routed.output.get("boundary_code")
                response = QAResponse(
                    answer="该请求超出监管知识库可验证回答的范围，系统需要补充可核验依据或改为资料库可支持的问题。",
                    citations=[],
                    confidence=0.0,
                    refused=True,
                    context_package=None,
                    answer_type=(
                        "clarification"
                        if boundary_code in {"missing_required_context", "missing_calculation_operands"}
                        else "refusal"
                    ),
                    generation_status="skipped",
                    claims=[],
                    grounding_validation={
                        "passed": False,
                        "status": "not_run",
                        "task_complete": False,
                        "reason": "explicit_request_boundary",
                    },
                    refusal_reason="explicit_request_boundary",
                    refusal_code=boundary_code,
                    degraded=False,
                )
                rag_service._save_qa_log(board.db, board.original_question, response)
                rag_service._log_qa_audit(
                    board.db, "qa_refused_explicit_boundary", board.original_question, response
                )
                self._record_trace(board, response)
                return response

            package = rag_service.build_context_package(
                board.db,
                board.question,
                options=board.options,
                progress_reporter=board.reporter,
            )
            board.context_package = package
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
                if board.answer_preview_reporter is None:
                    return
                preview_revision += 1
                cited_labels = {
                    citation_id
                    for item in preview_claims
                    for citation_id in item.citation_ids
                }
                board.answer_preview_reporter(QAAnswerPreview(
                    answer="\n\n".join(item.text for item in preview_claims),
                    citations=[
                        rag_service._citation_from_result(result)
                        for result in package.context_chunks
                        if result.citation_label in cited_labels
                    ],
                    verified_claim_count=len(preview_claims),
                    revision=preview_revision,
                ))

            rag_service._report_progress(
                board.reporter,
                {
                    "stage": "llm_generation",
                    "status": "running",
                    "title": "正在生成可信回答",
                    "detail": "正在基于已选证据生成结构化答案……",
                },
            )

            def resolver(kind: str, **kwargs: Any) -> Any:
                return self.registry.get("table").execute(
                    board, "deterministic", kind=kind, **kwargs
                ).output

            if package.retrieval_summary.get('clarification_reason'):
                # The planner asked for clarification, so the context package is
                # the empty stub.  Route straight to the response: this is the
                # answer agent's first gate (no context, insufficient context)
                # reproduced verbatim, without engaging generation.
                board.notes["generation_route"] = "clarification"
                prepared = AgentResult(output=refusal_answer(
                    'insufficient_context',
                    validation={'missing_requirements': rag_service._required_aspect_ids(package.retrieval_summary) or []},
                ), status="clarification")
            else:
                board.notes["generation_route"] = "answer_agent"
                prepared = self.registry.get("answer").execute(
                    board,
                    "prepare",
                    question=board.question,
                    context_chunks=package.context_chunks,
                    options=board.options,
                    option_labels=board.option_labels,
                    llm_prompt=package.llm_prompt,
                    has_sufficient_context=bool(package.retrieval_summary["has_sufficient_context"]),
                    answer_mode=rag_service._answer_mode(board.question, package.retrieval_summary),
                    required_aspect_ids=rag_service._required_aspect_ids(package.retrieval_summary),
                    requirements=[
                        EvidenceRequirement.model_validate(item)
                        for item in package.retrieval_summary.get('evidence_bundle', {}).get('requirements', [])
                    ] or None,
                    table_resolver=resolver,
                )
            if isinstance(prepared.output, GeneratedAnswer):
                generated = prepared.output
            else:
                ctx = prepared.output
                answer_agent = self.registry.get("answer")
                verifier = self.registry.get("verifier")

                def draft(ctx: Any, *, attempt: int, correction: str | None) -> GeneratedAnswer:
                    return answer_agent.execute(
                        board,
                        "draft",
                        attempt=attempt + 1,
                        ctx=ctx,
                        draft_attempt=attempt,
                        correction=correction,
                    ).output

                def verify(ctx: Any, generated: GeneratedAnswer) -> dict[str, Any]:
                    return verifier.execute(
                        board,
                        "verify",
                        attempt=len(board.validations) + 1,
                        ctx=ctx,
                        generated=generated,
                    ).output

                def judge(
                    ctx: Any,
                    generated: GeneratedAnswer,
                    validation: dict[str, Any],
                    attempt: int,
                ) -> Any:
                    return verifier.execute(
                        board,
                        "judge",
                        attempt=attempt + 1,
                        ctx=ctx,
                        generated=generated,
                        validation=validation,
                    ).output

                generated = run_generation_loop(ctx, draft=draft, verify=verify, judge=judge)
            # ``generate_answer`` is wrapped by ``qa_execution``, which stamps
            # the spent model budget onto the generated answer before the
            # response is assembled.  Reproduce that here because the loop is
            # now driven through agent primitives.
            generated.grounding_validation["model_call_count"] = model_call_count()

            if package.retrieval_summary.get('clarification_reason'):
                generated.answer_type = 'clarification'
                generated.answer = package.retrieval_summary['clarification_reason']
                generated.refusal_code = 'missing_required_context'
                generated.grounding_validation['failure_stage'] = 'planning'
            if generated.refused:
                missing = generated.grounding_validation.get('missing_requirements') or []
                requirements = (package.retrieval_summary.get('evidence_bundle') or {}).get('requirements', [])
                descriptions = {r['requirement_id']: ('《' + '》、《'.join(r['source_titles']) + '》的直接依据'
                                if r.get('source_titles') else r['question']) for r in requirements}
                labels = [descriptions.get(item, item) for item in missing]
                generated.grounding_validation['missing_required_aspect_ids'] = missing
                generated.grounding_validation['missing_requirements'] = labels
                if labels and generated.answer_type != 'service_error':
                    generated.answer = '当前证据不足以完整回答。缺少：' + '；'.join(labels)
            from backend.app.services.answer_validation_service import validation_passed
            if not generated.refused and not validation_passed(generated.grounding_validation):
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
                    "passed": False,
                    "reason": "table_evidence_not_found",
                    "table_refusal_reasons": table_refusal_reasons,
                }
            if generated.refused and not generated.refusal_code:
                # Keep refusal outcomes machine-readable even when the answer
                # model declines before it can emit its own structured code.
                generated.refusal_code = rag_service._refusal_code_for_request(board.question)
            if not generated.refused and validation_passed(generated.grounding_validation):
                for claim in generated.claims:
                    report_verified_claim(claim)
            rag_service._report_progress(
                board.reporter,
                {
                    "stage": "llm_generation",
                    "status": "completed" if generated.generation_status == "completed" else "skipped",
                    "title": "答案生成完成" if generated.generation_status == "completed" else "答案生成已跳过",
                    "detail": rag_service._generation_progress_detail(
                        generated.generation_status, generated.answer_type
                    ),
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
            grounding_passed = validation_passed(generated.grounding_validation)
            rag_service._report_progress(
                board.reporter,
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
            cited_results = rag_service._cited_context_results(
                package.context_chunks,
                answer=generated.answer,
                claims=generated.claims,
            )
            response = QAResponse(
                answer=generated.answer,
                citations=[rag_service._citation_from_result(result) for result in cited_results],
                confidence=rag_service._confidence_from_results(cited_results),
                refused=generated.refused,
                context_package=package if board.include_debug else None,
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
                evidence_coverage=rag_service._build_evidence_coverage(
                    package.retrieval_summary,
                    generated.claims,
                    cited_results,
                ),
            )
            from backend.app.services.answer_explanation_service import build_answer_explanation
            response.explanation = build_answer_explanation(
                question=board.original_question,
                generated=generated,
                package=package,
                cited_results=cited_results,
                progress_events=board.notes.get("progress_events") or [],
                started_at=board.notes.get("started_at"),
                first_feedback_ms=board.notes.get("first_feedback_ms"),
            )
            self._record_trace(board, response)
            rag_service._save_qa_log(board.db, board.original_question, response)
            rag_service._log_qa_audit(board.db, "qa_context_built", board.original_question, response)
            return response

    def _record_trace(self, board: Blackboard, response: Any) -> None:
        """Append the supervisor's own route entry and attach the request trace."""
        from backend.app.schemas.qa import AgentTraceEntry

        if not response.refused:
            status = "completed"
        elif response.answer_type == "clarification":
            status = "clarification"
        elif response.answer_type == "service_error":
            status = "failed"
        else:
            status = "refused"
        board.record(AgentTraceEntry(
            agent=self.name,
            role=self.role,
            task="route",
            status=status,
            summary={
                "route": board.route,
                "boundary_code": board.boundary_code,
                "aspect_routes": board.notes.get("aspect_routes", []),
                "generation_attempts": len(board.drafts),
                "validations": len(board.validations),
                "model_calls_total": board.total_model_calls(),
                "answer_type": response.answer_type,
                "refusal_reason": response.refusal_reason,
            },
        ))
        response.agent_trace = list(board.trace)
