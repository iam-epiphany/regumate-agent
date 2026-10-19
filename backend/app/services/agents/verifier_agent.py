"""Evidence traceability, unified answer validation and repair/refusal decisions.

The agent wraps the existing deterministic verification primitives without
changing their arguments, ordering or exception semantics.  ``rag_service`` is
imported inside the task that needs it to avoid a module-level import cycle.
"""
from __future__ import annotations

from typing import Any

from backend.app.services import answer_generation_service as generation
from backend.app.services.agents.base import SpecialistAgent
from backend.app.services.agents.contracts import AgentResult, Blackboard


class VerifierAgent(SpecialistAgent):
    name = "verifier"
    role = "核验"
    tasks = ('trace_evidence', 'verify', 'judge')
    capabilities = frozenset({'evidence_traceability', 'citation_integrity', 'fact_support', 'task_completeness', 'repair_decision'})

    def run(self, board: Blackboard, task: str, **inputs: Any) -> AgentResult:
        if task == "trace_evidence":
            return self._trace_evidence(board, **inputs)
        if task == "verify":
            return self._verify(board, **inputs)
        if task == "judge":
            return self._judge(board, **inputs)
        raise ValueError(f"{self.name} has no task {task!r}")

    def _trace_evidence(self, board: Blackboard, **inputs: Any) -> AgentResult:
        from backend.app.services import rag_service

        candidates = inputs.get("candidates") or []
        valid_chunks, citation_validation = rag_service._validate_context_chunks(board.db, candidates)
        return AgentResult(
            output=(valid_chunks, citation_validation),
            status="completed",
            summary={
                "checked": citation_validation.get("checked_chunks", len(candidates)),
                "valid": citation_validation.get("valid_chunks", len(valid_chunks)),
                "invalid": citation_validation.get("invalid_chunks", 0),
            },
        )

    def _verify(self, board: Blackboard, **inputs: Any) -> AgentResult:
        ctx = inputs.get("ctx") or board.generation
        generated = inputs.get("generated")
        if not isinstance(ctx, generation.GenerationContext) or generated is None:
            raise ValueError("verifier.verify requires a GenerationContext and a generated answer")
        validation = generation.verify_draft(ctx, generated)
        board.validations.append(validation)
        return AgentResult(
            output=validation,
            status="completed" if validation.get("passed") else "failed",
            summary={
                "passed": validation.get("passed"),
                "status": validation.get("verifier_status") or validation.get("status"),
                "method": validation.get("method"),
                "reasons": (validation.get("reasons") or [])[:3],
                "missing_requirements": len(validation.get("missing_requirements") or []),
            },
        )

    def _judge(self, board: Blackboard, **inputs: Any) -> AgentResult:
        ctx = inputs.get("ctx") or board.generation
        generated = inputs.get("generated")
        validation = inputs.get("validation")
        if (
            not isinstance(ctx, generation.GenerationContext)
            or generated is None
            or not isinstance(validation, dict)
        ):
            raise ValueError(
                "verifier.judge requires a GenerationContext, a generated answer and a validation report"
            )
        decision, payload = generation.judge_draft(
            ctx, generated, validation, int(inputs.get("attempt", 0))
        )
        status = "refused" if decision == "refuse" else "completed"
        return AgentResult(
            output=(decision, payload),
            status=status,
            summary={
                "decision": decision,
                "refusal_reason": getattr(payload, "refusal_reason", None),
            },
        )
