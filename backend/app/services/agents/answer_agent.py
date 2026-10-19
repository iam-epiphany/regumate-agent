"""Generation pre-checks and deterministic or grounded model drafts.

The agent is a thin, traced wrapper over the primitives in
``answer_generation_service``.  It never changes prompts, validation or
control flow; it only normalises its inputs and records what happened.
"""
from __future__ import annotations

from typing import Any

from backend.app.services import answer_generation_service as generation
from backend.app.services.agents.base import SpecialistAgent
from backend.app.services.agents.contracts import AgentResult, Blackboard


class AnswerAgent(SpecialistAgent):
    name = "answer"
    role = "答案生成"
    tasks = ('prepare', 'draft')
    capabilities = frozenset({'deterministic_answer', 'grounded_generation'})

    def run(self, board: Blackboard, task: str, **inputs: Any) -> AgentResult:
        if task == "prepare":
            return self._prepare(board, **inputs)
        if task == "draft":
            return self._draft(board, **inputs)
        raise ValueError(f"{self.name} has no task {task!r}")

    def _prepare(self, board: Blackboard, **inputs: Any) -> AgentResult:
        options = inputs.get("options")
        if options is None:
            options = board.options
        option_labels = inputs.get("option_labels")
        if option_labels is None:
            option_labels = board.option_labels
        normalized_options = [option for option in options or [] if str(option or "").strip()]
        normalized_labels = generation._normalize_option_labels(option_labels, len(normalized_options))
        question = inputs.get("question")
        if question is None:
            question = board.question
        requirements = inputs.get("requirements")
        required_aspect_ids = inputs.get("required_aspect_ids")
        answer_mode = inputs.get("answer_mode")
        if answer_mode is None:
            answer_mode = "text"
        prepared = generation.prepare_generation(
            question,
            inputs.get("context_chunks") or [],
            normalized_options,
            normalized_labels,
            llm_prompt=inputs.get("llm_prompt"),
            has_sufficient_context=bool(inputs.get("has_sufficient_context", False)),
            cancellation_checker=board.cancellation_checker,
            answer_mode=answer_mode,
            required_aspect_ids=required_aspect_ids,
            requirements=requirements,
            table_resolver=inputs.get("table_resolver"),
        )
        board.generation = prepared
        required = (
            [item.requirement_id for item in requirements]
            if requirements is not None
            else required_aspect_ids or []
        )
        if isinstance(prepared, generation.GenerationContext):
            return AgentResult(
                output=prepared,
                status="completed",
                summary={
                    "deterministic": prepared.deterministic is not None,
                    "answer_mode": prepared.answer_mode,
                    "required": len(prepared.required),
                },
            )
        status = "failed" if prepared.answer_type == "service_error" else "refused"
        return AgentResult(
            output=prepared,
            status=status,
            summary={
                "deterministic": prepared.answer_type in {
                    "table_deterministic", "formula_deterministic", "formula_refusal",
                },
                "answer_mode": answer_mode,
                "required": len(required),
                "refusal_reason": prepared.refusal_reason,
            },
        )

    def _draft(self, board: Blackboard, **inputs: Any) -> AgentResult:
        ctx = inputs.get("ctx")
        if ctx is None:
            ctx = board.generation
        if not isinstance(ctx, generation.GenerationContext):
            raise ValueError("answer.draft requires a GenerationContext")
        correction = inputs.get("correction")
        generated = generation.draft_answer(
            ctx,
            attempt=int(inputs.get("draft_attempt", inputs.get("attempt", 0))),
            correction=correction,
        )
        board.drafts.append(generated)
        status = "refused" if generated.refused else "completed"
        return AgentResult(
            output=generated,
            status=status,
            summary={
                "answer_type": generated.answer_type,
                "refused": generated.refused,
                "refusal_reason": generated.refusal_reason,
                "claims": len(generated.claims),
                "correction": bool(correction),
            },
        )
