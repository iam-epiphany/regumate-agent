"""Request triage: document identity, preprocessing and explicit request boundaries.

The three checks below are the original opening sequence of
``rag_service.answer_question``.  They only dispatch and record routing facts;
the decision logic itself stays in the existing service functions.
"""
from __future__ import annotations

from typing import Any

from backend.app.services.agents.base import SpecialistAgent
from backend.app.services.agents.contracts import AgentResult, Blackboard


class TriageAgent(SpecialistAgent):
    name = "triage"
    role = "分诊"
    tasks = ('route',)
    capabilities = frozenset({'document_identity_routing', 'question_preprocessing', 'request_boundary'})

    def run(self, board: Blackboard, task: str, **inputs: Any) -> AgentResult:
        if task != "route":
            raise ValueError(f"{self.name} has no task {task!r}")
        from backend.app.services import rag_service

        raw_options = inputs.get("raw_options")
        identity_response = rag_service.answer_document_identity_question(board.db, board.question)
        if identity_response is not None:
            board.notes["identity_response"] = identity_response
            board.route = "identity"
            return AgentResult(output={"route": "identity"}, summary={"route": "identity"})

        question_task = rag_service.build_question_task(board.question, raw_options)
        board.question = question_task.question
        board.options = list(question_task.options)
        board.option_labels = list(question_task.option_labels)
        boundary_code = rag_service._explicit_request_boundary_code(board.question)
        if boundary_code:
            board.boundary_code = boundary_code
            board.route = "boundary"
            return AgentResult(
                output={"route": "boundary", "boundary_code": boundary_code},
                summary={"route": "boundary", "boundary_code": boundary_code},
            )

        board.route = "pipeline"
        return AgentResult(output={"route": "pipeline"}, summary={"route": "pipeline"})
