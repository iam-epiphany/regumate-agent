"""Answer an explicit document identity question directly from the identity ledger."""
from __future__ import annotations

from typing import Any

from backend.app.services.agents.base import SpecialistAgent
from backend.app.services.agents.contracts import AgentResult, Blackboard


class IdentityAgent(SpecialistAgent):
    name = "identity"
    role = "文件身份"
    tasks = ('answer',)
    capabilities = frozenset({'document_identity', 'version_status'})

    def run(self, board: Blackboard, task: str, **inputs: Any) -> AgentResult:
        if task != "answer":
            raise ValueError(f"{self.name} has no task {task!r}")
        from backend.app.services import rag_service

        response = board.notes.get("identity_response")
        if response is None:
            raise ValueError("identity.answer requires a routed identity response")
        rag_service._save_qa_log(board.db, board.original_question, response)
        rag_service._log_qa_audit(board.db, "qa_identity_answered", board.original_question, response)
        return AgentResult(
            output=response,
            summary={
                "citations": len(response.citations),
                "answer_type": response.answer_type,
            },
        )
