"""Base class for specialist agents.

An agent wraps existing deterministic service steps behind a named task
interface.  ``execute`` records timing and model-call deltas on the request
blackboard and re-raises every exception unchanged, so budget, cancellation and
service-unavailable semantics of the mainline are preserved.
"""
from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, ClassVar, Sequence

from backend.app.schemas.qa import AgentTraceEntry
from backend.app.services.agents.contracts import AgentResult, Blackboard
from backend.app.services.performance_metrics import model_call_count

logger = logging.getLogger(__name__)


class SpecialistAgent:
    name: ClassVar[str] = "agent"
    role: ClassVar[str] = ""
    capabilities: ClassVar[frozenset[str]] = frozenset()
    tasks: ClassVar[tuple[str, ...]] = ()

    def run(self, board: Blackboard, task: str, **inputs: Any) -> AgentResult:
        raise NotImplementedError(f"{self.name} does not implement task {task!r}")

    def execute(
        self,
        board: Blackboard,
        task: str,
        *,
        attempt: int = 1,
        aspect_ids: Sequence[str] = (),
        **inputs: Any,
    ) -> AgentResult:
        if task not in self.tasks:
            raise ValueError(f"{self.name} has no task {task!r}")
        started = perf_counter()
        calls_before = model_call_count()
        previous = board.active_agent
        board.active_agent = self.name
        status = "failed"
        summary: dict[str, Any] = {}
        try:
            result = self.run(board, task, **inputs)
            status, summary = result.status, result.summary
            return result
        except Exception as exc:  # re-raised unchanged; the trace only notes the type
            summary = {"error_type": type(exc).__name__}
            raise
        finally:
            board.active_agent = previous
            self._record(board, task, status=status, attempt=attempt, started=started,
                         calls_before=calls_before, aspect_ids=aspect_ids, summary=summary)

    def _record(self, board: Blackboard, task: str, *, status: str, attempt: int, started: float,
                calls_before: int, aspect_ids: Sequence[str], summary: dict[str, Any]) -> None:
        """Record the trace entry; a failure here is logged and never propagated.

        The entry is observability only.  If building or merging it raised, that
        secondary error would replace the task's own exception (budget exhaustion,
        cancellation, service unavailability) or block a successful result.
        """
        try:
            board.record(AgentTraceEntry(
                agent=self.name,
                role=self.role,
                task=task,
                status=status,
                attempt=attempt,
                elapsed_ms=round((perf_counter() - started) * 1000, 2),
                model_calls=max(0, model_call_count() - calls_before),
                aspect_ids=list(aspect_ids),
                summary=summary,
            ))
        except Exception:  # noqa: BLE001 - observability must not alter outcomes
            logger.warning("agent trace entry for %s/%s could not be recorded", self.name, task, exc_info=True)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "capabilities": sorted(self.capabilities),
            "tasks": list(self.tasks),
        }
