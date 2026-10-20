"""Shared per-request state and result types for the agent collaboration layer.

The blackboard carries the three mainline contracts (QueryPlan, EvidenceBundle,
ValidationReport) between specialist agents.  It never stores prompts, secrets
or evaluation identifiers; trace summaries hold counts, status codes and
aspect identifiers only.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from backend.app.schemas.qa import AgentTraceEntry

ProgressReporter = Callable[[dict[str, Any]], None]


@dataclass
class AgentResult:
    """What one agent task produced, plus a compact trace summary."""

    output: Any
    status: str = "completed"  # completed | failed | skipped | refused | clarification
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class Blackboard:
    """Shared state for one question; one instance per request."""

    db: Any
    original_question: str
    question: str
    options: list[str] = field(default_factory=list)
    option_labels: list[str | None] = field(default_factory=list)
    include_debug: bool = False
    progress_reporter: ProgressReporter | None = None
    answer_preview_reporter: Callable[..., None] | None = None
    cancellation_checker: Callable[[], None] | None = None
    route: str | None = None
    boundary_code: str | None = None
    query_plan: Any = None
    document_chunk_cache: dict[str, Any] = field(default_factory=dict)
    aspect_retrievals: list[Any] = field(default_factory=list)
    evidence_bundle: Any = None
    context_package: Any = None
    generation: Any = None
    drafts: list[Any] = field(default_factory=list)
    validations: list[dict[str, Any]] = field(default_factory=list)
    trace: list[AgentTraceEntry] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)  # routing decisions and other compact orchestration facts
    active_agent: str | None = None
    standalone: bool = False

    def report(self, event: dict[str, Any]) -> None:
        """Forward a progress event, stamping and locally recording it.

        The local list feeds the display projection.  It is deliberately
        bounded and contains the same user-visible stage metadata already sent
        to the task store, never prompts or model reasoning.
        """

        if self.active_agent and not event.get("agent"):
            event = {**event, "agent": self.active_agent}
        events = self.notes.setdefault("progress_events", [])
        if isinstance(events, list):
            events.append(dict(event))
            del events[:-100]
        if "first_feedback_ms" not in self.notes:
            started_at = self.notes.get("started_at")
            if isinstance(started_at, (int, float)):
                from time import perf_counter
                self.notes["first_feedback_ms"] = round(
                    max(0.0, perf_counter() - float(started_at)) * 1000, 2
                )
        if self.progress_reporter is None:
            return
        self.progress_reporter(event)

    @property
    def reporter(self) -> ProgressReporter | None:
        """Reporter for legacy helpers so their unchanged events carry the active agent."""

        if self.progress_reporter is None:
            return None
        return self.report

    def record(self, entry: AgentTraceEntry) -> None:
        """Append a trace entry, merging consecutive entries of the same task and status."""

        last = self.trace[-1] if self.trace else None
        if last is None or (last.agent, last.task, last.attempt, last.status) != (
            entry.agent, entry.task, entry.attempt, entry.status
        ):
            self.trace.append(entry)
            return
        summary = dict(last.summary)
        for key, value in entry.summary.items():
            if isinstance(value, list) and isinstance(summary.get(key), list):
                summary[key] = summary[key] + value
            else:
                summary[key] = value
        summary["calls"] = int(last.summary.get("calls", 1)) + 1
        self.trace[-1] = last.model_copy(update={
            "elapsed_ms": round((last.elapsed_ms or 0.0) + (entry.elapsed_ms or 0.0), 2),
            "model_calls": last.model_calls + entry.model_calls,
            "aspect_ids": list(dict.fromkeys([*last.aspect_ids, *entry.aspect_ids])),
            "summary": summary,
        })

    def total_model_calls(self) -> int:
        return sum(entry.model_calls for entry in self.trace)


_CURRENT_BOARD: ContextVar[Blackboard | None] = ContextVar("regumate_agent_board", default=None)


def active_board() -> Blackboard | None:
    """The blackboard of the request being served, if any."""

    return _CURRENT_BOARD.get()


@contextmanager
def board_context(board: Blackboard) -> Iterator[Blackboard]:
    token = _CURRENT_BOARD.set(board)
    try:
        yield board
    finally:
        _CURRENT_BOARD.reset(token)


def board_for(
    db: Any,
    question: str,
    *,
    options: list[str] | None = None,
    progress_reporter: ProgressReporter | None = None,
) -> Blackboard:
    """Return the active request board, or a standalone board for a direct stage call.

    Direct calls (``/qa/retrieve``, replay tooling, tests) have no supervisor;
    they still get a board so the same agents run and their trace is returned
    on the stage result instead of being dropped.
    """

    board = _CURRENT_BOARD.get()
    if board is not None:
        return board
    return Blackboard(
        db=db,
        original_question=question,
        question=question,
        options=list(options or []),
        progress_reporter=progress_reporter,
        standalone=True,
    )
