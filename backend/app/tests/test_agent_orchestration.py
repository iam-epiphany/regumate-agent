from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from backend.app.schemas.qa import (
    AgentTraceEntry,
    AnswerClaim,
    LLMContextPackage,
    QAResponse,
    RagProgressEvent,
    RetrievalResult,
)
from backend.app.services.agents.base import SpecialistAgent
from backend.app.services.agents.contracts import Blackboard, AgentResult, board_context, board_for
from backend.app.services.agents.registry import AgentRegistry, default_registry
from backend.app.services.performance_metrics import model_call_budget, reserve_model_call


def _board(*, reporter=None) -> Blackboard:
    return Blackboard(
        db=SimpleNamespace(),
        original_question="蓝杉在雾港的许可条件是什么？",
        question="蓝杉在雾港的许可条件是什么？",
        progress_reporter=reporter,
    )


def test_blackboard_record_merges_consecutive_same_entries_and_keeps_status_boundaries() -> None:
    board = _board()
    first = AgentTraceEntry(
        agent="evidence", role="证据", task="collect", status="completed", attempt=1,
        elapsed_ms=1.25, model_calls=1, aspect_ids=["a", "b"],
        summary={"summary": ["one"], "calls": 1},
    )
    second = AgentTraceEntry(
        agent="evidence", role="证据", task="collect", status="completed", attempt=1,
        elapsed_ms=2.75, model_calls=2, aspect_ids=["b", "c"],
        summary={"summary": ["two"], "calls": 1},
    )
    failed = second.model_copy(update={"status": "failed", "elapsed_ms": 3.0})

    board.record(first)
    board.record(second)
    board.record(failed)

    assert len(board.trace) == 2
    merged = board.trace[0]
    assert merged.elapsed_ms == 4.0
    assert merged.model_calls == 3
    assert merged.aspect_ids == ["a", "b", "c"]
    assert merged.summary["summary"] == ["one", "two"]
    assert merged.summary["calls"] == 2
    assert board.trace[1].status == "failed"


class _CountingAgent(SpecialistAgent):
    name = "counter"
    role = "测试"
    tasks = ("work", "explode")

    def run(self, board: Blackboard, task: str, **inputs):
        if task == "explode":
            raise RuntimeError("蓝杉测试异常")
        reserve_model_call(1.0)
        return AgentResult(output="done", summary={"kind": "test"})


def test_specialist_execute_records_metrics_restores_active_agent_and_reraises() -> None:
    board = _board()
    board.active_agent = "supervisor"
    agent = _CountingAgent()

    with model_call_budget(calls=0):
        result = agent.execute(board, "work", attempt=2, aspect_ids=["a1"])
    assert result.output == "done"
    assert board.active_agent == "supervisor"
    assert len(board.trace) == 1
    assert board.trace[0].agent == "counter"
    assert board.trace[0].task == "work"
    assert board.trace[0].attempt == 2
    assert board.trace[0].elapsed_ms is not None
    assert board.trace[0].elapsed_ms >= 0
    assert board.trace[0].model_calls == 1
    assert board.trace[0].aspect_ids == ["a1"]

    with pytest.raises(RuntimeError, match="蓝杉测试异常"):
        agent.execute(board, "explode")
    assert board.active_agent == "supervisor"
    assert board.trace[-1].status == "failed"
    assert board.trace[-1].summary["error_type"] == "RuntimeError"

    with pytest.raises(ValueError):
        agent.execute(board, "unknown")


def test_blackboard_report_stamps_agent_without_overwriting_existing_agent() -> None:
    events: list[dict] = []
    board = _board(reporter=events.append)
    board.active_agent = "evidence"
    board.report({"stage": "retrieval", "status": "running", "title": "开始"})
    board.report({"stage": "retrieval", "status": "completed", "title": "完成", "agent": "table"})
    assert events[0]["agent"] == "evidence"
    assert events[1]["agent"] == "table"
    _board().report({"stage": "retrieval", "status": "running", "title": "无 reporter"})


def test_board_for_returns_standalone_board_and_board_context_reuses_active_board() -> None:
    standalone = board_for(SimpleNamespace(), "蓝杉问题")
    assert standalone.standalone is True
    assert board_for(SimpleNamespace(), "另一个问题") is not standalone
    with board_context(standalone):
        assert board_for(SimpleNamespace(), "忽略") is standalone


def test_default_registry_has_eight_specialists_with_complete_descriptors_and_rejects_duplicates() -> None:
    registry = default_registry()
    expected = {"triage", "identity", "planner", "table", "evidence", "verifier", "curation", "answer"}
    assert set(registry.names()) == expected
    descriptors = registry.describe()
    assert {item["name"] for item in descriptors} == expected
    for item in descriptors:
        assert item["role"]
        assert isinstance(item["capabilities"], list)
        assert isinstance(item["tasks"], list)
    with pytest.raises(ValueError):
        registry.register(next(iter(registry)))


def test_schema_agent_trace_defaults_progress_agent_and_rejects_invalid_status() -> None:
    response = QAResponse(answer=None, citations=[], confidence=0.0, refused=True)
    assert response.agent_trace == []
    event = RagProgressEvent(
        stage="retrieval", status="running", title="开始", detail="详情", agent="evidence"
    )
    assert event.agent == "evidence"
    with pytest.raises(ValueError):
        AgentTraceEntry(agent="x", role="x", task="x", status="invalid")


def _chunk(text: str = "blue cedar evidence") -> RetrievalResult:
    return RetrievalResult(
        chunk_id="chunk-blue-1",
        rank=1,
        score=0.9,
        source_doc="mist-harbor.txt",
        text=text,
        citation_label="C1",
        metadata={"document_id": "doc-blue-1", "evidence_role": "direct_evidence"},
    )


def _context_package(*, text: str = "blue cedar evidence") -> LLMContextPackage:
    return LLMContextPackage(
        query="What does Blue Cedar require in Mist Harbor?",
        instruction="answer from evidence",
        retrieval_summary={
            "has_sufficient_context": True,
            "used_chunks": 1,
            "query_plan": {"aspects": []},
            "evidence_bundle": {"requirements": [], "missing_requirements": []},
            "coverage": {},
            "missing_requirements": [],
            "final_prompt_chunk_ids": ["chunk-blue-1"],
        },
        context_chunks=[_chunk(text)],
        llm_prompt="answer from C1",
    )


def _generated_answer(answer: str = "Blue Cedar requires notice.", *, refused: bool = False):
    from backend.app.services.answer_generation_service import GeneratedAnswer

    return GeneratedAnswer(
        answer=answer,
        answer_type="refusal" if refused else "text",
        generation_status="skipped" if refused else "completed",
        claims=[] if refused else [AnswerClaim(text=answer, citation_ids=["C1"])],
        refused=refused,
        refusal_reason="model_refusal" if refused else None,
    )


def _disable_identity_and_logs(monkeypatch, rag_service) -> None:
    monkeypatch.setattr(rag_service, "answer_document_identity_question", lambda db, question: None)
    monkeypatch.setattr(rag_service, "_save_qa_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(rag_service, "_log_qa_audit", lambda *args, **kwargs: None)


def _query_plan_with_aspects(aspects):
    from backend.app.services.query_planner_service import QueryPlan

    return QueryPlan(
        original_question="Blue Cedar in Mist Harbor",
        aspects=tuple(aspects),
        planner="test",
    )


def _aspect(aspect_id: str, modality: str):
    from backend.app.services.query_planner_service import QueryAspect, QuerySearchQuery

    return QueryAspect(
        aspect_id=aspect_id,
        question=f"Blue Cedar {aspect_id}",
        search_queries=(QuerySearchQuery(f"Blue Cedar {aspect_id}", "semantic_question"),),
        evidence_need="direct_evidence",
        keywords=("Blue Cedar",),
        modality=modality,
        table_task="lookup" if modality in {"table", "mixed"} else "none",
    )


def test_integration_identity_route_skips_pipeline(monkeypatch) -> None:
    from backend.app.services import rag_service

    identity_response = QAResponse(
        answer="Blue Cedar is listed in Mist Harbor.",
        citations=[],
        confidence=0.8,
        refused=False,
        answer_type="identity",
        generation_status="completed",
    )
    monkeypatch.setattr(rag_service, "answer_document_identity_question", lambda db, question: identity_response)
    monkeypatch.setattr(rag_service, "_save_qa_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(rag_service, "_log_qa_audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        rag_service,
        "build_context_package",
        lambda *args, **kwargs: pytest.fail("identity route must not build context"),
    )

    response = rag_service.answer_question(SimpleNamespace(), "Which file lists Blue Cedar?")

    assert response.answer == identity_response.answer
    tasks = [(entry.agent, entry.task) for entry in response.agent_trace]
    assert ("triage", "route") in tasks
    assert ("identity", "answer") in tasks
    assert not any(agent in {"planner", "evidence", "answer"} for agent, _task in tasks)


def test_integration_explicit_boundary_uses_zero_model_calls(monkeypatch) -> None:
    from backend.app.services import answer_generation_service, rag_service

    _disable_identity_and_logs(monkeypatch, rag_service)
    monkeypatch.setattr(rag_service, "_explicit_request_boundary_code", lambda question: "unsupported_prediction")
    monkeypatch.setattr(
        answer_generation_service,
        "_call_llm",
        lambda *args, **kwargs: pytest.fail("boundary request must not call a model"),
    )

    response = rag_service.answer_question(SimpleNamespace(), "Please predict the future outcome for Blue Cedar.")

    assert response.refused is True
    assert response.grounding_validation.get("model_call_count", 0) == 0
    assert not any(entry.agent == "planner" for entry in response.agent_trace)


def test_integration_planning_clarification_skips_evidence_and_answer(monkeypatch) -> None:
    from backend.app.services import rag_service
    from backend.app.services.query_planner_service import QueryPlan

    _disable_identity_and_logs(monkeypatch, rag_service)
    monkeypatch.setattr(
        rag_service,
        "plan_query",
        lambda question, options=None: QueryPlan(
            original_question=question,
            aspects=(),
            planner="test",
            clarification_reason="Please name the relevant Mist Harbor period.",
        ),
    )
    monkeypatch.setattr(
        rag_service,
        "retrieve_evidence",
        lambda *args, **kwargs: pytest.fail("clarification must not collect evidence"),
    )

    events: list[dict] = []
    response = rag_service.answer_question(
        SimpleNamespace(), "Clarify Blue Cedar", progress_reporter=events.append
    )

    assert response.answer_type == "clarification"
    tasks = [(entry.agent, entry.task) for entry in response.agent_trace]
    assert ("planner", "plan") in tasks
    assert not any(agent in {"evidence", "answer"} for agent, _task in tasks)
    assert not any(entry.task in {"collect", "draft"} for entry in response.agent_trace)
    planning_events = [event for event in events if event.get("stage") == "planning"]
    assert [event["status"] for event in planning_events] == ["running", "completed"]
    assert planning_events[-1]["title"] == "需要补充问题信息"


def test_integration_retrieval_routes_table_text_and_mixed_with_legacy_progress(monkeypatch) -> None:
    from backend.app.services import rag_service, retrieval_support
    from backend.app.services.agents import evidence_agent

    table, text_aspect, mixed = (_aspect("table-1", "table"), _aspect("text-1", "text"), _aspect("mixed-1", "mixed"))
    plan = _query_plan_with_aspects((table, text_aspect, mixed))
    events: list[dict] = []
    board = _board(reporter=events.append)
    board.question = plan.original_question
    table_stage_type = retrieval_support._TableStageResult
    prep_type = retrieval_support._AspectPrepResult
    table_calls: list[str] = []

    def fake_table_stage(db, aspect):
        table_calls.append(aspect.aspect_id)
        if aspect.aspect_id == "table-1":
            return table_stage_type([], [{"match_status": "structured_cells_located"}], True, "structured_cells_located")
        return table_stage_type([], [], False, None)

    def fake_text_stage(db, aspect, table_stage, **kwargs):
        return prep_type(False, [], [], SimpleNamespace(search_queries=()))

    def fake_collect(self, board, prepared_items):
        return AgentResult(
            output={aspect.aspect_id: ([], []) for aspect, _prepared in prepared_items},
            summary={"aspects": len(prepared_items)},
        )

    monkeypatch.setattr(retrieval_support, "_prepare_table_stage", fake_table_stage)
    monkeypatch.setattr(retrieval_support, "_prepare_text_stage", fake_text_stage)
    monkeypatch.setattr(evidence_agent.EvidenceAgent, "_collect", fake_collect)
    monkeypatch.setattr(rag_service, "_expand_neighbor_matches", lambda *args, **kwargs: [])
    monkeypatch.setattr(rag_service, "_validate_context_chunks", lambda db, chunks: (chunks, {
        "checked_chunks": len(chunks), "valid_chunks": len(chunks), "invalid_chunks": 0,
    }))

    with board_context(board):
        retrievals = rag_service._retrieve_aspects(board.db, plan)

    assert [item.aspect.aspect_id for item in retrievals] == ["table-1", "text-1", "mixed-1"]
    assert table_calls == ["table-1", "mixed-1"]
    assert board.notes["aspect_routes"] == [
        {"aspect_id": "table-1", "agent": "table", "terminal": True, "terminal_reason": "structured_cells_located"},
        {"aspect_id": "text-1", "agent": "evidence", "terminal": False, "terminal_reason": None},
        {"aspect_id": "mixed-1", "agent": "evidence", "terminal": False, "terminal_reason": None},
    ]
    assert [(event["stage"], event["status"]) for event in events] == [
        ("retrieval", "running"),
        ("retrieval", "running"), ("retrieval", "running"), ("retrieval", "running"),
        ("retrieval", "failed"), ("retrieval", "failed"), ("retrieval", "failed"),
        ("retrieval", "completed"), ("rerank", "skipped"),
    ]
    assert all(isinstance(event.get("title"), str) and event["title"] for event in events)
    assert all(event.get("agent") for event in events)


def test_integration_structured_spreadsheet_refusal_skips_text_prepare_and_collect(monkeypatch) -> None:
    from backend.app.services import rag_service, retrieval_support
    from backend.app.services.agents import evidence_agent

    aspect = _aspect("table-refused", "table")
    plan = _query_plan_with_aspects((aspect,))
    stage_type = retrieval_support._TableStageResult
    called: list[str] = []

    monkeypatch.setattr(
        retrieval_support,
        "_prepare_table_stage",
        lambda db, aspect: stage_type([], [{"match_status": "structured_spreadsheet_refusal"}], True, "structured_spreadsheet_refusal"),
    )
    monkeypatch.setattr(
        retrieval_support,
        "_prepare_text_stage",
        lambda *args, **kwargs: (called.append("prepare") or pytest.fail("text prepare must be skipped")),
    )
    monkeypatch.setattr(
        evidence_agent.EvidenceAgent,
        "_collect",
        lambda *args, **kwargs: (called.append("collect") or pytest.fail("evidence collect must be skipped")),
    )
    monkeypatch.setattr(rag_service, "_expand_neighbor_matches", lambda *args, **kwargs: [])
    monkeypatch.setattr(rag_service, "_validate_context_chunks", lambda db, chunks: (chunks, {}))

    board = _board()
    with board_context(board):
        rag_service._retrieve_aspects(board.db, plan)
    assert called == []


def _run_supervisor_with_generation(monkeypatch, llm_results, validation_results):
    from backend.app.services import answer_generation_service, answer_validation_service, rag_service

    _disable_identity_and_logs(monkeypatch, rag_service)
    monkeypatch.setattr(rag_service, "build_context_package", lambda *args, **kwargs: _context_package())
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_ENABLED", True)
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", "test-key")
    calls = {"llm": 0, "validate": 0}

    def fake_llm(*args, **kwargs):
        calls["llm"] += 1
        reserve_model_call(1.0)
        return llm_results[calls["llm"] - 1]

    def fake_validate(*args, **kwargs):
        calls["validate"] += 1
        return validation_results[calls["validate"] - 1]

    monkeypatch.setattr(answer_generation_service, "_call_llm", fake_llm)
    monkeypatch.setattr(answer_validation_service, "validate_answer", fake_validate)
    response = rag_service.answer_question(SimpleNamespace(), "What does Blue Cedar require?")
    return response, calls


def test_integration_generation_verification_repair_refusal_and_model_decline(monkeypatch) -> None:
    from backend.app.services.answer_generation_service import GeneratedAnswer

    first, second = _generated_answer("first draft"), _generated_answer("repaired draft")
    response, calls = _run_supervisor_with_generation(
        monkeypatch,
        [first, second],
        [
            {"passed": False, "status": "failed", "task_complete": False, "reasons": ["unsupported"]},
            {"passed": True, "status": "passed", "task_complete": True, "citation_integrity": True, "facts_supported": True},
        ],
    )
    assert response.refused is False
    assert response.grounding_validation.get("repair_attempted") is True
    assert calls["llm"] == 2 and calls["validate"] == 2
    assert [(entry.agent, entry.task, entry.attempt) for entry in response.agent_trace if entry.agent == "answer" and entry.task == "draft"] == [("answer", "draft", 1), ("answer", "draft", 2)]
    assert sum(entry.agent == "verifier" and entry.task == "verify" for entry in response.agent_trace) == 2
    assert response.grounding_validation.get("model_call_count", 0) <= 5

    failed_response, failed_calls = _run_supervisor_with_generation(
        monkeypatch,
        [_generated_answer("bad one"), _generated_answer("bad two")],
        [
            {"passed": False, "status": "failed", "task_complete": False},
            {"passed": False, "status": "failed", "task_complete": False},
        ],
    )
    assert failed_response.refused is True
    assert failed_response.refusal_reason == "grounding_validation_failed"
    assert failed_calls["llm"] == 2
    assert not any(entry.agent == "answer" and entry.task == "draft" and entry.attempt == 3 for entry in failed_response.agent_trace)

    declined, declined_calls = _run_supervisor_with_generation(
        monkeypatch,
        [_generated_answer("declined", refused=True)],
        [],
    )
    assert declined.refused is True
    assert declined_calls["validate"] == 0
    assert not any(entry.agent == "verifier" and entry.task == "verify" for entry in declined.agent_trace)


def test_integration_generate_answer_and_supervisor_loop_are_equivalent(monkeypatch) -> None:
    from backend.app.services import answer_generation_service, answer_validation_service, rag_service

    _disable_identity_and_logs(monkeypatch, rag_service)
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_ENABLED", True)
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", "test-key")
    old_calls = [_generated_answer("equivalent answer")]
    def old_call(*args, **kwargs):
        reserve_model_call(1.0)
        return old_calls.pop(0)

    monkeypatch.setattr(answer_generation_service, "_call_llm", old_call)
    monkeypatch.setattr(answer_validation_service, "validate_answer", lambda *args, **kwargs: {
        "passed": True, "status": "passed", "task_complete": True,
        "citation_integrity": True, "facts_supported": True,
    })
    with model_call_budget(calls=0):
        legacy = answer_generation_service.generate_answer(
            "What does Blue Cedar require?", [_chunk()], has_sufficient_context=True,
        )

    captured: dict[str, object] = {}
    original_loop = answer_generation_service.run_generation_loop

    def capture_loop(ctx, **kwargs):
        result = original_loop(ctx, **kwargs)
        captured["generated"] = result
        return result

    monkeypatch.setattr(answer_generation_service, "run_generation_loop", capture_loop)
    def supervisor_call(*args, **kwargs):
        reserve_model_call(1.0)
        return _generated_answer("equivalent answer")

    monkeypatch.setattr(answer_generation_service, "_call_llm", supervisor_call)
    monkeypatch.setattr(rag_service, "build_context_package", lambda *args, **kwargs: _context_package())
    with model_call_budget(calls=0):
        rag_service.answer_question(SimpleNamespace(), "What does Blue Cedar require?")

    supervisor_result = captured["generated"]
    fields = ("answer", "answer_type", "generation_status", "refused", "refusal_reason", "grounding_validation")
    for field in fields:
        old_value = getattr(legacy, field)
        new_value = getattr(supervisor_result, field)
        if field == "grounding_validation":
            old_value = {k: v for k, v in old_value.items() if k != "model_call_count"}
            new_value = {k: v for k, v in new_value.items() if k != "model_call_count"}
        assert new_value == old_value


def test_integration_response_trace_is_compact_and_progress_has_agents(monkeypatch) -> None:
    from backend.app.services import answer_generation_service, rag_service

    _disable_identity_and_logs(monkeypatch, rag_service)
    monkeypatch.setattr(rag_service, "build_context_package", lambda *args, **kwargs: _context_package(text="SECRET_CONTEXT_TEXT"))
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_ENABLED", True)
    monkeypatch.setattr(answer_generation_service, "ANSWER_GENERATION_API_KEY", "test-key")
    monkeypatch.setattr(answer_generation_service, "_call_llm", lambda *args, **kwargs: _generated_answer())
    monkeypatch.setattr(
        "backend.app.services.answer_validation_service.validate_answer",
        lambda *args, **kwargs: {"passed": True, "status": "passed", "task_complete": True, "citation_integrity": True, "facts_supported": True},
    )
    events: list[dict] = []
    response = rag_service.answer_question(SimpleNamespace(), "What does Blue Cedar require?", progress_reporter=events.append)

    assert response.agent_trace
    serialized = str([entry.model_dump() for entry in response.agent_trace])
    assert "SECRET_CONTEXT_TEXT" not in serialized
    assert events and all(event.get("agent") for event in events)
    assert response.answer is not None
    assert response.citations is not None
    assert response.generation_status in {"completed", "skipped", "error"}


def test_integration_agents_endpoint_lists_supervisor_and_eight_specialists() -> None:
    from fastapi.testclient import TestClient
    from backend.main import app

    payload = TestClient(app).get("/api/qa/agents")
    assert payload.status_code == 200
    body = payload.json()
    assert body["supervisor"] == "supervisor"
    assert {item["name"] for item in body["agents"]} == {
        "triage", "identity", "planner", "table", "evidence", "verifier", "curation", "answer",
    }


def test_refused_draft_with_degraded_flag_takes_the_single_model_refusal_branch(monkeypatch) -> None:
    """A refused draft is normalised the same way whatever flags the seam sets on it."""
    from backend.app.services import answer_generation_service as generation

    chunk = RetrievalResult(chunk_id="c1", rank=1, source_doc="蓝杉规则.docx", text="蓝杉核验需双人复核。",
                            citation_label="[1]", metadata={"aspect_id": "a1", "prompt_matched_aspects": ["a1"]})
    monkeypatch.setattr(generation, "ANSWER_GENERATION_ENABLED", True)
    monkeypatch.setattr(generation, "ANSWER_GENERATION_API_KEY", "test-key")
    monkeypatch.setattr(generation, "_call_llm", lambda *args, **kwargs: generation.GeneratedAnswer(
        answer=None, answer_type="refusal", generation_status="completed", refused=True,
        refusal_reason=None, degraded=True, grounding_validation={"status": "error"}))
    result = generation.generate_answer("蓝杉核验要求？", [chunk], has_sufficient_context=True, required_aspect_ids=["a1"])
    assert result.refused and result.refusal_reason == "model_refusal"
    # Legacy order: the answer fallback is filled before the reason is defaulted.
    assert result.answer == "当前证据不足，无法完整回答。"
    assert {k: result.grounding_validation[k] for k in ("passed", "status", "task_complete", "reason", "failure_stage")} == {
        "passed": False, "status": "not_run", "task_complete": False, "reason": "model_refusal", "failure_stage": "generation"}


def test_generation_unavailable_is_decided_before_any_draft(monkeypatch) -> None:
    from backend.app.services import answer_generation_service as generation

    chunk = RetrievalResult(chunk_id="c1", rank=1, source_doc="蓝杉规则.docx", text="蓝杉核验需双人复核。",
                            citation_label="[1]", metadata={"aspect_id": "a1", "prompt_matched_aspects": ["a1"]})
    monkeypatch.setattr(generation, "ANSWER_GENERATION_API_KEY", None)
    monkeypatch.setattr(generation, "_call_llm", lambda *args, **kwargs: pytest.fail("no draft when generation is unavailable"))
    result = generation.generate_answer("蓝杉核验要求？", [chunk], has_sufficient_context=True, required_aspect_ids=["a1"])
    assert result.answer_type == "service_error" and result.refusal_reason == "generation_unavailable"
    assert result.degraded is True and result.grounding_validation["status"] == "error"


def test_integration_composite_preparation_seam_on_retrieval_support_is_honoured(monkeypatch) -> None:
    from backend.app.services import rag_service, retrieval_support

    aspect = _aspect("text-1", "text")
    plan = _query_plan_with_aspects((aspect,))
    seen: list[str] = []

    def composite(db, aspect, progress_reporter=None, document_chunk_cache=None):
        seen.append(aspect.aspect_id)
        return retrieval_support._AspectPrepResult(True, [], [], None)

    monkeypatch.setattr(retrieval_support, "_prepare_aspect_retrieval", composite)
    monkeypatch.setattr(retrieval_support, "_prepare_table_stage", lambda *a, **k: pytest.fail("composite seam must win"))
    monkeypatch.setattr(rag_service, "_validate_context_chunks", lambda db, candidates: (candidates, {}))
    retrievals = rag_service._retrieve_aspects(SimpleNamespace(), plan)
    assert seen == ["text-1"]
    assert [item.aspect.aspect_id for item in retrievals] == ["text-1"]


def test_trace_recording_failure_never_masks_the_task_outcome(monkeypatch) -> None:
    """A broken trace must not replace the primary exception or block a result."""
    from backend.app.services.performance_metrics import RequestBudgetExceeded

    class Boom(SpecialistAgent):
        name = "boom"
        role = "test"
        tasks = ("ok", "budget")

        def run(self, board, task, **inputs):
            if task == "budget":
                raise RequestBudgetExceeded("model call limit exhausted", stage="model_call", code="model_call_limit_exceeded")
            return AgentResult(output="fine", status="not-a-valid-status")

    board = _board()
    agent = Boom()
    assert agent.execute(board, "ok").output == "fine"          # invalid status only loses the entry
    assert board.trace == []
    monkeypatch.setattr(board, "record", lambda entry: (_ for _ in ()).throw(RuntimeError("trace store down")))
    with pytest.raises(RequestBudgetExceeded) as info:
        agent.execute(board, "budget")
    assert info.value.code == "model_call_limit_exceeded"
    assert board.active_agent is None
