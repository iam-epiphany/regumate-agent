import json
from types import SimpleNamespace

import pytest

from backend.app.core import rag_features
from backend.app.schemas.qa import AnswerClaim, EvidenceRequirement, RetrievalResult
from backend.app.services import answer_generation_service as generation
from backend.app.services import answer_validation_service as validation
from backend.app.services import query_planner_service as planner
from backend.app.services import rag_service, retrieval_support
from backend.app.services.evidence_contract_service import build_evidence_bundle, source_matches
from backend.app.services.llm_client import ChatCompletionConfig, ChatCompletionError, chat_completion_content
from backend.app.services.query_planner_service import QueryAspect


def _chunk(title: str, metadata: dict | None = None) -> RetrievalResult:
    return RetrievalResult(
        chunk_id="c", rank=1, source_doc=title, text="蓝杉服务应当完成核验。",
        citation_label="[1]", metadata=metadata or {},
    )


def _aspect(aspect_id: str, question: str) -> QueryAspect:
    return QueryAspect(aspect_id, question, (), "证据", ())


def test_planner_demotes_single_question_research_expansions() -> None:
    main = _aspect("A1", "第三档商业银行的最低资本要求是什么？")
    expansion = _aspect("A2", "第三档商业银行的认定标准、附加资本和过渡安排是什么？")
    marked = planner._mark_aspect_scope("第三档商业银行的最低资本要求是什么？", [main, expansion], 1)
    assert [item.mandatory for item in marked] == [True, False]


def test_planner_preserves_explicit_multi_part_scope() -> None:
    first = _aspect("A1", "甲事项的定义是什么？")
    second = _aspect("A2", "乙事项的审批流程是什么？")
    question = "请分别说明甲事项的定义和乙事项的审批流程。"
    marked = planner._mark_aspect_scope(question, [first, second], 2)
    assert all(item.mandatory for item in marked)


def test_period_qualifier_uses_validated_table_metadata() -> None:
    requested = "2025年商业银行主要指标分机构类情况表（季度）"
    actual = _chunk("2025年商业银行主要指标分机构类情况表",
                    {"column_label": "一季度 / 不良贷款余额"})
    assert source_matches([requested], actual)
    assert not source_matches(["2024年商业银行主要指标分机构类情况表（季度）"], actual)
    assert not source_matches(["2025年商业银行主要指标分机构类情况表（雾港版）"], actual)


def test_optional_requirement_missing_does_not_block_bundle() -> None:
    required = EvidenceRequirement(requirement_id="required", question="必答", mandatory=True)
    optional = EvidenceRequirement(requirement_id="optional", question="检索扩展", mandatory=False)
    chunk = _chunk("规则")
    chunk.metadata.update(aspect_id="required", prompt_matched_aspects=["required"])
    bundle = build_evidence_bundle([required, optional], [chunk])
    assert bundle.complete
    assert bundle.missing_requirements == []
    assert bundle.missing_optional_requirements == ["optional"]


def test_numbered_definition_does_not_swallow_following_section(monkeypatch) -> None:
    texts = ["（四）专业贷款是指符合特定条件的债权。", "1.其他无关事项。", "2.另一无关事项。"]
    chunks = [
        SimpleNamespace(
            chunk_id=f"d{i}", text=text, source_file="规则.docx", section_title=None,
            chunk_metadata="{}", document_id="doc", index_status="indexed", embedding_text=text,
        )
        for i, text in enumerate(texts)
    ]
    monkeypatch.setattr(retrieval_support, "_chunks_by_document", lambda *args, **kwargs: {"doc": chunks})
    anchor = _chunk("规则.docx", {"document_id": "doc"})
    anchor.chunk_id = "d0"
    selected = rag_service._retrieval_support_structural_context(
        SimpleNamespace(scalars=True), [anchor], question="“专业贷款”是如何定义的？"
    )
    assert [chunk.chunk_id for chunk in selected] == ["d0"]
    assert not selected[0].metadata.get("unit_incomplete")


class _Response:
    def __init__(self, body: dict):
        self._body = json.dumps(body, ensure_ascii=False).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return self._body


def _llm_config() -> ChatCompletionConfig:
    return ChatCompletionConfig("openai_compatible", "key", "https://example.test/v1", "m", 10)


def test_llm_client_marks_truncated_and_schema_responses() -> None:
    truncated = lambda *args, **kwargs: _Response(
        {"choices": [{"message": {"content": "{\"refused\":"}, "finish_reason": "length"}]}
    )
    with pytest.raises(ChatCompletionError) as exc:
        chat_completion_content(_llm_config(), [{"role": "user", "content": "hi"}], opener=truncated)
    assert exc.value.kind == "truncated"
    assert exc.value.finish_reason == "length"

    malformed = lambda *args, **kwargs: _Response({"choices": [{"message": {}}]})
    with pytest.raises(ChatCompletionError) as exc:
        chat_completion_content(_llm_config(), [{"role": "user", "content": "hi"}], opener=malformed)
    assert exc.value.kind == "response_schema"


def test_generation_parse_errors_are_categorized() -> None:
    with pytest.raises(generation.GenerationParseError) as exc:
        generation._parse_generated_answer_content(
            "not json", question="q", context_chunks=[], options=[], option_labels=[], answer_mode="text"
        )
    assert exc.value.kind == "no_json"
    with pytest.raises(generation.GenerationParseError) as exc:
        generation._parse_generated_answer_content(
            json.dumps({"refused": "no", "claims": []}), question="q", context_chunks=[],
            options=[], option_labels=[], answer_mode="text",
        )
    assert exc.value.kind == "schema_error"


def _generation_context() -> generation.GenerationContext:
    return generation.GenerationContext(
        question="问题", context_chunks=[], options=[], option_labels=[], llm_prompt=None,
        answer_mode="text", required=[], requirements=None, table_findings=[],
        deterministic=None, cancellation_checker=None,
    )


def test_format_failure_exhaustion_is_service_error(monkeypatch) -> None:
    monkeypatch.setattr(generation, "ANSWER_GENERATION_ENABLED", True)
    monkeypatch.setattr(generation, "ANSWER_GENERATION_API_KEY", "test")

    def draft(context, *, attempt, correction):
        raise generation.GenerationParseError("bad", kind="no_json")

    result = generation.run_generation_loop(_generation_context(), draft=draft, verify=lambda *a, **k: {})
    assert result.answer_type == "service_error"
    assert result.refusal_reason == "generation_format_failure"
    assert result.grounding_validation["error_type"] == "no_json"


def test_truncated_generation_retries_with_compact_output(monkeypatch) -> None:
    monkeypatch.setattr(generation, "ANSWER_GENERATION_ENABLED", True)
    monkeypatch.setattr(generation, "ANSWER_GENERATION_API_KEY", "test")
    attempts = []

    def draft(context, *, attempt, correction):
        attempts.append((attempt, bool(correction)))
        if attempt == 0:
            raise ChatCompletionError("truncated", kind="truncated", finish_reason="length")
        return generation.GeneratedAnswer(
            answer="答案为 A。", answer_type="llm_grounded", generation_status="completed",
            claims=[AnswerClaim(text="答案为 A。[1]", citation_ids=["[1]"])],
        )

    def verify(context, generated):
        return {"passed": True, "status": "passed", "task_complete": True,
                "facts_supported": True, "missing_requirements": [], "reasons": []}

    result = generation.run_generation_loop(_generation_context(), draft=draft, verify=verify)
    assert attempts == [(0, False), (1, True)]
    assert result.answer == "答案为 A。"


def test_validation_transport_failure_retries_once(monkeypatch) -> None:
    calls = []

    def fake_chat(config, messages, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ChatCompletionError("temporary", kind="transport")
        return json.dumps({"complete": True, "supported_claims": [1], "missing_requirements": [], "reasons": []})

    monkeypatch.setattr(validation, "chat_completion_content", fake_chat)
    monkeypatch.setattr(validation, "features", lambda: SimpleNamespace(semantic=True))
    chunk = _chunk("规则")
    claim = AnswerClaim(text="蓝杉服务应当完成核验。[1]", citation_ids=["[1]"])
    report = validation.validate_answer("q", claim.text, [claim], [chunk], requirements=[])
    assert report["status"] == "passed"
    assert report["validation_attempts"] == 2


def test_validation_schema_failure_is_not_transport_failure(monkeypatch) -> None:
    monkeypatch.setattr(validation, "chat_completion_content", lambda *a, **k: "not json")
    monkeypatch.setattr(validation, "features", lambda: SimpleNamespace(semantic=True))
    chunk = _chunk("规则")
    claim = AnswerClaim(text="蓝杉服务应当完成核验。[1]", citation_ids=["[1]"])
    report = validation.validate_answer("q", claim.text, [claim], [chunk], requirements=[])
    assert report["status"] == "error"
    assert report["reasons"] == ["validation_schema_failure"]
    assert report["validation_attempts"] == 1
