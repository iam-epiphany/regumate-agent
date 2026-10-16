import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from backend.app.services import reference_resolver_service as resolver
from backend.app.services import retrieval_advisor_service as advisor
from backend.app.services import retrieval_support as support
from backend.app.services.llm_client import ChatCompletionError
from backend.app.services.retrieval_service import RetrievalMatch
from backend.app.core.rag_features import RagFeatures
from backend.app.schemas.qa import Citation


def chunk(name, text, *, doc="document", page=None, title=None, status="indexed", **metadata):
    return SimpleNamespace(chunk_id=name, document_id=doc, text=text, embedding_text=text,
                           source_file="source.pdf", section_title=title, page_number=page,
                           index_status=status, chunk_metadata=json.dumps(metadata))


def resolved(anchor, chunks, hints=None):
    return resolver.resolve_references(anchor, [anchor, *chunks],
                                       hints if hints is not None else resolver.fallback_references(anchor.text))


@pytest.mark.parametrize("text,target,kind", [
    ("计算要求见第十条。", chunk("target", "第10条 计算要求"), "clause"),
    ("计算要求见第18页。", chunk("target", "计算要求", page=18), "page"),
    ("应参见附件2。", chunk("target", "附件二\n报告格式"), "attachment"),
    ("流程参见图2。", chunk("target", "图二 流程说明"), "figure"),
    ("数据见第18页表3。", chunk("target", "|项目|金额|", page=18, table_title="表三 金额", table_id="t"), "table"),
    ("数据见第十八页，表三。", chunk("target", "|项目|金额|", page=18, table_title="表三 金额", table_id="t"), "table"),
])
def test_stored_locators_resolve_across_numeral_spellings(text, target, kind):
    hits = resolved(chunk("anchor", text), [target])
    assert [(edge.chunk.chunk_id, edge.hint.kind) for edge in hits] == [("target", kind)]


def test_compound_locator_requires_both_page_and_printed_table_number():
    anchor = chunk("anchor", "详见第18页表3。")
    wrong_page = chunk("wrong-page", "|a|b|", page=17, table_id="a", table_title="表3 金额")
    wrong_number = chunk("wrong-number", "|a|b|", page=18, table_id="b", table_title="表4 金额")
    ingest_ordinal = chunk("ordinal", "|a|b|", page=18, table_id="c", table_index=3)
    assert resolved(anchor, [wrong_page, wrong_number, ingest_ordinal]) == []


def test_model_recognizes_implicit_reference_and_batches_sources(monkeypatch):
    anchors = [chunk("a", "操作依照后面给出的流程示意办理。"), chunk("b", "没有引用。")]
    calls = []

    def model(instruction, payload):
        calls.append(payload)
        return {"sources": [{"source": 0, "references": [{"kind": "figure", "quote": "后面给出的流程示意",
                                                          "direction": "next"}]}]}

    monkeypatch.setattr(resolver, "request_advice", model)
    hints = resolver.recognize_references(anchors)
    figure = chunk("figure", "图2 办理流程")
    assert resolved(anchors[0], [chunk("paragraph", "普通正文"), figure], hints["a"])[0].chunk == figure
    assert len(calls) == 1 and len(calls[0]) == 2
    assert hints["b"] == []


@pytest.mark.parametrize("response", [
    {}, {"sources": "invalid"}, {"sources": [{"source": 20, "references": []}]},
    {"sources": [{"source": 0, "references": [{"kind": "table", "quote": "原文没有的引用", "number": "3"}]}]},
    {"sources": [{"source": 0, "references": [{"kind": "table", "quote": "见下表", "number": "3"}]}]},
    {"sources": [{"source": 0, "references": [{"kind": "table", "quote": "见下表", "target_chunk_id": "invented"}]}]},
])
def test_invalid_model_output_keeps_explicit_fallback(monkeypatch, response):
    anchor = chunk("a", "见下表，另见第十条。")
    monkeypatch.setattr(resolver, "request_advice", lambda *args: response)
    hints = resolver.recognize_references([anchor])["a"]
    assert any(h.kind == "clause" and h.number == "十" for h in hints)
    assert not any(h.number == "3" for h in hints)


def test_ambiguous_or_missing_objects_and_foreign_documents_are_not_guessed():
    anchor = chunk("a", "参见表3。")
    one = chunk("t1", "|a|b|", table_title="表3 金额", table_id="t1", page=1)
    two = chunk("t2", "|a|b|", table_title="表3 金额", table_id="t2", page=2)
    foreign = chunk("foreign", "|a|b|", doc="other", table_title="表3 金额", table_id="foreign")
    deleted = chunk("deleted", "|a|b|", status="failed", table_title="表3 金额", table_id="deleted")
    mention = chunk("mention", "另一个计算方法也参见表3。")
    assert resolved(anchor, [one, two]) == []
    assert resolved(anchor, [foreign, deleted, mention]) == []
    external = chunk("external", "依据《另一办法》第十条。")
    assert resolved(external, [chunk("clause", "第十条 计算规则")]) == []
    external = chunk("external", "参见《另一办法》附件2。")
    assert resolved(external, [chunk("attachment", "附件2 申请格式")]) == []


def test_nearest_table_and_context_direction_preserve_document_order():
    anchor = chunk("a", "数据见下表。")
    farther = chunk("farther", "|a|b|", table_id="farther")
    nearest = chunk("nearest", "|a|b|", table_id="nearest")
    assert resolved(anchor, [nearest, farther])[0].chunk == nearest
    previous = chunk("previous", "前文结论")
    anchor = chunk("a", "详见上文。")
    assert resolver.resolve_references(anchor, [previous, anchor], resolver.fallback_references(anchor.text))[0].chunk == previous


def test_section_model_hint_resolves_existing_numbered_section():
    anchor = chunk("a", "应按照2.3节办理。")
    target = chunk("target", "办理规则", title="2.3 办理规则")
    hint = resolver.ReferenceHint(kind="section", quote="按照2.3节", number="2.3")
    assert resolved(anchor, [target], [hint])[0].chunk == target


def test_split_attachment_across_pages_and_merged_inline_table():
    anchor = chunk("a", "参见附件2。")
    parts = [chunk("part1", "申请格式", title="附件二 申请格式", page=3),
             chunk("part2", "申请格式续", title="附件二 申请格式", page=4)]
    assert [edge.chunk.chunk_id for edge in resolved(anchor, parts)] == ["part1", "part2"]
    inline = chunk("inline", "见下表\n|a|b|", table_context="见下表", table_id="inline")
    unrelated = chunk("unrelated", "|a|b|", table_id="other")
    assert resolved(inline, [unrelated]) == []


def test_dynamic_spreadsheet_evidence_skips_model(monkeypatch):
    def unexpected(*args):
        pytest.fail("spreadsheet fast path called the reference model")
    monkeypatch.setattr(resolver, "request_advice", unexpected)
    anchor = chunk("a", "单元格=100", table_id="t")
    citation = Citation(document_id="document", chunk_id="a", filename="source.xlsx", excerpt=anchor.text,
                        chunk_type="table_cell")
    match = RetrievalMatch(citation=citation, score=.9, rerank_score=.9, coverage_score=.8,
                           evidence_role="table_fact", evidence_text=anchor.text, metadata={"dynamic_table_evidence": True})
    assert support._expand_neighbor_matches(None, "金额", [match], {"document": [anchor]}) == [match]


def test_reference_expansion_has_priority_provenance_and_no_recursive_calls(monkeypatch):
    anchor = chunk("a", "金额的计算见第18页表3。")
    near = chunk("near", "金额计算的一般规则。")
    target = chunk("target", "|项目|金额|", page=18, table_title="表3 金额", table_id="t")
    citation = Citation(document_id="document", chunk_id="a", filename="source.pdf", excerpt=anchor.text)
    match = RetrievalMatch(citation=citation, score=.9, rerank_score=.9, coverage_score=.8,
                           evidence_role="regulatory_basis", evidence_text=anchor.text, metadata={})
    result = support._expand_neighbor_matches(None, "金额如何计算", [match], {"document": [anchor, near, target]})
    assert [item.citation.chunk_id for item in result][:2] == ["a", "target"]
    provenance = result[1].metadata["reference_resolution"]
    assert provenance["kind"] == "table" and provenance["source_chunk_id"] == "a"
    assert provenance["target_chunk_id"] == "target"


def test_reference_feature_switch_skips_model_and_edges(monkeypatch):
    monkeypatch.setattr(support, "features", lambda: replace(RagFeatures(), references=False))
    def unexpected(*args):
        pytest.fail("disabled references called the model")
    monkeypatch.setattr(support, "recognize_references", unexpected)
    anchor = chunk("a", "参见第十条。")
    target = chunk("target", "第十条 目标内容")
    assert not any(reason.startswith("referenced_") for _, reason in support._neighbor_candidates("a", [anchor, target], "目标"))
    citation = Citation(document_id="document", chunk_id="a", filename="source.pdf", excerpt=anchor.text)
    match = RetrievalMatch(citation=citation, score=.9, rerank_score=.9, coverage_score=.8,
                           evidence_role="regulatory_basis", evidence_text=anchor.text, metadata={})
    support._expand_neighbor_matches(None, "目标", [match], {"document": [anchor, target]})


@pytest.mark.parametrize("response", [{}, {"status": "sufficient", "missing_terms": ["x"]},
                                      {"status": "wrong", "missing_terms": []}])
def test_bad_crag_review_falls_back(monkeypatch, response):
    monkeypatch.setattr(advisor, "request_advice", lambda *args: response)
    assert advisor.review_coverage("如何办理", ["证据"])[1] == "rule_fallback"


def test_crag_model_terms_recover_only_real_same_document_evidence(monkeypatch):
    from backend.app.tests.test_retrieval_service import _intent_gap_aspect, _base_match, _fake_db_for_support, _snapshot_chunk
    monkeypatch.setattr(advisor, "request_advice", lambda *args: {"status": "insufficient", "missing_terms": ["办理流程"]})
    target = _snapshot_chunk("DOC-1-target", "DOC-1", text="办理流程应经过申请和审批。", section_title="流程")
    supplements = support._intent_gap_supplement(_fake_db_for_support(), _intent_gap_aspect(),
                                                [_base_match(evidence_text="一般规定")],
                                                document_chunk_cache={"DOC-1": [target]})
    assert any(item.citation.chunk_id == target.chunk_id for item in supplements)
    assert advisor.review_coverage("如何办理", ["一般规定"])[0].status == "insufficient"


def test_advisor_timeout_and_invalid_json_fall_back(monkeypatch):
    def unavailable(*args, **kwargs):
        raise ChatCompletionError("timeout")
    monkeypatch.setattr(advisor, "request_advice", unavailable)
    assert advisor.review_coverage("问题", ["证据"])[1] == "rule_fallback"


def test_advisor_request_uses_shared_client_with_bounds(monkeypatch):
    from backend.app.core import config
    captured = []
    def model(connection, messages, **kwargs):
        captured.append((connection, messages, kwargs))
        return '{"status":"sufficient","missing_terms":[]}'
    monkeypatch.setattr(config, "RETRIEVAL_ADVISOR_ENABLED", True)
    monkeypatch.setattr(advisor, "chat_completion_content", model)
    assert advisor.request_advice("inspect", {"text": "data"})["status"] == "sufficient"
    connection, messages, options = captured[0]
    assert connection.timeout_seconds == config.RETRIEVAL_ADVISOR_TIMEOUT_SECONDS
    assert connection.disable_thinking is True
    assert options == {"temperature": 0, "max_tokens": 1800}
    assert messages[0]["role"] == "system" and messages[1]["role"] == "user"
    monkeypatch.setattr(advisor, "chat_completion_content", lambda *args, **kwargs: "not JSON")
    assert advisor.review_coverage("问题", ["证据"])[1] == "rule_fallback"


def test_request_budget_is_shared_and_reset_after_retrieval(monkeypatch):
    from backend.app.core import config
    calls = []
    monkeypatch.setattr(config, "RETRIEVAL_ADVISOR_ENABLED", True)
    monkeypatch.setattr(config, "RETRIEVAL_ADVISOR_MAX_CALLS", 1)
    monkeypatch.setattr(advisor, "chat_completion_content", lambda *args, **kwargs: calls.append(1) or '{}')

    @advisor.with_advice_budget
    def retrieve():
        advisor.request_advice("first aspect", {})
        with pytest.raises(ChatCompletionError, match="budget exhausted"):
            advisor.request_advice("second aspect", {})

    retrieve()
    retrieve()
    assert calls == [1, 1]


def test_scope_terms_must_be_verbatim_from_question(monkeypatch):
    monkeypatch.setattr(advisor, "request_advice", lambda *args: {
        "status": "insufficient", "missing_terms": ["办理流程"], "scope_terms": ["问题没有这个限定"],
    })
    assert advisor.review_coverage("申请如何办理", ["办理规定"])[1] == "rule_fallback"


def test_unindexed_adjacent_context_is_not_skipped_to_farther_chunk():
    anchor = chunk("a", "参见下文。")
    missing = chunk("missing", "未成功索引的原文", status="failed")
    farther = chunk("farther", "其他段落")
    assert resolved(anchor, [missing, farther]) == []
