import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from backend.app.services import reference_resolver_service as resolver
from backend.app.services import retrieval_support as support
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
    monkeypatch.setattr(support, "recognize_references", unexpected)
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














def test_unindexed_adjacent_context_is_not_skipped_to_farther_chunk():
    anchor = chunk("a", "参见下文。")
    missing = chunk("missing", "未成功索引的原文", status="failed")
    farther = chunk("farther", "其他段落")
    assert resolved(anchor, [missing, farther]) == []
