import json

from backend.app.schemas.qa import Citation
from backend.app.services.structure_locator_service import (
    build_structure_locator,
    chinese_number,
    locator_from_citation,
    match_structure,
    structure_prefix_match,
)


def _locator(path: list[str], title: str = "消费金融公司管理办法") -> dict:
    return build_structure_locator(
        document_id="DOC-1",
        document_title=title,
        section_path=path,
        section_title=path[-1] if path else None,
    )


def test_chinese_number_handles_regulatory_numerals() -> None:
    assert chinese_number("十") == 10
    assert chinese_number("二十一") == 21
    assert chinese_number("一百零五") == 105
    assert chinese_number("2024") == 2024
    assert chinese_number("第十条") is None


def test_locator_resolves_chapter_and_article() -> None:
    locator = _locator(["第二章 监督管理", "第十条 业务规则"])

    assert locator["resolution_status"] == "resolved"
    assert [level["kind"] for level in locator["path"]] == ["chapter", "article"]
    assert locator["path"][1]["number"] == 10
    assert "第十条" in locator["display"]
    assert locator["key"].endswith("chapter:2|article:10")


def test_locator_keeps_article_suffix_distinct() -> None:
    locator = build_structure_locator(
        document_id="DOC-1",
        document_title="某办法",
        section_path=["第十条之一 特别规定"],
        section_title="第十条之一 特别规定",
    )

    assert locator["path"][0]["number"] == 10
    assert locator["path"][0]["suffix"] == "一"
    assert locator["path"][0]["label"] == "第十条之一"
    assert "article:10-一" in locator["key"]


def test_leading_article_in_chunk_text_sets_coordinate() -> None:
    locator = build_structure_locator(
        document_id="DOC-1",
        document_title="某办法",
        text="第十条 商业银行应当满足资本充足率要求。",
    )

    assert locator["resolution_status"] == "resolved"
    assert locator["path"][0]["kind"] == "article"
    assert locator["path"][0]["number"] == 10


def test_reference_text_does_not_become_the_chunk_coordinate() -> None:
    locator = build_structure_locator(
        document_id="DOC-1",
        document_title="某办法",
        text="根据第十条规定，商业银行应当满足资本充足率要求。",
    )

    assert locator["resolution_status"] == "unresolved"
    assert all(level["kind"] != "article" for level in locator["path"])


def test_strict_and_ancestor_modes_disagree_when_target_level_is_missing() -> None:
    gold = _locator(["第二章 监督管理", "第十条 业务规则"])
    chapter_only = _locator(["第二章 监督管理"])

    ancestor = match_structure(gold, chapter_only, depth="article", mode="ancestor")
    strict = match_structure(gold, chapter_only, depth="article", mode="strict")

    assert ancestor["matched"] is True
    assert "article" in ancestor["implicit_levels"]
    assert strict["matched"] is False
    assert strict["reason"] == "level_unavailable"


def test_document_mismatch_blocks_even_ancestor_match() -> None:
    gold = _locator(["第十条 业务规则"], title="办法甲")
    other = _locator(["第十条 业务规则"], title="办法乙")

    result = match_structure(gold, other, depth="article")

    assert result["matched"] is False
    assert result["reason"] == "document_mismatch"


def test_year_prefixed_titles_stay_distinct() -> None:
    gold = _locator(["第十条 业务规则"], title="2024年监管办法")
    other = _locator(["第十条 业务规则"], title="2025年监管办法")

    result = match_structure(gold, other, depth="article")

    assert result["matched"] is False
    assert result["reason"] == "document_mismatch"


def test_article_mismatch_is_reported_at_the_level() -> None:
    gold = _locator(["第十条 业务规则"])
    other = _locator(["第十一条 业务规则"])

    result = match_structure(gold, other, depth="article")

    assert result["matched"] is False
    assert result["mismatch_level"] == "article"


def test_prefix_match_uses_document_and_article() -> None:
    gold = _locator(["第二章 监督管理", "第十条 业务规则"])
    retrieved = build_structure_locator(
        document_id="DOC-1",
        document_title="消费金融公司管理办法.pdf",
        section_path=["第十条 业务规则"],
        section_title="第十条 业务规则",
    )

    assert structure_prefix_match(gold, retrieved, depth="article") is True


def test_citation_fills_locator_and_label_without_extra_lookup() -> None:
    citation = Citation(
        document_id="DOC-1",
        chunk_id="CHUNK-1",
        filename="396_消费金融公司管理办法.doc",
        source_title="消费金融公司管理办法",
        section_title="第十条 业务规则",
        section_path=["第二章 监督管理", "第十条 业务规则"],
        section_number="第十条",
        excerpt="本办法所称消费贷款是指……",
    )

    assert citation.structure_locator is not None
    assert citation.structure_locator["resolution_status"] == "resolved"
    assert citation.structure_label == citation.structure_locator["display"]
    assert locator_from_citation(citation)["key"] == citation.structure_locator["key"]


def test_locator_is_json_serialisable() -> None:
    locator = _locator(["第三章 业务规则", "第二十一条 风险管理"])

    payload = json.dumps(locator, ensure_ascii=False)

    assert "第二十一条" in payload
