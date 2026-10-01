import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from backend.app.core.database import Base
from backend.app.models.document import (
    Document,
    DocumentChunk,
    DocumentClause,
    DocumentDefinition,
)
from backend.app.services.document_structure_index_service import (
    clause_number_from_label,
    extract_definition_terms,
    lookup_definition_chunks,
    normalize_definition_term,
    rebuild_document_structure_index,
)


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    with session_factory() as session:
        yield session


def _add_document_with_chunks(
    db_session, *, document_id: str, chunks: list[dict]
) -> Document:
    document = Document(
        document_id=document_id,
        filename=f"{document_id}.docx",
        file_type="docx",
        size=100,
        storage_path=f"/tmp/{document_id}.docx",
    )
    db_session.add(document)
    db_session.flush()
    for index, item in enumerate(chunks, start=1):
        db_session.add(
            DocumentChunk(
                chunk_id=f"{document_id}-CHUNK-{index:04d}",
                document_id=document_id,
                text=item["text"],
                embedding_text=item.get("embedding_text"),
                source_file=f"{document_id}.docx",
                section_title=item.get("section_title"),
            )
        )
    db_session.flush()
    return document


def test_extract_definition_terms_covers_sentence_forms() -> None:
    rows = extract_definition_terms(
        "本办法所称贷款，是指经批准设立的机构发放的本金。"
        "核心一级资本充足率是指资本净额除以风险加权资产后的比率。"
        "（以下简称商业银行法）"
        "本办法所称消费金融公司是指不吸收公众存款的非银行金融机构。"
    )
    by_kind = {}
    for kind, term, _snippet in rows:
        by_kind.setdefault(kind, []).append(term)
    assert "贷款" in by_kind["所称"]
    assert "消费金融公司" in by_kind["所称"]
    assert "核心一级资本充足率" in by_kind["是指"]
    assert "商业银行法" in by_kind["简称"]


def test_extract_definition_terms_skips_prose_without_definitions() -> None:
    assert (
        extract_definition_terms("商业银行应当计提贷款损失准备。本条明确规定相关要求。")
        == []
    )


def test_clause_number_from_label_parses_chinese_and_digits() -> None:
    assert clause_number_from_label("四") == 4
    assert clause_number_from_label("十四") == 14
    assert clause_number_from_label("一百零五") == 105
    assert clause_number_from_label("105") == 105
    assert clause_number_from_label("廿") is None
    assert clause_number_from_label("") is None


def test_rebuild_is_idempotent_and_lookup_resolves_terms(db_session) -> None:
    document = _add_document_with_chunks(
        db_session,
        document_id="DOC-STRUCT-0001",
        chunks=[
            {
                "text": "本办法所称消费金融公司，是指不吸收公众存款的非银行金融机构。",
                "section_title": "第二条",
            },
            {"text": "商业银行应当按照规定计提准备。", "section_title": "第三条"},
            {
                "text": "资本净额是指扣除扣减项后的资本。",
                "section_title": "第一百零五条",
            },
        ],
    )
    first = rebuild_document_structure_index(db_session, document)
    second = rebuild_document_structure_index(db_session, document)
    assert first == second
    definitions = list(db_session.scalars(select(DocumentDefinition)))
    clauses = list(db_session.scalars(select(DocumentClause)))
    assert len(definitions) == 2
    assert len(clauses) == 3
    assert {clause.clause_number for clause in clauses} == {2, 3, 105}

    hits = lookup_definition_chunks(db_session, ["消费金融公司"])
    assert len(hits) == 1
    chunk, pattern = hits[0]
    assert chunk.section_title == "第二条"
    assert pattern == "所称消费金融公司"

    # Normalized lookup ignores case and full/half width parens differences.
    assert lookup_definition_chunks(db_session, ["消费金融公司（试行）"]) == []
    hits = lookup_definition_chunks(db_session, ["资本净额"])
    assert [pattern for _chunk, pattern in hits] == ["资本净额是指"]


def test_lookup_scopes_by_document_ids(db_session) -> None:
    for document_id in ("DOC-STRUCT-0001", "DOC-STRUCT-0002"):
        document = _add_document_with_chunks(
            db_session,
            document_id=document_id,
            chunks=[{"text": "测试机构是指用于验证检索范围划分的机构。"}],
        )
        rebuild_document_structure_index(db_session, document)
    for document_id in ("DOC-STRUCT-0001", "DOC-STRUCT-0002"):
        hits = lookup_definition_chunks(
            db_session, ["测试机构"], document_ids={document_id}
        )
        assert len(hits) == 1
        assert hits[0][0].document_id == document_id


def test_normalize_definition_term_unifies_width_and_case() -> None:
    assert normalize_definition_term("资本管理办法") == normalize_definition_term(
        "资本管理办法"
    )
    assert normalize_definition_term("资本（试行）办法") == "资本(试行)办法"


def test_clause_index_from_text_start_beats_title_heuristic(db_session) -> None:
    """正文条款块（text 以“第X条”开头）进入条款索引；章标题不误报。"""
    document = _add_document_with_chunks(
        db_session,
        document_id="DOC-CLAUSE-TEXT",
        chunks=[
            {
                "text": "第五条 保险期限一年及以下的意外险保单最低现金价值按照未满期净保费计算。",
                "section_title": "二、现金价值",
            },
            {
                "text": "保险公司应于每年末开展意外险业务回溯工作。",
                "section_title": "第十二条 回溯",  # 章标题含条号的真实形态也应命中
            },
            {"text": "第五条 这一条与第一条共用块去重校验。", "section_title": None},
        ],
    )
    _definition_count, clause_count = rebuild_document_structure_index(
        db_session, document
    )
    numbers = list(
        db_session.scalars(
            select(DocumentClause.clause_number).order_by(DocumentClause.id)
        )
    )
    assert clause_count == len(numbers)
    # 两个“第五条”开头块各占一行；同一块内 text-start 与 title 不重复
    assert numbers == [5, 12, 5]
