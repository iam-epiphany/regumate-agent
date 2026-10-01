"""Cross-document clause reference resolution ("《title》第X条" multi-hop)."""

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from backend.app.core.database import Base
from backend.app.models.document import Document, DocumentChunk, DocumentClause
from backend.app.services.document_structure_index_service import (
    rebuild_document_structure_index,
    resolve_document_ids_by_titles,
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


def _add_document(
    db_session, *, document_id: str, filename: str, title: str | None = None
) -> Document:
    document = Document(
        document_id=document_id,
        filename=filename,
        filename_norm=filename.casefold(),
        title=title,
        file_type="docx",
        size=100,
        storage_path=f"/tmp/{filename}",
    )
    db_session.add(document)
    db_session.flush()
    return document


def _add_clause_chunk(
    db_session,
    *,
    document_id: str,
    chunk_id: str,
    text: str,
    section_title: str,
) -> DocumentChunk:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        source_file=f"{document_id}.docx",
        section_title=section_title,
        index_status="indexed",
    )
    db_session.add(chunk)
    db_session.flush()
    return chunk


def test_resolve_document_ids_prefers_exact_then_shortest_filename(db_session) -> None:
    _add_document(
        db_session, document_id="DOC-MAIN", filename="商业银行资本管理办法.docx"
    )
    _add_document(
        db_session,
        document_id="DOC-ATTACH",
        filename="商业银行资本管理办法附件1.docx",
    )
    _add_document(
        db_session, document_id="DOC-OTHER", filename="流动性风险管理办法.docx"
    )

    resolved = resolve_document_ids_by_titles(db_session, ["商业银行资本管理办法"])
    assert list(resolved.values()) == ["DOC-MAIN"]

    resolved = resolve_document_ids_by_titles(
        db_session, ["资本管理办法", "流动性风险管理办法"]
    )
    assert set(resolved.values()) == {"DOC-MAIN", "DOC-OTHER"}

    assert resolve_document_ids_by_titles(db_session, ["不存在的办法"]) == {}


def test_resolve_document_ids_caps_result_count(db_session) -> None:
    for index in range(3):
        _add_document(
            db_session,
            document_id=f"DOC-CAP-{index}",
            filename=f"测试办法{index}.docx",
        )
    resolved = resolve_document_ids_by_titles(
        db_session, ["测试办法0", "测试办法1", "测试办法2"], max_documents=2
    )
    assert len(resolved) == 2


def test_cross_document_reference_resolves_via_clause_index(db_session) -> None:
    from backend.app.services.retrieval_support import (
        _cross_document_referenced_chunks,
        _DocumentChunkSnapshot,
    )

    document = _add_document(
        db_session,
        document_id="DOC-CAPITAL",
        filename="商业银行资本管理办法.docx",
        title="商业银行资本管理办法",
    )
    clause_chunk = _add_clause_chunk(
        db_session,
        document_id="DOC-CAPITAL",
        chunk_id="DOC-CAPITAL-CHUNK-0010",
        text="商业银行应当达到本办法规定的资本充足率监管要求。",
        section_title="第十条 监管要求",
    )
    rebuild_document_structure_index(db_session, document)

    target_snapshot = _DocumentChunkSnapshot(
        id=1,
        chunk_id=clause_chunk.chunk_id,
        document_id="DOC-CAPITAL",
        text=clause_chunk.text,
        embedding_text=clause_chunk.text,
        chunk_metadata=None,
        index_status="indexed",
        source_file=clause_chunk.source_file,
        page_number=None,
        section_title=clause_chunk.section_title,
    )
    anchor_text = "依据《商业银行资本管理办法》第10条的规定执行。"
    hits = _cross_document_referenced_chunks(
        db_session,
        anchor_text,
        document_chunk_cache={"DOC-CAPITAL": [target_snapshot]},
    )
    assert len(hits) == 1
    chunk, document_chunks = hits[0]
    assert chunk.chunk_id == "DOC-CAPITAL-CHUNK-0010"
    assert [c.chunk_id for c in document_chunks] == ["DOC-CAPITAL-CHUNK-0010"]


def test_cross_document_reference_falls_back_to_title_scan(db_session) -> None:
    """条款索引为空（未回填）时回退到条号字面扫描。"""
    from backend.app.services.retrieval_support import (
        _cross_document_referenced_chunks,
        _DocumentChunkSnapshot,
    )

    _add_document(
        db_session, document_id="DOC-CAPITAL", filename="商业银行资本管理办法.docx"
    )
    clause_chunk = _add_clause_chunk(
        db_session,
        document_id="DOC-CAPITAL",
        chunk_id="DOC-CAPITAL-CHUNK-0020",
        text="商业银行应当建立恢复与处置计划。",
        section_title="第二十五条 恢复计划",
    )
    target_snapshot = _DocumentChunkSnapshot(
        id=1,
        chunk_id=clause_chunk.chunk_id,
        document_id="DOC-CAPITAL",
        text=clause_chunk.text,
        embedding_text=clause_chunk.text,
        chunk_metadata=None,
        index_status="indexed",
        source_file=clause_chunk.source_file,
        page_number=None,
        section_title=clause_chunk.section_title,
    )
    anchor_text = "按照《商业银行资本管理办法》第二十五条执行。"
    hits = _cross_document_referenced_chunks(
        db_session,
        anchor_text,
        document_chunk_cache={"DOC-CAPITAL": [target_snapshot]},
    )
    assert [chunk.chunk_id for chunk, _chunks in hits] == ["DOC-CAPITAL-CHUNK-0020"]


def test_expand_neighbor_matches_appends_cross_document_clause(db_session) -> None:
    from backend.app.schemas.qa import Citation
    from backend.app.services.retrieval_service import RetrievalMatch
    from backend.app.services.retrieval_support import (
        _DocumentChunkSnapshot,
        _expand_neighbor_matches,
    )

    capital_document = _add_document(
        db_session, document_id="DOC-CAPITAL", filename="商业银行资本管理办法.docx"
    )
    clause_chunk = _add_clause_chunk(
        db_session,
        document_id="DOC-CAPITAL",
        chunk_id="DOC-CAPITAL-CHUNK-0010",
        text="商业银行应当达到本办法规定的资本充足率监管要求。",
        section_title="第十条 监管要求",
    )
    rebuild_document_structure_index(db_session, capital_document)
    anchor_text = "风险加权资产的计算依据《商业银行资本管理办法》第十条执行。"
    anchor_snapshot = _DocumentChunkSnapshot(
        id=2,
        chunk_id="DOC-RISK-CHUNK-0001",
        document_id="DOC-RISK",
        text=anchor_text,
        embedding_text=anchor_text,
        chunk_metadata=None,
        index_status="indexed",
        source_file="DOC-RISK.docx",
        page_number=None,
        section_title="第一条 计量规则",
    )
    capital_snapshot = _DocumentChunkSnapshot(
        id=1,
        chunk_id=clause_chunk.chunk_id,
        document_id="DOC-CAPITAL",
        text=clause_chunk.text,
        embedding_text=clause_chunk.text,
        chunk_metadata=None,
        index_status="indexed",
        source_file=clause_chunk.source_file,
        page_number=None,
        section_title=clause_chunk.section_title,
    )
    match = RetrievalMatch(
        citation=Citation(
            document_id="DOC-RISK",
            chunk_id="DOC-RISK-CHUNK-0001",
            filename="DOC-RISK.docx",
            section_title="第一条 计量规则",
            excerpt=anchor_text,
            score=0.9,
            rerank_score=0.9,
            evidence_role="bounded_lexical_support",
        ),
        score=0.9,
        rerank_score=0.9,
        coverage_score=0.5,
        evidence_role="bounded_lexical_support",
        evidence_text=anchor_text,
        metadata={"evidence_role": "bounded_lexical_support"},
    )
    expanded = _expand_neighbor_matches(
        db_session,
        "风险加权资产如何计算",
        [match],
        document_chunk_cache={
            "DOC-RISK": [anchor_snapshot],
            "DOC-CAPITAL": [capital_snapshot],
        },
    )
    cross_hits = [
        m
        for m in expanded
        if m.citation.chunk_id == "DOC-CAPITAL-CHUNK-0010"
        and m.metadata.get("expansion_reason") == "cross_document_referenced_clause"
    ]
    assert len(cross_hits) == 1
    assert cross_hits[0].citation.document_id == "DOC-CAPITAL"


def test_expand_neighbor_matches_respects_existing_budget(db_session) -> None:
    """锚点已有 2 个同文档邻居时不再追加跨文档条款。"""
    from backend.app.schemas.qa import Citation
    from backend.app.services.retrieval_service import RetrievalMatch
    from backend.app.services.retrieval_support import (
        _DocumentChunkSnapshot,
        _expand_neighbor_matches,
    )

    _add_document(
        db_session, document_id="DOC-CAPITAL", filename="商业银行资本管理办法.docx"
    )
    _add_clause_chunk(
        db_session,
        document_id="DOC-CAPITAL",
        chunk_id="DOC-CAPITAL-CHUNK-0010",
        text="商业银行应当达到本办法规定的资本充足率监管要求。",
        section_title="第十条 监管要求",
    )
    anchor_text = "风险加权资产的计算依据《商业银行资本管理办法》第十条执行，并参考上一条与下一条。"
    anchor_snapshot = _DocumentChunkSnapshot(
        id=2,
        chunk_id="DOC-RISK-CHUNK-0002",
        document_id="DOC-RISK",
        text=anchor_text,
        embedding_text=anchor_text,
        chunk_metadata=None,
        index_status="indexed",
        source_file="DOC-RISK.docx",
        page_number=None,
        section_title="第十条 计量规则",
    )
    neighbour_a = _DocumentChunkSnapshot(
        id=3,
        chunk_id="DOC-RISK-CHUNK-0001",
        document_id="DOC-RISK",
        text="上一条内容。",
        embedding_text="上一条内容。",
        chunk_metadata=None,
        index_status="indexed",
        source_file="DOC-RISK.docx",
        page_number=None,
        section_title="第九条 计量规则",
    )
    neighbour_b = _DocumentChunkSnapshot(
        id=4,
        chunk_id="DOC-RISK-CHUNK-0003",
        document_id="DOC-RISK",
        text="下一条内容。",
        embedding_text="下一条内容。",
        chunk_metadata=None,
        index_status="indexed",
        source_file="DOC-RISK.docx",
        page_number=None,
        section_title="第十一条 计量规则",
    )
    match = RetrievalMatch(
        citation=Citation(
            document_id="DOC-RISK",
            chunk_id="DOC-RISK-CHUNK-0002",
            filename="DOC-RISK.docx",
            section_title="第十条 计量规则",
            excerpt=anchor_text,
            score=0.9,
            rerank_score=0.9,
            evidence_role="bounded_lexical_support",
        ),
        score=0.9,
        rerank_score=0.9,
        coverage_score=0.5,
        evidence_role="bounded_lexical_support",
        evidence_text=anchor_text,
        metadata={"evidence_role": "bounded_lexical_support"},
    )
    expanded = _expand_neighbor_matches(
        db_session,
        "风险加权资产如何计算",
        [match],
        document_chunk_cache={
            "DOC-RISK": [neighbour_a, anchor_snapshot, neighbour_b],
            "DOC-CAPITAL": [],
        },
    )
    assert all(m.citation.document_id == "DOC-RISK" for m in expanded)


def test_document_clause_rows_written_for_all_documents(db_session) -> None:
    document = _add_document(
        db_session, document_id="DOC-CLAUSE", filename="条款测试办法.docx"
    )
    _add_clause_chunk(
        db_session,
        document_id="DOC-CLAUSE",
        chunk_id="DOC-CLAUSE-CHUNK-0001",
        text="总体要求。",
        section_title="第一条 总体要求",
    )
    _add_clause_chunk(
        db_session,
        document_id="DOC-CLAUSE",
        chunk_id="DOC-CLAUSE-CHUNK-0002",
        text="信息披露要求。",
        section_title="第一百零五条 信息披露",
    )
    definition_count, clause_count = rebuild_document_structure_index(
        db_session, document
    )
    assert definition_count == 0
    assert clause_count == 2
    numbers = list(
        db_session.scalars(
            select(DocumentClause.clause_number).order_by(DocumentClause.id)
        )
    )
    assert numbers == [1, 105]


def test_sibling_recovery_scope_includes_cited_documents(
    db_session, monkeypatch
) -> None:
    """证据引用《标题》而未被检索的文档进入兄弟文档补搜范围（不做全语料扫描）。"""
    from backend.app.schemas.qa import RetrievalResult
    from backend.app.services import rag_service
    from backend.app.services.query_planner_service import QueryAspect

    _add_document(db_session, document_id="DOC-RISK", filename="风险管理指引.docx")
    _add_document(
        db_session, document_id="DOC-CAPITAL", filename="商业银行资本管理办法.docx"
    )

    def _candidate(chunk_id: str, text: str) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=chunk_id,
            rank=1,
            score=0.9,
            source_doc="风险管理指引.docx",
            text=text,
            citation_label="[1]",
            metadata={"document_id": "DOC-RISK"},
        )

    aspects = [
        (
            "a1",
            "风险加权资产如何计算",
            "风险加权资产的计算依据《商业银行资本管理办法》第十条执行。",
        ),
        ("a2", "资本充足率监管要求是什么", "商业银行应当满足资本充足率要求。"),
    ]
    retrievals = [
        rag_service.AspectRetrieval(
            aspect=QueryAspect(
                aspect_id=aspect_id,
                question=question,
                search_queries=(),
                evidence_need="rule",
                keywords=(),
            ),
            candidates=[_candidate("DOC-RISK-CHUNK-0001", text)],
            diagnostics=[],
            citation_validation={},
            selected_chunk_ids=[],
            retrieval_covered=False,
            covered=False,
        )
        for aspect_id, question, text in aspects
    ]

    captured: dict[str, set[str]] = {}

    def fake_lexical(db, aspect, matches, *, document_ids, document_chunk_cache):
        captured.setdefault("document_ids", set()).update(document_ids)
        return []

    monkeypatch.setattr(
        rag_service, "_bounded_document_lexical_support_matches", fake_lexical
    )
    rag_service._recover_missing_aspects_from_sibling_documents(
        db_session, retrievals, document_chunk_cache={}
    )
    assert captured.get("document_ids") == {"DOC-CAPITAL"}
