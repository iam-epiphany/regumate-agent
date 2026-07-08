from backend.app.services.rerank_service import RerankedChunk
from backend.app.services.retrieval_service import retrieve_citations
from backend.app.services.vector_store_service import VectorSearchResult


def candidate() -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id="DOC-TEST-0001-CHUNK-0001",
        document_id="DOC-TEST-0001",
        filename="rules.txt",
        section_title="资产合计",
        page_number=1,
        text="资产合计应等于资产分项金额合计。",
        embedding_text="章节：资产合计\n\n资产合计应等于资产分项金额合计。",
        token_count=18,
        score=0.7,
        chunk_type="paragraph",
    )


def test_retrieve_citations_runs_hybrid_search_and_rerank(monkeypatch) -> None:
    calls = []
    item = candidate()

    monkeypatch.setattr("backend.app.services.retrieval_service.embed_query", lambda question: "query-vector")

    def fake_hybrid_search(query_embedding, *, limit: int):
        calls.append((query_embedding, limit))
        return [item]

    monkeypatch.setattr("backend.app.services.retrieval_service.hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(
        "backend.app.services.retrieval_service.rerank_candidates",
        lambda question, candidates, limit: [RerankedChunk(candidate=candidates[0], rerank_score=0.91)],
    )

    matches = retrieve_citations("资产合计怎么计算")

    assert calls == [("query-vector", 30)]
    assert matches[0].citation.chunk_id == "DOC-TEST-0001-CHUNK-0001"
    assert matches[0].rerank_score == 0.91
    assert matches[0].evidence_role == "direct_evidence"


def test_retrieve_citations_filters_low_rerank_scores(monkeypatch) -> None:
    item = candidate()

    monkeypatch.setattr("backend.app.services.retrieval_service.embed_query", lambda question: "query-vector")
    monkeypatch.setattr("backend.app.services.retrieval_service.hybrid_search", lambda query_embedding, *, limit: [item])
    monkeypatch.setattr(
        "backend.app.services.retrieval_service.rerank_candidates",
        lambda question, candidates, limit: [RerankedChunk(candidate=candidates[0], rerank_score=-0.2)],
    )

    assert retrieve_citations("资产合计怎么计算") == []


def test_retrieve_citations_prefers_direct_paragraph_over_neighbor_table(monkeypatch) -> None:
    paragraph = VectorSearchResult(
        chunk_id="DOC-TEST-0001-CHUNK-0003",
        document_id="DOC-TEST-0001",
        filename="rules.md",
        section_title="普惠小微贷款统计口径",
        page_number=None,
        text=(
            "普惠小微贷款统计应同时满足客户范围、贷款用途和金额口径三类条件。"
            "同一借款人在同一机构存在多笔贷款时，应按借款人维度汇总判断余额口径。"
            "若借款人部分贷款符合用途要求、部分贷款不符合用途要求，应仅统计符合用途要求的贷款余额。"
        ),
        embedding_text="章节：普惠小微贷款统计口径\n\n同一借款人在同一机构存在多笔贷款时，应按借款人维度汇总判断余额口径。",
        token_count=120,
        score=0.7,
        chunk_type="paragraph",
    )
    table = VectorSearchResult(
        chunk_id="DOC-TEST-0001-CHUNK-0005",
        document_id="DOC-TEST-0001",
        filename="rules.md",
        section_title="不应纳入普惠小微统计的情形",
        page_number=None,
        text="表格：\n| 情形 | 处理口径 |\n| --- | --- |\n| 借款用途为个人住房装修 | 不纳入普惠小微贷款 |",
        embedding_text="章节：不应纳入普惠小微统计的情形\n内容类型：表格\n\n表格：\n| 情形 | 处理口径 |",
        token_count=80,
        score=0.9,
        chunk_type="table",
    )

    monkeypatch.setattr("backend.app.services.retrieval_service.embed_query", lambda question: "query-vector")
    monkeypatch.setattr("backend.app.services.retrieval_service.hybrid_search", lambda query_embedding, *, limit: [table, paragraph])
    monkeypatch.setattr(
        "backend.app.services.retrieval_service.rerank_candidates",
        lambda question, candidates, limit: [
            RerankedChunk(candidate=table, rerank_score=0.96),
            RerankedChunk(candidate=paragraph, rerank_score=0.9),
        ],
    )

    matches = retrieve_citations("同一个借款人有多笔贷款时，普惠小微贷款余额应该怎么统计？")

    assert matches[0].citation.chunk_id == "DOC-TEST-0001-CHUNK-0003"
    assert matches[0].evidence_role == "direct_evidence"
    assert all(match.citation.evidence_role != "table_context" for match in matches)


def test_retrieve_citations_filters_high_rerank_low_coverage(monkeypatch) -> None:
    item = VectorSearchResult(
        chunk_id="DOC-TEST-0001-CHUNK-0010",
        document_id="DOC-TEST-0001",
        filename="rules.md",
        section_title="绿色信贷标识",
        page_number=None,
        text="绿色信贷标识应基于贷款资金投向、项目性质和支持材料确定。",
        embedding_text="章节：绿色信贷标识\n\n绿色信贷标识应基于贷款资金投向、项目性质和支持材料确定。",
        token_count=50,
        score=0.9,
        chunk_type="paragraph",
    )

    monkeypatch.setattr("backend.app.services.retrieval_service.embed_query", lambda question: "query-vector")
    monkeypatch.setattr("backend.app.services.retrieval_service.hybrid_search", lambda query_embedding, *, limit: [item])
    monkeypatch.setattr(
        "backend.app.services.retrieval_service.rerank_candidates",
        lambda question, candidates, limit: [RerankedChunk(candidate=item, rerank_score=1.0)],
    )

    assert retrieve_citations("同一个借款人有多笔贷款时，普惠小微贷款余额应该怎么统计？") == []
