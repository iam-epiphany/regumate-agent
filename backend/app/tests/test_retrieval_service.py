from backend.app.services.rerank_service import RerankedChunk
from backend.app.services.retrieval_service import (
    get_last_retrieval_diagnostics,
    retrieval_queries,
    retrieve_citations,
)
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


def fake_embed_texts(monkeypatch):
    calls = []

    def _fake(texts: list[str]):
        calls.append(texts)
        return [f"query-vector-{index}" for index, _text in enumerate(texts)]

    monkeypatch.setattr("backend.app.services.retrieval_service.embed_texts", _fake)
    return calls


def test_retrieve_citations_runs_hybrid_search_and_rerank(monkeypatch) -> None:
    calls = []
    item = candidate()

    embed_calls = fake_embed_texts(monkeypatch)

    def fake_hybrid_search(query_embedding, *, limit: int):
        calls.append((query_embedding, limit))
        return [item]

    monkeypatch.setattr("backend.app.services.retrieval_service.hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(
        "backend.app.services.retrieval_service.rerank_candidates",
        lambda question, candidates, limit: [RerankedChunk(candidate=candidates[0], rerank_score=0.91)],
    )

    matches = retrieve_citations("资产合计怎么计算")

    assert embed_calls == [["资产合计怎么计算"]]
    assert calls == [("query-vector-0", 50)]
    assert matches[0].citation.chunk_id == "DOC-TEST-0001-CHUNK-0001"
    assert matches[0].rerank_score == 0.91
    assert matches[0].evidence_role == "direct_evidence"
    diagnostics = get_last_retrieval_diagnostics()
    assert diagnostics.query_count == 1
    assert diagnostics.candidate_count == 1
    assert diagnostics.reranked_count == 1


def test_retrieve_citations_filters_low_rerank_scores(monkeypatch) -> None:
    item = candidate()

    fake_embed_texts(monkeypatch)
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

    fake_embed_texts(monkeypatch)
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

    fake_embed_texts(monkeypatch)
    monkeypatch.setattr("backend.app.services.retrieval_service.hybrid_search", lambda query_embedding, *, limit: [item])
    monkeypatch.setattr(
        "backend.app.services.retrieval_service.rerank_candidates",
        lambda question, candidates, limit: [RerankedChunk(candidate=item, rerank_score=1.0)],
    )

    assert retrieve_citations("同一个借款人有多笔贷款时，普惠小微贷款余额应该怎么统计？") == []


def test_retrieve_citations_accepts_table_evidence_for_comparison_question(monkeypatch) -> None:
    table = VectorSearchResult(
        chunk_id="DOC-TEST-0001-CHUNK-0020",
        document_id="DOC-TEST-0001",
        filename="rules.md",
        section_title="逾期贷款与不良贷款区别",
        page_number=None,
        text=(
            "表格行证据：比较项目为“逾期贷款与不良贷款”时，区别为“逾期统计强调时间状态，"
            "不良贷款或风险分类强调还款能力和资产质量，两者不能直接等同”。"
        ),
        embedding_text=(
            "章节：逾期贷款与不良贷款区别\n内容类型：表格\n内容形态：表格行级证据\n\n"
            "表格行证据：比较项目为“逾期贷款与不良贷款”时，区别为“逾期统计强调时间状态，"
            "不良贷款或风险分类强调还款能力和资产质量，两者不能直接等同”。"
        ),
        token_count=80,
        score=0.88,
        chunk_type="table",
    )

    fake_embed_texts(monkeypatch)
    monkeypatch.setattr("backend.app.services.retrieval_service.hybrid_search", lambda query_embedding, *, limit: [table])
    monkeypatch.setattr(
        "backend.app.services.retrieval_service.rerank_candidates",
        lambda question, candidates, limit: [RerankedChunk(candidate=table, rerank_score=0.94)],
    )

    matches = retrieve_citations("逾期与不良的区别")

    assert len(matches) == 1
    assert matches[0].citation.chunk_id == "DOC-TEST-0001-CHUNK-0020"
    assert matches[0].evidence_role == "table_evidence"


def test_retrieval_queries_decompose_asset_difference_foreign_currency_question() -> None:
    queries = retrieval_queries("资产合计差异应该优先排查哪些问题？如果差异来自外币折算，需要保留什么依据？")

    assert "资产合计 差异 优先检查 币种折算 四舍五入 科目映射 重复汇总" in queries
    assert "外币折算 保留 汇率日期 折算规则 原币金额来源" in queries


def test_retrieve_citations_batches_expanded_query_embeddings(monkeypatch) -> None:
    item = VectorSearchResult(
        chunk_id="DOC-TEST-0001-CHUNK-0030",
        document_id="DOC-TEST-0001",
        filename="rules.md",
        section_title="资产合计差异处理",
        page_number=None,
        text=(
            "资产合计差异应优先检查币种折算、四舍五入、科目映射和重复汇总。"
            "若差异来自外币折算，应保留汇率日期、折算规则和原币金额来源。"
        ),
        embedding_text=(
            "章节：资产合计差异处理\n\n资产合计差异应优先检查币种折算、四舍五入、科目映射和重复汇总。"
            "若差异来自外币折算，应保留汇率日期、折算规则和原币金额来源。"
        ),
        token_count=100,
        score=0.9,
        chunk_type="paragraph",
    )
    embed_calls = fake_embed_texts(monkeypatch)
    hybrid_calls = []

    def fake_hybrid_search(query_embedding, *, limit: int):
        hybrid_calls.append((query_embedding, limit))
        return [item]

    monkeypatch.setattr("backend.app.services.retrieval_service.hybrid_search", fake_hybrid_search)
    monkeypatch.setattr(
        "backend.app.services.retrieval_service.rerank_candidates",
        lambda question, candidates, limit: [RerankedChunk(candidate=candidates[0], rerank_score=0.93)],
    )

    matches = retrieve_citations("资产合计差异应该优先排查哪些问题？如果差异来自外币折算，需要保留什么依据？")

    assert matches
    assert len(embed_calls) == 1
    assert len(embed_calls[0]) > 1
    assert len(hybrid_calls) == len(embed_calls[0])
    diagnostics = get_last_retrieval_diagnostics()
    assert diagnostics.query_count == len(embed_calls[0])
    assert diagnostics.raw_candidate_count == len(embed_calls[0])
    assert diagnostics.candidate_count == 1


def test_retrieve_citations_limits_candidates_before_rerank(monkeypatch) -> None:
    items = [
        VectorSearchResult(
            chunk_id=f"DOC-TEST-0001-CHUNK-{index:04d}",
            document_id="DOC-TEST-0001",
            filename="quality_rules.md",
            section_title="资产合计差异处理",
            page_number=None,
            text=(
                "资产合计差异应优先检查币种折算、四舍五入、科目映射和重复汇总问题。"
                "若差异来自外币折算，应保留汇率日期、折算规则和原币金额来源。"
            ),
            embedding_text=(
                "章节：资产合计差异处理\n\n"
                "资产合计差异应优先检查币种折算、四舍五入、科目映射和重复汇总问题。"
                "若差异来自外币折算，应保留汇率日期、折算规则和原币金额来源。"
            ),
            token_count=100,
            score=1.0 - index * 0.001,
            chunk_type="paragraph",
        )
        for index in range(40)
    ]
    rerank_inputs = []

    fake_embed_texts(monkeypatch)
    monkeypatch.setattr("backend.app.services.retrieval_service.hybrid_search", lambda query_embedding, *, limit: items)

    def fake_rerank_candidates(question, candidates, limit):
        rerank_inputs.append(candidates)
        return [
            RerankedChunk(candidate=candidate, rerank_score=0.95 - index * 0.001)
            for index, candidate in enumerate(candidates[:limit])
        ]

    monkeypatch.setattr("backend.app.services.retrieval_service.rerank_candidates", fake_rerank_candidates)

    matches = retrieve_citations("资产合计差异应该优先排查哪些问题？如果差异来自外币折算，需要保留什么依据？")

    assert matches
    assert len(rerank_inputs) == 1
    assert len(rerank_inputs[0]) == 24
    assert [item.chunk_id for item in rerank_inputs[0]] == [item.chunk_id for item in items[:24]]
    diagnostics = get_last_retrieval_diagnostics()
    assert diagnostics.query_count > 1
    assert diagnostics.raw_candidate_count == 40 * diagnostics.query_count
    assert diagnostics.candidate_count == 40
    assert diagnostics.rerank_input_count == 24
    assert diagnostics.reranked_count == 20
