from dataclasses import replace

import pytest

from backend.app.services.rerank_service import RerankedChunk
from backend.app.services.retrieval_service import (
    get_last_retrieval_diagnostics,
    limit_rerank_candidates,
    retrieval_queries,
    retrieve_citations,
)
from backend.app.services.vector_store_service import VectorSearchResult


@pytest.fixture(autouse=True)
def keep_synthetic_candidates_active(monkeypatch):
    monkeypatch.setattr(
        "backend.app.services.retrieval_service.filter_active_candidates",
        lambda candidates: candidates,
    )


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


def test_limit_rerank_candidates_can_preserve_rrf_order(monkeypatch) -> None:
    monkeypatch.setattr("backend.app.services.retrieval_service.RERANK_CANDIDATE_LIMIT", 2)
    items = [
        replace(candidate(), chunk_id="rrf-first", score=0.2),
        replace(candidate(), chunk_id="rrf-second", score=0.3),
        replace(candidate(), chunk_id="raw-high", score=0.99),
    ]

    selected = limit_rerank_candidates(items, preserve_order=True)

    assert [item.chunk_id for item in selected] == ["rrf-first", "rrf-second"]


def test_limit_rerank_candidates_accepts_effective_limit(monkeypatch) -> None:
    monkeypatch.setattr("backend.app.services.retrieval_service.RERANK_CANDIDATE_LIMIT", 24)
    items = [
        replace(candidate(), chunk_id="rrf-first", score=0.2),
        replace(candidate(), chunk_id="rrf-second", score=0.3),
        replace(candidate(), chunk_id="raw-high", score=0.99),
    ]

    selected = limit_rerank_candidates(items, preserve_order=True, limit=2)

    assert [item.chunk_id for item in selected] == ["rrf-first", "rrf-second"]


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


def _coverage_candidate(chunk_id: str, document_id: str) -> "object":
    from backend.app.services.vector_store_service import VectorSearchResult

    return VectorSearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        filename=f"{document_id}.doc",
        section_title=None,
        page_number=None,
        text="证据文本",
        embedding_text="证据文本",
        token_count=5,
        score=1.0,
    )


def test_select_with_document_coverage_keeps_secondary_source() -> None:
    """多文件相关时，次要文件至少 1 个进入最终证据（不被高分文件挤掉）。"""
    from backend.app.services.retrieval_service import (
        _AnnotatedChunk,
        _select_with_document_coverage,
    )

    items = [
        _AnnotatedChunk(candidate=_coverage_candidate(f"A{i}", "DOC-A"), rerank_score=0.9, coverage_score=0.5, evidence_role="direct_evidence")
        for i in range(8)
    ] + [
        _AnnotatedChunk(candidate=_coverage_candidate(f"B{i}", "DOC-B"), rerank_score=0.8, coverage_score=0.5, evidence_role="direct_evidence")
        for i in range(4)
    ]
    selected = _select_with_document_coverage(items, limit=6)
    assert len(selected) == 6
    docs = {item.candidate.document_id for item in selected}
    assert "DOC-A" in docs and "DOC-B" in docs
    # 第一轮每文件最高分 1 个（A0/B0），随后按序回填至 limit。
    assert [item.candidate.chunk_id for item in selected] == ["A0", "B0", "A1", "A2", "A3", "A4"]


def test_select_with_document_coverage_does_not_trim_single_file() -> None:
    """单文件问题不受影响：回填后仍满额且顺序不变。"""
    from backend.app.services.retrieval_service import (
        _AnnotatedChunk,
        _select_with_document_coverage,
    )

    items = [
        _AnnotatedChunk(candidate=_coverage_candidate(f"A{i}", "DOC-A"), rerank_score=0.9, coverage_score=0.5, evidence_role="direct_evidence")
        for i in range(12)
    ]
    selected = _select_with_document_coverage(items, limit=6)
    assert [item.candidate.chunk_id for item in selected] == [f"A{i}" for i in range(6)]


def test_rerank_input_with_document_coverage_keeps_secondary_source() -> None:
    """rerank 输入阶段文件覆盖：次要文件的最高分候选进入重排。"""
    from backend.app.services.retrieval_service import rerank_input_with_document_coverage

    candidates = [
        _coverage_candidate(f"A{i}", "DOC-A")
        for i in range(10)
    ] + [
        _coverage_candidate(f"B{i}", "DOC-B")
        for i in range(3)
    ]
    bounded = rerank_input_with_document_coverage(candidates, limit=8)
    assert len(bounded) == 8
    docs = {candidate.document_id for candidate in bounded}
    assert "DOC-A" in docs and "DOC-B" in docs


# ---- 阶段二：CRAG-lite（意图缺口补充 + 列表完整性） ----

def _fake_db_for_support() -> object:
    """db 仅为占位：所有文档必须由 document_chunk_cache 覆盖。"""

    class _FakeDb:
        def scalars(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("document_chunk_cache should cover all documents")

    return _FakeDb()


def _intent_gap_aspect():
    from backend.app.services.query_planner_service import QueryAspect, QuerySearchQuery

    return QueryAspect(
        aspect_id="derivatives_exemption",
        question="哪些非集中清算衍生品交易可以不适用初始保证金要求？",
        search_queries=(
            QuerySearchQuery("非集中清算衍生品交易 初始保证金 豁免", "keyword_anchor", "术语兜底"),
        ),
        evidence_need="非集中清算衍生品交易不适用初始保证金要求的豁免情形",
        keywords=("非集中清算衍生品交易", "初始保证金要求"),
    )


def _snapshot_chunk(
    chunk_id: str, document_id: str, text: str, section_title: str = "第二章 保证金要求"
):
    from backend.app.services.retrieval_support import _DocumentChunkSnapshot

    return _DocumentChunkSnapshot(
        id=1,
        chunk_id=chunk_id,
        document_id=document_id,
        text=text,
        embedding_text=f"章节：{section_title}\n\n{text}",
        chunk_metadata=None,
        index_status="indexed",
        source_file="办法.md",
        page_number=None,
        section_title=section_title,
    )


def _base_match(evidence_text: str = "担保品应当满足集中清算要求。"):
    from backend.app.schemas.qa import Citation
    from backend.app.services.retrieval_service import RetrievalMatch

    return RetrievalMatch(
        citation=Citation(
            document_id="DOC-1",
            chunk_id="DOC-1-CHUNK-0001",
            filename="办法.md",
            section_title="第四章 风险管理",
            page_number=None,
            excerpt=evidence_text,
            score=0.9,
            rerank_score=0.9,
            chunk_type="paragraph",
            evidence_role="direct_evidence",
        ),
        score=0.9,
        rerank_score=0.9,
        coverage_score=1.0,
        evidence_role="direct_evidence",
        evidence_text=evidence_text,
    )


def test_enumeration_clause_detector_styles() -> None:
    """列表条款检测：中文序号、数字序号、圈号均识别；比率数字不误判。"""
    from backend.app.services.retrieval_support import _looks_like_enumeration_clause

    assert _looks_like_enumeration_clause(
        "（一）甲应当报经批准；（二）乙应当报经批准；（三）丙应当报经批准。"
    )
    assert _looks_like_enumeration_clause(
        "(一)甲应当报经批准；(二)乙应当报经批准；(三)丙应当报经批准。"
    )
    assert _looks_like_enumeration_clause(
        "1．甲应当报经批准；2．乙应当报经批准；3．丙应当报经批准。"
    )
    assert _looks_like_enumeration_clause("①甲应当报经批准；②乙应当报经批准；③丙应当报经批准。")
    assert not _looks_like_enumeration_clause(
        "存款利率1.5%，贷款利率2.8%，净息差1.2%，加权平均收益率2.6%。"
    )
    assert not _looks_like_enumeration_clause("本条规定了最低资本要求的计算口径，与附件二保持一致。")


def test_intent_gap_extracts_quoted_terms() -> None:
    """回归：heredoc 转义曾破坏引号正则，引号术语必须能被提取并参与缺口判定。"""
    from backend.app.services.retrieval_support import _intent_gap_supplement

    aspect = replace(_intent_gap_aspect(), question="哪些“实物结算”衍生品交易可以不适用初始保证金要求？")
    matches = [_base_match(evidence_text="担保品应当满足集中清算要求。")]
    supplements = _intent_gap_supplement(
        _fake_db_for_support(),
        aspect,
        matches,
        document_chunk_cache={
            "DOC-1": [
                _snapshot_chunk(
                    "DOC-1-CHUNK-0002",
                    "DOC-1",
                    "实物结算的非集中清算衍生品交易不适用初始保证金要求。",
                )
            ]
        },
    )
    assert len(supplements) == 1
    assert supplements[0].citation.chunk_id == "DOC-1-CHUNK-0002"
    assert supplements[0].metadata.get("evidence_role") == "intent_gap_support"
    assert supplements[0].metadata.get("intent_gap_term") in ("初始保证金要求", "实物结算")


def test_intent_gap_prohibition_adds_institution_clause() -> None:
    """禁止性问题额外构造“{机构}不得”模式。"""
    from backend.app.services.retrieval_support import _intent_gap_supplement

    aspect = replace(
        _intent_gap_aspect(),
        question="消费金融公司防范欺诈风险有什么禁止性规定？",
        keywords=("消费金融公司", "欺诈风险"),
    )
    matches = [_base_match(evidence_text="担保品应当满足集中清算要求。")]
    supplements = _intent_gap_supplement(
        _fake_db_for_support(),
        aspect,
        matches,
        document_chunk_cache={
            "DOC-1": [
                _snapshot_chunk(
                    "DOC-1-CHUNK-0003",
                    "DOC-1",
                    "消费金融公司不得参与欺诈相关的资金往来活动。",
                )
            ]
        },
    )
    assert len(supplements) == 1
    assert supplements[0].metadata.get("intent_gap_term") in ("消费金融公司", "消费金融公司不得")


def test_list_completeness_promotes_enumeration_chunk() -> None:
    """列表型问题：含主题词的枚举条款即使未被召回也应被补充。"""
    from backend.app.services.retrieval_support import _list_completeness_supplement

    aspect = _intent_gap_aspect()
    matches = [_base_match(evidence_text="担保品应当满足集中清算要求。")]
    supplements = _list_completeness_supplement(
        _fake_db_for_support(),
        aspect,
        matches,
        document_chunk_cache={
            "DOC-1": [
                _snapshot_chunk(
                    "DOC-1-CHUNK-0004",
                    "DOC-1",
                    "下列非集中清算衍生品交易可以不适用初始保证金要求："
                    "（一）实物结算的交易；（二）单一交易对手的交易；（三）集团内部交易。",
                ),
                _snapshot_chunk(
                    "DOC-1-CHUNK-0005",
                    "DOC-1",
                    "担保品应当满足集中清算要求，防止重复使用。",
                ),
            ]
        },
    )
    assert len(supplements) == 1
    assert supplements[0].citation.chunk_id == "DOC-1-CHUNK-0004"
    assert supplements[0].metadata.get("evidence_role") == "list_completeness_support"


def test_list_completeness_skips_non_list_questions() -> None:
    """非列表型问题不触发列表补充。"""
    from backend.app.services.retrieval_support import _list_completeness_supplement

    aspect = replace(_intent_gap_aspect(), question="担保品的估值应如何处理？")
    matches = [_base_match(evidence_text="担保品应当满足集中清算要求。")]
    supplements = _list_completeness_supplement(
        _fake_db_for_support(),
        aspect,
        matches,
        document_chunk_cache={
            "DOC-1": [
                _snapshot_chunk(
                    "DOC-1-CHUNK-0006",
                    "DOC-1",
                    "下列情形之一应当调整估值：（一）市场波动；（二）信用变化；（三）其他。",
                )
            ]
        },
    )
    terms = [s.metadata.get("intent_gap_term") for s in supplements]
    assert "不符合" not in terms


def test_intent_gap_fallback_clause_for_unclassified_assets() -> None:
    """“不符合…标准的资产如何处理”由兜底条款回答（含否定词+处理动词）。"""
    from backend.app.services.retrieval_support import _intent_gap_supplement

    aspect = replace(
        _intent_gap_aspect(),
        question="商业银行采用内部评级法时，不符合各类风险暴露分类标准的资产应如何处理？",
        keywords=("内部评级法", "风险暴露", "分类标准", "不符合", "处理", "监管资本计量"),
    )
    matches = [_base_match(evidence_text="商业银行应建立内部评级法风险暴露分类和调整的报告制度。")]
    supplements = _intent_gap_supplement(
        _fake_db_for_support(),
        aspect,
        matches,
        document_chunk_cache={
            "DOC-1": [
                _snapshot_chunk(
                    "DOC-1-CHUNK-0007",
                    "DOC-1",
                    "对不符合主权风险暴露、金融机构风险暴露、零售风险暴露、股权风险暴露、"
                    "其他风险暴露划分标准且存在信用风险的资产，应纳入公司风险暴露处理。",
                ),
                _snapshot_chunk(
                    "DOC-1-CHUNK-0008",
                    "DOC-1",
                    "本附件规定了内部评级法风险暴露分类标准的适用范围。",
                ),
            ]
        },
    )
    chunk_ids = [s.citation.chunk_id for s in supplements]
    assert "DOC-1-CHUNK-0007" in chunk_ids
    terms = {s.metadata.get("intent_gap_term") for s in supplements}
    assert "不符合" in terms
    assert all(s.metadata.get("evidence_role") == "intent_gap_support" for s in supplements)


def test_intent_gap_fallback_requires_treatment_verb() -> None:
    """兜底条款必须同时含处理动词，避免把一般性“不符合”表述误注入。"""
    from backend.app.services.retrieval_support import _intent_gap_supplement

    aspect = replace(
        _intent_gap_aspect(),
        question="商业银行采用内部评级法时，不符合各类风险暴露分类标准的资产应如何处理？",
        keywords=("内部评级法", "风险暴露", "分类标准", "不符合", "处理", "监管资本计量"),
    )
    matches = [_base_match(evidence_text="商业银行应建立内部评级法风险暴露分类和调整的报告制度。")]
    supplements = _intent_gap_supplement(
        _fake_db_for_support(),
        aspect,
        matches,
        document_chunk_cache={
            "DOC-1": [
                _snapshot_chunk(
                    "DOC-1-CHUNK-0009",
                    "DOC-1",
                    "不符合分类标准的资产不适用本附件规定的计量方法。",
                ),
            ]
        },
    )
    terms = [s.metadata.get("intent_gap_term") for s in supplements]
    assert "不符合" not in terms


def test_to_retrieval_results_keeps_highest_role_version() -> None:
    """同一 chunk 的多个 role 版本去重时保留高价值 role（mcq_exact_support）。"""
    from backend.app.schemas.qa import Citation
    from backend.app.services.retrieval_support import _to_retrieval_results
    from backend.app.services.retrieval_service import RetrievalMatch

    text = "商业银行应按给定格式或自定义格式填写可变表格。如使用自定义格式披露信息，应提供与给定格式等效可比的信息内容。"
    base = dict(
        document_id="DOC-1", chunk_id="DOC-1-CHUNK-0006", filename="417.docx",
        section_title="二、披露内容", page_number=None, excerpt=text,
        score=0.92, rerank_score=0.92, chunk_type="paragraph",
    )
    weak = RetrievalMatch(
        citation=Citation(**base, evidence_role="bounded_lexical_support"),
        score=0.92, rerank_score=0.92, coverage_score=0.0,
        evidence_role="bounded_lexical_support", evidence_text=text,
        metadata={"evidence_role": "bounded_lexical_support", "fusion_method": "bounded_document_lexical_support"},
    )
    strong = RetrievalMatch(
        citation=Citation(**base, evidence_role="mcq_exact_support"),
        score=0.90, rerank_score=0.90, coverage_score=0.0,
        evidence_role="mcq_exact_support", evidence_text=text,
        metadata={"evidence_role": "mcq_exact_support", "fusion_method": "mcq_exact_support"},
    )
    results = _to_retrieval_results([weak, strong])
    assert len(results) == 1
    assert results[0].metadata["evidence_role"] == "mcq_exact_support"


def test_expand_neighbor_matches_keeps_highest_role_version() -> None:
    """邻居扩展前按 chunk 去重时保留高价值 role 版本。"""
    from backend.app.schemas.qa import Citation
    from backend.app.services.retrieval_support import _expand_neighbor_matches
    from backend.app.services.retrieval_service import RetrievalMatch

    text = "商业银行应按给定格式或自定义格式填写可变表格。如使用自定义格式披露信息，应提供与给定格式等效可比的信息内容。"
    base = dict(
        document_id="DOC-1", chunk_id="DOC-1-CHUNK-0006", filename="417.docx",
        section_title="二、披露内容", page_number=None, excerpt=text,
        score=0.92, rerank_score=0.92, chunk_type="paragraph",
    )
    weak = RetrievalMatch(
        citation=Citation(**base, evidence_role="bounded_lexical_support"),
        score=0.92, rerank_score=0.92, coverage_score=0.0,
        evidence_role="bounded_lexical_support", evidence_text=text,
        metadata={"evidence_role": "bounded_lexical_support"},
    )
    strong = RetrievalMatch(
        citation=Citation(**base, evidence_role="mcq_exact_support"),
        score=0.90, rerank_score=0.90, coverage_score=0.0,
        evidence_role="mcq_exact_support", evidence_text=text,
        metadata={"evidence_role": "mcq_exact_support"},
    )
    # Fake db: cache must cover DOC-1 so no DB hit happens.
    class _FakeDb:
        def scalars(self, *args, **kwargs):
            raise AssertionError("no db expected")
    from backend.app.services.retrieval_support import _DocumentChunkSnapshot

    snapshot = _DocumentChunkSnapshot(
        id=1, chunk_id="DOC-1-CHUNK-0006", document_id="DOC-1", text=text,
        embedding_text=text, chunk_metadata=None, index_status="indexed",
        source_file="417.docx", page_number=None, section_title="二、披露内容",
    )
    expanded = _expand_neighbor_matches(
        _FakeDb(), "测试问题", [weak, strong],
        document_chunk_cache={"DOC-1": [snapshot]},
    )
    roles = {m.metadata.get("evidence_role") for m in expanded if m.citation.chunk_id == "DOC-1-CHUNK-0006"}
    assert "mcq_exact_support" in roles
    assert "bounded_lexical_support" not in roles


def test_referenced_clause_chunks_resolves_cross_section_citation() -> None:
    """“第X条”引用解析：命中同文档被引条款（跨章节多跳）。"""
    from backend.app.services.retrieval_support import _referenced_clause_chunks

    anchor = _snapshot_chunk(
        "DOC-1-CHUNK-0010",
        "DOC-1",
        "按照本办法第二十八条的规定执行。",
        section_title="第十二条 定义",
    )
    target = _snapshot_chunk(
        "DOC-1-CHUNK-0020",
        "DOC-1",
        "商业银行应按照监管并表范围披露相关信息。",
        section_title="第二十八条 并表范围",
    )
    unrelated = _snapshot_chunk(
        "DOC-1-CHUNK-0030",
        "DOC-1",
        "其他内容。",
        section_title="第三十五条 过渡安排",
    )
    hits = _referenced_clause_chunks(anchor, [anchor, target, unrelated])
    assert [chunk.chunk_id for chunk in hits] == ["DOC-1-CHUNK-0020"]


def test_referenced_clause_skips_own_section_title() -> None:
    """锚点自身的条号标题不被当作引用。"""
    from backend.app.services.retrieval_support import _referenced_clause_chunks

    anchor = _snapshot_chunk(
        "DOC-1-CHUNK-0010",
        "DOC-1",
        "本条规定了报告制度。",
        section_title="第二十八条 并表范围",
    )
    hits = _referenced_clause_chunks(anchor, [anchor])
    assert hits == []
