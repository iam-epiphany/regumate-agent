import json

from backend.app.services import query_planner_service
from backend.app.services.query_planner_service import plan_query


def test_query_planner_fallback_decomposes_asset_difference_question(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query("资产合计差异应该优先排查哪些问题？如果差异来自外币折算，需要保留什么依据？")

    assert plan.fallback_used is True
    assert [aspect.aspect_id for aspect in plan.aspects] == [
        "asset_total_difference_check",
        "foreign_currency_evidence",
    ]
    first_queries = plan.aspects[0].search_queries
    assert [query.query_type for query in first_queries] == [
        "semantic_question",
        "document_style_statement",
        "keyword_anchor",
    ]
    assert first_queries[0].query == "资产合计与分项合计存在差异时应优先检查哪些原因"
    assert "资产合计校验差异处理流程" in first_queries[1].query
    assert plan.aspects[1].evidence_need == "外币折算差异的留痕依据或支持材料"
    assert all(aspect.modality == "text" for aspect in plan.aspects)


def test_query_planner_fallback_recognizes_excel_lookup_question(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query("根据 Excel 附件《2024年一季度全国各地区原保险保费收入情况表》，全国合计的原保险保费收入是多少？")

    assert plan.fallback_used is True
    assert len(plan.aspects) == 1
    aspect = plan.aspects[0]
    assert aspect.modality == "table"
    assert aspect.table_task == "lookup"
    assert aspect.operation == "none"
    assert aspect.table_filters["source_title"] == "2024年一季度全国各地区原保险保费收入情况表"
    assert aspect.table_filters["year"] == 2024
    assert aspect.table_filters["quarter"] == 1
    assert aspect.table_filters["row_label"] == "全国合计"
    assert any(query.query_type == "table_locator" for query in aspect.search_queries)


def test_query_planner_builds_deterministic_mcq_title_and_option_queries(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)
    options = [
        "消费金融公司是非银行金融机构。",
        "消费贷款不包括住房和汽车贷款。",
        "名称中应当标明消费金融字样。",
        "核心数据遭到泄露属于特别重大数据安全事件。",
    ]

    plan = plan_query(
        "检索《数据安全事件分级》后，以下哪一项与材料内容一致？",
        options=options,
    )

    assert plan.planner == "deterministic-mcq"
    assert len(plan.aspects) == 1
    aspect = plan.aspects[0]
    assert aspect.question == "数据安全事件分级"
    assert [query.query_type for query in aspect.search_queries[:2]] == [
        "keyword_anchor",
        "semantic_question",
    ]
    assert all(query.query_type == "document_style_statement" for query in aspect.search_queries[2:])
    option_queries = "\n".join(query.query for query in aspect.search_queries[2:])
    assert all(option.rstrip("。") in option_queries for option in options)

    pdf_plan = plan_query(
        "检索《银行函证工作操作指引（PDF）》后，以下哪一项与材料内容一致？",
        options=options,
    )
    assert pdf_plan.aspects[0].question == "银行函证工作操作指引"
    assert pdf_plan.aspects[0].table_filters["file_type"] == "pdf"

    repeated = "共同事实。；"
    multi_fact_plan = plan_query(
        "关于《意外伤害保险业务监管办法》，下列哪组表述均属于材料内容？",
        options=[
            repeated + "保险期限一年及以下的个人意外险平均附加费用率上限为35%。",
            repeated + "无关事实甲。",
            repeated + "无关事实乙。",
            repeated + "无关事实丙。",
        ],
    )
    option_query = "\n".join(query.query for query in multi_fact_plan.aspects[0].search_queries[2:])
    assert option_query.count("共同事实") == 1
    assert "35%" in option_query


def test_query_planner_fallback_recognizes_excel_compare_and_calculate(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    compare_plan = plan_query("Excel 附件里哪个地区原保险保费收入最高？")
    calculate_plan = plan_query("Excel 附件里北京和上海的原保险保费收入差值是多少？")

    assert compare_plan.aspects[0].modality == "table"
    assert compare_plan.aspects[0].table_task == "compare"
    assert compare_plan.aspects[0].operation == "max"
    assert calculate_plan.aspects[0].table_task == "calculate"
    assert calculate_plan.aspects[0].operation == "difference"


def test_query_planner_fallback_recognizes_mixed_excel_policy_question(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query("根据监管制度口径和 Excel 附件，2024年一季度全国合计原保险保费收入是多少？")

    assert plan.aspects[0].modality == "mixed"


def test_query_planner_llm_returns_structured_multi_view_queries(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", "test-key")

    llm_payload = {
        "aspects": [
            {
                "aspect_id": "asset_total_difference_check",
                "question": "资产合计差异应该优先排查哪些问题？",
                "evidence_need": "资产合计差异处理流程、优先排查原因或校验规则",
                "search_queries": [
                    {
                        "query": "资产合计与分项合计存在差异时应优先检查哪些原因",
                        "query_type": "semantic_question",
                        "rationale": "贴近用户意图",
                    },
                    {
                        "query": "资产合计校验差异处理流程 币种折算 四舍五入 科目映射 重复汇总",
                        "query_type": "document_style_statement",
                        "rationale": "贴近填报说明证据句",
                    },
                    {
                        "query": "资产合计 差异 币种折算 四舍五入 科目映射 重复汇总",
                        "query_type": "keyword_anchor",
                        "rationale": "术语兜底",
                    },
                ],
                "keywords": ["资产合计", "差异", "币种折算"],
            }
        ]
    }

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": json.dumps(llm_payload, ensure_ascii=False)}}]},
                ensure_ascii=False,
            ).encode("utf-8")

    monkeypatch.setattr(query_planner_service.urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())

    plan = plan_query("资产合计差异应该优先排查哪些问题？")

    assert plan.fallback_used is False
    assert plan.planner.startswith("deepseek:")
    aspect = plan.aspects[0]
    assert aspect.evidence_need == "资产合计差异处理流程、优先排查原因或校验规则"
    assert [query.query_type for query in aspect.search_queries] == [
        "semantic_question",
        "document_style_statement",
        "keyword_anchor",
    ]
    assert aspect.search_queries[0].query.endswith("应优先检查哪些原因")
    assert aspect.search_queries[0].query != "资产合计 差异"


def test_query_planner_accepts_legacy_string_search_queries() -> None:
    aspects = query_planner_service._aspects_from_payload(
        "资产合计怎么填报",
        {
            "aspects": [
                {
                    "aspect_id": "asset_total",
                    "question": "资产合计怎么填报",
                    "expected_evidence_type": "填报口径",
                    "search_queries": ["资产合计填报口径", "资产合计填报口径"],
                    "keywords": ["资产合计"],
                }
            ]
        },
    )

    assert len(aspects) == 1
    assert aspects[0].evidence_need == "填报口径"
    assert len(aspects[0].search_queries) == 1
    assert aspects[0].search_queries[0].query == "资产合计填报口径"
    assert aspects[0].search_queries[0].query_type == "legacy"


def test_query_planner_dynamic_budget_detects_complex_question_aspects(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)
    question = (
        "系统应如何分别处理普惠小微贷款纳入、绿色信贷识别、逾期与不良贷款关系、"
        "资产合计差异排查、外币折算依据保留、历史差错更正记录、多期间影响说明，"
        "以及在依据不足时是否可以直接判断银行违规或生成正式监管报告？"
    )

    plan = plan_query(question)

    assert plan.fallback_used is True
    assert [aspect.aspect_id for aspect in plan.aspects] == [
        "inclusive_micro_loan_scope",
        "green_credit_identification",
        "overdue_nonperforming_relationship",
        "asset_total_difference_check",
        "foreign_currency_evidence",
        "historical_error_correction",
        "multi_period_impact",
        "insufficient_evidence_safety_boundary",
    ]
    assert plan.budget is not None
    assert plan.budget["detected_item_count"] == 8
    assert plan.budget["max_aspects"] == 8
    assert plan.budget["capacity_limited"] is False


def test_query_planner_budget_marks_capacity_limit(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_MAX_ASPECTS", 3)

    budget = query_planner_service.plan_query_budget(
        "系统应如何分别处理普惠小微贷款纳入、绿色信贷识别、逾期与不良贷款关系、资产合计差异排查？"
    )

    assert budget.max_aspects == 3
    assert budget.capacity_limited is True
    assert budget.omitted_or_merged_items == ("资产合计差异排查",)
