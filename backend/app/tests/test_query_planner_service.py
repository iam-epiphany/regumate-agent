import json
import re

from backend.app.services import query_planner_service, rag_service
from backend.app.schemas.qa import Citation, RetrievalResult
from backend.app.services.query_planner_service import QueryAspect, QuerySearchQuery, plan_query


def test_query_planner_keeps_user_stated_table_locator_when_llm_guesses_period() -> None:
    """Explicit table metadata is authoritative over a probabilistic plan."""

    question = (
        "根据监管统计表《2023年3季度保险业资金运用情况表》的工作表"
        "“2023年3季度保险资金运用情况表”，请给出“资金运用余额”在“账面余额”列的数值。"
    )
    aspects = query_planner_service._aspects_from_payload(
        question,
        {
            "aspects": [
                {
                    "aspect_id": "table_evidence",
                    "question": question,
                    "evidence_need": "table cell",
                    "search_queries": [question],
                    "keywords": ["资金运用余额"],
                    "modality": "table",
                    "table_task": "lookup",
                    "table_filters": {
                        "source_title": "2023年3季度保险业资金运用情况表",
                        "year": 2023,
                        "month": 3,
                        "quarter": 1,
                    },
                }
            ]
        },
    )

    filters = aspects[0].table_filters
    assert filters["source_title"] == "2023年3季度保险业资金运用情况表"
    assert filters["year"] == 2023
    assert filters["quarter"] == 3
    assert "month" not in filters


def test_query_planner_parses_explicit_sheet_row_and_column_lookup() -> None:
    question = (
        "根据监管统计表《148_2023年3季度保险业资金运用情况表》的工作表"
        "“2023年3季度保险资金运用情况表”，请给出“资金运用余额”在“账面余额”列的数值。"
    )

    aspect = query_planner_service._fallback_aspects(question, None)[0]

    assert aspect.table_filters["sheet"] == "2023年3季度保险资金运用情况表"
    assert aspect.table_filters["row_label"] == "资金运用余额"
    assert aspect.table_filters["column_label"] == "账面余额"
    assert list(aspect.selectors) == [{"row_or_indicator": "资金运用余额", "column_label": "账面余额"}]


def test_refusal_code_contract_classifies_explicit_request_limits() -> None:
    assert rag_service._refusal_code_for_request("请预测未来十二个月的监管数据。") == "unsupported_prediction"
    assert rag_service._refusal_code_for_request("没有给出余额和期限时，请计算利息。") == "missing_calculation_operands"
    assert rag_service._refusal_code_for_request("请提供资料库之外的实时汇率。") == "out_of_scope_or_realtime"
    assert rag_service._explicit_request_boundary_code("请概述资本充足率的监管口径。") is None


def test_prompt_aspect_marking_preserves_retrieval_origin() -> None:
    first = QueryAspect(
        aspect_id="document_a",
        question="根据《附件甲》说明甲条款。",
        evidence_need="甲条款",
        search_queries=(QuerySearchQuery("甲条款", "keyword_anchor", ""),),
        keywords=("甲条款",),
    )
    second = QueryAspect(
        aspect_id="document_b",
        question="根据《附件乙》说明乙条款。",
        evidence_need="乙条款",
        search_queries=(QuerySearchQuery("乙条款", "keyword_anchor", ""),),
        keywords=("乙条款",),
    )
    chunk = RetrievalResult(
        chunk_id="D-A-C1",
        document_id="D-A",
        text="甲条款规定商业银行应当落实甲事项。",
        score=1.0,
        rank=1,
        source_doc="附件甲",
        citation_label="[1]",
        citation=Citation(
            chunk_id="D-A-C1",
            document_id="D-A",
            filename="附件甲.docx",
            excerpt="甲条款规定商业银行应当落实甲事项。",
        ),
        metadata={"aspect_id": "document_a", "evidence_role": "bounded_lexical_support"},
    )

    rag_service._mark_chunk_for_aspect(chunk, first)
    rag_service._mark_chunk_for_aspect(chunk, second)

    assert chunk.metadata["retrieval_aspect_id"] == "document_a"
    assert chunk.metadata["aspect_id"] == "document_a"
    assert chunk.metadata["prompt_aspect_questions"] == {
        "document_a": first.question,
        "document_b": second.question,
    }
    assert rag_service._chunk_matches_query_aspect(chunk, first) is True
    assert rag_service._chunk_matches_query_aspect(chunk, second) is False
    assert rag_service._prompt_chunk_covers_aspect(chunk, second) is False
    assert rag_service._best_shared_candidate([chunk], [chunk], second) is None


def test_exact_support_is_isolated_by_explicit_source_title() -> None:
    first = QueryAspect(
        aspect_id="document_a",
        question="根据《附件甲》说明共同条款。",
        evidence_need="共同条款",
        search_queries=(QuerySearchQuery("共同条款", "keyword_anchor", ""),),
        keywords=("共同条款",),
        table_filters={"source_title": "附件甲"},
    )
    second = QueryAspect(
        aspect_id="document_b",
        question="根据《附件乙》说明共同条款。",
        evidence_need="共同条款",
        search_queries=(QuerySearchQuery("共同条款", "keyword_anchor", ""),),
        keywords=("共同条款",),
        table_filters={"source_title": "附件乙"},
    )
    chunk = RetrievalResult(
        chunk_id="D-A-C1",
        rank=1,
        score=1.0,
        source_doc="附件甲.docx",
        text="共同条款规定机构应当完成甲事项。",
        citation_label="[1]",
        metadata={
            "retrieval_aspect_id": "document_b",
            "evidence_role": "exact_anchor_support",
            "source_title": "附件甲",
        },
    )

    assert rag_service._chunk_matches_query_aspect(chunk, first) is True
    assert rag_service._chunk_matches_query_aspect(chunk, second) is False


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


def test_query_planner_does_not_route_policy_threshold_question_to_table(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    threshold_plan = plan_query("资产合计差异超过多少万元时需要复核？复核期限是多少？")
    scope_plan = plan_query("普惠小微贷款余额是否包括个人住房贷款？")

    assert all(aspect.modality == "text" for aspect in threshold_plan.aspects)
    assert all(aspect.table_task == "none" for aspect in threshold_plan.aspects)
    assert all(aspect.modality == "text" for aspect in scope_plan.aspects)
    assert all(aspect.table_task == "none" for aspect in scope_plan.aspects)


def test_query_planner_does_not_treat_policy_table_wording_as_spreadsheet(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    format_plan = plan_query(
        "根据《商业银行信息披露内容和要求》，商业银行以表格形式进行第三支柱信息披露包括哪些表格？"
    )
    frequency_plan = plan_query(
        "根据《商业银行信息披露内容和要求》，商业银行应根据表格要求分别按什么频率披露？"
    )

    assert all(aspect.modality == "text" for aspect in format_plan.aspects)
    assert all(aspect.modality == "text" for aspect in frequency_plan.aspects)


def test_query_planner_recognizes_monthly_report_lookup_and_difference(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    lookup_plan = plan_query("根据自制月报，2026年1月资产合计是多少？")
    difference_plan = plan_query("根据自制月报，2026年1月贷款余额比证券投资多多少万元？")
    sum_plan = plan_query("结合自制制度和自制月报，2026年1月资产合计是否等于四个分项之和？")
    missing_period_plan = plan_query("请根据资料说明 2027 年 3 月绿色信贷余额是多少？")

    lookup = lookup_plan.aspects[0]
    assert lookup.modality == "table"
    assert lookup.table_filters["year"] == 2026
    assert lookup.table_filters["month"] == 1
    assert lookup.table_filters["indicator"] == "资产合计"

    difference = difference_plan.aspects[0]
    assert difference.table_task == "calculate"
    assert difference.operation == "difference"
    # "贷款余额比证券投资多" → 证券投资在前(逆序)并标记 ordered_transition,
    # 计算时按 second - first = 贷款余额 - 证券投资。
    assert list(difference.selectors) == [
        {"row_or_indicator": "证券投资", "ordered_transition": True},
        {"row_or_indicator": "贷款余额"},
    ]

    table, regulation = sum_plan.aspects
    assert table.modality == "table"
    assert table.table_task == "calculate"
    assert table.operation == "sum"
    assert [selector["row_or_indicator"] for selector in table.selectors] == ["现金及存放同业", "贷款余额", "证券投资", "其他资产"]
    assert regulation.modality == "text"

    missing_period = missing_period_plan.aspects[0]
    assert missing_period.modality == "table"
    assert missing_period.table_filters["year"] == 2027
    assert missing_period.table_filters["month"] == 3
    assert missing_period.table_filters["indicator"] == "绿色信贷余额"


def test_query_planner_sanitizes_llm_table_filters_for_policy_question() -> None:
    aspects = query_planner_service._aspects_from_payload(
        "资产合计差异超过多少万元时需要复核？复核期限是多少？",
        {
            "aspects": [
                {
                    "aspect_id": "asset_threshold",
                    "question": "资产合计差异超过多少万元时需要复核？复核期限是多少？",
                    "modality": "mixed",
                    "table_task": "lookup",
                    "table_filters": {
                        "indicator": "资产合计",
                        "row_label": "资产合计",
                        "column_label": "复核期限",
                        "unit": "万元",
                    },
                    "operation": "none",
                    "search_queries": ["资产合计差异复核阈值和期限"],
                    "keywords": ["资产合计", "复核期限"],
                }
            ]
        },
    )

    assert aspects[0].modality == "text"
    assert aspects[0].table_task == "none"
    assert aspects[0].operation == "none"
    assert aspects[0].table_filters == {}
    assert aspects[0].explicit_filter_keys == ()


def test_query_planner_sanitizes_llm_numeric_policy_value_without_table_intent() -> None:
    aspects = query_planner_service._aspects_from_payload(
        "寿险折现率曲线终极利率暂定多少？",
        {
            "aspects": [
                {
                    "aspect_id": "ultimate_rate",
                    "question": "寿险折现率曲线终极利率暂定多少？",
                    "modality": "mixed",
                    "table_task": "locate",
                    "table_filters": {"indicator": "终极利率", "row_label": "寿险折现率曲线"},
                    "operation": "none",
                    "search_queries": ["寿险折现率曲线终极利率暂定值"],
                }
            ]
        },
    )

    assert aspects[0].modality == "text"
    assert aspects[0].table_task == "none"
    assert aspects[0].table_filters == {}


def test_query_planner_drops_unasserted_llm_metadata_filter_from_text_aspect() -> None:
    aspects = query_planner_service._aspects_from_payload(
        "重大数据安全事件的影响条件是什么？",
        {
            "aspects": [
                {
                    "aspect_id": "incident",
                    "question": "重大数据安全事件的影响条件是什么？",
                    "modality": "text",
                    "table_task": "none",
                    "table_filters": {"regulatory_topic": "数据安全"},
                    "search_queries": ["重大数据安全事件 影响条件"],
                }
            ]
        },
    )

    assert aspects[0].table_filters == {}
    assert aspects[0].explicit_filter_keys == ()


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
    assert len(aspect.search_queries) <= query_planner_service.MCQ_OPTION_SEARCH_QUERY_BUDGET + 2

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


def test_query_planner_preserves_all_open_comparison_operands_and_excel_title(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query(
        "制度侧根据《恢复计划示例》说明要求；报表侧根据 Excel 附件"
        "《2023年商业银行主要指标分机构类情况表（季度）》，在“农村商业银行”口径下，"
        "请比较“损失类贷款余额”、“不良贷款余额”、“可疑类贷款余额”、“次级类贷款余额”，"
        "找出数值最高的项目。"
    )

    table = plan.aspects[0]
    assert table.table_task == "compare"
    assert table.table_filters["source_title"] == "2023年商业银行主要指标分机构类情况表（季度）"
    assert table.table_filters["row_label"] == "农村商业银行"
    assert "indicator" not in table.table_filters
    assert "column_label" not in table.table_filters
    assert [selector["label"] for selector in table.selectors] == [
        "损失类贷款余额",
        "不良贷款余额",
        "可疑类贷款余额",
        "次级类贷款余额",
    ]


def test_query_planner_keeps_fixed_column_scope_and_unquoted_comparison_rows(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query(
        "在《2024年9月全国各地区原保险保费收入情况表》的健康险口径下，"
        "比较全国合计、北京、天津、公司本级，给出最大项、数值和单位。"
    )

    table = plan.aspects[0]
    assert table.table_filters["column_label"] == "健康险"
    assert "row_label" not in table.table_filters
    assert [selector["label"] for selector in table.selectors] == [
        "全国合计",
        "北京",
        "天津",
        "公司本级",
    ]


def test_query_planner_builds_ordered_period_selectors_with_shared_year(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query(
        "分别从2023年10月与12月人身险公司经营情况表读取原保险保费收入本年累计值，"
        "并计算12月值比10月值增加多少亿元。"
    )

    table = plan.aspects[0]
    assert table.table_task == "calculate"
    assert table.operation == "difference"
    assert "year" not in table.table_filters
    assert "month" not in table.table_filters
    assert "source_title" not in table.table_filters
    assert [selector["month"] for selector in table.selectors] == [10, 12]
    assert [selector["source_title"] for selector in table.selectors] == [
        "2023年10月人身险公司经营情况表",
        "2023年12月人身险公司经营情况表",
    ]


def test_query_planner_extracts_natural_table_title_in_mixed_question(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query(
        "联合核验。制度侧：说明数据事件分级。报表侧：读取2023年10月人身险公司经营情况表中"
        "原保险保费收入本年累计值。"
    )

    table = next(aspect for aspect in plan.aspects if aspect.modality in {"table", "mixed"})
    assert table.table_filters["source_title"] == "2023年10月人身险公司经营情况表"


def test_mixed_table_plan_splits_same_line_regulation_and_report_labels(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query(
        "联合核验。制度侧：说明账簿转换的禁止理由和不可撤销例外。"
        "报表侧：在2023年保障房贷款表四项汇总中找最大值。"
        "最后说明能否据此认定单一机构合规。"
    )

    table, regulation = plan.aspects
    assert table.modality == "table"
    assert regulation.question == "说明账簿转换的禁止理由和不可撤销例外。"
    assert "保障房贷款表" not in regulation.question
    assert regulation.search_queries[1].query == regulation.question.rstrip("。")


def test_llm_postprocessor_splits_two_coordinated_requirement_categories() -> None:
    assert query_planner_service._coordinated_aspect_questions(
        "处置计划还应说明自救资金原则和沟通对象"
    ) == ["处置计划还应说明自救资金原则", "处置计划还应说明沟通对象"]
    assert query_planner_service._coordinated_aspect_questions(
        "说明大额风险暴露门槛及处置自救原则"
    ) == ["说明大额风险暴露门槛", "处置自救原则"]
    assert query_planner_service._coordinated_aspect_questions(
        "说明意外伤害保险定义及保费厘定要求"
    ) == ["说明意外伤害保险定义", "保费厘定要求"]


def test_llm_postprocessor_splits_bare_threshold_and_order_categories() -> None:
    question = "持续经营触发阈值与二级资本损失吸收顺序是什么？"

    assert query_planner_service._coordinated_aspect_questions(question) == [
        "持续经营触发阈值是什么？",
        "二级资本损失吸收顺序是什么？",
    ]


def test_query_budget_counts_nested_coordination_across_semicolon_parts() -> None:
    budget = query_planner_service.plan_query_budget(
        "分别回答：持续经营触发阈值与二级资本损失吸收顺序是什么；"
        "账簿划分政策和程序的内部审计频率及留档要求是什么？"
    )

    assert budget.max_aspects == 3


def test_llm_postprocessor_splits_explicit_follow_up_requirement() -> None:
    assert query_planner_service._coordinated_aspect_questions(
        "商业银行大额风险暴露相对一级资本净额的门槛是多少，并说明处置策略基本原则。"
    ) == [
        "商业银行大额风险暴露相对一级资本净额的门槛是多少",
        "说明处置策略基本原则。",
    ]


def test_llm_postprocessor_splits_strong_categories_after_planner_rephrasing() -> None:
    assert query_planner_service._coordinated_aspect_questions(
        "商业银行大额风险暴露相对一级资本净额的门槛及处置策略基本原则"
    ) == [
        "商业银行大额风险暴露相对一级资本净额的门槛",
        "处置策略基本原则",
    ]
    assert query_planner_service._coordinated_aspect_questions(
        "说明定义、口径、范围和报送要求"
    ) == ["说明定义、口径、范围和报送要求"]


def test_llm_postprocessor_preserves_shared_question_tail_when_splitting() -> None:
    assert query_planner_service._coordinated_aspect_questions(
        "处置计划还应说明自救资金原则和沟通对象的依据是什么？"
    ) == [
        "处置计划还应说明自救资金原则的依据是什么？",
        "处置计划还应说明沟通对象的依据是什么？",
    ]


def test_llm_postprocessor_splits_two_topics_inside_related_requirements() -> None:
    parts = query_planner_service._coordinated_aspect_questions(
        "请综合概括《消费金融公司管理办法》中与关键人员薪酬递延和消费者催收保护相关的要求，"
        "不能漏掉比例、期限及禁止对象。"
    )

    assert len(parts) == 2
    assert "关键人员薪酬递延相关的要求" in parts[0]
    assert "消费者催收保护相关的要求" in parts[1]
    assert all("不能漏掉比例、期限及禁止对象" in part for part in parts)
    assert query_planner_service.plan_query_budget(
        "请综合概括《消费金融公司管理办法》中与关键人员薪酬递延和消费者催收保护相关的要求，"
        "不能漏掉比例、期限及禁止对象。"
    ).max_aspects == 2


def test_single_short_anchor_keeps_shared_classification_context() -> None:
    question = "区分重要数据事件‘特别重大’与‘重大’的影响条件。"

    assert query_planner_service._question_for_single_anchor(question, "特别重大") == (
        "区分重要数据事件“特别重大”的影响条件。"
    )
    assert query_planner_service._question_for_single_anchor(question, "重大") == (
        "区分重要数据事件“重大”的影响条件。"
    )


def test_coordinated_definition_and_pricing_inherits_subject() -> None:
    assert query_planner_service._coordinated_aspect_questions("意外伤害保险的定义与定价原则是什么？") == [
        "意外伤害保险的定义是什么？",
        "意外伤害保险的定价原则是什么？",
    ]


def test_mixed_text_aspects_do_not_inherit_report_source_title(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query(
        "联合核验。制度侧：说明处置资金自救原则和沟通策略对象。"
        "报表侧：读取2023年10月全国各地区原保险保费收入表全国合计的合计值。"
    )

    text_aspects = [aspect for aspect in plan.aspects if aspect.modality == "text"]
    assert len(text_aspects) == 2
    assert all("source_title" not in aspect.table_filters for aspect in text_aspects)
    assert any("自救原则" in aspect.question for aspect in text_aspects)
    assert any("沟通策略对象" in aspect.question for aspect in text_aspects)


def test_mixed_plan_keeps_two_short_quoted_regulatory_levels(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query(
        "联合核验。制度侧：区分重要数据事件‘特别重大’与‘重大’的影响条件。"
        "报表侧：读取2023年10月人身险公司经营情况表中原保险保费收入本年累计值。"
    )

    text_aspects = [aspect for aspect in plan.aspects if aspect.modality == "text"]
    assert len(text_aspects) == 2
    assert any("特别重大" in aspect.question for aspect in text_aspects)
    assert any("重大" in aspect.question for aspect in text_aspects)
    assert all("source_title" not in aspect.table_filters for aspect in text_aspects)


def test_table_plan_extracts_explicit_total_column_after_total_row(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query(
        "读取2023年10月全国各地区原保险保费收入表全国合计的合计值。"
    )

    table = next(aspect for aspect in plan.aspects if aspect.modality in {"table", "mixed"})
    assert table.table_filters["row_label"] == "全国合计"
    assert table.table_filters["column_label"] == "合计"


def test_llm_postprocessor_splits_definition_and_pricing_but_preserves_title_scope() -> None:
    parts = query_planner_service._coordinated_aspect_questions(
        "请用一段话概括《意外伤害保险业务监管办法》对意外伤害保险的定义，"
        "以及保险费厘定所遵循的精算原则与定价假设。"
    )

    assert len(parts) == 2
    assert "意外伤害保险的定义" in parts[0]
    assert "《意外伤害保险业务监管办法》中保险费厘定" in parts[1]


def test_query_planner_expands_four_loan_balance_comparison(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query(
        "报表侧：在2023年商业银行主要指标分机构类表的农村商业银行口径下比较四类贷款余额并给出最大项。"
    )

    table = plan.aspects[0]
    assert table.table_filters["row_label"] == "农村商业银行"
    assert "column_label" not in table.table_filters
    assert [selector["label"] for selector in table.selectors] == [
        "损失类贷款余额",
        "不良贷款余额",
        "可疑类贷款余额",
        "次级类贷款余额",
    ]


def test_query_planner_does_not_route_word_appendix_formula_to_spreadsheet(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query("请依据《附件7：信用风险缓释要求》中的 Word 公式计算 LGD*：LGD_s=0.4。")

    assert plan.aspects
    assert all(aspect.modality == "text" for aspect in plan.aspects)
    assert all(aspect.table_task == "none" for aspect in plan.aspects)


def test_query_planner_fallback_recognizes_mixed_excel_policy_question(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query("根据监管制度口径和 Excel 附件，2024年一季度全国合计原保险保费收入是多少？")

    assert [aspect.modality for aspect in plan.aspects] == ["table", "text"]
    assert plan.aspects[0].aspect_id == "table_evidence"
    assert plan.aspects[1].aspect_id == "regulatory_basis"


def test_mixed_table_plan_uses_only_report_side_for_filters_and_preserves_nested_sheet_name(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)
    question = (
        "请联合核验制度文件与统计报表。\n"
        "制度侧：根据《寿险合同负债评估折现率曲线》，说明与"
        "“万能险、投资连结险、变额年金及中短存续期产品适用30BP综合溢价”相关的规定。\n"
        "报表侧：根据 Excel 附件《2023年10月全国各地区原保险保费收入情况表》"
        "（工作表：各地区数据（月度）），“全国合计”在“合计”口径下的数值是多少？\n"
        "证据边界：能否据此认定单一机构合规？"
    )

    plan = plan_query(question)

    table, regulation = plan.aspects
    assert table.table_filters["source_title"] == "2023年10月全国各地区原保险保费收入情况表"
    assert table.table_filters["sheet"] == "各地区数据（月度）"
    assert table.table_filters["indicator"] == "全国合计"
    assert table.table_filters["column_label"] == "合计"
    assert "30BP" not in str(table.table_filters)
    assert list(table.selectors) == [{"row_or_indicator": "全国合计", "column_label": "合计"}]
    assert regulation.table_filters["source_title"] == "寿险合同负债评估折现率曲线"


def test_mixed_table_plan_splits_each_quoted_regulatory_condition(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)
    question = (
        "请联合核验制度文件与统计报表。\n"
        "制度侧：根据《处置计划建议示例》，请说明与"
        "“处置资金应优先使用自有资产”、“沟通策略应覆盖监管部门和客户”相关的规定。\n"
        "报表侧：根据 Excel 附件《贷款情况表》（工作表：汇总），比较“A”、“B”，找出最高项。"
    )

    plan = plan_query(question)

    assert [aspect.aspect_id for aspect in plan.aspects] == [
        "table_evidence",
        "regulatory_basis_1",
        "regulatory_basis_2",
    ]
    assert "处置资金应优先使用自有资产" in plan.aspects[1].question
    assert "沟通策略应覆盖监管部门和客户" in plan.aspects[2].question
    assert all(
        aspect.table_filters.get("source_title") == "处置计划建议示例"
        for aspect in plan.aspects[1:]
    )


def test_fallback_plan_splits_multiple_quoted_anchors_in_same_document_aspect(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)
    question = (
        "请跨文件分别回答以下两个事项，并为每项给出对应来源。\n"
        "关于前一份文件：根据《恢复计划示例（商业银行版）》，请说明与"
        "“恢复计划压力测试情景设置至少”相关的明确规定。\n"
        "关于后一份文件：根据《处置计划建议示例（商业银行版）》，请说明与"
        "“处置资金使用上应以使用金融机构自有资产或市场化渠道筹集资金开展自救为原则”、"
        "“处置计划沟通策略关注”相关的明确规定。"
    )

    plan = plan_query(question)
    questions = [aspect.question for aspect in plan.aspects]

    assert any("恢复计划压力测试情景设置至少" in item for item in questions)
    assert any("处置资金使用上应以使用金融机构自有资产" in item for item in questions)
    assert any("处置计划沟通策略关注" in item for item in questions)
    assert len(plan.aspects) >= 4


def test_fallback_does_not_split_standard_conditions_scope_suffix(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query(
        "根据《消费金融公司管理办法》，请说明与“消费金融公司可以”相关的明确规定。"
        "请完整给出材料明确写明的条件、范围、期限或例外。"
    )

    assert len(plan.aspects) == 1


def test_named_pdf_suffix_becomes_explicit_file_type_filter(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query(
        "根据《银行函证工作操作指引（PDF）》，请说明与“函证范围和回函用章”相关的规定。"
    )

    assert plan.aspects[0].table_filters["file_type"] == "pdf"
    assert "file_type" in plan.aspects[0].explicit_filter_keys


def test_query_planner_llm_returns_structured_multi_view_queries(monkeypatch) -> None:
    # The LLM planner is opt-in (QUERY_PLANNER_ENABLED defaults to False for
    # determinism); this test exercises the opt-in path explicitly.
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", "test-key")
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_ENABLED", True)

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
    assert plan.planner.startswith("openai_compatible:")
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


def test_query_planner_budget_honors_explicit_three_item_summary(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_MAX_ASPECTS", 8)

    budget = query_planner_service.plan_query_budget(
        "形成审查摘要：风险暴露划分、估值调整留档、内部审计安排三项要求分别是什么？"
    )

    assert len(budget.detected_items) == 3
    assert budget.max_aspects >= 3


def test_query_planner_budget_honors_explicit_four_item_joint_check(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_MAX_ASPECTS", 8)

    budget = query_planner_service.plan_query_budget(
        "联合核验四项内容：数字化回函效力、10个工作日回复、全国健康险比较、依据不足边界。"
    )

    assert len(budget.detected_items) == 4
    assert budget.max_aspects >= 4


def test_policy_question_with_numbers_is_not_misrouted_to_spreadsheet(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query(
        "形成审查摘要：终极利率暂定值是多少；重大数据安全事件的影响条件是什么；"
        "账簿划分政策内部审计如何安排并留档。"
    )

    assert plan.planner != "deterministic-table"
    assert all(aspect.modality == "text" for aspect in plan.aspects)


def test_cross_period_operands_are_selectors_not_global_filters(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query(
        "比较2024年9月、2025年3月和2025年9月三张人身险公司经营表的"
        "原保险保费收入本年累计值，指出最高期间、数值和单位。"
    )

    aspect = plan.aspects[0]
    assert plan.planner == "deterministic-table"
    assert aspect.table_task == "compare"
    assert aspect.operation == "max"
    assert len(aspect.selectors) == 3
    assert [selector["month"] for selector in aspect.selectors] == [9, 3, 9]
    assert all("人身险公司" in selector["source_title"] for selector in aspect.selectors)
    assert not {"year", "month", "quarter"} & set(aspect.table_filters)


def test_reconciliation_builds_three_ordered_source_selectors(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query(
        "分别读取2025年3月保险业合计、人身险和财产险原保险保费收入累计值，"
        "计算勾稽差额：保险业合计－人身险－财产险。"
    )

    aspect = plan.aspects[0]
    assert aspect.table_task == "calculate"
    assert aspect.operation == "difference"
    assert [selector["source_title"] for selector in aspect.selectors] == [
        "2025年3月保险业经营情况表",
        "2025年3月人身险公司经营情况表",
        "2025年3月财产保险公司经营情况表",
    ]
    assert all(selector["row_label"] == "原保险保费收入" for selector in aspect.selectors)


def test_unlabelled_joint_check_separates_table_and_regulation_clauses(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query(
        "联合核验：说明数字化回函效力与10个工作日回复要求；"
        "再比较2024年9月全国健康险的全国合计、北京、天津和公司本级，"
        "给出最大项、数值和单位。最后说明能否据此认定某家机构合规。"
    )

    assert plan.planner == "deterministic-table"
    assert plan.aspects[0].modality == "table"
    assert plan.aspects[0].question.startswith("再比较2024年9月")
    assert list(plan.aspects[0].selectors) == [
        {"label": "全国合计", "column_label": "健康险"},
        {"label": "北京", "column_label": "健康险"},
        {"label": "天津", "column_label": "健康险"},
        {"label": "公司本级", "column_label": "健康险"},
    ]
    assert any("数字化回函效力" in aspect.question for aspect in plan.aspects[1:])
    assert all("最后说明" not in aspect.question for aspect in plan.aspects)


def test_table_compare_uses_source_local_labels(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query(
        "比较2025年3月财产保险公司经营情况表中的原保险保费收入、赔款支出和承保利润，给出最大项。"
    )

    aspect = plan.aspects[0]
    assert aspect.table_task == "compare"
    assert [selector["label"] for selector in aspect.selectors] == ["原保险保费收入", "赔款支出", "承保利润"]


def test_table_short_report_family_sets_source_title_filter(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    property_plan = plan_query("比较2025年9月财产险表中原保险保费收入、机动车辆保险和责任险，给出最大项。")
    insurance_plan = plan_query("比较2025年9月保险业经营表的人身险、财产险与原保险保费收入，给出最大项。")

    assert property_plan.aspects[0].table_filters["source_title"] == "2025年9月财产保险公司经营情况表"
    assert insurance_plan.aspects[0].table_filters["source_title"] == "2025年9月保险业经营情况表"


def test_cross_period_regional_health_selectors_keep_period_and_metric(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query(
        "比较2024年9月、2025年3月和2025年9月全国各地区原保险保费收入情况表的全国健康险合计本年累计值。"
    )

    aspect = plan.aspects[0]
    assert aspect.table_task == "compare"
    assert [selector["source_title"] for selector in aspect.selectors] == [
        "2024年9月全国各地区原保险保费收入情况表",
        "2025年3月全国各地区原保险保费收入情况表",
        "2025年9月全国各地区原保险保费收入情况表",
    ]
    assert all(selector["row_label"] == "全国合计" for selector in aspect.selectors)
    assert all(selector["column_label"] == "健康险" for selector in aspect.selectors)


def test_bounded_lexical_phrases_split_document_style_support_anchors() -> None:
    aspect = query_planner_service.QueryAspect(
        aspect_id="regulatory_basis",
        question="说明处置自救原则",
        search_queries=(
            query_planner_service.QuerySearchQuery(
                query="商业银行版 处置计划建议示例 说明处置策略建议 坚持 自救为本 基本原则",
                query_type="document_style_statement",
                rationale="以指定材料和显式条件定位制度原文",
            ),
        ),
        evidence_need="制度原文",
        keywords=("说明处置自救原则", "自救为本"),
        modality="text",
    )

    phrases = rag_service._bounded_lexical_phrases(aspect)

    assert "处置策略建议坚持自救为本基本原则" in phrases
    assert "处置策略建议坚持自救为本" in phrases
    assert "处置策略建议" in phrases


def test_bounded_lexical_phrases_do_not_use_regulatory_keywords_for_table_aspects() -> None:
    aspect = query_planner_service.QueryAspect(
        aspect_id="table_evidence",
        question="比较2025年9月人身险表中意外险、寿险和原保险保费收入",
        search_queries=(
            query_planner_service.QuerySearchQuery(
                query="2025 9",
                query_type="table_locator",
                rationale="表格定位",
            ),
        ),
        evidence_need="表格单元格",
        keywords=("大额风险暴露", "自救为本", "2025"),
        modality="table",
    )

    phrases = rag_service._bounded_lexical_phrases(aspect)

    assert "大额风险暴露" not in phrases
    assert "自救为本" not in phrases


def test_insurance_reconciliation_minus_expression_plans_three_operand_calculation(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", "unused-key")

    plan = plan_query(
        "读取2024年9月保险业合计、人身险、财产险原保险保费收入累计值，计算保险业合计－人身险－财产险。"
    )

    aspect = plan.aspects[0]
    assert plan.planner == "deterministic-table"
    assert aspect.table_task == "calculate"
    assert aspect.operation == "difference"
    assert [selector["source_title"] for selector in aspect.selectors] == [
        "2024年9月保险业经营情况表",
        "2024年9月人身险公司经营情况表",
        "2024年9月财产保险公司经营情况表",
    ]
    assert all(selector["row_label"] == "原保险保费收入" for selector in aspect.selectors)
    assert all(selector["column_label"] == "本年累计" for selector in aspect.selectors)


def test_execution_constraint_is_not_converted_to_retrieval_aspect() -> None:
    aspects = query_planner_service._aspects_from_payload(
        "依据 Word 公式计算 SES；若执行器不支持平方根，必须明确拒答而非估算。",
        {
            "aspects": [
                {
                    "aspect_id": "formula",
                    "question": "市场风险规则中 SES 的 Word 公式是什么？",
                    "search_queries": ["SES 公式"],
                },
                {
                    "aspect_id": "instruction",
                    "question": "若执行器不支持平方根运算，必须明确拒答而非估算。",
                    "search_queries": ["拒答要求"],
                },
            ]
        },
    )

    assert [aspect.aspect_id for aspect in aspects] == ["formula"]


# ---------------------------------------------------------------------------
# composite row label table positioning (insurance fund asset classes)
# ---------------------------------------------------------------------------


def test_table_filters_quoted_scope_term_is_not_indicator_or_column(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    # "本年累计" / "截至当期" describe the measurement scope, not a queried
    # row; the quoted indicator must come out as "银行存款".
    plan = plan_query("根据《2025年四季度保险公司资金运用情况表》，“银行存款”在“本年累计/截至当期”口径下的数值是多少？")

    assert plan.fallback_used is True
    table = plan.aspects[0]
    assert table.modality == "table"
    assert table.table_filters["indicator"] == "银行存款"
    # The scope phrase survives as a column constraint, not an indicator.
    assert table.table_filters["column_label"] == "本年累计/截至当期"


def test_table_filters_recognises_insurance_asset_class_indicators(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query("根据《2025年四季度保险公司资金运用情况表》，财产险公司债券在“截至当期-账面余额”口径下的数值是多少？")

    table = plan.aspects[0]
    assert table.modality == "table"
    assert table.table_filters["indicator"] == "债券"
    # 机构类别词并入行定位词（通用能力：让财产险公司行优先于保险公司合计行）。
    assert table.table_filters["row_label"] == "财产险公司 债券"


def test_fallback_calculate_drops_global_column_filter_for_operands(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    # Calculation operands carry their own columns; a global column filter
    # would discard all but one operand before the executor can calculate.
    plan = plan_query("根据《2025年四季度保险公司资金运用情况表》，财产险公司资金运用余额与银行存款的差值是多少？")

    table = plan.aspects[0]
    assert table.modality == "table"
    assert table.table_task == "calculate"
    assert table.operation == "difference"
    assert "column_label" not in table.table_filters
    assert table.table_filters.get("indicator") == "资金运用余额"


def test_explicit_sheet_row_and_column_are_kept_in_distinct_table_fields(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query(
        "根据《示例月报》工作表“各地区数据（月度）”，"
        "行项目“全国合计”在列“保费收入 / 合计”的值是多少？"
    )

    aspect = plan.aspects[0]
    assert aspect.table_task == "lookup"
    assert aspect.table_filters["source_title"] == "示例月报"
    assert aspect.table_filters["sheet"] == "各地区数据（月度）"
    assert aspect.table_filters["row_label"] == "全国合计"
    assert aspect.table_filters["column_label"] == "保费收入 / 合计"
    assert aspect.table_filters.get("indicator") != "各地区数据（月度）"
    assert list(aspect.selectors) == [
        {"row_label": "全国合计", "column_label": "保费收入 / 合计"}
    ]


def test_difference_selectors_ignore_quoted_sheet_and_preserve_operand_order(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query(
        "根据《示例季报》工作表“资产负债季度”，计算行项目“同比增长率”"
        "在列“2023年 / 一季度”与列“2023年 / 四季度”之间的差额"
        "（后者减前者）。"
    )

    aspect = plan.aspects[0]
    assert aspect.table_task == "calculate"
    assert aspect.operation == "difference"
    assert list(aspect.selectors) == [
        {"row_label": "同比增长率", "column_label": "2023年 / 一季度"},
        {"row_label": "同比增长率", "column_label": "2023年 / 四季度"},
    ]
    assert "quarter" not in aspect.table_filters
    assert aspect.table_filters.get("indicator") != "资产负债季度"


def test_percentage_wording_builds_ordered_ratio_operands(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query(
        "根据《示例月报》工作表“地区数据”，计算行项目“全国合计”的"
        "“健康险”数值占“合计”数值的百分比。"
    )

    aspect = plan.aspects[0]
    assert aspect.table_task == "calculate"
    assert aspect.operation == "ratio"
    assert list(aspect.selectors) == [
        {"row_label": "全国合计", "column_label": "健康险"},
        {"row_label": "全国合计", "column_label": "合计"},
    ]


def test_joint_policy_report_uses_separate_explicit_document_titles(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query(
        "制度—报表联合核验：先依据《资本工具标准》相关条款说明监管要求，"
        "再从《资产负债季报》工作表“资产负债季度”提取行项目“同比增长率”"
        "在列“2023年 / 一季度”的值。请分别引用制度和报表来源。"
    )

    table_aspect = plan.aspects[0]
    assert table_aspect.modality == "table"
    assert table_aspect.table_filters["source_title"] == "资产负债季报"
    assert "quarter" not in table_aspect.table_filters
    regulation_aspects = [aspect for aspect in plan.aspects[1:] if aspect.modality == "text"]
    assert regulation_aspects
    assert all(
        aspect.table_filters.get("source_title") == "资本工具标准"
        for aspect in regulation_aspects
    )
    assert all("资产负债季度" not in aspect.question for aspect in regulation_aspects)


def test_explicit_missing_operands_and_subjective_decision_boundaries_are_general() -> None:
    assert (
        rag_service._explicit_request_boundary_code(
            "请比较两家机构的不良率差异，但未给出机构、期间和对应数值。"
        )
        == "missing_calculation_operands"
    )
    assert (
        rag_service._explicit_request_boundary_code(
            "请换算监管报表金额，但没有给出原币金额、币种或汇率。"
        )
        == "missing_calculation_operands"
    )
    assert (
        rag_service._explicit_request_boundary_code(
            "请根据监管材料替我决定明年的产品投放方向。"
        )
        == "subjective_business_advice"
    )


def test_refusal_boundaries_cover_common_missing_context_phrasings() -> None:
    assert rag_service._explicit_request_boundary_code(
        "请算出两个未提供数值的报告期资本充足率差异。"
    ) == "missing_calculation_operands"
    assert rag_service._explicit_request_boundary_code(
        "请把一笔外币负债折算成人民币，但没有金额、币种或汇率。"
    ) == "missing_calculation_operands"
    assert rag_service._explicit_request_boundary_code(
        "请从Excel取数，但没有文件名、工作表、行和列定位信息。"
    ) == "missing_required_context"
    assert rag_service._explicit_request_boundary_code(
        "请查询尚未入库的今日监管处罚清单。"
    ) == "out_of_scope_or_realtime"
    assert rag_service._explicit_request_boundary_code(
        "请断言下一版监管规则必然会降低某项比例要求。"
    ) == "unsupported_prediction"
    assert rag_service._explicit_request_boundary_code(
        "请依据监管文件设计保证盈利的营销方案。"
    ) == "subjective_business_advice"


def test_quoted_sum_word_in_row_label_does_not_turn_lookup_into_calculation() -> None:
    question = (
        "根据《2012年机构范围和指标解释》工作表“机构范围解释和指标解释”，"
        "行项目“不良贷款余额 / 次级类、可疑类和损失类贷款之和”"
        "在列“基本指标解释 / 序号”的值是多少？"
    )

    aspect = query_planner_service._fallback_aspects(question, None)[0]

    assert aspect.table_task == "lookup"
    assert aspect.operation == "none"
    assert list(aspect.selectors) == [
        {
            "row_label": "不良贷款余额 / 次级类、可疑类和损失类贷款之和",
            "column_label": "基本指标解释 / 序号",
        }
    ]
    assert aspect.table_filters["year"] == 2012


def test_multi_document_subquestions_keep_their_own_explicit_source_titles() -> None:
    question = (
        "请分别依据《规则甲》第六条和《规则乙》第六条，"
        "列出两份文件各自的核心监管要求并区分来源。"
    )
    aspects = query_planner_service._aspects_from_payload(
        question,
        {
            "aspects": [
                {
                    "aspect_id": "rule_a",
                    "question": "《规则甲》第六条的核心监管要求是什么？",
                    "search_queries": ["规则甲 第六条"],
                    "modality": "text",
                },
                {
                    "aspect_id": "rule_b",
                    "question": "《规则乙》第六条的核心监管要求是什么？",
                    "search_queries": ["规则乙 第六条"],
                    "modality": "text",
                },
            ]
        },
    )

    assert [aspect.table_filters["source_title"] for aspect in aspects] == ["规则甲", "规则乙"]


def test_merged_two_document_aspect_is_split_with_independent_source_filters() -> None:
    aspect = query_planner_service.QueryAspect(
        aspect_id="two_rules",
        question=(
            "请分别依据《规则甲》相关条款和《规则乙》相关条款，"
            "列出两份文件各自的核心监管要求，并明确区分来源。"
        ),
        search_queries=(
            query_planner_service.QuerySearchQuery("规则甲 核心要求", "semantic_question", ""),
            query_planner_service.QuerySearchQuery("规则乙 核心要求", "semantic_question", ""),
        ),
        evidence_need="两份制度的核心要求",
        keywords=("规则甲", "规则乙"),
        modality="text",
        table_filters={"source_title": "规则甲"},
        explicit_filter_keys=("source_title",),
    )
    budget = query_planner_service.QueryBudget(
        detected_items=("规则甲", "规则乙"),
        max_aspects=2,
        system_max_aspects=8,
        capacity_limited=False,
    )

    split = query_planner_service._split_merged_llm_aspects([aspect], budget)

    assert len(split) == 2
    assert [item.table_filters["source_title"] for item in split] == ["规则甲", "规则乙"]
    assert all(len(re.findall(r"《[^》]+》", item.question)) == 1 for item in split)


def test_fallback_planner_splits_explicit_cross_document_request(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query(
        "请分别依据《规则甲》相关条款和《规则乙》相关条款，"
        "列出两份文件各自的核心监管要求，并明确区分来源。"
    )

    assert len(plan.aspects) == 2
    assert [aspect.table_filters["source_title"] for aspect in plan.aspects] == ["规则甲", "规则乙"]


def test_quoted_regulatory_decision_and_estimate_are_not_request_boundaries() -> None:
    decision_question = (
        "请分别依据《规则甲》中“监管机构决定调整计量范围”相关条款，"
        "以及《规则乙》中“机构应当报告”的条款，列出各自要求。"
    )
    estimate_question = (
        "制度—报表联合核验：依据《规则甲》中“会计政策和会计估计”说明要求；"
        "再读取《行业报表》工作表“数据”的指定值，不要据此直接认定合规。"
    )

    assert rag_service._explicit_request_boundary_code(decision_question) is None
    assert rag_service._explicit_request_boundary_code(estimate_question) is None


def test_explicit_document_anchor_pairs_are_isolated_before_generic_splitting() -> None:
    question = (
        "请分别依据《规则甲》中“一、总体要求，机构应当完整验证”相关条款，"
        "以及《规则乙》中“监管机构决定调整计量范围”相关条款，"
        "列出两份文件各自的监管要求并明确区分来源。"
    )

    plan = plan_query(question)

    assert len(plan.aspects) == 2
    assert [aspect.table_filters["source_title"] for aspect in plan.aspects] == [
        "规则甲",
        "规则乙",
    ]
    assert [aspect.table_filters["indicator"] for aspect in plan.aspects] == [
        "一、总体要求，机构应当完整验证",
        "监管机构决定调整计量范围",
    ]


def test_hierarchical_row_period_is_not_a_global_lookup_filter() -> None:
    question = (
        "根据《2024年商业银行主要指标分机构类情况表》"
        "工作表“商业银行分机构类情况表”，"
        "行项目“一季度 / 不良贷款率”在列“大型商业银行”的值是多少？"
    )

    aspect = plan_query(question).aspects[0]

    assert aspect.table_task == "lookup"
    assert aspect.table_filters["row_label"] == "一季度 / 不良贷款率"
    assert aspect.table_filters["column_label"] == "大型商业银行"
    assert "quarter" not in aspect.table_filters


def test_quoted_share_column_is_lookup_not_ratio_calculation() -> None:
    question = (
        "根据《2025年四季度保险公司资金运用情况表》工作表“Sheet1”，"
        "行项目“保险公司 / 资金运用余额”在列“截至当期 / 占比”的值是多少？"
    )

    aspect = plan_query(question).aspects[0]

    assert aspect.table_task == "lookup"
    assert aspect.operation == "none"
    assert aspect.selectors == (
        {
            "row_label": "保险公司 / 资金运用余额",
            "column_label": "截至当期 / 占比",
        },
    )


def test_explicit_definition_anchor_uses_definition_style_search() -> None:
    plan = plan_query(
        "根据《市场风险内部模型法监管要求》相关条款，“交易台”的完整定义是什么？"
    )

    aspect = plan.aspects[0]

    assert plan.planner == "deterministic-document-anchor"
    assert "完整定义" in aspect.question
    assert any(
        "是指" in search.query and "定义" in search.query
        for search in aspect.search_queries
    )


def test_table_filters_keep_institution_category_in_row_label(monkeypatch) -> None:
    """机构类别词（财产险公司/人身险公司）必须并入行定位词。"""
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query("根据《2025年四季度保险公司资金运用情况表》，财产险公司资金运用余额在'截至当期-账面余额'口径下的数值是多少？")
    table = plan.aspects[0]
    assert table.table_filters["indicator"] == "资金运用余额"
    assert table.table_filters["row_label"] == "财产险公司 资金运用余额"
    # ASCII 单引号列口径从引号内干净提取（原 column_scope 正则吞词 bug）。
    assert table.table_filters["column_label"] == "截至当期-账面余额"
    assert "财产险公司" not in str(table.table_filters.get("column_label") or "")

    plan = plan_query("根据《2025年四季度保险公司资金运用情况表》，人身险公司资金运用余额在'截至当期-账面余额'口径下的数值是多少？")
    assert plan.aspects[0].table_filters["row_label"] == "人身险公司 资金运用余额"


def test_table_filters_keep_aggregate_institution_row_label(monkeypatch) -> None:
    """问合计行（保险公司）时机构词为'保险公司'，要求行标签含该词。"""
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query("《2025年四季度保险公司资金运用情况表》中保险公司资金运用余额是多少？")
    assert plan.aspects[0].table_filters["row_label"] == "保险公司 资金运用余额"


def test_table_filters_without_institution_word_unchanged(monkeypatch) -> None:
    """无机构类别词时 row_label 保持裸指标（历史行为）。"""
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query("根据《2025年四季度保险公司资金运用情况表》，银行存款在'截至当期-账面余额'口径下的数值是多少？")
    table = plan.aspects[0]
    assert table.table_filters["row_label"] == "银行存款"
    assert "财产险" not in str(table.table_filters.get("row_label") or "")


def test_institution_row_terms_ignores_title_embedded_words() -> None:
    """无书名号自然标题中的机构词（'…保险公司资金运用情况表中…'）不得成为行定位词。"""
    terms = query_planner_service._institution_row_terms(
        "2025年四季度保险公司资金运用情况表中财产险公司资金运用余额",
        "资金运用余额",
    )
    assert terms == ["财产险公司"]

    terms = query_planner_service._institution_row_terms(
        "根据，财产险公司资金运用余额在'截至当期-账面余额'口径下的数值是多少？",
        "资金运用余额",
    )
    assert terms == ["财产险公司"]


def test_compare_question_never_gets_sheet_institution_row_label(monkeypatch) -> None:
    """compare（哪一项数值最高）无指标词：工作表名中的机构词不得成为行定位词。"""
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query(
        "根据 Excel 附件《2024年9月人身险公司经营情况表》（工作表：人身保险公司（月度） ），"
        "在“本年累计/截至当期”口径下，以下哪一项数值最高？"
    )
    table = plan.aspects[0]
    assert table.table_filters.get("row_label") in (None, "")
    assert table.table_filters["column_label"] == "本年累计/截至当期"


def test_ratio_phrase_not_read_as_difference(monkeypatch) -> None:
    """“占…的比例…多少”不能因“比例”的“比”+“多少”的“多”被误判为差值。"""
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query("根据《2025年四季度保险公司资金运用情况表》，财产险公司银行存款占资金运用余额的比例约为多少？")
    table = plan.aspects[0]
    assert table.operation == "ratio"
    assert list(table.selectors) == [
        {"row_or_indicator": "财产险公司 银行存款"},
        {"row_or_indicator": "财产险公司 资金运用余额"},
    ]
    # 同结构换指标/机构仍为 ratio（通用形态）。
    plan = plan_query("根据《2025年四季度保险公司资金运用情况表》，人身险公司债券占资金运用余额的比例约为多少？")
    assert plan.aspects[0].operation == "ratio"


def test_planner_detects_multiplier_ratio(monkeypatch) -> None:
    """“是…的多少倍”是 ratio 形态，不是 lookup。"""
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query("根据《2025年12月保险业经营情况表》，原保险保费收入是赔付支出的多少倍？")
    table = plan.aspects[0]
    assert table.operation == "ratio"
    assert [s["row_or_indicator"] for s in table.selectors] == ["原保险保费收入", "赔付支出"]


def test_multi_institution_difference_selectors(monkeypatch) -> None:
    """“X公司与Y公司…的差值”：每个机构词限定同一指标的一行（问题顺序）。"""
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query("根据《2025年四季度保险公司资金运用情况表》，人身险公司与财产险公司资金运用余额的差值是多少？")
    table = plan.aspects[0]
    assert table.operation == "difference"
    assert list(table.selectors) == [
        {"row_or_indicator": "人身险公司 资金运用余额"},
        {"row_or_indicator": "财产险公司 资金运用余额"},
    ]
    # 换机构/指标同样生效。
    plan = plan_query("根据《2025年四季度保险公司资金运用情况表》，城市商业银行与民营银行净利润的差值是多少？")
    assert [s["row_or_indicator"] for s in plan.aspects[0].selectors] == [
        "城市商业银行 净利润",
        "民营银行 净利润",
    ]


def test_cross_table_selectors_carry_source_titles(monkeypatch) -> None:
    """两张《表》的差值：每个操作数限定在各自表。"""
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query(
        "根据《2025年12月人身险公司经营情况表》和《2025年12月财产保险公司经营情况表》，"
        "人身险公司与财产险公司原保险保费收入（本年累计）的差值是多少？"
    )
    table = plan.aspects[0]
    assert list(table.selectors) == [
        {"row_or_indicator": "原保险保费收入", "source_title": "2025年12月人身险公司经营情况表"},
        {"row_or_indicator": "原保险保费收入", "source_title": "2025年12月财产保险公司经营情况表"},
    ]
    # 连词“与”不进入机构词；row_label 并入第一个机构词。
    assert table.table_filters["row_label"] == "人身险公司 原保险保费收入"


def test_institution_term_drops_conjunction_prefix(monkeypatch) -> None:
    """“人身险公司与财产险公司”中的“与”是连词，不是机构名一部分。"""
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    terms = query_planner_service._institution_row_terms(
        "人身险公司与财产险公司原保险保费收入的差值",
        "原保险保费收入",
    )
    assert terms == ["人身险公司", "财产险公司"]


def test_indicator_substring_dedup() -> None:
    """“不良贷款余额”同时包含“贷款余额”，不能拆成两个操作数。"""
    indicators = query_planner_service._mentioned_table_indicators(
        "农村商业银行一季度不良贷款余额与四季度不良贷款余额的差值"
    )
    assert indicators == ["不良贷款余额"]


def test_period_selectors_use_global_year(monkeypatch) -> None:
    """《2025年…表》中的年份不紧邻季度词时，跨期选择器仍归属该年份。"""
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query(
        "根据《2025年商业银行主要指标分机构类情况表（季度）》，"
        "农村商业银行一季度与四季度不良贷款余额的差值是多少？"
    )
    table = plan.aspects[0]
    assert table.operation == "difference"
    # 文本序（一季度在前），且无 ordered_transition 标记（先 − 后）。
    assert list(table.selectors) == [
        {"year": 2025, "quarter": 1, "row_label": "不良贷款余额"},
        {"year": 2025, "quarter": 4, "row_label": "不良贷款余额"},
    ]
