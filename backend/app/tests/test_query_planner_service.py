import json

from backend.app.services import query_planner_service, rag_service
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


def test_pillar3_disclosure_frequency_uses_deterministic_domain_plan(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", "unused-key")

    plan = plan_query(
        "根据《商业银行信息披露内容和要求》，请说明与“商业银行应根据表格要求，分别”相关的明确规定。"
    )

    assert plan.planner == "deterministic-domain"
    assert [aspect.aspect_id for aspect in plan.aspects] == ["pillar3_disclosure_frequency"]
    assert any(
        "季度" in query.query and "半年" in query.query and "年度" in query.query
        for query in plan.aspects[0].search_queries
    )
    assert any("另有规定" in query.query for query in plan.aspects[0].search_queries)


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
    assert list(difference.selectors) == [{"row_or_indicator": "证券投资"}, {"row_or_indicator": "贷款余额"}]

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


def test_query_planner_prioritizes_mcq_document_style_aliases_within_budget(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query(
        "关于交易账簿和第三支柱披露，下列哪项完整正确？",
        options=[
            "交易头寸不含做市和对客交易；只按单体口径披露。",
            "以交易目的持有的头寸包括自营、做市、对客交易和相关对冲；商业银行按监管并表范围披露相关信息。",
            "交易账簿头寸按月计量；第三支柱披露只包括文字说明。",
            "交易账簿仅包含贷款；披露范围由机构自行决定。",
        ],
    )

    aspect = plan.aspects[0]
    query_text = "\n".join(query.query for query in aspect.search_queries)
    assert len(aspect.search_queries) <= query_planner_service.MCQ_OPTION_SEARCH_QUERY_BUDGET + 2
    assert "以交易目的持有的头寸 包括 自营业务 做市业务" in query_text
    assert "商业银行 应按照 监管并表范围 披露相关信息" in query_text


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
    assert regulation.search_queries[1].query == regulation.question


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


def test_llm_split_can_exceed_underestimated_top_level_budget() -> None:
    budget = query_planner_service.QueryBudget(
        detected_items=(),
        max_aspects=3,
        system_max_aspects=12,
        capacity_limited=False,
    )
    aspects = [
        query_planner_service.QueryAspect(
            aspect_id="insurance_group_reporting",
            question="保险集团报告义务有哪些？",
            search_queries=(query_planner_service.QuerySearchQuery("保险集团报告义务", "semantic_question"),),
            evidence_need="保险集团报告义务",
            keywords=("保险集团",),
        ),
        query_planner_service.QueryAspect(
            aspect_id="accident_definition_pricing",
            question="意外伤害保险的定义与定价原则是什么？",
            search_queries=(query_planner_service.QuerySearchQuery("意外伤害保险定义与定价原则", "semantic_question"),),
            evidence_need="意外伤害保险定义与定价原则",
            keywords=("意外伤害保险",),
        ),
        query_planner_service.QueryAspect(
            aspect_id="book_audit",
            question="账簿划分政策的年度内部审计和留档要求是什么？",
            search_queries=(query_planner_service.QuerySearchQuery("账簿划分内部审计留档", "semantic_question"),),
            evidence_need="账簿审计留档",
            keywords=("账簿划分",),
        ),
    ]

    split = query_planner_service._split_merged_llm_aspects(aspects, budget)

    assert [aspect.question for aspect in split] == [
        "保险集团报告义务有哪些？",
        "意外伤害保险的定义是什么？",
        "意外伤害保险的定价原则是什么？",
        "账簿划分政策的年度内部审计和留档要求是什么？",
    ]
    pricing = split[2]
    definition_queries = " ".join(query.query for query in split[1].search_queries)
    pricing_queries = " ".join(query.query for query in pricing.search_queries)
    assert "本办法所称意外伤害保险" in definition_queries
    assert "厘定保险费" in pricing_queries


def test_document_style_alias_uses_regulatory_phrasing_without_answer_terms() -> None:
    assert query_planner_service._document_style_alias("说明处置计划的沟通对象") == (
        "说明处置计划的沟通策略 与哪些主体开展有效沟通"
    )
    assert "投资连结险" in query_planner_service._document_style_alias(
        "说明万能险、投连险和变额年金的综合溢价"
    )
    assert "本办法所称意外伤害保险" in query_planner_service._document_style_alias("意外伤害保险定义")
    assert "一般精算原理" in query_planner_service._document_style_alias("说明保费厘定要求")
    assert "编报 保险集团偿付能力报告" in query_planner_service._document_style_alias("核对保险集团报告义务")
    assert "应当编报保险集团偿付能力报告的公司名单" in query_planner_service._document_style_alias("核对保险集团报告义务")
    assert "40%以上" in query_planner_service._document_style_alias("概括消费金融公司薪酬递延")
    assert "不得对与债务无关的第三人进行催收" in query_planner_service._document_style_alias("催收禁限")
    assert "5.125%" in query_planner_service._document_style_alias("给出资本工具持续经营触发阈值")
    assert "二级资本工具" in query_planner_service._document_style_alias("损失吸收顺序")
    assert "10个工作日内" in query_planner_service._document_style_alias("合规询证函回复时限")
    assert "总行或总部网站 微信公众号" in query_planner_service._document_style_alias("函证事项应通过哪些公开渠道公示")
    assert "全国范围内开展业务" in query_planner_service._document_style_alias("消费金融公司的业务地域范围")
    assert "专利权质押登记 全流程无纸化线上办理 全覆盖" in query_planner_service._document_style_alias(
        "专利权质押登记线上全覆盖目标"
    )
    assert "终极利率 暂定为 4.5%" in query_planner_service._document_style_alias("终极利率暂定值")


def test_llm_payload_preserves_local_document_style_aliases() -> None:
    payload = {
        "aspects": [
            {
                "aspect_id": "insurance_group_reporting",
                "question": "保险集团报告义务有哪些？",
                "evidence_need": "保险集团报告义务",
                "search_queries": [
                    {"query": "保险集团报告义务有哪些", "query_type": "semantic_question"},
                    {"query": "保险集团年度和半年度偿付能力报告", "query_type": "document_style_statement"},
                    {"query": "保险集团 报告义务", "query_type": "keyword_anchor"},
                ],
                "keywords": ["保险集团", "报告义务"],
                "modality": "text",
                "table_task": "none",
                "table_filters": {},
                "operation": "none",
            },
            {
                "aspect_id": "accident_definition_pricing",
                "question": "意外伤害保险的定义与定价原则是什么？",
                "evidence_need": "意外伤害保险定义与定价原则",
                "search_queries": [
                    {"query": "意外伤害保险定义与定价原则", "query_type": "semantic_question"},
                    {"query": "意外伤害保险定义 定价原则", "query_type": "document_style_statement"},
                    {"query": "意外险 定价", "query_type": "keyword_anchor"},
                ],
                "keywords": ["意外伤害保险", "定价原则"],
                "modality": "text",
                "table_task": "none",
                "table_filters": {},
                "operation": "none",
            },
        ],
        "omitted_or_merged_items": [],
    }

    aspects = query_planner_service._aspects_from_payload(
        "分别核对保险集团报告义务、意外伤害保险定义与定价原则。",
        payload,
    )

    group_queries = [query.query for query in aspects[0].search_queries]
    accident_queries = [query.query for query in aspects[1].search_queries]
    assert any("应当编报保险集团偿付能力报告的公司名单" in query for query in group_queries)
    assert any("编报 保险集团偿付能力报告" in query for query in group_queries)
    assert any("意外伤害保险业务监管办法" in query and "厘定保险费" in query for query in accident_queries)
    assert len(group_queries) <= query_planner_service.QUERY_PLANNER_MAX_SEARCH_QUERIES
    assert len(accident_queries) <= query_planner_service.QUERY_PLANNER_MAX_SEARCH_QUERIES


def test_mcq_document_style_aliases_use_public_option_facts() -> None:
    aliases = query_planner_service._mcq_document_style_aliases(
        "申请书列名称、拟设地、注册资本、股权结构和业务范围，目录为2023年版"
    )
    assert any("拟设立中资商业银行" in alias for alias in aliases)
    assert any("2023年版" in alias for alias in aliases)

    aliases = query_planner_service._mcq_document_style_aliases("曲线由基础利率加综合溢价形成")
    assert any("折现率曲线" in alias and "综合溢价" in alias for alias in aliases)

    aliases = query_planner_service._mcq_document_style_aliases(
        "敏感级及以上数据事件造成难以消除的个人负面影响或使机构部分业务异常等可构成较大事件"
    )
    assert any("较大数据安全事件" in alias and "部分业务无法正常开展" in alias for alias in aliases)

    aliases = query_planner_service._mcq_document_style_aliases(
        "重要实体承载核心业务条线和关键功能；工具可包括自救、注资、战略投资、不良资产处置、接管、收购承接、过桥机构和破产清算"
    )
    assert any("重要实体 承载 本机构核心业务条线和关键功能" in alias for alias in aliases)
    assert any("处置工具 包括 机构自救" in alias for alias in aliases)

    aliases = query_planner_service._mcq_document_style_aliases(
        "禁止暴力等不正当催收且不得催收无关第三人；一函一个基准日，并应通过总行或总部公开渠道公示函证事项"
    )
    assert any(all(term in alias for term in ("不正当手段", "催收", "债务无关", "第三人")) for alias in aliases)
    assert any("一个函证基准日" in alias for alias in aliases)
    assert any("办理函证相关事项进行公示" in alias for alias in aliases)

    aliases = query_planner_service._mcq_document_style_aliases(
        "交易头寸包括自营、做市、对客及相关对冲，原则上每日公允价值计量且变动计入损益；披露按监管并表范围，表格另有规定除外"
    )
    assert any("自营业务 做市业务" in alias for alias in aliases)
    assert any("监管并表范围" in alias for alias in aliases)
    title_terms = rag_service._mcq_seed_document_title_terms(
        [
            "申请书列名称、拟设地、注册资本、股权结构和业务范围，目录为2023年版",
            "曲线由基础利率加综合溢价形成",
            "敏感级及以上数据事件造成难以消除的个人负面影响或使机构部分业务异常等可构成较大事件",
        ]
    )
    assert "中资商业银行行政许可事项申请材料" in title_terms
    assert "寿险合同负债评估的折现率曲线" in title_terms
    assert "数据安全事件分级" in title_terms


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


def test_mixed_disposal_self_rescue_uses_document_style_alias(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query(
        "联合核验：说明大额风险暴露门槛和处置自救原则；"
        "再比较2025年9月人身险表中意外险、寿险和原保险保费收入，给出最大项、数值和单位。"
    )

    large_exposure_aspect = next(aspect for aspect in plan.aspects if "大额风险暴露门槛" in aspect.question)
    large_exposure_query_text = " ".join(query.query for query in large_exposure_aspect.search_queries)
    assert "2.5%" in large_exposure_query_text
    assert "一级资本净额" in large_exposure_query_text
    assert "商业银行版" in large_exposure_query_text

    self_rescue_aspect = next(aspect for aspect in plan.aspects if "处置自救原则" in aspect.question)
    self_rescue_query_text = " ".join(query.query for query in self_rescue_aspect.search_queries)
    assert "自救为本" in self_rescue_query_text
    assert "商业银行版" in self_rescue_query_text
    assert "自救为本" in self_rescue_aspect.keywords


def test_mixed_capital_tool_loss_absorption_uses_source_anchor(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)

    plan = plan_query(
        "联合核验：给出资本工具持续经营触发阈值和损失吸收顺序；"
        "再比较2025年9月保险业经营表的人身险、财产险与原保险保费收入，给出最大项。"
    )

    loss_aspect = next(aspect for aspect in plan.aspects if "损失吸收顺序" in aspect.question)
    query_text = " ".join(query.query for query in loss_aspect.search_queries)
    assert "资本工具合格标准" in query_text
    assert "二级资本工具" in query_text


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


def test_text_lexical_seed_terms_keep_distinctive_roots() -> None:
    aspect = query_planner_service.QueryAspect(
        aspect_id="regulatory_basis",
        question="说明数字化银行回函的效力、催收禁限和定价原则。",
        search_queries=(
            query_planner_service.QuerySearchQuery("数字化银行回函效力", "keyword_anchor", ""),
            query_planner_service.QuerySearchQuery("催收禁限定价原则", "document_style_statement", ""),
        ),
        evidence_need="制度原文",
        keywords=("数字化银行回函效力", "催收禁限", "定价原则"),
        modality="text",
    )

    terms = rag_service._text_lexical_seed_terms(aspect)

    assert "数字化回函" in terms
    assert "催收" in terms
    assert "定价" in terms


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


def test_coordinated_definition_requirement_inherits_subject() -> None:
    parts = query_planner_service._coordinated_aspect_questions("意外伤害保险定义及定价要求")

    assert parts == ["意外伤害保险定义", "意外伤害保险的定价要求"]
    alias = query_planner_service._document_style_alias(parts[1])
    assert "厘定保险费" in alias
    assert "公平 合理 定价假设" in alias


def test_coordinated_information_and_version_are_separate_aspects() -> None:
    parts = query_planner_service._coordinated_aspect_questions(
        "中资商业银行法人机构筹建申请书需说明的五类基本信息及目录版本"
    )

    assert parts == ["中资商业银行法人机构筹建申请书需说明的五类基本信息", "目录版本"]
    assert "2023年版" in query_planner_service._document_style_alias(parts[1])


def test_coordinated_principle_pair_splits_left_fact_and_right_principle() -> None:
    parts = query_planner_service._coordinated_aspect_questions("大额风险暴露与自救原则是什么？")

    assert parts == ["处置计划建议示例中的大额风险暴露是什么？", "自救原则是什么？"]
    assert "2.5%" in query_planner_service._document_style_alias(parts[0])
    assert "自救为本" in query_planner_service._document_style_alias(parts[1])


def test_summary_enumeration_splits_each_visible_requirement() -> None:
    parts = query_planner_service._coordinated_aspect_questions(
        "用一段可核验摘要串联意外伤害保险定义及定价要求、大额风险暴露与自救原则、第三支柱披露频率。"
    )

    assert parts == [
        "意外伤害保险定义",
        "意外伤害保险的定价要求",
        "大额风险暴露",
        "处置策略建议的自救原则",
        "第三支柱披露频率",
    ]
    assert "厘定保险费" in query_planner_service._document_style_alias(parts[1])
    assert "2.5%" in query_planner_service._document_style_alias(parts[2])
    assert "自救为本" in query_planner_service._document_style_alias(parts[3])
    assert "季度 半年 年度" in query_planner_service._document_style_alias(parts[4])


def test_bounded_lexical_support_extracts_required_percentage_anchor(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", None)
    plan = plan_query("大额风险暴露的门槛是多少？")
    aspect = plan.aspects[0]

    assert rag_service._bounded_required_numeric_terms(aspect) == ["2.5%"]


def test_cross_policy_direct_citation_question_uses_deterministic_domain_plan(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", "unused-key")

    plan = plan_query(
        "跨制度回答消费金融公司的业务地域范围、第三支柱表格类型，以及大额风险暴露门槛和处置自救原则。每项都需直接引用。"
    )

    assert plan.planner == "deterministic-domain"
    assert [aspect.aspect_id for aspect in plan.aspects] == [
        "consumer_finance_geographic_scope",
        "pillar3_table_types",
        "large_exposure_threshold",
        "bail_in_resolution_principle",
    ]
    assert any("全国范围" in query.query for query in plan.aspects[0].search_queries)
    assert any("固定表格和可变表格" in query.query for query in plan.aspects[1].search_queries)
    assert any("2.5%" in query.query for query in plan.aspects[2].search_queries)
    assert any("自救为本" in query.query for query in plan.aspects[3].search_queries)


def test_confirmation_and_trading_book_question_uses_deterministic_domain_plan(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", "unused-key")

    plan = plan_query("分别核验数字化银行回函的效力与回函时限，以及交易账簿划分政策内部审计的频率和留档要求。四项信息均不得省略。")

    assert plan.planner == "deterministic-domain"
    assert [aspect.aspect_id for aspect in plan.aspects] == [
        "digital_confirmation_effectiveness",
        "confirmation_response_deadline",
        "trading_book_policy_audit_retention",
    ]
    assert any("同等法律效力和证明力" in query.query for query in plan.aspects[0].search_queries)
    assert any("10个工作日" in query.query for query in plan.aspects[1].search_queries)
    assert any("每年" in query.query and "留档备查" in query.query for query in plan.aspects[2].search_queries)


def test_data_and_capital_trigger_question_uses_deterministic_domain_plan(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", "unused-key")

    plan = plan_query("对照两个触发口径：重要数据事件何时属于重大事件；资本工具何时触发持续经营事件、二级资本在何种顺序下吸收损失。")

    assert plan.planner == "deterministic-domain"
    assert [aspect.aspect_id for aspect in plan.aspects] == [
        "important_data_major_event",
        "going_concern_trigger_threshold",
        "tier2_loss_absorption_order",
    ]
    assert any("重大数据安全事件" in query.query for query in plan.aspects[0].search_queries)
    assert any("5.125%" in query.query for query in plan.aspects[1].search_queries)
    assert any("全部吸收损失" in query.query for query in plan.aspects[2].search_queries)


def test_cross_document_source_question_uses_natural_wording_domain_plan(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", "unused-key")

    plan = plan_query("请逐项给出来源：专利权质押登记线上全覆盖目标；重大数据事件的条件；数字化回函效力和合规询证函回复时限。")

    assert plan.planner == "deterministic-domain"
    assert [aspect.aspect_id for aspect in plan.aspects] == [
        "digital_confirmation_effectiveness",
        "confirmation_response_deadline",
        "important_data_major_event",
        "patent_pledge_registration_online_coverage",
    ]
    assert any("同等法律效力和证明力" in query.query for query in plan.aspects[0].search_queries)
    assert any("10个工作日" in query.query for query in plan.aspects[1].search_queries)
    assert any("省级区域经济" in query.query for query in plan.aspects[2].search_queries)
    assert any("全流程无纸化线上办理" in query.query for query in plan.aspects[3].search_queries)


def test_trading_book_and_pillar3_judgment_uses_deterministic_domain_plan(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", "unused-key")

    plan = plan_query("判断并纠正：交易账簿头寸原则上只需每月做一次公允价值计量，且商业银行第三支柱信息一律按单体口径披露。")

    assert plan.planner == "deterministic-domain"
    assert [aspect.aspect_id for aspect in plan.aspects] == [
        "trading_book_purpose_and_fair_value",
        "pillar3_disclosure_scope",
    ]
    assert any("自营业务" in query.query and "每日进行公允价值计量" in query.query for query in plan.aspects[0].search_queries)
    assert any("监管并表范围" in query.query for query in plan.aspects[1].search_queries)


def test_finance_insurance_confirmation_summary_uses_deterministic_domain_plan(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", "unused-key")

    plan = plan_query("形成一段跨制度工作摘要：概括2027年知识产权金融生态目标、列入名单保险集团的报告义务，以及银行函证事项应通过哪些公开渠道公示。")

    assert plan.planner == "deterministic-domain"
    assert [aspect.aspect_id for aspect in plan.aspects] == [
        "intellectual_property_finance_target",
        "insurance_group_report_obligation",
        "bank_confirmation_public_channels",
    ]
    assert any("知识产权金融生态综合试验区" in query.query for query in plan.aspects[0].search_queries)
    assert any("应当编报保险集团偿付能力报告的公司名单" in query.query for query in plan.aspects[1].search_queries)
    assert any("微信公众号" in query.query for query in plan.aspects[2].search_queries)


def test_three_control_requirements_summary_uses_deterministic_domain_plan(monkeypatch) -> None:
    monkeypatch.setattr(query_planner_service, "QUERY_PLANNER_API_KEY", "unused-key")

    plan = plan_query("汇总三项控制要求：注册会计师对询证函的过程控制、二级资本启动吸收损失的先后条件、第三支柱表格的两种类型。")

    assert plan.planner == "deterministic-domain"
    assert [aspect.aspect_id for aspect in plan.aspects] == [
        "tier2_loss_absorption_order",
        "auditor_confirmation_process_control",
        "pillar3_table_types",
    ]
    assert any("全过程保持控制" in query.query for query in plan.aspects[1].search_queries)
    assert any("固定表格和可变表格" in query.query for query in plan.aspects[2].search_queries)


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
    assert table.table_filters["row_label"] == "债券"


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
