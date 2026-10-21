from backend.app.schemas.qa import EvidenceRequirement, RetrievalResult
from backend.app.services.query_planner_service import QueryAspect, QueryPlan
from backend.app.services.answer_generation_service import (
    _catalogue_deterministic_answer,
    _composition_list_answer,
    _disclosure_list_item_answer,
    _formation_sentence_answer,
    _joint_duty_answer,
    _list_intro_answer,
    _minimum_list_answer,
    _procedure_sentence_answer,
    _range_deterministic_answer,
    _scope_rule_deterministic_answer,
    _section_list_deterministic_answer,
    _table_row_deterministic_answer,
)
from backend.app.services.rag_service import _bind_cross_aspect_evidence


def _chunk(
    chunk_id: str,
    text: str,
    *,
    metadata: dict | None = None,
    section_title: str | None = None,
    citation_label: str = "[1]",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        rank=1,
        source_doc="制度.pdf",
        section_title=section_title,
        text=text,
        citation_label=citation_label,
        metadata=metadata or {},
    )


def test_catalogue_answer_lists_categories_and_attribution() -> None:
    toc = _chunk(
        "toc",
        "目 录 一、机构设立 二、机构变更",
        metadata={
            "catalogue_summary": "一、机构设立；二、机构变更；三、机构终止；"
            "四、调整业务范围和增加业务品种；五、董事和高级管理人员任职资格审核；"
            "六、申请材料格式要求",
        },
        section_title="目 录",
    )
    body = _chunk(
        "body",
        "1.1 中资商业银行法人机构筹建审批 申请材料目录",
        metadata={
            "section_path": ["一、机构设立", "（一）法人机构设立", "1.1 中资商业银行法人机构筹建审批"],
        },
        section_title="1.1 中资商业银行法人机构筹建审批",
        citation_label="[2]",
    )
    answer = _catalogue_deterministic_answer(
        "中资商业银行行政许可事项申请材料目录包括哪些类别，"
        "中资商业银行法人机构筹建审批属于哪一类行政许可事项？",
        [toc, body],
    )
    assert answer is not None
    assert "机构设立" in answer.answer
    assert "机构变更" in answer.answer
    assert "中资商业银行法人机构筹建审批属于机构设立类行政许可事项" in answer.answer
    assert "申请材料格式要求" not in answer.answer


def test_range_answer_reads_numeric_interval() -> None:
    chunk = _chunk("clause", "终极利率过渡曲线适用于20<t≤40的区间，其中t表示年度。")
    answer = _range_deterministic_answer(
        "终极利率过渡曲线适用于哪个期限区间？", [chunk]
    )
    assert answer is not None
    assert answer.answer == "终极利率过渡曲线适用于20年到40年的区间。"


def test_scope_rule_answer_prefers_operative_sentence() -> None:
    chunk = _chunk(
        "scope",
        "表头：目的：a、b列披露并表范围差异。\n"
        "商业银行应按照本办法规定的并表范围披露相关信息。表格中另有规定的除外。",
    )
    answer = _scope_rule_deterministic_answer(
        "商业银行披露信息时应按什么并表范围进行，何种情形可以例外？", [chunk]
    )
    assert answer is not None
    assert "按照本办法规定的并表范围披露相关信息" in answer.answer
    assert "表格中另有规定的除外" in answer.answer
    assert "表头" not in answer.answer


def test_table_row_answer_uses_named_row() -> None:
    personal = _chunk(
        "personal",
        "表格行证据：业务类型“个人”；保险期限一年及以下的意外险=“35%”。",
        metadata={
            "row_cells": {
                "业务类型": "个人",
                "保险期限一年及以下的意外险": "35%",
                "保险期限一年以上期交": "35%",
            }
        },
    )
    group = _chunk(
        "group",
        "表格行证据：业务类型“团体”；保险期限一年及以下的意外险=“25%”。",
        metadata={
            "row_cells": {
                "业务类型": "团体",
                "保险期限一年及以下的意外险": "25%",
                "保险期限一年以上期交": "15%",
            }
        },
        citation_label="[2]",
    )
    answer = _table_row_deterministic_answer(
        "个人意外险与团体意外险，保险期限一年及以下的平均附加费用率上限分别是多少？",
        [personal, group],
    )
    assert answer is not None
    assert "个人意外险平均附加费用率上限为35%" in answer.answer
    assert "团体意外险平均附加费用率上限为25%" in answer.answer


def test_section_list_answer_chooses_matching_section() -> None:
    wrong = _chunk(
        "wrong",
        "恢复措施",
        metadata={
            "section_path": ["三、恢复措施"],
            "section_list_summary": "（一）恢复措施概览；（二）恢复措施分析；（三）恢复能力",
        },
    )
    right = _chunk(
        "right",
        "处置计划实施所需信息和数据",
        metadata={
            "section_path": ["四、处置计划实施所需信息和数据"],
            "section_list_summary": "（一）资产质量；（二）股东股权及关联交易情况；（三）内部交易情况",
        },
        citation_label="[2]",
    )
    answer = _section_list_deterministic_answer(
        "处置计划实施中所需的信息和数据应涵盖哪些方面？", [wrong, right]
    )
    assert answer is not None
    assert answer.answer.startswith("处置计划实施所需信息和数据包括")
    assert "资产质量" in answer.answer
    assert "恢复措施" not in answer.answer


def test_section_list_scopes_items_to_named_section_and_prefixes_source() -> None:
    chunk = _chunk(
        "summary",
        "（一）机构概况\n1.经营情况。机构整体经营情况。",
        metadata={
            "section_path": ["一、概要", "（一）机构概况"],
            "source_filename": "502_制度_附件3：处置计划建议示例（商业银行版）.docx",
            "section_list_summary": (
                "（一）机构概况；1.经营情况。机构整体经营情况；"
                "2.组织架构。机构设置情况；3.主要分支机构和子公司情况。列表及持股比例；"
                "（二）处置计划建议更新情况；本次更新的主要内容"
            ),
        },
    )
    answer = _section_list_deterministic_answer(
        "《处置计划建议示例（商业银行版）》中，机构概况部分应说明哪些方面的信息？",
        [chunk],
    )
    assert answer is not None
    assert answer.answer == (
        "处置计划建议示例的机构概况包括经营情况、组织架构、主要分支机构和子公司情况。"
    )


def test_formation_answer_uses_longest_overlap_and_trims_explanation() -> None:
    chunk = _chunk(
        "formation",
        "该折现率曲线由基础利率曲线加综合溢价形成，基础利率曲线为即期曲线并由三段组成。",
    )
    answer = _formation_sentence_answer(
        "寿险合同负债评估中计算现金流现值所采用的折现率曲线是如何构成的？",
        [chunk],
    )
    assert answer is not None
    assert answer.answer == "该折现率曲线由基础利率曲线加综合溢价形成。"


def test_section_list_reads_numbered_items_from_chunk_text() -> None:
    chunk = _chunk(
        "section-text",
        "（一）机构概况\n1.经营情况。机构整体经营情况。\n"
        "2.组织架构。机构设置情况。\n3.主要分支机构和子公司情况。列表及持股比例。",
        metadata={
            "section_path": ["一、概要", "（一）机构概况"],
            "source_filename": "502_制度_附件3：处置计划建议示例（商业银行版）.docx",
        },
    )
    answer = _section_list_deterministic_answer(
        "《处置计划建议示例（商业银行版）》中，机构概况部分应说明哪些方面的信息？",
        [chunk],
    )
    assert answer is not None
    assert answer.answer == (
        "处置计划建议示例的机构概况包括经营情况、组织架构、主要分支机构和子公司情况。"
    )


def test_minimum_list_answer_returns_at_least_sentence() -> None:
    chunk = _chunk(
        "minimum",
        "恢复计划压力测试情景设置至少包括系统性压力情景、自身压力情景和混合压力情景。",
    )
    answer = _minimum_list_answer(
        "商业银行在恢复计划中开展压力测试时，压力情景设置至少应涵盖哪些类别？",
        [chunk],
    )
    assert answer is not None
    assert answer.answer == "恢复计划压力测试情景设置至少包括系统性压力情景、自身压力情景和混合压力情景。"


def test_formation_answer_accepts_three_part_composition() -> None:
    chunk = _chunk(
        "curve",
        "基础利率曲线为即期曲线，由750日移动平均国债收益率曲线、终极利率过渡曲线和终极利率三段组成。",
    )
    answer = _formation_sentence_answer(
        "寿险合同负债评估中的基础利率曲线采用哪类曲线，并由哪些组成部分构成？",
        [chunk],
    )
    assert answer is not None
    assert "750日移动平均国债收益率曲线" in answer.answer
    assert "三段组成" in answer.answer


def test_catalogue_answer_maps_items_from_printed_toc_without_paths() -> None:
    toc = _chunk(
        "toc-fallback",
        "|一、机构设立|（一）法人机构设立|"
        "1.2中资商业银行法人机构开业核准................3|"
        "（二）境内分支机构设立|"
        "1.3中资商业银行分行筹建审批....................4|",
    )
    answer = _catalogue_deterministic_answer(
        "中资商业银行法人机构开业核准与分行筹建审批分别归入哪一类行政许可事项？",
        [toc],
    )
    assert answer is not None
    assert "中资商业银行法人机构开业核准属于机构设立类行政许可事项" in answer.answer
    assert "中资商业银行分行筹建审批属于机构设立类行政许可事项" in answer.answer


def test_joint_duty_answer_keeps_merged_source_clause() -> None:
    chunk = _chunk(
        "duty",
        "1.与处置计划相关的公司治理，包括董事会、高管层、相关部门在处置计划建议的"
        "制定、审批、更新和处置计划执行中承担的职责；处置机构的职责。",
    )
    answer = _joint_duty_answer(
        "商业银行处置计划治理架构中，董事会、高管层及相关部门分别在处置计划建议的"
        "哪些环节承担职责？",
        [chunk],
    )
    assert answer is not None
    assert answer.answer.startswith("处置计划治理架构中的职责分工包括")
    assert "董事会、高管层、相关部门" in answer.answer
    assert "处置计划执行" in answer.answer


def test_composition_list_answer_reads_three_segments() -> None:
    chunk = _chunk(
        "curve-segments",
        "一、基础利率曲线为即期曲线，由以下三段组成: "
        "750 日移动平均国债收益率曲线 0 < t≤20 "
        "终极利率过渡曲线 20 < t≤40 终极利率 t > 40",
    )
    answer = _composition_list_answer(
        "寿险合同负债评估中的基础利率曲线采用哪类曲线，并由哪些组成部分构成？",
        [chunk],
    )
    assert answer is not None
    assert answer.answer == (
        "基础利率曲线为即期曲线，由750日移动平均国债收益率曲线、"
        "终极利率过渡曲线和终极利率三段组成。"
    )


def test_composition_list_answer_requires_matching_subject() -> None:
    chunk = _chunk(
        "curve-segments",
        "一、基础利率曲线为即期曲线，由以下三段组成: "
        "750 日移动平均国债收益率曲线 0 < t≤20 "
        "终极利率过渡曲线 20 < t≤40 终极利率 t > 40",
    )
    answer = _composition_list_answer(
        "寿险合同负债评估中计算现金流现值所采用的折现率曲线由哪些部分构成？",
        [chunk],
    )
    assert answer is None


def test_list_intro_answer_returns_operative_list_sentence() -> None:
    chunk = _chunk(
        "list-intro",
        "应当编报保险集团偿付能力报告的公司名单。"
        "下列保险集团应当按照《保险公司偿付能力监管规则第19号：保险集团》"
        "有关规定，编报保险集团偿付能力报告。",
    )
    answer = _list_intro_answer(
        "检索《应当编报保险集团偿付能力报告的公司名单》后，"
        "以下哪一项与材料内容一致。请直接给出规定内容。",
        [chunk],
    )
    assert answer is not None
    assert answer.answer.startswith("下列保险集团应当按照")
    assert "编报保险集团偿付能力报告" in answer.answer


def test_procedure_answer_returns_verbatim_how_to_sentence() -> None:
    chunk = _chunk(
        "procedure",
        "会计师事务所应当安排专门部门集中发送、收回银行询证函，"
        "采用公示地址作为邮寄地址，直接发出银行询证函并直接获取回函（跟函除外）。",
    )
    answer = _procedure_sentence_answer(
        "会计师事务所应如何确定询证函的邮寄地址，并完成询证函的发出与回函收取？",
        [chunk],
    )
    assert answer is not None
    assert "采用公示地址作为邮寄地址" in answer.answer
    assert "直接发出银行询证函" in answer.answer
    assert "跟函除外" not in answer.answer


def test_disclosure_list_item_selects_question_relevant_item() -> None:
    chunk = _chunk(
        "disclosure",
        "公示信息包括：1.各种函证方式下办理回函工作的机构及其联系方式。"
        "2.受理函证事项（1—14项及附表）及其办理机构。"
        "在公示后，对于符合公示条件的申请不得拒绝。3.函证范围和回函用章。",
    )
    answer = _disclosure_list_item_answer(
        "银行业金融机构公示信息中，与函证受理相关的具体内容是什么？",
        [chunk],
    )
    assert answer is not None
    assert answer.answer == "银行业金融机构公示信息包括受理函证事项及其办理机构。"


def test_cross_aspect_binding_marks_evidence_found_by_another_aspect() -> None:
    chunk = _chunk(
        "combined",
        "银行业金融机构应当在其总行或总部网站、微信公众号等公开渠道"
        "就办理函证相关事项进行公示。",
        metadata={"prompt_matched_aspects": ["A1"], "aspect_ids": ["A1"]},
    )
    plan = QueryPlan(
        original_question="A？B？",
        aspects=(
            QueryAspect(
                aspect_id="A1", question="银行询证函如何列示？",
                search_queries=(), evidence_need="", keywords=(),
            ),
            QueryAspect(
                aspect_id="A2", question="应通过哪些公开渠道公示？",
                search_queries=(), evidence_need="", keywords=(),
            ),
        ),
        requirements=(
            EvidenceRequirement(
                requirement_id="A1",
                question="银行询证函如何列示？",
                source_titles=["制度"],
                target_terms=["询证函"],
            ),
            EvidenceRequirement(
                requirement_id="A2",
                question="应通过哪些公开渠道公示？",
                source_titles=["制度"],
                target_terms=["公开渠道就办理函证相关事项进行公示"],
            ),
        ),
        planner="test",
    )
    _bind_cross_aspect_evidence([chunk], plan)
    assert "A2" in chunk.metadata["prompt_matched_aspects"]
