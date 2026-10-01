"""Generate gold part 4: cross-document, supplement rule/threshold, refusal."""

import json
from pathlib import Path

rows = []

rows.append({
    "id": "BW-X01", "question": "《恢复计划示例（商业银行版）》和《处置计划建议示例（商业银行版）》对“关键功能”的定义是否一致？", "answerable": True,
    "question_type": "cross_document_evidence", "difficulty": "hard", "business_relevance": "恢复与处置计划编制中对关键功能识别的口径一致性",
    "canonical_answer": "两份文件对关键功能的定义一致：关键功能是提供给第三方的关键业务或产品等金融服务，当这些金融服务出现突发中断时将带来严重影响，可能引发市场风险传染或恐慌。",
    "required_conclusions": ["关键功能是提供给第三方的关键业务或产品等金融服务，当这些金融服务出现突发中断时将带来严重影响，可能引发市场风险传染或恐慌"],
    "required_evidence_aspects": ["关键功能定义跨文件一致性"],
    "required_sources": [
        {"relative_path": "500_银行保险机构恢复和处置计划实施暂行办法_附件1：恢复计划示例（商业银行版）.docx", "evidence_text": "关键功能是提供给第三方的关键业务或产品等金融服务，当这些金融服务出现突发中断时将带来严重影响，可能引发市场风险传染或恐慌。"},
        {"relative_path": "502_银行保险机构恢复和处置计划实施暂行办法_附件3：处置计划建议示例（商业银行版）.docx", "evidence_text": "关键功能是提供给第三方的关键业务或产品等金融服务，当这些金融服务出现突发中断时将带来严重影响，可能引发市场风险传染或恐慌。"},
    ],
    "calculation": None, "expected_refusal_code": None, "refusal_rationale": None,
})
rows.append({
    "id": "BW-X02", "question": "《恢复计划示例（商业银行版）》和《恢复计划示例（保险公司版）》对恢复计划治理架构中职责分工的规定是否一致？", "answerable": True,
    "question_type": "cross_document_evidence", "difficulty": "hard", "business_relevance": "恢复计划编制中治理架构要求的一致性",
    "canonical_answer": "两份文件的规定一致：恢复计划治理架构中的职责分工包括董事会、高管层、相关部门在恢复计划制定、审批、更新、执行等工作中承担的职责。",
    "required_conclusions": ["恢复计划治理架构中的职责分工包括董事会、高管层、相关部门在恢复计划制定、审批、更新、执行等工作中承担的职责"],
    "required_evidence_aspects": ["治理架构跨文件一致性"],
    "required_sources": [
        {"relative_path": "500_银行保险机构恢复和处置计划实施暂行办法_附件1：恢复计划示例（商业银行版）.docx", "evidence_text": "恢复计划治理架构中的职责分工包括董事会、高管层、相关部门在恢复计划制定、审批、更新、执行等工作中承担的职责。"},
        {"relative_path": "501_银行保险机构恢复和处置计划实施暂行办法_附件2：恢复计划示例（保险公司版）.docx", "evidence_text": "恢复计划治理架构中的职责分工包括董事会、高管层、相关部门在恢复计划制定、审批、更新、执行等工作中承担的职责。"},
    ],
    "calculation": None, "expected_refusal_code": None, "refusal_rationale": None,
})
rows.append({
    "id": "BW-X03", "question": "《消费金融公司管理办法》规定消费金融公司可以申请发行资本工具，发行时应当符合什么监管要求？", "answerable": True,
    "question_type": "cross_document_evidence", "difficulty": "hard", "business_relevance": "消费金融公司资本工具发行合规",
    "canonical_answer": "符合条件的消费金融公司可以申请发行资本工具，并应当符合监管要求的相关合格标准；按《商业银行资本管理办法》附件《资本工具合格标准》，核心一级资本工具应当直接发行且实缴。",
    "required_conclusions": ["符合条件的消费金融公司可以申请发行资本工具，并应当符合监管要求的相关合格标准"],
    "required_evidence_aspects": ["消费金融公司资本工具合格标准"],
    "required_sources": [
        {"relative_path": "396_消费金融公司管理办法_消费金融公司管理办法.doc", "evidence_text": "第十七条 符合条件的消费金融公司可以申请发行资本工具，并应当符合监管要求的相关合格标准。"},
        {"relative_path": "400_商业银行资本管理办法_附件1：资本工具合格标准.docx", "evidence_text": "一、核心一级资本工具的合格标准 （一）直接发行且实缴的。"},
    ],
    "calculation": None, "expected_refusal_code": None, "refusal_rationale": None,
})
rows.append({
    "id": "BW-X04", "question": "《保险公司偿付能力监管规则第3号：寿险合同负债评估》和《市场风险（利率风险）》监管规则中，寿险合同负债评估的折现率曲线是如何构成的？", "answerable": True,
    "question_type": "cross_document_evidence", "difficulty": "hard", "business_relevance": "寿险合同负债评估中折现率曲线的理解",
    "canonical_answer": "寿险合同负债评估中计算现金流现值所采用的折现率曲线由基础利率曲线加综合溢价形成；综合溢价由银保监会综合考虑行业整体资产配置情况、国债收益率的税收效应、流动性补偿及逆周期调整等因素设定。",
    "required_conclusions": ["折现率曲线由基础利率曲线加综合溢价形成", "综合溢价由银保监会综合考虑行业整体资产配置情况、国债收益率的税收效应、流动性补偿及逆周期调整等因素设定"],
    "required_evidence_aspects": ["折现率曲线构成"],
    "required_sources": [
        {"relative_path": "474_中国银保监会关于印发保险公司偿付能力_监管规则（Ⅱ）的通知_保险公司偿付能力监管规则第3号：寿险合同负债评估_.pdf", "evidence_text": "计算现金流现值所采用的折现率曲线由基础利率曲线加综合溢价形成。"},
        {"relative_path": "479_中国银保监会关于印发保险公司偿付能力_监管规则（Ⅱ）的通知_保险公司偿付能力监管规则第8号：市场风险最.docx", "evidence_text": "综合溢价由银保监会综合考虑行业整体资产配置情况、国债收益率的税收效应、流动性补偿及逆周期调整等因素设定。"},
    ],
    "calculation": None, "expected_refusal_code": None, "refusal_rationale": None,
})
rows.append({
    "id": "BW-R19", "question": "《商业银行信息披露内容和要求》中，商业银行使用自定义格式披露可变表格信息时有什么要求？", "answerable": True,
    "question_type": "rule_scope", "difficulty": "medium", "business_relevance": "第三支柱可变表格披露合规",
    "canonical_answer": "如使用自定义格式披露信息，应提供与给定格式等效可比的信息内容，并达到给定格式的颗粒度水平。",
    "required_conclusions": ["如使用自定义格式披露信息，应提供与给定格式等效可比的信息内容，并达到给定格式的颗粒度水平"],
    "required_evidence_aspects": ["自定义格式披露要求"],
    "required_sources": [{"relative_path": "417_商业银行资本管理办法_附件22：商业银行信息披露内容和要求.docx", "evidence_text": "如使用自定义格式披露信息，应提供与给定格式等效可比的信息内容，并达到给定格式的颗粒度水平。"}],
    "calculation": None, "expected_refusal_code": None, "refusal_rationale": None,
})
rows.append({
    "id": "BW-T13", "question": "《消费金融公司管理办法》中，金融机构作为消费金融公司主要出资人应当具有多长的消费金融领域经营经验？", "answerable": True,
    "question_type": "threshold_rule", "difficulty": "easy", "business_relevance": "消费金融公司主要出资人资质审查",
    "canonical_answer": "金融机构作为消费金融公司的主要出资人，应当具有5年以上消费金融领域的经营经验。",
    "required_conclusions": ["金融机构作为消费金融公司的主要出资人，应当具有5年以上消费金融领域的经营经验"],
    "required_evidence_aspects": ["主要出资人经营经验要求"],
    "required_sources": [{"relative_path": "396_消费金融公司管理办法_消费金融公司管理办法.doc", "evidence_text": "（一）具有5年以上消费金融领域的经营经验"}],
    "calculation": None, "expected_refusal_code": None, "refusal_rationale": None,
})

refusal_templates = [
    ("BW-F01", "请计算2026年二季度银行业金融机构资本充足率。", "missing_calculation_operands", "medium", "语料中的资本充足率数据仅覆盖至2025年，2026年二季度数据未收录，无法计算。"),
    ("BW-F02", "请计算2026年全年全国原保险保费收入。", "missing_calculation_operands", "medium", "语料中的保险业经营数据仅覆盖至2025年，2026年全年数据不存在。"),
    ("BW-F03", "某商业银行2025年末总资产为3.5万亿元、不良贷款余额为120亿元，请计算该行不良贷款率。", "missing_calculation_operands", "medium", "题面给出的机构数值不在语料中，且未指明是哪家银行，无法从资料库核验并计算。"),
    ("BW-F04", "请计算2027年一季度银行业金融机构普惠型小微企业贷款余额。", "missing_calculation_operands", "medium", "语料中的普惠型小微企业贷款数据仅覆盖至2025年，2027年一季度数据不存在。"),
    ("BW-F05", "请计算2025年保险业的综合成本率。", "missing_calculation_operands", "medium", "语料中的保险业经营情况表不含综合成本率指标，缺少计算操作数。"),
    ("BW-F06", "请提供当前最新的一年期LPR报价。", "out_of_corpus_scope", "easy", "LPR报价属于外部实时市场数据，不在监管资料库范围内。"),
    ("BW-F07", "请提供中国工商银行2025年年度报告中的净利润数据。", "out_of_corpus_scope", "easy", "上市银行年报数据不在监管统计资料库范围内。"),
    ("BW-F08", "请说明某银行内部信贷审批流程的具体要求。", "out_of_corpus_scope", "easy", "机构内部信贷审批流程属于机构内部资料，不在监管资料库范围内。"),
    ("BW-F09", "请提供招商银行与建设银行2025年不良贷款率的对比数据。", "out_of_corpus_scope", "easy", "个别上市银行的自有财务数据不在监管统计资料库范围内。"),
    ("BW-F10", "请提供当前商业银行一年期定期存款利率。", "out_of_corpus_scope", "easy", "存款利率属于外部实时市场数据，不在监管资料库范围内。"),
    ("BW-F11", "请预测2026年一季度保险业原保险保费收入增速。", "prediction_request", "hard", "保费增速预测属于前瞻性预测，超出资料库可验证范围。"),
    ("BW-F12", "请预测未来一年商业银行不良贷款率走势。", "prediction_request", "hard", "不良贷款率走势预测属于前瞻性预测，超出资料库可验证范围。"),
    ("BW-F13", "请预测利率下行对银行资本充足率的影响幅度。", "prediction_request", "hard", "利率变动的量化影响预测属于前瞻性分析，资料库无法提供预测依据。"),
    ("BW-F14", "请预计某银行明年净利润规模。", "prediction_request", "hard", "净利润预测属于前瞻性预测，超出资料库可验证范围。"),
    ("BW-F15", "请预测《资本管理办法》新规实施后银行风险加权资产的变化比例。", "prediction_request", "hard", "新规影响的量化预测属于前瞻性分析，资料库无法提供预测依据。"),
    ("BW-F16", "请判断某银行是否应当新设一家县域支行。", "subjective_business_judgment", "medium", "机构布局决策属于银行经营自主决策事项，资料库无法给出主观判断。"),
    ("BW-F17", "请建议某银行是否应上调消费贷款产品定价。", "subjective_business_judgment", "medium", "产品定价调整属于银行经营自主决策事项，资料库无法给出主观建议。"),
    ("BW-F18", "请建议某银行是否应增加二级资本工具发行规模。", "subjective_business_judgment", "medium", "资本补充时点与规模属于银行经营决策事项，资料库无法给出主观建议。"),
    ("BW-F19", "请评估某银行收购一家农村商业银行的可行性。", "subjective_business_judgment", "medium", "并购决策属于银行经营自主决策事项，资料库无法给出主观评估。"),
    ("BW-F20", "请建议某银行是否应下调其信用风险权重计量方法。", "subjective_business_judgment", "medium", "计量方法选择属于银行经营决策事项，资料库无法给出主观建议。"),
]
for refusal_id, question, code, difficulty, rationale in refusal_templates:
    rows.append({
        "id": refusal_id, "question": question, "answerable": False,
        "question_type": "refusal", "difficulty": difficulty, "business_relevance": "拒答边界测试",
        "canonical_answer": "", "required_conclusions": [], "required_evidence_aspects": [],
        "required_sources": [], "calculation": None,
        "expected_refusal_code": code, "refusal_rationale": rationale,
    })

out_path = Path("ReguMate-Eval-Private/banking_workbench_100/candidates/gold_part4_cross_refusal.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print("wrote", len(rows), "rows")
