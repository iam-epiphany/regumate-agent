"""Remove document anchors from the 56 text questions (definition/rule/
threshold/scope), keeping subject+aspect constraints so each question stays
answerable.  Table questions keep their workbook names; cross-document
questions keep their natural document-name hints without 《》 formatting."""

import json
import re
from pathlib import Path

GOLD = Path("ReguMate-Eval-Private/banking_workbench_100/accepted/gold.jsonl")

# Manual de-anchored questions per case id (keep subject + aspect complete)
QUESTION_FIXES = {
    "BW-D01": "在银行业监管语境下，“消费金融公司”是如何定义的？",
    "BW-D02": "“消费贷款”的范围是如何界定的？",
    "BW-D03": "消费金融公司设立中，“主要出资人”是如何定义的？",
    "BW-D04": "银行监管资本计量中，“交易账簿”包括哪些内容？",
    "BW-D05": "其他一级资本工具的“持续经营触发事件”是指什么？",
    "BW-D06": "其他一级资本工具的“无法生存触发事件”包括哪些情形？",
    "BW-D07": "信用风险权重法下，“主权风险暴露”是指什么？",
    "BW-D08": "资产证券化监管计量中，“传统型资产证券化”是指什么？",
    "BW-D09": "寿险合同负债评估中，“寿险合同”包括哪些类型？",
    "BW-D10": "寿险合同负债评估中，“综合溢价”是如何确定的？",
    "BW-D11": "“一般数据安全事件”是如何界定的？",
    "BW-D12": "恢复计划编制中，“关键功能”是如何定义的？",
    "BW-D13": "恢复计划编制中，“关键共享服务”是如何定义的？",
    "BW-D14": "恢复计划编制中，“核心业务条线”是如何定义的？",
    "BW-D15": "银行函证工作中，对会计师事务所实施银行函证的集中化组织安排有什么要求？",
    "BW-R01": "消费金融公司名称的标示和审批有什么规定？",
    "BW-R02": "消费金融公司发生哪些变更事项应当报经国家金融监督管理总局或其派出机构批准？",
    "BW-R03": "金融机构入股消费金融公司的资金来源有什么要求？",
    "BW-R04": "消费金融公司防范欺诈风险有什么要求？",
    "BW-R05": "核心一级资本工具的发行方式有什么要求？",
    "BW-R06": "核心一级资本工具的抵押或保证安排有什么禁止性规定？",
    "BW-R07": "核心一级资本工具在破产清算时的受偿顺序有什么规定？",
    "BW-R08": "核心一级资本工具的收益分配有什么规定？",
    "BW-R09": "其他一级资本工具的到期日和赎回激励有什么要求？",
    "BW-R10": "其他一级资本工具的减记或转股条款有什么要求？",
    "BW-R11": "其他一级资本工具与二级资本工具的损失吸收顺序是怎样的？",
    "BW-R12": "二级资本工具的受偿顺序有什么规定？",
    "BW-R13": "一份银行询证函的函证基准日列示有什么要求？",
    "BW-R14": "数字化回函与纸质回函的法律效力关系是什么？",
    "BW-R15": "商业银行在固定表格中披露更多内容时有什么限制？",
    "BW-R16": "商业银行信息披露的并表口径有什么规定？",
    "BW-R17": "恢复计划压力测试的情景设置有什么最低要求？",
    "BW-R18": "意外险产品预定附加费用率的设定有什么规定？",
    "BW-R19": "商业银行使用自定义格式披露可变表格信息时有什么要求？",
    "BW-T01": "申请设立消费金融公司，注册资本有什么要求？",
    "BW-T02": "金融机构作为消费金融公司主要出资人，最近1个会计年度末总资产应达到什么标准？",
    "BW-T03": "消费金融公司向单一借款人发放消费贷款的授信额度上限有什么规定？",
    "BW-T04": "消费金融公司对具备消费金融业务管理和风险控制经验的出资人有什么要求？",
    "BW-T05": "二级资本工具的原始期限有什么最低要求？",
    "BW-T06": "发行银行赎回二级资本工具的最早时点有什么规定？",
    "BW-T07": "银行业金融机构回函的时限有什么要求？",
    "BW-T08": "什么情形构成特别重大数据安全事件？",
    "BW-T09": "金融机构作为消费金融公司一般出资人的注册资本有什么要求？",
    "BW-T10": "金融机构权益性投资余额占本企业净资产的比例有什么限制？",
    "BW-T11": "意外险业务回溯工作和整改时限有什么要求？",
    "BW-T12": "意外险产品任一渠道的年度佣金费用率超出平均附加费用率上限的情形有什么监管要求？",
    "BW-T13": "金融机构作为消费金融公司主要出资人应当具有多长的消费金融领域经营经验？",
    "BW-L01": "应当编报保险集团偿付能力报告的公司名单是如何规定的？",
    "BW-L02": "银行业金融机构的函证公示信息包括哪些内容？",
    "BW-L03": "数据安全事件分为哪几个级别？",
    "BW-L04": "恢复计划治理架构中的职责分工包括哪些主体？",
    "BW-L05": "商业银行第三支柱信息披露采用什么形式？",
    "BW-L06": "哪些情形发生时资本工具须设定无法生存触发事件？",
    "BW-L07": "划入交易账簿的头寸在估值与损益处理上有什么原则性要求？",
    "BW-L08": "知识产权金融生态综合试点工作的重点任务包括哪些方面？",
    "BW-L09": "银行询证函的函证方式包括哪些？",
}

DEANCHOR_PATTERN = re.compile(r"^(?:根据|关于|检索)《[^》]+》[,，]?\s*")


def main() -> None:
    rows = [json.loads(line) for line in GOLD.read_text(encoding="utf-8").splitlines() if line.strip()]
    changed = 0
    for row in rows:
        question = str(row.get("question") or "")
        if row["id"] in QUESTION_FIXES:
            row["question"] = QUESTION_FIXES[row["id"]]
            changed += 1
        else:
            # cross-document questions: remove 《》 formatting but keep names
            if row["question_type"] == "cross_document_evidence":
                row["question"] = re.sub(r"[《》]", "", question)
                changed += 1
    with open(GOLD, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("de-anchored", changed, "questions")


if __name__ == "__main__":
    main()
