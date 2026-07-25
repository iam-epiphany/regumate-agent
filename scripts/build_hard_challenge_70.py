"""Build a 70-case hard challenge with exactly 20 refusal cases.

The generated files are offline evaluation artifacts.  The public questions file
does not contain gold answers, evidence, calculations or forbidden conclusions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

try:
    from scripts.build_hard_challenge_50 import _clone, _read_jsonl, _write_jsonl
    from scripts.build_hard_challenge_50_round2 import (
        _combine,
        _formula_case,
        _judgment,
        _mixed,
        _multiple_choice,
        _reconciliation,
        _table_selection,
    )
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from build_hard_challenge_50 import _clone, _read_jsonl, _write_jsonl
    from build_hard_challenge_50_round2 import (
        _combine,
        _formula_case,
        _judgment,
        _mixed,
        _multiple_choice,
        _reconciliation,
        _table_selection,
    )


ROOT = Path(__file__).resolve().parents[1]
SOURCE_GOLD = ROOT / "data/evaluation/trust_challenge_100/gold.jsonl"
PRIOR_QUESTION_FILES = (
    ROOT / "data/evaluation/trust_challenge_100/questions.jsonl",
    ROOT / "data/evaluation/hard_challenge_50/round_1/questions.jsonl",
    ROOT / "data/evaluation/hard_challenge_50/round_2/questions.jsonl",
)
OUTPUT_DIR = ROOT / "data/evaluation/hard_challenge_70/round_1"

EXPECTED_TYPE_COUNTS = {
    "fact": 5,
    "multiple_choice": 5,
    "judgment": 5,
    "summary": 10,
    "cross_document": 5,
    "table_comparison": 5,
    "table_cross_period_calculation": 5,
    "mixed_regulation_table": 5,
    "formula": 5,
    "version_refusal": 8,
    "table_refusal": 8,
    "formula_refusal": 4,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_question(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _set_all_split(cases: list[dict[str, Any]]) -> None:
    for case in cases:
        case["difficulty"] = "hard"
        case["split"] = "all"
        case["review_status"] = "codex_verified"
        case["expert_reviewed"] = False


def _refusal_clone(
    source: dict[str, Any],
    case_id: str,
    question_type: str,
    scenario: str,
    question: str,
    *,
    canonical_answer: str | None = None,
    required_conclusions: list[str] | None = None,
    query: str | None = None,
) -> dict[str, Any]:
    case = _clone(
        source,
        case_id,
        question_type,
        scenario,
        question,
        answerable=False,
        canonical_answer=canonical_answer,
        required_conclusions=required_conclusions,
    )
    if query is not None:
        case["corpus_search"] = {
            "scope": "all_500_document_chunks_and_spreadsheet_cells",
            "query": query,
            "match_count": 0,
            "insufficiency_type": "nonexistent_indicator_or_period",
        }
    return case


def build_cases(g: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    # 1-5: multi-document fact cards.
    fact_specs = [
        ([g["TC001"], g["TC002"]], "同时核对知识产权金融生态试点到2027年的建设目标，以及重要数据事件构成特别重大的省级区域与后果条件。"),
        ([g["TC003"], g["TC009"]], "分别说明消费金融公司的业务地域范围，以及中资商业银行法人机构筹建申请书必须列明的五类基本信息和目录版本。"),
        ([g["TC004"], g["TC018"]], "核对银行询证函全过程控制主体，并说明函证事项应通过总行或总部哪些公开渠道公示。"),
        ([g["TC006"], g["TC020"]], "分别说明二级资本工具启动吸收损失的顺序，以及持续经营触发事件的核心一级资本充足率阈值。"),
        ([g["TC010"], g["TC022"]], "核对寿险折现率曲线终极利率暂定值，并列出第三支柱补充披露的频率要求。"),
    ]
    for index, (sources, question) in enumerate(fact_specs, start=1):
        cases.append(_combine(sources, f"H70-{index:03d}", "fact", "跨制度事实核验", question))

    # 6-10: options require complete evidence chains.
    cases.extend([
        _multiple_choice(g["TC053"], "H70-006", "下列哪组同时正确？\nA. 法人机构筹建申请书无需业务范围，寿险折现率曲线只有综合溢价\nB. 筹建申请书应说明名称、拟设地、注册资本、股权结构和业务范围等，目录为2023年版；寿险折现率曲线由基础利率加综合溢价形成\nC. 目录为2022年版，终极利率暂定为3%\nD. 两份材料均未规定", "B"),
        _multiple_choice(g["TC055"], "H70-007", "关于重要实体与处置工具，下列哪项完整？\nA. 重要实体只承载非核心业务\nB. 处置工具只包括破产清算\nC. 重要实体承载核心业务条线和关键功能；处置工具包括自救、注资、战略投资、不良资产处置、接管、收购承接、过桥机构和破产清算等\nD. 文件未说明处置工具", "C"),
        _multiple_choice(g["TC057"], "H70-008", "消费金融催收和银行函证要求中，哪项同时正确？\nA. 可向无关第三人催收，询证函可列多个基准日\nB. 禁止暴力等不正当催收且不得催收无关第三人；一函一个基准日并通过总行或总部公开渠道公示函证事项\nC. 只需口头公示函证事项\nD. 催收对象由外包机构自行确定", "B"),
        _multiple_choice(g["TC059"], "H70-009", "关于交易账簿与第三支柱披露，下列哪项正确？\nA. 交易头寸只含自营头寸\nB. 交易账簿原则上每月计量，第三支柱均按单体披露\nC. 交易头寸包括自营、做市、对客及相关对冲，原则上每日公允价值计量且变动计入损益；第三支柱按监管并表范围披露，表格另有规定除外\nD. 两项均由银行任意选择", "C"),
        _multiple_choice(g["TC060"], "H70-010", "关于行政许可和寿险负债折现，下列哪组正确？\nA. 分行筹建不是机构设立事项，折现率曲线只由基础利率形成\nB. 法人机构开业核准和分行筹建审批均属于机构设立事项；寿险合同负债折现率曲线由基础利率曲线加综合溢价形成\nC. 目录未列机构设立事项，终极利率暂定为5%\nD. 两份材料都不能回答", "B"),
    ])

    # 11-15: true/false with corrections.
    judgment_specs = [
        (g["TC007"], "H70-011", "判断并纠正：商业银行账簿转换只需高级管理层批准，且无需监管认可。", "说法错误。商业银行账簿转换应经高级管理层批准并经国家金融监督管理总局或其派出机构认可。", ["说法错误", *g["TC007"]["required_conclusions"]]),
        (g["TC010"], "H70-012", "判断并纠正：寿险合同负债折现率曲线终极利率暂定为5%。", "说法错误。寿险合同负债评估折现率曲线的终极利率暂定为3%。", ["说法错误", *g["TC010"]["required_conclusions"]]),
        (g["TC017"], "H70-013", "判断并说明依据：消费金融公司高级管理人员及对风险有重要影响岗位员工的绩效薪酬，是否至少40%需要延期支付且期限一般不少于3年？", "说法正确。消费金融公司高级管理人员以及对风险有重要影响岗位上的员工，绩效薪酬的40%以上应采取延期支付方式，且延期支付期限一般不少于3年。", ["说法正确", "绩效薪酬的40%以上应采取延期支付方式", "延期支付期限一般不少于3年"]),
        (g["TC021"], "H70-014", "判断并纠正：商业银行每两年对账簿划分政策和程序开展一次内部审计即可，结果无须留档。", "说法错误。商业银行应每年对账簿划分政策和程序开展内部审计，内部审计结果需留档备查。", ["说法错误", *g["TC021"]["required_conclusions"]]),
        (g["TC062"], "H70-015", "判断并说明依据：恢复计划压力测试至少覆盖系统性、自身和混合压力情景；处置计划沟通策略应覆盖监管部门、地方政府、股东、客户、员工和社会公众。", f"说法正确。{g['TC062']['canonical_answer']}", ["说法正确", *g["TC062"]["required_conclusions"]]),
    ]
    for source, case_id, question, canonical, conclusions in judgment_specs:
        cases.append(_judgment(source, case_id, question, canonical, conclusions))

    # 16-25: summaries, including three-source synthesis.
    summary_specs = [
        ([g["TC001"], g["TC011"], g["TC018"]], "请形成监管工作摘要：概括知识产权金融生态试点目标、保险集团名单报告义务和银行函证事项公示渠道。"),
        ([g["TC003"], g["TC013"], g["TC015"]], "请汇总消费金融业务地域范围、核心业务条线经营失败影响，以及试验区专利权质押登记线上办理目标。"),
        ([g["TC004"], g["TC006"], g["TC008"]], "请汇总银行询证函过程控制、二级资本损失吸收启动顺序和第三支柱表格类型三项要求。"),
        ([g["TC012"], g["TC014"], g["TC022"]], "请串联意外伤害保险定义及定价要求、商业银行大额风险暴露门槛与处置自救原则、第三支柱披露频率。"),
        ([g["TC016"], g["TC020"], g["TC021"]], "请形成一段审查摘要：重要数据事件构成重大事件的条件、资本工具持续经营触发阈值、账簿划分内部审计要求。"),
        ([g["TC019"], g["TC021"]], "请概括数字化银行回函效力、合规询证函回复时限，以及交易账簿划分政策内部审计频率和留档要求。"),
        ([g["TC008"], g["TC022"]], "请说明第三支柱披露表格类型和披露频率，并区分固定表格与可变表格。"),
        ([g["TC012"], g["TC017"]], "请分别概括意外伤害保险定义和保费厘定要求，以及消费金融公司薪酬递延和催收保护边界。"),
        ([g["TC014"], g["TC020"]], "请对照商业银行处置计划中的大额风险暴露门槛、自救原则，以及资本工具持续经营触发和损失吸收顺序。"),
        ([g["TC002"], g["TC016"]], "请比较重要数据事件在特别重大和重大两个层级中的影响范围与后果条件，避免混同省级区域门槛。"),
    ]
    for index, (sources, question) in enumerate(summary_specs, start=16):
        cases.append(_combine(sources, f"H70-{index:03d}", "summary", "跨制度综合摘要", question))

    # 26-30: explicit source-by-source cross-document tasks.
    cross_specs = [
        ([g["TC004"], g["TC006"], g["TC010"]], "分别引用三份材料回答：谁控制银行询证函全过程；何时启动二级资本吸收损失；寿险终极利率暂定多少。"),
        ([g["TC013"], g["TC018"], g["TC022"]], "逐项核对：核心业务条线经营失败影响、函证事项公开渠道、第三支柱披露频率。"),
        ([g["TC015"], g["TC016"], g["TC019"]], "分别说明试验区专利权质押登记线上办理目标、重大数据事件条件、数字化回函效力及回复时限。"),
        ([g["TC011"], g["TC012"], g["TC021"]], "分别核对保险集团报告义务、意外伤害保险定义与定价原则、账簿划分政策年度内部审计和留档。"),
        ([g["TC003"], g["TC008"], g["TC014"]], "跨制度回答消费金融业务地域范围、第三支柱表格类型、大额风险暴露门槛与处置自救原则。"),
    ]
    for index, (sources, question) in enumerate(cross_specs, start=26):
        cases.append(_combine(sources, f"H70-{index:03d}", "cross_document", "跨制度文件推理", question))

    # 31-35: table comparison across locked values.
    cases.extend([
        _table_selection([g["TC041"], g["TC045"], g["TC049"]], "H70-031", "比较2024年9月、2025年3月、2025年9月人身险公司原保险保费收入本年累计值，指出最高期间、数值和单位。", "三期中2025年9月最高，为38433.67亿元。", "38433.67", "2024-09、2025-03、2025-09"),
        _table_selection([g["TC042"], g["TC046"], g["TC050"]], "H70-032", "比较2024年9月、2025年3月、2025年9月保险业经营情况表原保险保费收入本年累计值，指出最高期间、数值和单位。", "三期中2025年9月最高，为52145.77亿元。", "52145.77", "2024-09、2025-03、2025-09"),
        _table_selection([g["TC044"], g["TC048"], g["TC052"]], "H70-033", "比较2024年9月、2025年3月、2025年9月财产险公司原保险保费收入本年累计值，指出最高期间、数值和单位。", "三期中2025年9月最高，为13712.1亿元。", "13712.1", "2024-09、2025-03、2025-09"),
        _table_selection([g["TC043"], g["TC047"], g["TC051"]], "H70-034", "比较2024年9月、2025年3月、2025年9月全国健康险合计，指出最高期间、数值和单位。", "三期中2025年9月最高，为8426.99亿元。", "8426.99", "2024-09、2025-03、2025-09"),
        _table_selection([g["TC032"], g["TC023"], g["TC026"]], "H70-035", "比较2023年8月、10月、12月人身险公司原保险保费收入本年累计值，指出最高月份、数值和单位。", "三个月中2023年12月最高，为35378.91亿元。", "35378.91", "2023-08、2023-10、2023-12"),
    ])

    # 36-40: cross-table Decimal calculations.
    cases.extend([
        _reconciliation([g["TC024"], g["TC023"], g["TC025"]], "H70-036", "读取2023年10月保险业合计、人身险、财产险原保险保费收入累计值，计算保险业合计－人身险－财产险的勾稽差额。", {"保险业合计": "45167.98", "人身险": "31739.18", "财产险": "13428.79"}, "0.01", "2023-10"),
        _reconciliation([g["TC033"], g["TC032"], g["TC035"]], "H70-037", "读取2023年8月保险业合计、人身险、财产险原保险保费收入累计值，计算保险业合计－人身险－财产险的勾稽差额。", {"保险业合计": "38731.77", "人身险": "27679.08", "财产险": "11052.68"}, "0.01", "2023-08"),
        _reconciliation([g["TC042"], g["TC041"], g["TC044"]], "H70-038", "读取2024年9月保险业合计、人身险、财产险原保险保费收入累计值，计算保险业合计－人身险－财产险。", {"保险业合计": "47945.35", "人身险": "34878.78", "财产险": "13066.56"}, "0.01", "2024-09"),
        _reconciliation([g["TC046"], g["TC045"], g["TC048"]], "H70-039", "读取2025年3月保险业合计、人身险、财产险原保险保费收入累计值，计算有符号勾稽差额：保险业合计－人身险－财产险。", {"保险业合计": "21745", "人身险": "16590.18", "财产险": "5155"}, "-0.18", "2025-03"),
        _reconciliation([g["TC050"], g["TC049"], g["TC052"]], "H70-040", "读取2025年9月保险业合计、人身险、财产险原保险保费收入累计值，计算保险业合计－人身险－财产险。", {"保险业合计": "52145.77", "人身险": "38433.67", "财产险": "13712.1"}, "0.00", "2025-09"),
    ])

    # 41-45: regulation/table pairings with explicit boundary.
    cases.extend([
        _mixed(g["TC019"], g["TC043"], "H70-041", "联合核验：说明数字化回函效力与10个工作日回复要求；比较2024年9月全国健康险全国合计、北京、天津、公司本级并给出最大项；最后说明能否据此认定某家机构合规。"),
        _mixed(g["TC014"], g["TC049"], "H70-042", "联合核验：说明大额风险暴露门槛和处置自救原则；比较2025年9月人身险表中意外险、寿险和原保险保费收入并给出最大项；不得作单一机构合规推断。"),
        _mixed(g["TC017"], g["TC052"], "H70-043", "联合核验：概括消费金融薪酬递延与催收禁限；比较2025年9月财产险表中原保险保费收入、机动车辆保险和责任险并给出最大项；最后说明证据边界。"),
        _mixed(g["TC020"], g["TC050"], "H70-044", "联合核验：给出资本工具持续经营触发阈值和损失吸收顺序；比较2025年9月保险业经营表的人身险、财产险与原保险保费收入并给出最大项；行业汇总不得外推到单一机构。"),
        _mixed(g["TC012"], g["TC051"], "H70-045", "联合核验：说明意外伤害保险定义及保费厘定要求；比较2025年9月全国健康险天津、北京、全国和河北值并给出最大项；最后说明证据边界。"),
    ])

    # 46-50: formula calculations with new operands.
    cases.extend([
        _formula_case(g["TC075"], "H70-046", "依据锁定 Word 公式计算 RWA：E=88，R=0.625；写出代入关系并给出精确结果。", {"E": "88", "R": "0.625"}, "687.5", canonical_answer="RWA=88×0.625×12.5=687.5。", required_conclusions=["RWA=88×0.625×12.5", "结果为687.5"]),
        _formula_case(g["TC075"], "H70-047", "依据 RWA=E×R×12.5 的锁定公式，E=64、R=0.25 时 RWA 等于多少？", {"E": "64", "R": "0.25"}, "200", canonical_answer="RWA=64×0.25×12.5=200。", required_conclusions=["RWA=64×0.25×12.5", "结果为200"]),
        _formula_case(g["TC077"], "H70-048", "依据锁定 Word 公式计算 LGD*：LGD_s=0.3，E_s=80，E=100，H_e=0.25，LGD_u=0.7，E_u=20。", {"LGD_s": "0.3", "E_s": "80", "E": "100", "H_e": "0.25", "LGD_u": "0.7", "E_u": "20"}, "0.304"),
        _formula_case(g["TC078"], "H70-049", "依据第三档商业银行资本监管 Word 公式，核心一级资本1600、扣除项100、调整后表内外资产余额20000时，计算杠杆率。", {"核心一级资本": "1600", "核心一级资本扣除项": "100", "调整后表内外资产余额": "20000"}, "7.5%"),
        _formula_case(g["TC075"], "H70-050", "再次使用 RWA=E×R×12.5 的锁定公式，E=120、R=0.4 时 RWA 等于多少？", {"E": "120", "R": "0.4"}, "600", canonical_answer="RWA=120×0.4×12.5=600。", required_conclusions=["RWA=120×0.4×12.5", "结果为600"]),
    ])

    # 51-70: exactly 20 refusal cases.
    for offset, source_id in enumerate(["TC083", "TC084", "TC085", "TC086", "TC087", "TC088", "TC089", "TC090"], start=51):
        title = g[source_id]["required_documents"][0]["filename"].split("_", 1)[-1]
        cases.append(_refusal_clone(
            g[source_id],
            f"H70-{offset:03d}",
            "version_refusal",
            "时效与版本拒答",
            f"只依据已确认文档身份字段，能否断言《{title}》当前现行、失效或已被替代？身份字段不足时必须说明未知并拒绝作法律效力断言。",
        ))

    table_refusals = [
        ("TC091", "2101年10月人身险公司经营情况表", "2101年"),
        ("TC092", "审慎监测不存在字段H70-01", "审慎监测不存在字段H70-01"),
        ("TC093", "2102年银行业金融机构保障性安居工程贷款情况表", "2102年"),
        ("TC094", "审慎监测不存在字段H70-02", "审慎监测不存在字段H70-02"),
        ("TC095", "2103年商业银行主要指标分机构类情况表", "2103年"),
        ("TC096", "审慎监测不存在字段H70-03", "审慎监测不存在字段H70-03"),
        ("TC098", "审慎监测不存在字段H70-04", "审慎监测不存在字段H70-04"),
        ("TC100", "审慎监测不存在字段H70-05", "审慎监测不存在字段H70-05"),
    ]
    for index, (source_id, label, query) in enumerate(table_refusals, start=59):
        answer = f"当前知识库中不存在{label}，依据不足，无法给出确定数值。"
        cases.append(_refusal_clone(
            g[source_id],
            f"H70-{index:03d}",
            "table_refusal",
            "可信拒答",
            f"请查询“{label}”对应的监管报表数值；若知识库和结构化表格中均无该期间或字段，必须明确拒答，不能返回相似指标或猜测数值。",
            canonical_answer=answer,
            required_conclusions=["依据不足", "无法给出确定数值"],
            query=query,
        ))

    formula_refusals = [
        ("TC079", "H70-067", "依据中央交易对手风险暴露资本计量规则中的复杂 Word 公式，给定 K_CCP=12、DF_i=18、pref=1、DF_CCP=30、DF_CM=4，若可信执行器不支持该运算，必须拒绝给出数值。"),
        ("TC080", "H70-068", "依据资产证券化风险加权资产计量规则中的 Word 公式计算 l：A=16、K_IRB=7；若 max 运算不受支持，必须说明能力边界并拒答。"),
        ("TC081", "H70-069", "依据资产管理产品风险加权资产计量规则中的 Word 公式计算 RWA_investment：Avg=3、RW_fund=4、Lvg=5、Equity=120、Investment=2；若 min 运算不受支持，必须拒答。"),
        ("TC082", "H70-070", "依据市场风险内部模型法监管要求中的 Word 公式尝试计算 SES：ISES_NM=4、SES_NM=9、ρ=0.25；若平方根运算未被可信执行器支持，必须明确拒答而非估算。"),
    ]
    for source_id, case_id, question in formula_refusals:
        cases.append(_refusal_clone(g[source_id], case_id, "formula_refusal", "公式能力边界", question))

    _set_all_split(cases)
    return cases


def validate(cases: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if len(cases) != 70:
        errors.append(f"expected 70 cases, got {len(cases)}")
    if [case["id"] for case in cases] != [f"H70-{index:03d}" for index in range(1, 71)]:
        errors.append("IDs/order must be H70-001..H70-070")
    prior_questions = [row for path in PRIOR_QUESTION_FILES for row in _read_jsonl(path)]
    prior_norm = {_normalize_question(row["question"]) for row in prior_questions}
    if any(_normalize_question(case["question"]) in prior_norm for case in cases):
        errors.append("one or more questions exactly duplicate a prior challenge")
    if len({_normalize_question(case["question"]) for case in cases}) != len(cases):
        errors.append("duplicate question text within hard70")
    if any(case["difficulty"] != "hard" or case["split"] != "all" for case in cases):
        errors.append("every case must be hard and split=all")
    answerable_count = sum(bool(case.get("answerable")) for case in cases)
    if answerable_count != 50:
        errors.append(f"hard70 must contain 50 answerable and 20 refusal cases, got {answerable_count}/{70 - answerable_count}")
    type_counts = {key: sum(case["question_type"] == key for case in cases) for key in EXPECTED_TYPE_COUNTS}
    if type_counts != EXPECTED_TYPE_COUNTS:
        errors.append(f"question type distribution mismatch: {type_counts}")
    if any(case["answerable"] and not case["evidence"] for case in cases):
        errors.append("answerable case without evidence")
    return errors


def main() -> int:
    source_rows = _read_jsonl(SOURCE_GOLD)
    cases = build_cases({row["id"]: row for row in source_rows})
    errors = validate(cases)
    if errors:
        raise ValueError("; ".join(errors))
    questions = [
        {key: case[key] for key in (
            "id", "scenario", "question_type", "difficulty", "split",
            "question", "answerable", "scoring_type",
        )}
        for case in cases
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    questions_path = OUTPUT_DIR / "questions.jsonl"
    gold_path = OUTPUT_DIR / "gold.jsonl"
    _write_jsonl(questions_path, questions)
    _write_jsonl(gold_path, cases)
    manifest = {
        "status": "built_not_yet_independently_audited",
        "profile": "hard70",
        "round": 1,
        "case_count": 70,
        "difficulty": {"hard": 70},
        "answerability": {"answerable": 50, "refusal": 20},
        "question_types": {
            question_type: sum(case["question_type"] == question_type for case in cases)
            for question_type in sorted({case["question_type"] for case in cases})
        },
        "design_constraints": {
            "prior_question_files_checked": [str(path.relative_to(ROOT)) for path in PRIOR_QUESTION_FILES],
            "gold_is_offline_only": True,
            "must_freeze_before_system_run": True,
        },
        "hashes": {
            "questions_sha256": _sha256(questions_path),
            "gold_sha256": _sha256(gold_path),
            "builder_sha256": _sha256(Path(__file__)),
        },
    }
    (OUTPUT_DIR / "build_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
