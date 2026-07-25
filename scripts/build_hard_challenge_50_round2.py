"""Build Hard-50 round 2 from new evidence combinations over the frozen corpus.

The runner never imports this file or the offline gold.  Round 2 deliberately
changes the reasoning task (new document combinations, reconciliations and
formula operands) instead of paraphrasing round-1 questions.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any

try:
    from scripts.build_hard_challenge_50 import (
        _clone,
        _compose,
        _read_jsonl,
        _write_jsonl,
    )
except ModuleNotFoundError:  # direct ``python scripts/...`` execution
    from build_hard_challenge_50 import (
        _clone,
        _compose,
        _read_jsonl,
        _write_jsonl,
    )


ROOT = Path(__file__).resolve().parents[1]
SOURCE_GOLD = ROOT / "data/evaluation/trust_challenge_100/gold.jsonl"
PRIOR_QUESTION_FILES = (
    ROOT / "data/evaluation/trust_challenge_100/questions.jsonl",
    ROOT / "data/evaluation/hard_challenge_50/round_1/questions.jsonl",
)
OUTPUT_DIR = ROOT / "data/evaluation/hard_challenge_50/round_2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_question(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _view(source: dict[str, Any], indexes: list[int]) -> dict[str, Any]:
    result = deepcopy(source)
    result["evidence"] = [deepcopy(source["evidence"][index]) for index in indexes]
    document_ids = {item["document_id"] for item in result["evidence"]}
    result["required_documents"] = [
        deepcopy(item) for item in source["required_documents"] if item["document_id"] in document_ids
    ]
    return result


def _combined_answer(sources: list[dict[str, Any]]) -> tuple[str, list[str]]:
    return (
        "\n".join(str(source["canonical_answer"]) for source in sources),
        [item for source in sources for item in source["required_conclusions"]],
    )


def _combine(
    sources: list[dict[str, Any]],
    case_id: str,
    question_type: str,
    scenario: str,
    question: str,
    *,
    scoring_type: str = "multi_assertion",
) -> dict[str, Any]:
    answer, conclusions = _combined_answer(sources)
    return _compose(
        sources,
        case_id,
        question_type,
        scenario,
        question,
        answer,
        conclusions,
        scoring_type=scoring_type,
    )


def _set_answer(
    case: dict[str, Any],
    canonical: str,
    conclusions: list[str],
    *,
    forbidden: list[str] | None = None,
) -> dict[str, Any]:
    case["canonical_answer"] = canonical
    case["equivalent_answers"] = [canonical]
    case["allowed_expressions"] = [canonical]
    case["required_conclusions"] = conclusions
    if forbidden is not None:
        case["forbidden_conclusions"] = forbidden
    return case


def _multiple_choice(
    source: dict[str, Any],
    case_id: str,
    question: str,
    option: str,
) -> dict[str, Any]:
    case = _clone(
        source,
        case_id,
        "multiple_choice",
        "跨文件选择题",
        question,
        scoring_type="multiple_choice",
    )
    canonical = f"正确选项为：{option}。{source['canonical_answer']}"
    return _set_answer(case, canonical, [f"正确选项为：{option}", *source["required_conclusions"]])


def _judgment(
    source: dict[str, Any],
    case_id: str,
    question: str,
    canonical: str,
    conclusions: list[str],
    *,
    evidence_indexes: list[int] | None = None,
) -> dict[str, Any]:
    case = _clone(
        source,
        case_id,
        "judgment",
        "多条件判断与纠错",
        question,
        scoring_type="judgment",
        evidence_indexes=evidence_indexes,
    )
    return _set_answer(case, canonical, conclusions)


def _table_selection(
    sources: list[dict[str, Any]],
    case_id: str,
    question: str,
    conclusion: str,
    result: str,
    periods: str,
) -> dict[str, Any]:
    calculation = {
        "operands": {
            str(source["id"]): str((source.get("calculation") or {}).get("result") or "")
            for source in sources
        },
        "unit": "亿元",
        "period": {"raw": periods},
        "formula": "",
        "rounding": "按各源表锁定值的原始精度比较",
        "result": result,
    }
    return _compose(
        sources,
        case_id,
        "table_comparison",
        "跨文件表格比较",
        question,
        conclusion,
        [conclusion],
        scoring_type="numeric",
        calculation=calculation,
    )


def _reconciliation(
    sources: list[dict[str, Any]],
    case_id: str,
    question: str,
    operands: dict[str, str],
    result: str,
    period: str,
) -> dict[str, Any]:
    conclusion = f"按‘保险业合计－人身险－财产险’计算的差额为{result}亿元。"
    calculation = {
        "operands": operands,
        "unit": "亿元",
        "period": {"raw": period},
        "formula": "保险业合计-人身险-财产险",
        "rounding": "使用三张源表锁定值做 Decimal 减法并保留结果精度",
        "result": result,
    }
    return _compose(
        sources,
        case_id,
        "table_cross_period_calculation",
        "跨表勾稽计算",
        question,
        conclusion,
        [conclusion],
        scoring_type="numeric",
        calculation=calculation,
    )


def _mixed(
    regulation: dict[str, Any],
    table: dict[str, Any],
    case_id: str,
    question: str,
) -> dict[str, Any]:
    table_calculation = deepcopy(table.get("calculation"))
    table_unit = str((table_calculation or {}).get("unit") or "").strip()

    def with_unit(value: str) -> str:
        if not table_unit or table_unit in value:
            return value
        return re.sub(
            r"(数值为\s*-?\d+(?:\.\d+)?)。",
            rf"\1{table_unit}。",
            value,
        )

    boundary = "制度条款和行业汇总值只能支持各自事实，不能据此直接认定某一家机构合规。"
    table_answer = with_unit(str(table["canonical_answer"]))
    answer = f"{regulation['canonical_answer']}\n{table_answer}\n{boundary}"
    conclusions = [
        *regulation["required_conclusions"],
        *(with_unit(str(item)) for item in table["required_conclusions"]),
        boundary,
    ]
    return _compose(
        [regulation, table],
        case_id,
        "mixed_regulation_table",
        "制度与报表联合判断",
        question,
        answer,
        conclusions,
        calculation=table_calculation,
        forbidden_conclusions=["可以直接认定该机构合规", "能够证明该机构合规"],
    )


def _formula_case(
    source: dict[str, Any],
    case_id: str,
    question: str,
    operands: dict[str, str],
    result: str,
    *,
    canonical_answer: str | None = None,
    required_conclusions: list[str] | None = None,
) -> dict[str, Any]:
    case = _clone(source, case_id, "formula", "Word/PDF公式", question, scoring_type="formula_numeric")
    case["calculation"] = deepcopy(source["calculation"])
    case["calculation"]["operands"] = operands
    case["calculation"]["result"] = result
    return _set_answer(
        case,
        canonical_answer or result,
        required_conclusions or [result],
    )


def build_cases(g: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    # 1-5: new multi-document fact cards (not one-source paraphrases).
    fact_specs = [
        ([g["TC009"], g["TC010"]], "制作一张跨制度核验卡：列出中资商业银行法人机构筹建申请书需说明的五类基本信息及目录版本，并列出寿险合同负债折现率曲线的终极利率暂定值。"),
        ([g["TC019"], g["TC021"]], "分别核验数字化银行回函的效力与回函时限，以及交易账簿划分政策内部审计的频率和留档要求。四项信息均不得省略。"),
        ([g["TC008"], g["TC022"]], "第三支柱信息披露台账应如何区分固定表格与可变表格，并分别按哪些周期披露？请把表格类型和三种频率同时列全。"),
        ([g["TC012"], g["TC014"]], "跨制度建立口径卡：先给出意外伤害保险定义及保费厘定原则，再给出商业银行大额风险暴露门槛和处置策略基本原则。"),
        ([g["TC016"], g["TC020"]], "对照两个触发口径：重要数据事件何时属于重大事件；资本工具何时触发持续经营事件、二级资本在何种顺序下吸收损失。"),
    ]
    for index, (sources, question) in enumerate(fact_specs, start=1):
        cases.append(_combine(sources, f"HC{index:03d}", "fact", "跨制度事实核验", question))

    # 6-10: options combine full evidence chains; only one option is complete.
    cases.extend([
        _multiple_choice(g["TC053"], "HC006", "下列哪一组同时正确描述行政许可申请材料和寿险折现率曲线？\nA. 申请书无需股权结构，曲线仅含基础利率\nB. 申请书列名称、拟设地、注册资本、股权结构和业务范围，目录为2023年版；曲线由基础利率加综合溢价形成\nC. 目录为2022年版，曲线只含综合溢价\nD. 两份材料均未规定", "B"),
        _multiple_choice(g["TC055"], "HC007", "关于重要实体与处置工具，下列哪项同时符合材料？\nA. 重要实体与关键功能无关，工具仅限破产清算\nB. 重要实体只承载非核心业务，禁止引入战略投资者\nC. 重要实体承载核心业务条线和关键功能；工具可包括自救、注资、战略投资、不良资产处置、接管、收购承接、过桥机构和破产清算\nD. 处置工具只能由股东注资", "C"),
        _multiple_choice(g["TC056"], "HC008", "下列哪组完整对应知识产权融资措施与较大数据安全事件条件？\nA. 只允许抵押贷款；任何一般数据泄露均为较大事件\nB. 可综合使用小额信用、知识产权质押和创新积分贷款；敏感级及以上数据事件造成难以消除的个人负面影响或使机构部分业务异常等可构成较大事件\nC. 只允许创新积分贷款；较大事件必须影响两个省\nD. 融资措施与事件分级均无明确内容", "B"),
        _multiple_choice(g["TC057"], "HC009", "消费金融催收与银行函证要求中，哪一组全部正确？\nA. 可向无关第三人催收；询证函可列多个基准日\nB. 禁止暴力等不正当催收且不得催收无关第三人；一函一个基准日，并应通过总行或总部公开渠道公示函证事项\nC. 只禁止暴力催收；函证无需公示\nD. 可由催收机构自行决定对象；公示仅限纸质公告", "B"),
        _multiple_choice(g["TC059"], "HC010", "关于交易账簿和第三支柱披露，下列哪项完整正确？\nA. 交易头寸不含做市和对客交易；按月计量；只按单体口径披露\nB. 交易头寸包括自营、做市、对客及相关对冲，原则上每日公允价值计量且变动计入损益；披露按监管并表范围，表格另有规定除外\nC. 交易头寸只含自营；公允价值变动不计损益\nD. 两项均由机构任意选择", "B"),
    ])

    # 11-15: multi-condition propositions, including explicit corrections.
    cases.extend([
        _judgment(g["TC009"], "HC011", "判断并纠正：中资商业银行法人机构筹建申请书可以省略股权结构和业务范围，而且所用目录不是2023年版。", "说法错误。申请书应说明名称、拟设地、注册资本、股权结构和业务范围等基本信息，目录版本为2023年版。", ["说法错误", *g["TC009"]["required_conclusions"]]),
        _judgment(g["TC019"], "HC012", "判断并说明依据：数字化回函与纸质回函具有同等法律效力和证明力，银行收到合规询证函后应在10个工作日内直接回复会计师事务所。", f"说法正确。{g['TC019']['canonical_answer']}", ["说法正确", *g["TC019"]["required_conclusions"]]),
        _judgment(g["TC059"], "HC013", "判断并纠正：交易账簿头寸原则上只需每月做一次公允价值计量，且商业银行第三支柱信息一律按单体口径披露。", f"说法错误。{g['TC059']['canonical_answer']}", ["说法错误", *g["TC059"]["required_conclusions"]]),
        _judgment(g["TC067"], "HC014", "判断并纠正：消费金融公司金融机构主要出资人的相关经营经验只需3年，且不必有具备5年以上管理与风控经验并达到规定出资比例的出资人。", "说法错误。金融机构主要出资人应有5年以上消费金融领域经营经验；至少应有1名具备5年以上消费金融业务管理和风险控制经验、出资比例不低于拟设公司全部股本三分之一的出资人。", ["说法错误", g["TC067"]["required_conclusions"][0]], evidence_indexes=[0, 1]),
        _judgment(g["TC070"], "HC015", "判断并说明依据：其他一级资本工具没有到期日，并且不得含有利率跳升机制及其他赎回激励。", "说法正确。其他一级资本工具没有到期日，并且不得含有利率跳升机制及其他赎回激励。", ["说法正确", g["TC070"]["required_conclusions"][0]], evidence_indexes=[0]),
    ])

    # 16-20: genuinely new three-source synthesis combinations.
    summary_specs = [
        ([g["TC001"], g["TC011"], g["TC018"]], "形成一段跨制度工作摘要：概括2027年知识产权金融生态目标、列入名单保险集团的报告义务，以及银行函证事项应通过哪些公开渠道公示。"),
        ([g["TC003"], g["TC013"], g["TC015"]], "用一段话同时说明消费金融公司的业务地域范围、核心业务条线经营失败的影响，以及试验区专利权质押登记线上办理目标。"),
        ([g["TC004"], g["TC006"], g["TC008"]], "汇总三项控制要求：注册会计师对询证函的过程控制、二级资本启动吸收损失的先后条件、第三支柱表格的两种类型。"),
        ([g["TC010"], g["TC016"], g["TC021"]], "形成审查摘要：终极利率暂定值是多少；重大数据安全事件的影响条件是什么；账簿划分政策内部审计如何安排并留档。"),
        ([g["TC012"], g["TC014"], g["TC022"]], "用一段可核验摘要串联意外伤害保险定义及定价要求、大额风险暴露与自救原则、第三支柱披露频率。"),
    ]
    for index, (sources, question) in enumerate(summary_specs, start=16):
        cases.append(_combine(sources, f"HC{index:03d}", "summary", "跨制度综合摘要", question))

    # 21-25: new document combinations with source-by-source attribution.
    cross_specs = [
        ([g["TC004"], g["TC006"], g["TC010"]], "分别引用三份材料回答：谁应控制银行询证函全过程；二级资本在何时启动吸收损失；寿险折现率曲线终极利率暂定多少。"),
        ([g["TC013"], g["TC018"], g["TC022"]], "跨文件核验：核心业务条线经营失败可能造成何种损失；函证事项通过何种公开渠道公示；第三支柱披露有哪些频率。"),
        ([g["TC015"], g["TC016"], g["TC019"]], "请逐项给出来源：专利权质押登记线上全覆盖目标；重大数据事件的条件；数字化回函效力和合规询证函回复时限。"),
        ([g["TC011"], g["TC012"], g["TC021"]], "分别核对保险集团报告义务、意外伤害保险定义与定价原则、账簿划分政策的年度内部审计和留档。"),
        ([g["TC003"], g["TC008"], g["TC014"]], "跨制度回答消费金融公司的业务地域范围、第三支柱表格类型，以及大额风险暴露门槛和处置自救原则。每项都需直接引用。"),
    ]
    for index, (sources, question) in enumerate(cross_specs, start=21):
        cases.append(_combine(sources, f"HC{index:03d}", "cross_document", "跨制度文件推理", question))

    # 26-30: compare the same locked indicator across three or four files.
    cases.extend([
        _table_selection([g["TC041"], g["TC045"], g["TC049"]], "HC026", "比较2024年9月、2025年3月和2025年9月三张人身险公司经营表的原保险保费收入本年累计值，指出最高期间、数值和单位。", "三期中2025年9月最高，为38433.67亿元。", "38433.67", "2024-09、2025-03、2025-09"),
        _table_selection([g["TC042"], g["TC046"], g["TC050"]], "HC027", "比较2024年9月、2025年3月和2025年9月保险业经营情况表的原保险保费收入本年累计值，指出最高期间、数值和单位。", "三期中2025年9月最高，为52145.77亿元。", "52145.77", "2024-09、2025-03、2025-09"),
        _table_selection([g["TC044"], g["TC048"], g["TC052"]], "HC028", "比较2024年9月、2025年3月和2025年9月财产险公司经营表的原保险保费收入本年累计值，指出最高期间、数值和单位。", "三期中2025年9月最高，为13712.1亿元。", "13712.1", "2024-09、2025-03、2025-09"),
        _table_selection([g["TC043"], g["TC047"], g["TC051"]], "HC029", "比较2024年9月、2025年3月和2025年9月全国各地区原保险保费收入表中的全国健康险合计，指出最高期间、数值和单位。", "三期中2025年9月最高，为8426.99亿元。", "8426.99", "2024-09、2025-03、2025-09"),
        _table_selection([g["TC032"], g["TC023"], g["TC026"]], "HC030", "比较2023年8月、10月和12月人身险公司原保险保费收入本年累计值，指出最高月份、数值和单位。", "三个月中2023年12月最高，为35378.91亿元。", "35378.91", "2023-08、2023-10、2023-12"),
    ])

    # 31-35: three-file accounting reconciliations with signed Decimal residuals.
    cases.extend([
        _reconciliation([g["TC024"], g["TC023"], g["TC025"]], "HC031", "分别读取2023年10月保险业合计、人身险和财产险原保险保费收入累计值，计算勾稽差额：保险业合计－人身险－财产险。", {"保险业合计": "45167.98", "人身险": "31739.18", "财产险": "13428.79"}, "0.01", "2023-10"),
        _reconciliation([g["TC033"], g["TC032"], g["TC035"]], "HC032", "分别读取2023年8月保险业合计、人身险和财产险原保险保费收入累计值，计算勾稽差额：保险业合计－人身险－财产险。", {"保险业合计": "38731.77", "人身险": "27679.08", "财产险": "11052.68"}, "0.01", "2023-08"),
        _reconciliation([g["TC042"], g["TC041"], g["TC044"]], "HC033", "分别读取2024年9月保险业合计、人身险和财产险原保险保费收入累计值，计算勾稽差额：保险业合计－人身险－财产险。", {"保险业合计": "47945.35", "人身险": "34878.78", "财产险": "13066.56"}, "0.01", "2024-09"),
        _reconciliation([g["TC046"], g["TC045"], g["TC048"]], "HC034", "分别读取2025年3月保险业合计、人身险和财产险原保险保费收入累计值，计算有符号勾稽差额：保险业合计－人身险－财产险。", {"保险业合计": "21745", "人身险": "16590.18", "财产险": "5155"}, "-0.18", "2025-03"),
        _reconciliation([g["TC050"], g["TC049"], g["TC052"]], "HC035", "分别读取2025年9月保险业合计、人身险和财产险原保险保费收入累计值，计算勾稽差额：保险业合计－人身险－财产险。", {"保险业合计": "52145.77", "人身险": "38433.67", "财产险": "13712.1"}, "0.00", "2025-09"),
    ])

    # 36-40: new regulation/table pairings with an explicit evidence boundary.
    cases.extend([
        _mixed(g["TC019"], g["TC043"], "HC036", "联合核验：说明数字化回函效力与10个工作日回复要求；再比较2024年9月全国健康险的全国合计、北京、天津和公司本级，给出最大项、数值和单位。最后说明能否据此认定某家机构合规。"),
        _mixed(g["TC014"], g["TC049"], "HC037", "联合核验：说明大额风险暴露门槛和处置自救原则；再比较2025年9月人身险表中意外险、寿险和原保险保费收入，给出最大项、数值和单位。不得作单一机构合规推断。"),
        _mixed(g["TC017"], g["TC052"], "HC038", "联合核验：概括消费金融公司薪酬递延与催收禁限；再比较2025年9月财产险表中原保险保费收入、机动车辆保险和责任险，给出最大项、数值和单位。最后给出证据边界。"),
        _mixed(g["TC020"], g["TC050"], "HC039", "联合核验：给出资本工具持续经营触发阈值和损失吸收顺序；再比较2025年9月保险业经营表的人身险、财产险与原保险保费收入，给出最大项、数值和单位。行业汇总不得外推到单一机构。"),
        _mixed(g["TC012"], g["TC051"], "HC040", "联合核验：说明意外伤害保险定义及保费厘定要求；再比较2025年9月全国健康险的天津、北京、全国和河北值，给出最大项、数值和单位。最后说明证据边界。"),
    ])

    # 41-44: same locked formulas, entirely new operands and independently recomputed gold.
    cases.extend([
        _formula_case(
            g["TC075"],
            "HC041",
            "依据锁定 Word 公式计算 RWA：E=96，R=0.75；写出代入关系并给出精确结果。",
            {"E": "96", "R": "0.75"},
            "900",
            canonical_answer="RWA=96×0.75×12.5=900。",
            required_conclusions=["RWA=96×0.75×12.5", "结果为900"],
        ),
        _formula_case(g["TC076"], "HC042", "依据交易对手信用风险规则中的 Word 公式计算 d：SD=0.32，Notional=375。", {"SD": "0.32", "Notional": "375"}, "120"),
        _formula_case(g["TC077"], "HC043", "依据锁定 Word 公式计算 LGD*：LGD_s=0.35，E_s=70，E=100，H_e=0.25，LGD_u=0.65，E_u=30。", {"LGD_s": "0.35", "E_s": "70", "E": "100", "H_e": "0.25", "LGD_u": "0.65", "E_u": "30"}, "0.352"),
        _formula_case(g["TC078"], "HC044", "依据第三档商业银行资本监管 Word 公式，核心一级资本1250、扣除项50、调整后表内外资产余额15000时，计算杠杆率。", {"核心一级资本": "1250", "核心一级资本扣除项": "50", "调整后表内外资产余额": "15000"}, "8%"),
    ])

    # 45-50: new unknown-version documents, absent indicators, and a new formula boundary.
    cases.extend([
        _clone(g["TC085"], "HC045", "version_refusal", "时效与版本拒答", "只依据身份卡，能否确认《附件2：人身保险公司县域机构统计表》当前现行、失效或已被替代？字段未知时不得按文件名推断。"),
        _clone(g["TC087"], "HC046", "version_refusal", "时效与版本拒答", "核验《细则第5号附件：非寿险业务准备金回溯分析报表》的当前法律效力；若身份卡没有明确状态和替代关系，应如何回答？"),
        _clone(g["TC088"], "HC047", "version_refusal", "时效与版本拒答", "《细则第6号附件1.准备金报表填表说明》是否仍然现行有效？仅允许依据已确认身份元数据，不得从日期或上传顺序推断。"),
        _clone(g["TC094"], "HC048", "table_refusal", "可信拒答", "在《2025年9月人身险公司经营情况表》中查询指标‘审慎监测不存在字段R2-01’；全库无此字段时应拒答，不能返回相似指标。", canonical_answer="当前知识库中不存在指标‘审慎监测不存在字段R2-01’，依据不足，无法给出确定数值。", required_conclusions=["依据不足", "无法给出确定数值"]),
        _clone(g["TC098"], "HC049", "table_refusal", "可信拒答", "在《商业银行主要指标分机构类情况表(季度)(2023年)》查询‘审慎监测不存在字段R2-02’；若字段不存在，明确说明不足并拒绝猜测。", canonical_answer="当前知识库中不存在指标‘审慎监测不存在字段R2-02’，依据不足，无法给出确定数值。", required_conclusions=["依据不足", "无法给出确定数值"]),
        _clone(g["TC082"], "HC050", "formula_refusal", "公式能力边界", "依据《市场风险内部模型法监管要求》的 Word 公式尝试计算 SES：ISES_NM=4，SES_NM=9，ρ=0.25；若可信执行器不支持平方根运算，必须明确拒答而非估算。"),
    ])
    for case, query in ((cases[47], "审慎监测不存在字段R2-01"), (cases[48], "审慎监测不存在字段R2-02")):
        case["corpus_search"] = {
            "scope": "all_500_document_chunks_and_spreadsheet_cells",
            "query": query,
            "match_count": 0,
            "insufficiency_type": "nonexistent_indicator",
        }

    for case in cases:
        case["split"] = "independent_round_2"
    return cases


def validate(cases: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if len(cases) != 50:
        errors.append(f"expected 50 cases, got {len(cases)}")
    if [case["id"] for case in cases] != [f"HC{index:03d}" for index in range(1, 51)]:
        errors.append("IDs/order must be HC001..HC050")
    prior_questions = [row for path in PRIOR_QUESTION_FILES for row in _read_jsonl(path)]
    prior_norm = {_normalize_question(row["question"]) for row in prior_questions}
    if any(_normalize_question(case["question"]) in prior_norm for case in cases):
        errors.append("one or more questions exactly duplicate a prior challenge")
    if len({_normalize_question(case["question"]) for case in cases}) != len(cases):
        errors.append("duplicate question text within round 2")
    if any(case["difficulty"] != "hard" or case["split"] != "independent_round_2" for case in cases):
        errors.append("every case must be hard and independent_round_2")
    if sum(bool(case["answerable"]) for case in cases) != 44:
        errors.append("round 2 must contain 44 answerable and 6 refusal cases")
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
        "round": 2,
        "case_count": 50,
        "difficulty": {"hard": 50},
        "answerability": {"answerable": 44, "refusal": 6},
        "question_types": {
            question_type: sum(case["question_type"] == question_type for case in cases)
            for question_type in sorted({case["question_type"] for case in cases})
        },
        "new_task_design": {
            "prior_question_files_checked": [str(path.relative_to(ROOT)) for path in PRIOR_QUESTION_FILES],
            "cross_file_table_reconciliations": 5,
            "new_formula_operands": 4,
            "new_regulation_table_pairings": 5,
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
