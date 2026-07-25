"""Build an independently auditable 50-case hard challenge from locked official evidence."""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_GOLD = ROOT / "data/evaluation/trust_challenge_100/gold.jsonl"
SOURCE_QUESTIONS = ROOT / "data/evaluation/trust_challenge_100/questions.jsonl"
OUTPUT_DIR = ROOT / "data/evaluation/hard_challenge_50/round_1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_question(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _dedupe_records(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        value = str(record.get(key) or "")
        if value and value not in seen:
            output.append(deepcopy(record))
            seen.add(value)
    return output


def _base_case(
    case_id: str,
    question_type: str,
    scenario: str,
    question: str,
    *,
    scoring_type: str,
    answerable: bool,
    canonical_answer: str,
    required_conclusions: list[str],
    forbidden_conclusions: list[str],
    allowed_expressions: list[str],
    required_documents: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    calculation: dict[str, Any] | None,
    expected_refusal_code: str | None,
    source_case_ids: list[str],
    corpus_search: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "scenario": scenario,
        "question_type": question_type,
        "difficulty": "hard",
        "split": "independent_round_1",
        "question": question,
        "scoring_type": scoring_type,
        "answerable": answerable,
        "review_status": "codex_verified",
        "expert_reviewed": False,
        "canonical_answer": canonical_answer,
        "equivalent_answers": list(dict.fromkeys([canonical_answer, *allowed_expressions])),
        "allowed_expressions": list(dict.fromkeys(allowed_expressions)),
        "required_conclusions": required_conclusions,
        "forbidden_conclusions": forbidden_conclusions,
        "expected_refusal_code": expected_refusal_code,
        "primary_document_id": required_documents[0]["document_id"] if required_documents else None,
        "required_documents": required_documents,
        "evidence": evidence,
        "calculation": calculation,
        "corpus_search": corpus_search,
        "source_case_ids": source_case_ids,
        "scoring_rules": {
            "semantic_equivalence_allowed": True,
            "all_required_conclusions_required": True,
            "forbidden_conclusion_policy": "any_asserted_forbidden_fails",
            "citation_policy": "each_required_evidence_aspect_must_have_direct_citation",
            "numeric_policy": "Decimal value, unit, period, direction and rounding must agree",
        },
    }


def _clone(
    source: dict[str, Any],
    case_id: str,
    question_type: str,
    scenario: str,
    question: str,
    *,
    scoring_type: str | None = None,
    canonical_answer: str | None = None,
    required_conclusions: list[str] | None = None,
    forbidden_conclusions: list[str] | None = None,
    allowed_expressions: list[str] | None = None,
    answerable: bool | None = None,
    expected_refusal_code: str | None = None,
    evidence_indexes: list[int] | None = None,
) -> dict[str, Any]:
    evidence = source["evidence"]
    if evidence_indexes is not None:
        evidence = [evidence[index] for index in evidence_indexes]
    document_ids = {item["document_id"] for item in evidence}
    # A refusal can be bound to a concrete official document while correctly
    # having no positive evidence chunk (for example, unknown version status).
    documents = (
        [item for item in source["required_documents"] if item["document_id"] in document_ids]
        if evidence
        else deepcopy(source["required_documents"])
    )
    score_type = scoring_type or source["scoring_type"]
    is_answerable = answerable if answerable is not None else score_type not in {"refusal", "formula_refusal", "version_uncertainty"}
    return _base_case(
        case_id,
        question_type,
        scenario,
        question,
        scoring_type=score_type,
        answerable=is_answerable,
        canonical_answer=canonical_answer or source["canonical_answer"],
        required_conclusions=required_conclusions or deepcopy(source["required_conclusions"]),
        forbidden_conclusions=forbidden_conclusions if forbidden_conclusions is not None else deepcopy(source["forbidden_conclusions"]),
        allowed_expressions=allowed_expressions or deepcopy(source["allowed_expressions"]),
        required_documents=deepcopy(documents),
        evidence=deepcopy(evidence),
        calculation=deepcopy(source.get("calculation")),
        expected_refusal_code=(expected_refusal_code if expected_refusal_code is not None else source.get("expected_refusal_code")),
        source_case_ids=[source["id"]],
        corpus_search=deepcopy(source.get("corpus_search")),
    )


def _compose(
    sources: list[dict[str, Any]],
    case_id: str,
    question_type: str,
    scenario: str,
    question: str,
    canonical_answer: str,
    required_conclusions: list[str],
    *,
    scoring_type: str = "multi_assertion",
    calculation: dict[str, Any] | None = None,
    forbidden_conclusions: list[str] | None = None,
) -> dict[str, Any]:
    documents = _dedupe_records(
        [item for source in sources for item in source["required_documents"]],
        "document_id",
    )
    evidence = _dedupe_records([item for source in sources for item in source["evidence"]], "chunk_id")
    return _base_case(
        case_id,
        question_type,
        scenario,
        question,
        scoring_type=scoring_type,
        answerable=True,
        canonical_answer=canonical_answer,
        required_conclusions=required_conclusions,
        forbidden_conclusions=forbidden_conclusions or [],
        allowed_expressions=[canonical_answer],
        required_documents=documents,
        evidence=evidence,
        calculation=calculation,
        expected_refusal_code=None,
        source_case_ids=[source["id"] for source in sources],
    )


def _difference_case(
    first: dict[str, Any],
    second: dict[str, Any],
    case_id: str,
    question: str,
    result: str,
    formula: str,
    operands: dict[str, str],
) -> dict[str, Any]:
    calculation = {
        "operands": operands,
        "unit": "亿元",
        "period": {"raw": "跨期间"},
        "formula": formula,
        "rounding": "按两个来源表的锁定精度做 Decimal 减法，结果保留可验证精度",
        "result": result,
        "evidence_cells": sorted(
            {
                cell
                for source in (first, second)
                for evidence in source["evidence"]
                for cell in evidence.get("cells") or []
            }
        ),
    }
    return _compose(
        [first, second],
        case_id,
        "table_cross_period_calculation",
        "跨表期间计算",
        question,
        f"增加额为{result}亿元。",
        [f"增加额为{result}亿元"],
        scoring_type="numeric",
        calculation=calculation,
    )


def build_cases(source_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    g = source_by_id
    cases: list[dict[str, Any]] = []

    # 1-5: precise facts with full, non-ambiguous evidence anchors.
    cases.extend([
        _clone(g["TC002"], "HC001", "fact", "制度事实", "《数据安全事件分级》中，重要数据事件在什么影响范围和严重程度下构成特别重大事件？请同时给出区域数量门槛和影响后果。"),
        _clone(g["TC014"], "HC002", "fact", "主体与适用范围", "请依据《处置计划建议示例（商业银行版）》分别说明大额风险暴露的量化门槛，以及处置策略建议应坚持的基本原则。"),
        _clone(g["TC017"], "HC003", "fact", "条件、期限与禁止行为", "依据《消费金融公司管理办法》，同时概括关键岗位绩效薪酬延期支付的比例与期限要求，以及催收行为的禁止边界。"),
        _clone(g["TC020"], "HC004", "fact", "触发条件与顺序", "依据《资本工具合格标准》，说明持续经营触发事件的核心一级资本充足率阈值，并说明二级资本工具相对于其他一级资本工具的损失吸收顺序。"),
        _clone(g["TC061"], "HC005", "fact", "主体、禁止行为与期限", "请分别核对：中国太平洋保险（集团）股份有限公司在偿付能力报告名单中的类别；团体意外险销售对象限制；保单查询服务的最低保留期限。"),
    ])

    # 6-10: MCQ, with one complete option and three controlled contradictions.
    mcq_specs = [
        ("HC006", "TC002", "关于《数据安全事件分级》，下列哪项表述正确？\nA. 影响任一省级区域即一律属于特别重大事件\nB. 重要数据遭泄露等并对2个及以上省级区域经济运行秩序造成特别严重影响，属于特别重大事件\nC. 只要影响银行保险行业安全就属于特别重大事件\nD. 特别重大事件不要求说明影响后果", "答案为B。"),
        ("HC007", "TC014", "关于《处置计划建议示例（商业银行版）》，下列哪组同时正确？\nA. 大额风险暴露门槛为一级资本净额5%，处置以外部救助为本\nB. 门槛为2.5%，处置以外部救助为本\nC. 门槛为一级资本净额2.5%，处置策略建议坚持自救为本\nD. 门槛不涉及一级资本净额，处置原则未规定", "答案为C。"),
        ("HC008", "TC020", "关于资本工具触发与损失吸收顺序，下列哪项正确？\nA. 核心一级资本充足率降至5.125%或以下构成持续经营触发事件，其他一级资本工具全部吸收损失后再启动二级资本工具\nB. 阈值为8%，二级资本先于其他一级资本吸收损失\nC. 阈值为4.5%，两类工具无先后顺序\nD. 文件只规定阈值，未规定顺序", "答案为A。"),
        ("HC009", "TC061", "下列哪组同时符合两份材料？\nA. 太平洋保险集团属于非保险控股型集团；查询服务保留一个月\nB. 属于保险控股型集团；可向团体外个人销售团体意外险\nC. 属于非保险控股型集团；查询服务保留三个月\nD. 属于应编报的保险控股型集团；不得向特定团体成员以外个人销售团体意外险，查询服务至少保留至责任结束后三个月", "答案为D。"),
        ("HC010", "TC072", "关于第三支柱补充披露，下列哪项完整列出了材料要求？\nA. 只要求真实性和准确性\nB. 要求真实性、准确性、完整性、一致性和可比性\nC. 要求及时性、盈利性和保密性\nD. 可由商业银行自行删除可比性要求", "答案为B。"),
    ]
    for case_id, source_id, question, verdict in mcq_specs:
        source = g[source_id]
        source_conclusions = source["required_conclusions"]
        evidence_indexes = None
        if source_id == "TC072":
            # The MCQ asks only about the regulatory disclosure requirement;
            # the source mixed case's spreadsheet and evidence-boundary parts
            # are intentionally excluded from this question and its gold.
            source_conclusions = source_conclusions[:1]
            evidence_indexes = [0]
        case = _clone(
            source,
            case_id,
            "multiple_choice",
            "选择题",
            question,
            scoring_type="multiple_choice",
            canonical_answer=f"{verdict}{source_conclusions[0] if source_id == 'TC072' else source['canonical_answer']}",
            required_conclusions=[verdict.rstrip("。"), *source_conclusions],
            allowed_expressions=[verdict, f"{verdict}{source_conclusions[0] if source_id == 'TC072' else source['canonical_answer']}"],
            evidence_indexes=evidence_indexes,
        )
        if source_id == "TC072":
            case["calculation"] = None
        cases.append(case)

    # 11-15: true/false and correction, including explicit negative statements.
    judgment_specs = [
        ("HC011", "TC007", "判断并说明依据：商业银行进行账簿转换，只需高级管理层批准，无需国家金融监督管理总局或其派出机构认可。", ["只需高级管理层批准"], None),
        ("HC012", "TC010", "判断并纠正：寿险合同负债评估折现率曲线中的终极利率暂定为5%。", ["终极利率暂定为5%"], None),
        ("HC013", "TC017", "判断并说明依据：消费金融公司关键岗位员工绩效薪酬40%以上应延期支付，延期支付期限一般不少于3年。", [], None),
        ("HC014", "TC021", "判断并纠正：商业银行只需每两年对账簿划分政策和程序开展一次内部审计。", ["每两年"], None),
        ("HC015", "TC062", "判断并说明依据：恢复计划压力测试情景至少同时覆盖系统性、自身和混合压力情景；处置计划还应说明自救资金原则和沟通对象。", [], None),
    ]
    for case_id, source_id, question, forbidden, _ in judgment_specs:
        source = g[source_id]
        case = _clone(
            source,
            case_id,
            "judgment",
            "判断与纠错",
            question,
            scoring_type="judgment",
            forbidden_conclusions=[*source["forbidden_conclusions"], *forbidden],
        )
        if case_id == "HC013":
            case["canonical_answer"] = (
                "该说法正确。消费金融公司高级管理人员以及对风险有重要影响岗位上的员工，"
                "绩效薪酬的40%以上应采取延期支付方式，且延期支付期限一般不少于3年。"
            )
            case["equivalent_answers"] = [case["canonical_answer"]]
            case["allowed_expressions"] = [case["canonical_answer"]]
            case["required_conclusions"] = [
                "说法正确",
                "消费金融公司高级管理人员以及对风险有重要影响岗位上的员工，绩效薪酬的40%以上应采取延期支付方式，且延期支付期限一般不少于3年",
            ]
            case["evidence"] = case["evidence"][:1]
        cases.append(case)

    # 16-20: summaries requiring multiple paragraphs/conditions.
    cases.extend([
        _clone(g["TC012"], "HC016", "summary", "跨段摘要", "请用一段话概括《意外伤害保险业务监管办法》对意外伤害保险的定义，以及保险费厘定所遵循的精算原则与定价假设。"),
        _clone(g["TC017"], "HC017", "summary", "跨段摘要", "请综合概括《消费金融公司管理办法》中与关键人员薪酬递延和消费者催收保护相关的要求，不能漏掉比例、期限及禁止对象。"),
        _clone(g["TC020"], "HC018", "summary", "跨段摘要", "请概括资本工具持续经营触发阈值与损失吸收先后顺序，并明确阈值方向。"),
        _clone(g["TC062"], "HC019", "summary", "跨文件摘要", "请形成一段可核验摘要：恢复计划压力测试需设置哪些情景；处置计划的资金自救原则是什么；沟通策略应覆盖哪些对象。"),
        _clone(g["TC062"], "HC020", "summary", "同文件跨段摘要", "只依据《处置计划建议示例（商业银行版）》，概括处置资金使用原则与沟通策略对象。", canonical_answer="处置资金使用上应以使用金融机构自有资产或市场化渠道筹集资金开展自救为原则；处置计划沟通策略关注在实施过程中与监管部门、地方政府、股东、客户、员工和社会公众开展有效沟通。", required_conclusions=["处置资金使用上应以使用金融机构自有资产或市场化渠道筹集资金开展自救为原则", "处置计划沟通策略关注在实施过程中与监管部门、地方政府、股东、客户、员工和社会公众开展有效沟通"], evidence_indexes=[1, 2]),
    ])

    # 21-25: cross-document reasoning with explicit, complete subquestions.
    cases.extend([
        _clone(g["TC054"], "HC021", "cross_document", "跨制度文件推理", "请分别引用两份文件回答：复星保德信人寿与复星联合健康的偿付能力报告编报主体是谁；意外伤害保险如何定义；厘定保险费须遵循何种原则与假设。"),
        _clone(g["TC060"], "HC022", "cross_document", "跨制度文件推理", "请分别引用两份文件说明：法人机构开业核准与分行筹建审批在中资商业银行行政许可目录中的上位类别；寿险合同负债现金流现值的折现率曲线由哪两部分形成。"),
        _clone(g["TC061"], "HC023", "cross_document", "跨制度文件推理", "跨文件核对太平洋保险集团的名单类别，并同时说明团体意外险销售对象禁限与查询服务保留期限。每项结论需对应来源。"),
        _compose([g["TC020"], g["TC021"]], "HC024", "cross_document", "跨制度文件推理", "分别依据《资本工具合格标准》和《账簿划分和名词解释》回答：持续经营触发阈值与二级资本损失吸收顺序是什么；账簿划分政策和程序的内部审计频率及留档要求是什么。", "持续经营触发事件指商业银行核心一级资本充足率降至5.125%或以下；所有其他一级资本工具全部吸收损失后，再启动二级资本工具吸收损失；商业银行应每年对划分政策和程序开展内部审计，内部审计结果需留档备查。", [*g["TC020"]["required_conclusions"], *g["TC021"]["required_conclusions"]]),
        _compose([g["TC002"], g["TC014"]], "HC025", "cross_document", "跨制度文件推理", "跨制度核对两个量化门槛：特别重大重要数据事件涉及多少个省级区域及何种后果；商业银行大额风险暴露相对一级资本净额的门槛是多少，并说明处置策略基本原则。", f"{g['TC002']['canonical_answer']}；{g['TC014']['canonical_answer']}", [*g["TC002"]["required_conclusions"], *g["TC014"]["required_conclusions"]]),
    ])

    # 26-30: table comparison, rephrased and operand-complete.
    table_sources = ["TC038", "TC039", "TC040", "TC043", "TC047"]
    table_questions = [
        "在《2023年商业银行主要指标分机构类情况表（季度）》的‘农村商业银行’列中，对损失类、不良、可疑类和次级类贷款余额做四项比较，报告最大项、原始值和单位。",
        "读取《2023年银行业金融机构保障性安居工程贷款情况表(季度)》‘保障房贷款’工作表，在银行业金融机构合计、商业银行合计、大型商业银行、股份制商业银行四项中确定最大值，并给出坐标可追溯的项目、值和单位。",
        "读取《2023年银行业金融机构普惠型小微企业贷款情况（季度）》并比较城市商业银行、股份制商业银行、银行业金融机构合计、大型商业银行四项，哪项最高？给出数值和单位。",
        "在《2024年9月全国各地区原保险保费收入情况表》的健康险口径下，比较全国合计、北京、天津、公司本级，给出最大项、数值和单位。",
        "在《2025年3月全国各地区原保险保费收入情况表》的健康险口径下，比较北京、全国、天津、河北，给出最大项、数值和单位。",
    ]
    for offset, (source_id, question) in enumerate(zip(table_sources, table_questions), start=26):
        cases.append(_clone(g[source_id], f"HC{offset:03d}", "table_comparison", "表格比较", question))

    # 31-35: cross-file period arithmetic, with Decimal-locked direction.
    cases.extend([
        _difference_case(g["TC023"], g["TC026"], "HC031", "分别从2023年10月与12月人身险公司经营情况表读取原保险保费收入本年累计值，并计算12月值比10月值增加多少亿元。", "3639.73", "35378.91 - 31739.18", {"2023-10": "31739.18", "2023-12": "35378.91"}),
        _difference_case(g["TC025"], g["TC029"], "HC032", "分别从2023年10月与12月财产保险公司经营情况表读取原保险保费收入本年累计值，并计算12月值比10月值增加多少亿元。", "2439.00", "15867.79 - 13428.79", {"2023-10": "13428.79", "2023-12": "15867.79"}),
        _difference_case(g["TC030"], g["TC031"], "HC033", "对比2023年3季度与4季度保险业资金运用情况表的资金运用余额账面值，计算4季度比3季度增加多少亿元，保持原始精度。", "6342.9840", "281573.6094 - 275230.6254", {"2023-Q3": "275230.6254", "2023-Q4": "281573.6094"}),
        _difference_case(g["TC032"], g["TC036"], "HC034", "分别读取2023年8月和9月人身险公司原保险保费收入本年累计值，计算9月比8月增加额。", "2467.51", "30146.59 - 27679.08", {"2023-08": "27679.08", "2023-09": "30146.59"}),
        _difference_case(g["TC033"], g["TC037"], "HC035", "分别读取2023年8月和9月保险业经营情况表的原保险保费收入本年累计值，计算9月比8月增加额。", "3795.08", "42526.85 - 38731.77", {"2023-08": "38731.77", "2023-09": "42526.85"}),
    ])

    # 36-40: mixed regulation/table/evidence-boundary tasks.
    mixed_sources = ["TC063", "TC064", "TC066", "TC071", "TC074"]
    mixed_questions = [
        "联合核验但不得作机构合规推断。制度侧：列出恢复计划压力测试的三类最低情景。报表侧：在2023年商业银行主要指标分机构类表的农村商业银行口径下比较四类贷款余额并给出最大项。最后说明证据边界。",
        "联合核验。制度侧：说明处置资金自救原则和沟通策略对象。报表侧：在2023年保障房贷款表四类机构汇总项中找最大值。最后判断这些材料能否直接证明某一家机构合规。",
        "联合核验。制度侧：区分重要数据事件‘特别重大’与‘重大’的影响条件。报表侧：读取2023年10月人身险公司经营情况表中原保险保费收入本年累计值。不得把行业汇总直接用于单一机构结论。",
        "联合核验。制度侧：说明账簿转换的禁止理由和不可撤销例外。报表侧：在2023年保障房贷款表四项汇总中找最大值。最后说明能否据此认定单一机构合规。",
        "联合核验。制度侧：说明万能险、投连险、变额年金和中短存续期产品的综合溢价。报表侧：读取2023年10月全国各地区原保险保费收入表全国合计的合计值。最后给出证据边界。",
    ]
    for offset, (source_id, question) in enumerate(zip(mixed_sources, mixed_questions), start=36):
        cases.append(_clone(g[source_id], f"HC{offset:03d}", "mixed_regulation_table", "制度与报表联合判断", question))

    # 41-44: supported Word-formula calculations.
    formula_questions = [
        "依据《信用风险权重法表内资产风险权重、表外项目信用转换系数及合格信用风险缓释工具》的 Word 公式，在 E=80、R=0.5 时计算 RWA；写出代入式、结果和引用。",
        "依据《交易对手信用风险加权资产计量规则》的 Word 公式，在 SD=0.4、Notional=250 时求 d；写出代入过程。",
        "依据《信用风险内部评级法风险缓释监管要求》的 Word 公式，代入 LGD_s=0.4、E_s=60、E=100、H_e=0.2、LGD_u=0.6、E_u=40，计算 LGD*。",
        "依据《第三档商业银行资本监管规定》的 Word 公式，核心一级资本900、扣除项0、调整后表内外资产余额10000时，计算杠杆率并给出百分比。",
    ]
    for offset, (source_id, question) in enumerate(zip(["TC075", "TC076", "TC077", "TC078"], formula_questions), start=41):
        cases.append(_clone(g[source_id], f"HC{offset:03d}", "formula", "Word/PDF公式", question))

    # 45-50: answerability/refusal, no fabricated document uploads.
    cases.extend([
        _clone(g["TC083"], "HC045", "version_refusal", "时效与版本拒答", "仅根据当前文档身份卡，能否断言《附件1：集中度风险指标统计表》现行有效、已经失效或已被替代？如果版本字段未知，必须拒绝作法律效力结论。"),
        _clone(g["TC084"], "HC046", "version_refusal", "时效与版本拒答", "请核验《附件1：财产保险公司县域机构统计表》的当前法律效力状态。不得从文件名、年份或上传顺序推断；元数据不足时说明未知。"),
        _clone(g["TC086"], "HC047", "version_refusal", "时效与版本拒答", "《附件：关于融资租赁业务统计口径的说明》是否已被后续文件替代？只允许依据身份卡中的明确版本关系；没有关系时拒绝断言。"),
        _clone(g["TC092"], "HC048", "table_refusal", "可信拒答", "在《2023年08月保险业经营情况表》中查询指标‘监管压力测试虚拟项QX-50’的数值；若全库及该表均无该字段，明确拒答且不得给出猜测数值。", canonical_answer="当前知识库中不存在指标‘监管压力测试虚拟项QX-50’，依据不足，无法给出确定数值。", required_conclusions=["依据不足", "无法给出确定数值"]),
        _clone(g["TC096"], "HC049", "table_refusal", "可信拒答", "在《2024年9月财产险公司经营情况表》中查询指标‘监管压力测试虚拟项QX-51’；如不存在，应给出可解释拒答而不是返回邻近指标。", canonical_answer="当前知识库中不存在指标‘监管压力测试虚拟项QX-51’，依据不足，无法给出确定数值。", required_conclusions=["依据不足", "无法给出确定数值"]),
        _clone(g["TC079"], "HC050", "formula_refusal", "公式能力边界", "依据《中央交易对手风险暴露资本计量规则》的 Word 公式，给定 K_CCP=10、DF_i=20、pref=1、DF_CCP=30、DF_CM=5，尝试计算 K_CM_i；若可信执行器不支持该运算，必须说明能力边界并拒绝给出数值。"),
    ])

    # Update refusal corpus-search records for the two brand-new absent indicators.
    for case, query in ((cases[47], "监管压力测试虚拟项QX-50"), (cases[48], "监管压力测试虚拟项QX-51")):
        case["corpus_search"] = {
            "scope": "all_500_document_chunks_and_spreadsheet_cells",
            "query": query,
            "match_count": 0,
            "insufficiency_type": "nonexistent_indicator",
        }
    return cases


def validate(cases: list[dict[str, Any]], old_questions: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if len(cases) != 50:
        errors.append(f"expected 50 cases, got {len(cases)}")
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        errors.append("duplicate case IDs")
    old_norm = {_normalize_question(item["question"]) for item in old_questions}
    duplicate_questions = [case["id"] for case in cases if _normalize_question(case["question"]) in old_norm]
    if duplicate_questions:
        errors.append(f"questions duplicate prior challenge: {duplicate_questions}")
    if any(case["difficulty"] != "hard" for case in cases):
        errors.append("all cases must be hard")
    if any(case["review_status"] != "codex_verified" or case["expert_reviewed"] is not False for case in cases):
        errors.append("review status must be codex_verified and expert_reviewed=false")
    if any(case["answerable"] and not case["evidence"] for case in cases):
        errors.append("answerable case without evidence")
    required_types = {
        "fact", "multiple_choice", "judgment", "summary", "cross_document",
        "table_comparison", "table_cross_period_calculation", "mixed_regulation_table",
        "formula", "version_refusal", "table_refusal", "formula_refusal",
    }
    actual_types = {case["question_type"] for case in cases}
    missing_types = sorted(required_types - actual_types)
    if missing_types:
        errors.append(f"missing question types: {missing_types}")
    return errors


def main() -> int:
    source_rows = _read_jsonl(SOURCE_GOLD)
    source_by_id = {row["id"]: row for row in source_rows}
    cases = build_cases(source_by_id)
    errors = validate(cases, _read_jsonl(SOURCE_QUESTIONS))
    if errors:
        raise ValueError("; ".join(errors))

    questions = [
        {
            key: case[key]
            for key in (
                "id", "scenario", "question_type", "difficulty", "split",
                "question", "answerable", "scoring_type",
            )
        }
        for case in cases
    ]
    questions_path = OUTPUT_DIR / "questions.jsonl"
    gold_path = OUTPUT_DIR / "gold.jsonl"
    _write_jsonl(questions_path, questions)
    _write_jsonl(gold_path, cases)
    manifest = {
        "status": "built_not_yet_independently_audited",
        "case_count": len(cases),
        "difficulty": {"hard": len(cases)},
        "question_types": {
            question_type: sum(case["question_type"] == question_type for case in cases)
            for question_type in sorted({case["question_type"] for case in cases})
        },
        "answerability": {
            "answerable": sum(case["answerable"] for case in cases),
            "refusal": sum(not case["answerable"] for case in cases),
        },
        "source_case_count": len({source_id for case in cases for source_id in case["source_case_ids"]}),
        "hashes": {
            "questions_sha256": _sha256(questions_path),
            "gold_sha256": _sha256(gold_path),
            "builder_sha256": _sha256(Path(__file__)),
        },
    }
    (OUTPUT_DIR / "build_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
