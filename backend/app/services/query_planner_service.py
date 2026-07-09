from dataclasses import dataclass
import json
import re
import urllib.error
import urllib.request
from typing import Any

from backend.app.core.config import (
    QUERY_PLANNER_API_KEY,
    QUERY_PLANNER_BASE_URL,
    QUERY_PLANNER_ENABLED,
    QUERY_PLANNER_MAX_ASPECTS,
    QUERY_PLANNER_MAX_SEARCH_QUERIES,
    QUERY_PLANNER_MODEL,
    QUERY_PLANNER_TIMEOUT_SECONDS,
)


class QueryPlannerError(RuntimeError):
    pass


VALID_QUERY_TYPES = {"semantic_question", "document_style_statement", "keyword_anchor", "legacy", "fallback"}


@dataclass(frozen=True)
class QuerySearchQuery:
    query: str
    query_type: str
    rationale: str = ""

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "query_type": self.query_type,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class QueryAspect:
    aspect_id: str
    question: str
    search_queries: tuple[QuerySearchQuery, ...]
    evidence_need: str
    keywords: tuple[str, ...]

    @property
    def expected_evidence_type(self) -> str:
        return self.evidence_need

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "aspect_id": self.aspect_id,
            "question": self.question,
            "evidence_need": self.evidence_need,
            "expected_evidence_type": self.evidence_need,
            "search_queries": [query.to_debug_dict() for query in self.search_queries],
            "keywords": list(self.keywords),
        }


@dataclass(frozen=True)
class QueryPlan:
    original_question: str
    aspects: tuple[QueryAspect, ...]
    planner: str
    fallback_used: bool = False
    error: str | None = None
    budget: dict[str, Any] | None = None

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "original_question": self.original_question,
            "planner": self.planner,
            "fallback_used": self.fallback_used,
            "error": self.error,
            "budget": self.budget or {},
            "aspects": [aspect.to_debug_dict() for aspect in self.aspects],
        }


@dataclass(frozen=True)
class QueryBudget:
    detected_items: tuple[str, ...]
    max_aspects: int
    system_max_aspects: int
    capacity_limited: bool
    omitted_or_merged_items: tuple[str, ...] = ()

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "detected_items": list(self.detected_items),
            "detected_item_count": len(self.detected_items),
            "max_aspects": self.max_aspects,
            "system_max_aspects": self.system_max_aspects,
            "capacity_limited": self.capacity_limited,
            "omitted_or_merged_items": list(self.omitted_or_merged_items),
        }


def plan_query(question: str) -> QueryPlan:
    cleaned_question = question.strip()
    if not cleaned_question:
        return QueryPlan(original_question="", aspects=(), planner="empty", fallback_used=True)

    budget = plan_query_budget(cleaned_question)

    if QUERY_PLANNER_ENABLED and QUERY_PLANNER_API_KEY:
        try:
            aspects, omitted_or_merged_items = _plan_with_deepseek(cleaned_question, budget)
            if aspects:
                budget = _budget_with_omissions(budget, omitted_or_merged_items)
                return QueryPlan(
                    original_question=cleaned_question,
                    aspects=tuple(aspects),
                    planner=f"deepseek:{QUERY_PLANNER_MODEL}",
                    fallback_used=False,
                    budget=budget.to_debug_dict(),
                )
        except QueryPlannerError as exc:
            fallback = _fallback_aspects(cleaned_question, budget)
            return QueryPlan(
                original_question=cleaned_question,
                aspects=tuple(fallback),
                planner="fallback",
                fallback_used=True,
                error=str(exc),
                budget=budget.to_debug_dict(),
            )

    return QueryPlan(
        original_question=cleaned_question,
        aspects=tuple(_fallback_aspects(cleaned_question, budget)),
        planner="fallback",
        fallback_used=True,
        budget=budget.to_debug_dict(),
    )


def plan_query_budget(question: str) -> QueryBudget:
    detected_items = tuple(_detect_requested_items(question))
    desired = max(len(detected_items), len(_split_question_parts(question)))
    desired = max(desired, 1)
    max_aspects = min(desired, QUERY_PLANNER_MAX_ASPECTS)
    return QueryBudget(
        detected_items=detected_items,
        max_aspects=max_aspects,
        system_max_aspects=QUERY_PLANNER_MAX_ASPECTS,
        capacity_limited=desired > QUERY_PLANNER_MAX_ASPECTS,
        omitted_or_merged_items=detected_items[QUERY_PLANNER_MAX_ASPECTS:],
    )


def _budget_with_omissions(budget: QueryBudget, omitted_or_merged_items: list[str]) -> QueryBudget:
    merged = tuple(_dedupe([*budget.omitted_or_merged_items, *omitted_or_merged_items]))
    return QueryBudget(
        detected_items=budget.detected_items,
        max_aspects=budget.max_aspects,
        system_max_aspects=budget.system_max_aspects,
        capacity_limited=budget.capacity_limited or bool(merged),
        omitted_or_merged_items=merged,
    )


def _plan_with_deepseek(question: str, budget: QueryBudget) -> tuple[list[QueryAspect], list[str]]:
    endpoint = f"{QUERY_PLANNER_BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": QUERY_PLANNER_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是 ReguMate 的检索前 QueryPlanner。"
                    "你的唯一任务是把银行监管制度、统计报表填报说明、指标口径问题拆成检索 aspect，生成检索计划。"
                    "禁止回答用户问题，禁止给出结论，禁止补充知识库外事实。"
                    "你要生成有利于向量检索命中制度原文 chunk 的自然语言查询，不要只挑关键词。"
                    "用户枚举多个处理对象时，原则上一项一个 aspect。"
                    "不得丢弃依据不足、违规判断、正式监管报告生成等安全边界问题。"
                    "只输出 JSON 对象。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "请按本次预算把下面问题拆成 1 到 "
                    f"{budget.max_aspects} 个 aspect，系统安全上限为 {budget.system_max_aspects}。每个 aspect 包含："
                    "aspect_id、question、evidence_need、search_queries、keywords。"
                    "如果用户明确枚举了多个处理对象，一般每个处理对象单独成为一个 aspect；"
                    "如果因为预算必须合并或遗漏项目，把项目名称写入 omitted_or_merged_items。"
                    "输出 JSON 顶层必须包含 aspects 和 omitted_or_merged_items。"
                    "search_queries 必须是对象数组，每个对象包含 query、query_type、rationale。"
                    "query_type 只能是 semantic_question、document_style_statement、keyword_anchor。"
                    "每个 aspect 生成 2 到 "
                    f"{QUERY_PLANNER_MAX_SEARCH_QUERIES} 条 search_queries："
                    "semantic_question 贴近用户意图；document_style_statement 像制度原文、填报说明标题或证据句；"
                    "keyword_anchor 只用少量关键术语兜底。"
                    "优先生成可能出现在制度文档/填报说明中的自然语言证据查询，禁止只输出关键词串。"
                    "evidence_need 描述要找章节定义、填报口径、处理流程、例外条件或留痕依据。"
                    "如果问题只有一个主题，也返回一个 aspect。"
                    "示例输入：资产合计差异应该优先排查哪些问题？如果差异来自外币折算，需要保留什么依据？"
                    "示例输出：{\"aspects\":["
                    "{\"aspect_id\":\"asset_total_difference_check\","
                    "\"question\":\"资产合计差异应该优先排查哪些问题？\","
                    "\"evidence_need\":\"资产合计差异处理流程、优先排查原因或校验规则\","
                    "\"search_queries\":["
                    "{\"query\":\"资产合计与分项合计存在差异时应优先检查哪些原因\",\"query_type\":\"semantic_question\",\"rationale\":\"贴近用户意图\"},"
                    "{\"query\":\"资产合计校验差异处理流程 币种折算 四舍五入 科目映射 重复汇总\",\"query_type\":\"document_style_statement\",\"rationale\":\"贴近填报说明证据句\"},"
                    "{\"query\":\"资产合计 差异 币种折算 四舍五入 科目映射 重复汇总\",\"query_type\":\"keyword_anchor\",\"rationale\":\"术语兜底\"}],"
                    "\"keywords\":[\"资产合计\",\"差异\",\"币种折算\",\"四舍五入\",\"科目映射\",\"重复汇总\"]},"
                    "{\"aspect_id\":\"foreign_currency_evidence\","
                    "\"question\":\"如果差异来自外币折算，需要保留什么依据？\","
                    "\"evidence_need\":\"外币折算差异的留痕依据、支持材料或填报说明\","
                    "\"search_queries\":["
                    "{\"query\":\"外币折算导致资产合计差异时需要保留哪些支持材料\",\"query_type\":\"semantic_question\",\"rationale\":\"贴近用户意图\"},"
                    "{\"query\":\"外币折算差异 留痕依据 汇率日期 折算规则 原币金额来源\",\"query_type\":\"document_style_statement\",\"rationale\":\"贴近制度材料表述\"},"
                    "{\"query\":\"外币折算 汇率日期 折算规则 原币金额来源\",\"query_type\":\"keyword_anchor\",\"rationale\":\"术语兜底\"}],"
                    "\"keywords\":[\"外币折算\",\"留痕依据\",\"汇率日期\",\"折算规则\",\"原币金额来源\"]}"
                    "]}\n"
                    f"本次规则识别到的用户处理对象：{json.dumps(list(budget.detected_items), ensure_ascii=False)}\n"
                    "输出格式：{\"aspects\":[...],\"omitted_or_merged_items\":[]}\n\n"
                    f"用户问题：{question}"
                ),
            },
        ],
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={
            "Authorization": f"Bearer {QUERY_PLANNER_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=QUERY_PLANNER_TIMEOUT_SECONDS,
        ) as response:
            response_body = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise QueryPlannerError("LLM query planner 调用失败") from exc

    try:
        body = json.loads(response_body)
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(_extract_json_object(content))
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise QueryPlannerError("LLM query planner 返回格式不可解析") from exc

    return _aspects_from_payload(question, parsed, max_aspects=budget.max_aspects), _clean_string_list(
        parsed.get("omitted_or_merged_items")
    )


def _aspects_from_payload(
    question: str,
    payload: dict[str, Any],
    max_aspects: int | None = None,
) -> list[QueryAspect]:
    raw_aspects = payload.get("aspects")
    if not isinstance(raw_aspects, list):
        raise QueryPlannerError("LLM query planner 未返回 aspects")

    aspects: list[QueryAspect] = []
    limit = max_aspects if max_aspects is not None else QUERY_PLANNER_MAX_ASPECTS
    for index, raw_aspect in enumerate(raw_aspects[:limit], start=1):
        if not isinstance(raw_aspect, dict):
            continue
        sub_question = _clean_text(raw_aspect.get("question")) or question
        search_queries = _clean_search_queries(raw_aspect.get("search_queries"))
        keywords = _clean_string_list(raw_aspect.get("keywords"))
        if not search_queries:
            search_queries = [QuerySearchQuery(query=sub_question, query_type="fallback", rationale="LLM 未返回检索查询")]
        if not keywords:
            keywords = _keywords_from_text(" ".join([sub_question, *[query.query for query in search_queries]]))
        aspects.append(
            QueryAspect(
                aspect_id=_clean_aspect_id(raw_aspect.get("aspect_id"), index),
                question=sub_question,
                search_queries=tuple(_dedupe_search_queries(search_queries)[:QUERY_PLANNER_MAX_SEARCH_QUERIES]),
                evidence_need=(
                    _clean_text(raw_aspect.get("evidence_need"))
                    or _clean_text(raw_aspect.get("expected_evidence_type"))
                    or "相关制度依据"
                ),
                keywords=tuple(_dedupe(keywords)[:8]),
            )
        )

    if not aspects:
        raise QueryPlannerError("LLM query planner 未产生有效 aspect")
    return aspects


def _fallback_aspects(question: str, budget: QueryBudget | None = None) -> list[QueryAspect]:
    budget = budget or plan_query_budget(question)
    heuristic_aspects = _domain_heuristic_aspects(question, max_aspects=budget.max_aspects)
    if heuristic_aspects:
        return heuristic_aspects

    parts = _split_question_parts(question)
    if not parts:
        parts = [question]

    aspects: list[QueryAspect] = []
    for index, part in enumerate(parts[:budget.max_aspects], start=1):
        keywords = _keywords_from_text(part)
        aspects.append(
            QueryAspect(
                aspect_id=f"aspect_{index}",
                question=part,
                search_queries=_fallback_search_queries(part),
                evidence_need=_infer_evidence_type(part),
                keywords=tuple(keywords),
            )
        )
    return aspects


def _domain_heuristic_aspects(question: str, max_aspects: int | None = None) -> list[QueryAspect]:
    normalized = re.sub(r"\s+", "", question)
    aspects: list[QueryAspect] = []
    detected = _detect_requested_items(question)
    if not detected:
        return []
    for item in detected:
        aspect = _heuristic_aspect_for_item(item, normalized)
        if aspect is not None:
            aspects.append(aspect)
        if max_aspects is not None and len(aspects) >= max_aspects:
            break
    return _dedupe_aspects(aspects)


def _heuristic_aspect_for_item(item: str, normalized: str) -> QueryAspect | None:
    if item == "文档版本信息":
        return QueryAspect(
            aspect_id="document_version_info",
            question="文件版本是什么",
            search_queries=(
                QuerySearchQuery("文档重要声明中的文件版本是什么", "semantic_question", "贴近用户意图"),
                QuerySearchQuery("重要声明 文件版本 发布日期 适用范围", "document_style_statement", "贴近文档元数据表述"),
                QuerySearchQuery("文件版本 版本号 发布日期", "keyword_anchor", "术语兜底"),
            ),
            evidence_need="文档重要声明、文件版本、发布日期或适用范围",
            keywords=("文件版本", "版本号", "发布日期", "重要声明", "适用范围"),
        )
    if item == "统计报表定义":
        return QueryAspect(
            aspect_id="statistical_report_definition",
            question="统计报表是什么意思",
            search_queries=(
                QuerySearchQuery("统计报表在本文档中是什么意思", "semantic_question", "贴近用户意图"),
                QuerySearchQuery("统计报表是指 指定模板 期间 币种 机构层级 指标口径", "document_style_statement", "贴近制度定义句"),
                QuerySearchQuery("统计报表 定义 指标口径", "keyword_anchor", "术语兜底"),
            ),
            evidence_need="统计报表术语定义和字段口径",
            keywords=("统计报表", "定义", "指定模板", "期间", "币种", "指标口径"),
        )
    if item == "校验规则分类":
        return QueryAspect(
            aspect_id="check_rule_classification",
            question="校验规则分为哪几类",
            search_queries=(
                QuerySearchQuery("校验规则分为哪几类", "semantic_question", "贴近用户意图"),
                QuerySearchQuery("校验规则分为格式规则 完整性规则 数值规则 勾稽规则 跨期规则 跨表规则 依据规则", "document_style_statement", "贴近制度分类句"),
                QuerySearchQuery("校验规则 分类 格式 完整性 数值 勾稽 跨期 跨表 依据", "keyword_anchor", "术语兜底"),
            ),
            evidence_need="校验规则分类说明",
            keywords=("校验规则", "格式规则", "完整性规则", "数值规则", "勾稽规则", "跨期规则", "跨表规则", "依据规则"),
        )
    if item.startswith("规则编号查询:"):
        rule_code = item.split(":", maxsplit=1)[1]
        return QueryAspect(
            aspect_id=f"rule_{_clean_aspect_id(rule_code, 1).lower()}",
            question=f"规则 {rule_code} 的触发条件和处理建议是什么",
            search_queries=(
                QuerySearchQuery(f"规则 {rule_code} 的触发条件和处理建议是什么", "semantic_question", "贴近用户意图"),
                QuerySearchQuery(f"规则编号 {rule_code} 规则名称 触发条件 处理建议", "document_style_statement", "贴近规则表字段"),
                QuerySearchQuery(f"{rule_code} 触发条件 处理建议", "keyword_anchor", "术语兜底"),
            ),
            evidence_need=f"规则表中 {rule_code} 的触发条件和处理建议",
            keywords=(rule_code, "触发条件", "处理建议", "规则编号"),
        )
    if item == "普惠小微贷款纳入":
        return QueryAspect(
            aspect_id="inclusive_micro_loan_scope",
            question="普惠小微贷款纳入应如何处理",
            search_queries=(
                QuerySearchQuery("普惠小微贷款纳入的认定标准和处理流程", "semantic_question", "贴近用户意图"),
                QuerySearchQuery("普惠小微贷款 纳入 口径 条件 处理流程", "document_style_statement", "贴近填报说明证据句"),
                QuerySearchQuery("普惠小微贷款 纳入 口径 条件", "keyword_anchor", "术语兜底"),
            ),
            evidence_need="普惠小微贷款纳入的口径、条件和处理流程",
            keywords=("普惠小微贷款", "普惠小微", "纳入", "口径", "条件", "处理流程"),
        )
    if item == "绿色信贷识别":
        return QueryAspect(
            aspect_id="green_credit_identification",
            question="绿色信贷识别应如何处理",
            search_queries=(
                QuerySearchQuery("绿色信贷识别的标准和方法有哪些", "semantic_question", "贴近用户意图"),
                QuerySearchQuery("绿色信贷 识别 标准 分类 认定方法", "document_style_statement", "贴近制度原文"),
                QuerySearchQuery("绿色信贷 识别 标准 分类", "keyword_anchor", "术语兜底"),
            ),
            evidence_need="绿色信贷识别的标准、分类和识别方法",
            keywords=("绿色信贷", "识别", "标准", "分类", "认定方法"),
        )
    if item == "逾期与不良贷款关系":
        return QueryAspect(
            aspect_id="overdue_nonperforming_relationship",
            question="逾期与不良贷款关系应如何处理",
            search_queries=(
                QuerySearchQuery("逾期贷款与不良贷款之间的认定关系和转换规则", "semantic_question", "贴近用户意图"),
                QuerySearchQuery("逾期贷款 不良贷款 关系 认定标准 转换规则", "document_style_statement", "贴近填报说明证据句"),
                QuerySearchQuery("逾期 不良 认定 转换", "keyword_anchor", "术语兜底"),
            ),
            evidence_need="逾期贷款与不良贷款的关系、认定标准和转换规则",
            keywords=("逾期贷款", "不良贷款", "关系", "认定标准", "转换规则"),
        )
    if item == "资产合计差异排查" or ("资产合计" in normalized and "差异" in normalized and not item):
        return QueryAspect(
            aspect_id="asset_total_difference_check",
            question="资产合计差异应该优先排查哪些问题",
            search_queries=(
                QuerySearchQuery("资产合计与分项合计存在差异时应优先检查哪些原因", "semantic_question", "贴近用户问题的处理口径"),
                QuerySearchQuery("资产合计校验差异处理流程 币种折算 四舍五入 科目映射 重复汇总", "document_style_statement", "贴近填报说明中的证据句"),
                QuerySearchQuery("资产合计 差异 币种折算 四舍五入 科目映射 重复汇总", "keyword_anchor", "关键术语兜底"),
            ),
            evidence_need="资产合计差异处理或优先排查章节",
            keywords=("资产合计", "差异", "优先检查", "优先排查", "币种折算", "四舍五入", "科目映射", "重复汇总"),
        )
    if item == "外币折算依据保留" or ("外币折算" in normalized and not item):
        return QueryAspect(
            aspect_id="foreign_currency_evidence",
            question="外币折算差异需要保留什么依据",
            search_queries=(
                QuerySearchQuery("外币折算导致资产合计差异时需要保留哪些支持材料", "semantic_question", "贴近用户问题的留痕依据"),
                QuerySearchQuery("外币折算差异 留痕依据 汇率日期 折算规则 原币金额来源", "document_style_statement", "贴近制度材料表述"),
                QuerySearchQuery("外币折算 汇率日期 折算规则 原币金额来源", "keyword_anchor", "关键术语兜底"),
            ),
            evidence_need="外币折算差异的留痕依据或支持材料",
            keywords=("外币折算", "保留", "依据", "汇率日期", "折算规则", "原币金额", "来源"),
        )
    if item == "历史差错更正记录":
        return QueryAspect(
            aspect_id="historical_error_correction",
            question="历史差错更正记录应如何处理",
            search_queries=(
                QuerySearchQuery("历史差错更正记录需要保留哪些更正依据", "semantic_question", "贴近用户意图"),
                QuerySearchQuery("历史差错 更正记录 更正原因 更正前后数据 审批痕迹", "document_style_statement", "贴近制度材料表述"),
                QuerySearchQuery("历史差错 更正记录 更正原因", "keyword_anchor", "术语兜底"),
            ),
            evidence_need="历史差错更正记录、原因、前后数据和审批痕迹要求",
            keywords=("历史差错", "差错更正", "更正记录", "更正原因", "审批痕迹"),
        )
    if item == "多期间影响说明":
        return QueryAspect(
            aspect_id="multi_period_impact",
            question="多期间影响说明应如何处理",
            search_queries=(
                QuerySearchQuery("多期间影响说明需要覆盖哪些期间和影响口径", "semantic_question", "贴近用户意图"),
                QuerySearchQuery("多期间影响说明 跨期影响 涉及期间 调整口径", "document_style_statement", "贴近制度材料表述"),
                QuerySearchQuery("多期间 影响说明 跨期 期间", "keyword_anchor", "术语兜底"),
            ),
            evidence_need="多期间影响说明、跨期影响和涉及期间要求",
            keywords=("多期间", "影响说明", "跨期", "涉及期间", "调整口径"),
        )
    if item == "依据不足违规报告边界":
        return QueryAspect(
            aspect_id="insufficient_evidence_safety_boundary",
            question="依据不足时是否可以直接判断银行违规或生成正式监管报告",
            search_queries=(
                QuerySearchQuery("依据不足时是否可以判断银行违规或生成正式监管报告", "semantic_question", "贴近用户意图"),
                QuerySearchQuery("依据不足 无法判断 银行违规 正式监管报告 不得生成", "document_style_statement", "贴近系统边界表述"),
                QuerySearchQuery("依据不足 违规 正式监管报告", "keyword_anchor", "术语兜底"),
            ),
            evidence_need="依据不足时的拒答边界、违规判断边界和正式监管报告生成边界",
            keywords=("依据不足", "无法判断", "银行违规", "正式监管报告", "不得生成"),
        )
    return None


def _detect_requested_items(question: str) -> list[str]:
    normalized = re.sub(r"\s+", "", question)
    detectors = [
        ("文档版本信息", ("文件版本",)),
        ("文档版本信息", ("版本", "模拟制度")),
        ("统计报表定义", ("统计报表", "意思")),
        ("统计报表定义", ("统计报表", "定义")),
        ("校验规则分类", ("校验规则", "几类")),
        ("校验规则分类", ("校验规则", "分类")),
        ("普惠小微贷款纳入", ("普惠小微", "纳入")),
        ("绿色信贷识别", ("绿色信贷", "识别")),
        ("逾期与不良贷款关系", ("逾期", "不良")),
        ("资产合计差异排查", ("资产合计", "差异")),
        ("资产合计差异排查", ("资产合计", "不一致")),
        ("资产合计差异排查", ("资产合计", "分项加总")),
        ("资产合计差异排查", ("资产合计", "分项合计")),
        ("外币折算依据保留", ("外币折算",)),
        ("历史差错更正记录", ("历史差错", "更正")),
        ("多期间影响说明", ("多期间", "影响")),
        ("依据不足违规报告边界", ("依据不足",)),
    ]
    items = [label for label, terms in detectors if all(term in normalized for term in terms)]
    for rule_code in re.findall(r"R-[A-Z]+-\d+", question.upper()):
        items.append(f"规则编号查询:{rule_code}")
    if any(term in normalized for term in ["违规", "监管报告", "正式监管报告"]) and "依据不足违规报告边界" not in items:
        items.append("依据不足违规报告边界")
    return _dedupe(items)


def _split_question_parts(question: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"[？?。；;]|如果|若|以及|并且|同时|，且|、", question)
        if part.strip()
    ]


def _dedupe_aspects(aspects: list[QueryAspect]) -> list[QueryAspect]:
    deduped: list[QueryAspect] = []
    seen: set[str] = set()
    for aspect in aspects:
        if aspect.aspect_id in seen:
            continue
        seen.add(aspect.aspect_id)
        deduped.append(aspect)
    return deduped


def _extract_json_object(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    match = re.search(r"\{.*\}", stripped, flags=re.S)
    if not match:
        raise json.JSONDecodeError("missing JSON object", stripped, 0)
    return match.group(0)


def _clean_aspect_id(value: Any, index: int) -> str:
    text = _clean_text(value)
    if not text:
        return f"aspect_{index}"
    cleaned = re.sub(r"[^A-Za-z0-9_\-]+", "_", text).strip("_")
    return cleaned or f"aspect_{index}"


def _clean_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        return []
    return [cleaned for item in items if (cleaned := _clean_text(item))]


def _clean_search_queries(value: Any) -> list[QuerySearchQuery]:
    if isinstance(value, str):
        cleaned = _clean_text(value)
        return [QuerySearchQuery(query=cleaned, query_type="legacy", rationale="兼容旧字符串格式")] if cleaned else []
    if not isinstance(value, list):
        return []

    queries: list[QuerySearchQuery] = []
    for item in value:
        if isinstance(item, str):
            cleaned = _clean_text(item)
            if cleaned:
                queries.append(
                    QuerySearchQuery(query=cleaned, query_type="legacy", rationale="兼容旧字符串格式")
                )
            continue
        if not isinstance(item, dict):
            continue
        query = _clean_text(item.get("query"))
        if not query:
            continue
        query_type = _clean_text(item.get("query_type")) or "semantic_question"
        if query_type not in VALID_QUERY_TYPES:
            query_type = "semantic_question"
        queries.append(
            QuerySearchQuery(
                query=query,
                query_type=query_type,
                rationale=_clean_text(item.get("rationale")),
            )
        )
    return queries


def _fallback_search_queries(question: str) -> tuple[QuerySearchQuery, ...]:
    keywords = " ".join(_keywords_from_text(question))
    queries = [
        QuerySearchQuery(query=question, query_type="semantic_question", rationale="fallback 子问题检索"),
    ]
    if keywords and keywords != question:
        queries.append(
            QuerySearchQuery(query=keywords, query_type="keyword_anchor", rationale="fallback 关键术语兜底")
        )
    return tuple(queries[:QUERY_PLANNER_MAX_SEARCH_QUERIES])


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _dedupe_search_queries(items: list[QuerySearchQuery]) -> list[QuerySearchQuery]:
    deduped: list[QuerySearchQuery] = []
    seen: set[str] = set()
    for item in items:
        normalized = re.sub(r"\s+", "", item.query)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(item)
    return deduped


def _keywords_from_text(text: str) -> list[str]:
    domain_phrases = [
        "资产合计",
        "差异",
        "优先检查",
        "优先排查",
        "币种折算",
        "外币折算",
        "四舍五入",
        "科目映射",
        "重复汇总",
        "保留",
        "依据",
        "汇率日期",
        "折算规则",
        "原币金额",
        "逾期贷款",
        "不良贷款",
        "风险分类",
        "绿色信贷",
        "普惠小微",
    ]
    normalized = re.sub(r"\s+", "", text)
    keywords = [phrase for phrase in domain_phrases if phrase in normalized]
    keywords.extend(token for token in re.findall(r"[A-Za-z0-9_]{2,}", text))
    if keywords:
        return _dedupe(keywords)
    return _dedupe(re.findall(r"[\u4e00-\u9fff]{2,8}", text))[:6]


def _infer_evidence_type(text: str) -> str:
    normalized = re.sub(r"\s+", "", text)
    if any(term in normalized for term in ["保留", "依据", "留痕", "材料"]):
        return "留痕依据或支持材料"
    if any(term in normalized for term in ["怎么", "如何", "处理", "排查", "检查"]):
        return "处理流程或排查要求"
    if any(term in normalized for term in ["哪些", "情形", "条件", "口径"]):
        return "分类条件或填报口径"
    return "相关制度依据"
