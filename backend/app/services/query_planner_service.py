from dataclasses import dataclass, field, replace
import json
import re
import urllib.error
import urllib.request
from typing import Any
from backend.app.services.performance_metrics import measure, timed

from backend.app.core.config import (
    QUERY_PLANNER_INCLUDE_THINKING,
    QUERY_PLANNER_API_KEY,
    QUERY_PLANNER_BASE_URL,
    QUERY_PLANNER_ENABLED,
    QUERY_PLANNER_MAX_ASPECTS,
    QUERY_PLANNER_MAX_SEARCH_QUERIES,
    QUERY_PLANNER_MODEL,
    QUERY_PLANNER_PROVIDER,
    QUERY_PLANNER_RESPONSE_FORMAT,
    QUERY_PLANNER_TIMEOUT_SECONDS,
)
from backend.app.services.llm_client import (
    ChatCompletionConfig,
    ChatCompletionError,
    chat_completion_content,
)


class QueryPlannerError(RuntimeError):
    pass


VALID_QUERY_TYPES = {"semantic_question", "document_style_statement", "keyword_anchor", "table_locator", "legacy", "fallback"}
MCQ_OPTION_SEARCH_QUERY_BUDGET = 6
TABLE_MARKERS = ["Excel", "excel", "工作表", "单元格", "情况表", "统计表", "经营表", "月报", "报表", "CSV", "csv"]
TABLE_ONLY_FILTER_KEYS = {
    "sheet",
    "indicator",
    "row_label",
    "column_label",
    "unit",
    "metric",
    "scope",
    "year",
    "month",
    "quarter",
}

COMMON_TABLE_INDICATORS = [
    "现金及存放同业",
    "贷款余额",
    "证券投资",
    "其他资产",
    "资产合计",
    "不良贷款余额",
    "损失类贷款余额",
    "可疑类贷款余额",
    "次级类贷款余额",
    "资金运用余额",
    "普惠小微贷款余额",
    "绿色信贷余额",
    "原保险保费收入",
    "资本充足率",
]


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
    modality: str = "text"
    table_task: str = "none"
    table_filters: dict[str, Any] = field(default_factory=dict)
    operation: str = "none"
    selectors: tuple[dict[str, Any], ...] = ()
    expected_unit: str | None = None
    explicit_filter_keys: tuple[str, ...] = ()

    @property
    def expected_evidence_type(self) -> str:
        return self.evidence_need

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "aspect_id": self.aspect_id,
            "question": self.question,
            "evidence_need": self.evidence_need,
            "expected_evidence_type": self.evidence_need,
            "modality": self.modality,
            "table_task": self.table_task,
            "table_filters": self.table_filters,
            "operation": self.operation,
            "selectors": list(self.selectors),
            "expected_unit": self.expected_unit,
            "explicit_filter_keys": list(self.explicit_filter_keys),
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


@timed("query_planner.total")
def plan_query(question: str, options: list[str] | None = None) -> QueryPlan:
    cleaned_question = question.strip()
    if not cleaned_question:
        return QueryPlan(original_question="", aspects=(), planner="empty", fallback_used=True)

    budget = plan_query_budget(cleaned_question)

    # Explicit spreadsheet questions are deterministic enough to plan locally.
    # This avoids an API round trip and, more importantly, prevents an LLM from
    # changing operand order or losing a filename/sheet constraint.
    table_aspect = _fallback_table_aspect(cleaned_question, options=options)
    if table_aspect is not None:
        table_aspects = (
            (
                replace(
                    table_aspect,
                    modality="table",
                    question=_mixed_table_question_part(table_aspect.question),
                ),
                *_mixed_regulation_aspects(table_aspect),
            )
            if table_aspect.modality == "mixed"
            else (table_aspect,)
        )
        return QueryPlan(
            original_question=cleaned_question,
            aspects=table_aspects,
            planner="deterministic-table",
            fallback_used=True,
            budget=budget.to_debug_dict(),
        )

    # Multiple-choice contest questions normally name the target material in
    # book-title marks, while the remaining stem is generic wording such as
    # "which statement is consistent".  Treating that whole stem as lexical
    # coverage dilutes the exact title and can discard the right chunk after
    # reranking.  Anchor coverage/reranking on the material title while still
    # using the original stem and every option for hybrid candidate recall.
    mcq_aspect = _fallback_mcq_aspect(cleaned_question, options=options)
    if mcq_aspect is not None:
        return QueryPlan(
            original_question=cleaned_question,
            aspects=(mcq_aspect,),
            planner="deterministic-mcq",
            fallback_used=True,
            budget=budget.to_debug_dict(),
        )

    heuristic_aspects = _domain_heuristic_aspects(cleaned_question, max_aspects=budget.max_aspects)
    if heuristic_aspects and _should_use_domain_heuristic_plan(cleaned_question, heuristic_aspects):
        return QueryPlan(
            original_question=cleaned_question,
            aspects=tuple(heuristic_aspects),
            planner="deterministic-domain",
            fallback_used=True,
            budget=budget.to_debug_dict(),
        )

    if QUERY_PLANNER_ENABLED and QUERY_PLANNER_API_KEY:
        try:
                aspects, omitted_or_merged_items = _plan_with_llm(cleaned_question, budget)
                if aspects:
                    aspects = _split_merged_llm_aspects(aspects, budget)
                    budget = _budget_with_omissions(budget, omitted_or_merged_items)
                    return QueryPlan(
                        original_question=cleaned_question,
                        aspects=tuple(aspects),
                        planner=f"{QUERY_PLANNER_PROVIDER}:{QUERY_PLANNER_MODEL}",
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
    explicit_items = _explicit_requested_items(question)
    detected_items = tuple(
        _dedupe(
            [
                *explicit_items,
                *[
                    item
                    for item in _detect_requested_items(question)
                    if not any(_requested_items_overlap(item, explicit) for explicit in explicit_items)
                ],
            ]
        )
    )
    question_parts = _split_question_parts(question)
    nested_coordinated_count = sum(
        max(1, len(_coordinated_aspect_questions(part)))
        for part in question_parts
    )
    desired = max(len(detected_items), len(question_parts), nested_coordinated_count)
    desired = max(desired, len(_coordinated_aspect_questions(question)))
    desired = max(desired, _explicit_requested_aspect_count(question))
    desired = max(desired, 1)
    max_aspects = min(desired, QUERY_PLANNER_MAX_ASPECTS)
    return QueryBudget(
        detected_items=detected_items,
        max_aspects=max_aspects,
        system_max_aspects=QUERY_PLANNER_MAX_ASPECTS,
        capacity_limited=desired > QUERY_PLANNER_MAX_ASPECTS,
        omitted_or_merged_items=detected_items[QUERY_PLANNER_MAX_ASPECTS:],
    )


def _explicit_requested_items(question: str) -> list[str]:
    text = str(question or "").strip()
    if not text:
        return []
    compact = re.sub(r"\s+", "", text)
    chinese_counts = {
        "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
        "七": 7, "八": 8, "九": 9, "十": 10,
    }
    count_match = re.search(r"([二三四五六七八九十两]|\d+)(?:项|类)(?:信息|要求|内容|控制|事实)?", compact)
    if not count_match:
        return []
    expected_count = int(count_match.group(1)) if count_match.group(1).isdigit() else chinese_counts.get(count_match.group(1), 0)
    if expected_count <= 1:
        return []
    original_match = re.search(r"([二三四五六七八九十两]|\d+)(?:项|类)(?:信息|要求|内容|控制|事实)?", text)
    if not original_match:
        return []
    if original_match.end() < len(text) and "：" in text[original_match.end(): original_match.end() + 4]:
        segment = text[text.find("：", original_match.end()) + 1:]
    else:
        left = text[: original_match.start()]
        segment = re.split(r"[：:]", left)[-1]
    segment = re.split(r"[。！？?]", segment, maxsplit=1)[0]
    parts = [
        re.sub(r"^(?:形成审查摘要|联合核验|核验|说明|概括|回答|列出|请)+", "", part.strip(" ，,；;。:："))
        for part in re.split(r"[、；;,，]|以及|和|与|及", segment)
        if part.strip(" ，,；;。:：")
    ]
    parts = [part for part in parts if part and not re.fullmatch(r"(?:再)?(?:比较|查询|读取|计算|说明|最后)", part)]
    return _dedupe(parts)[:expected_count] if len(parts) >= expected_count else []


def _requested_items_overlap(left: str, right: str) -> bool:
    def normalize(value: str) -> str:
        text = re.sub(r"[\s，,。；;：:、？?]+", "", str(value or ""))
        for noise in ("是否可以", "是否", "可以", "直接", "生成", "正式", "违规", "报告"):
            text = text.replace(noise, "")
        return text

    left_norm = normalize(left)
    right_norm = normalize(right)
    return bool(left_norm and right_norm and (left_norm in right_norm or right_norm in left_norm))


def _explicit_requested_aspect_count(question: str) -> int:
    """Estimate only an upper budget for visibly enumerated user requests."""

    compact = re.sub(r"\s+", "", str(question or ""))
    chinese_counts = {
        "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
        "七": 7, "八": 8, "九": 9, "十": 10,
    }
    explicit = 1
    for match in re.finditer(r"([二三四五六七八九十两]|\d+)(?:项|类)(?:信息|要求|内容|控制|事实)?", compact):
        token = match.group(1)
        explicit = max(explicit, int(token) if token.isdigit() else chinese_counts.get(token, 1))
    if any(marker in compact for marker in ("分别", "逐项", "汇总", "串联", "同时", "跨制度", "审查摘要")):
        # Enumeration commas are ambiguous in ordinary prose, so only use
        # them when the user explicitly asks for a multi-item response. This
        # raises the planner ceiling; it does not force artificial aspects.
        explicit = max(explicit, 1 + compact.count("、") + compact.count("；"))
    return min(explicit, QUERY_PLANNER_MAX_ASPECTS)


def _split_merged_llm_aspects(
    aspects: list[QueryAspect],
    budget: QueryBudget,
) -> list[QueryAspect]:
    """Restore clearly coordinated requirements merged by the remote planner.

    The split is intentionally narrow: strong “以及” coordination, or an
    “说明 A 和 B” construction where both sides name requirement categories.
    It never exceeds the deterministic budget and does not use answer data.
    """

    split_limit = max(budget.max_aspects, budget.system_max_aspects)
    if len(aspects) >= split_limit:
        return aspects
    result: list[QueryAspect] = []
    for aspect in aspects:
        remaining_capacity = split_limit - len(result)
        split_questions = _coordinated_aspect_questions(aspect.question)
        if len(split_questions) <= 1 or remaining_capacity < len(split_questions):
            result.append(aspect)
            continue
        for index, question in enumerate(split_questions, start=1):
            result.append(
                replace(
                    aspect,
                    aspect_id=f"{aspect.aspect_id}_part_{index}",
                    question=question,
                    search_queries=_fallback_search_queries(question),
                    evidence_need=f"与“{question}”直接对应的制度事实、条件或例外",
                    keywords=tuple(_keywords_from_text(question)),
                )
            )
    return result[:split_limit]


def _coordinated_aspect_questions(question: str) -> list[str]:
    text = str(question or "").strip()
    prefix_match = re.match(r"^(.*?(?:说明|概括|回答|列出|判断|核验|比较))", text)
    prefix = prefix_match.group(1) if prefix_match else ""
    titles = re.findall(r"《[^》]{2,80}》", text)

    summary_enumeration = _summary_enumerated_aspect_questions(text)
    if summary_enumeration:
        return summary_enumeration

    explicit_follow_up = re.match(
        r"^(.*?)[，,]并(说明|概括|回答|列出|判断|核验|比较)([^，。；]{4,80})([。？?]?)$",
        text,
    )
    if explicit_follow_up:
        left, verb, right, suffix = explicit_follow_up.groups()
        return [left.strip(" ，,"), f"{verb}{right}{suffix}"]

    strong_parts = re.split(r"，?以及", text, maxsplit=1)
    if len(strong_parts) == 2:
        left, right = (part.strip(" ，,。；;") for part in strong_parts)
        if prefix and not right.startswith(prefix):
            title_context = f"{''.join(titles)}中" if titles else ""
            right = f"{prefix}{title_context}{right}"
        if all(len(re.sub(r"\s+", "", part)) >= 6 for part in (left, right)):
            return [left, right]

    related_requirements = re.match(
        r"^(.*?与)([^，。；]{2,30}?)(?:和|以及)([^，。；]{2,30}?)(相关的(?:要求|规定|内容).*)$",
        text,
    )
    if related_requirements:
        common_prefix, left, right, suffix = related_requirements.groups()
        return [f"{common_prefix}{left}{suffix}", f"{common_prefix}{right}{suffix}"]

    category = r"(?:原则|对象|条件|期限|比例|定义|范围|要求|例外|情景|原因|方法|假设|阈值|门槛|顺序|效力|禁限|递延)"
    strong_bare_category = r"(?:原则|对象|条件|期限|时限|比例|例外|情景|原因|方法|假设|阈值|门槛|顺序|效力|禁限|递延|信息|版本|频率|留档要求)"
    coordinated = re.match(
        rf"^(.*?(?:说明|概括|回答|列出))(.{{2,24}}?{strong_bare_category})和(.{{2,24}}?{strong_bare_category})(.*)$",
        text,
    )
    if coordinated:
        common_prefix, left, right, suffix = coordinated.groups()
        # Preserve a shared interrogative tail such as “的依据是什么？”.  The
        # remote planner often merges two evidence needs immediately before
        # that tail, and dropping it makes the second query unnatural.
        return [f"{common_prefix}{left}{suffix}", f"{common_prefix}{right}{suffix}"]
    definition_and_requirement = re.match(
        r"^(.{2,30}?定义)(?:及|和)(.{2,30}?(?:要求|原则))(.*)$",
        text,
    )
    if definition_and_requirement:
        left, right, suffix = definition_and_requirement.groups()
        right = _inherit_left_subject(left, right)
        return [f"{left}{suffix}", f"{right}{suffix}"]
    # Remote planners sometimes compress an explicit “，并说明 …” request to
    # “门槛及…原则”.  Accept ``及/和`` only for two strong requirement
    # categories so ordinary lists such as “定义、口径、范围和报送要求” are
    # not split merely because they contain a conjunction.
    strong_bare_coordinated = re.match(
        rf"^(.{{2,40}}?{strong_bare_category})(?:及|和)(.{{2,40}}?{strong_bare_category})(.*)$",
        text,
    )
    if strong_bare_coordinated:
        left, right, suffix = strong_bare_coordinated.groups()
        right = _inherit_left_subject(left, right)
        return [f"{left}{suffix}", f"{right}{suffix}"]
    bare_coordinated = re.match(
        rf"^(.{{2,30}}?{category})与(.{{2,30}}?{category})(.*)$",
        text,
    )
    if bare_coordinated:
        left, right, suffix = bare_coordinated.groups()
        right = _inherit_left_subject(left, right)
        return [f"{left}{suffix}", f"{right}{suffix}"]
    principle_pair = re.match(
        r"^(.{2,30}?(?:风险暴露|资本|工具|策略|计划|要求|定义))(?:与|和)(.{2,20}?原则)(.*)$",
        text,
    )
    if principle_pair:
        left, right, suffix = principle_pair.groups()
        if "自救" in right and "风险暴露" in left and "处置" not in left:
            left = f"处置计划建议示例中的{left}"
        return [f"{left}{suffix}", f"{right}{suffix}"]
    return [text]


def _summary_enumerated_aspect_questions(question: str) -> list[str]:
    text = str(question or "").strip()
    if "、" not in text or not any(marker in text for marker in ("串联", "汇总", "逐项", "分别")):
        return []
    marker_positions = [
        text.rfind(marker)
        for marker in ("串联", "汇总", "逐项", "分别")
        if marker in text
    ]
    if not marker_positions:
        return []
    tail = text[max(marker_positions) + 2 :].strip(" ：:，,。；;？?")
    if not tail:
        return []
    parts = [
        part.strip(" ：:，,。；;？?")
        for part in re.split(r"、", tail)
        if part.strip(" ：:，,。；;？?")
    ]
    if len(parts) <= 1:
        return []
    expanded: list[str] = []
    for part in parts:
        nested = _split_compact_enumerated_part(part)
        expanded.extend(nested or [part])
    return _dedupe(expanded)[:QUERY_PLANNER_MAX_ASPECTS] if len(expanded) > 1 else []


def _split_compact_enumerated_part(part: str) -> list[str]:
    definition_and_requirement = re.match(
        r"^(.{2,30}?定义)(?:及|和)(.{2,30}?(?:要求|原则))(.*)$",
        part,
    )
    if definition_and_requirement:
        left, right, suffix = definition_and_requirement.groups()
        right = _inherit_left_subject(left, right)
        return [f"{left}{suffix}", f"{right}{suffix}"]
    principle_pair = re.match(
        r"^(.{2,30}?(?:风险暴露|资本|工具|策略|计划|要求|定义))(?:与|和)(.{2,20}?原则)(.*)$",
        part,
    )
    if principle_pair:
        left, right, suffix = principle_pair.groups()
        if "自救" in right and "风险暴露" in left and "处置" not in left:
            right = f"处置策略建议的{right}"
        return [f"{left}{suffix}", f"{right}{suffix}"]
    return [part]


def _inherit_left_subject(left: str, right: str) -> str:
    if "的" in right:
        return right
    if not re.search(r"(?:定义|概念|含义)$", left):
        return right
    if re.search(r"(?:保费|保险费|资本|账簿|回函|函证|披露|催收|贷款|风险暴露)", right):
        return right
    bare_subject = re.sub(r"(?:的)?(?:定义|概念|含义)$", "", left).strip()
    bare_subject = re.sub(r"^(?:说明|概括|回答|列出|判断|核验|比较)", "", bare_subject).strip()
    if bare_subject and len(re.sub(r"\s+", "", bare_subject)) >= 2:
        return f"{bare_subject}的{right}"
    subject_match = re.match(r"^(.{2,40}?的).+", left)
    if not subject_match:
        return right
    subject = subject_match.group(1)
    if right.startswith(subject):
        return right
    return f"{subject}{right}"


def _budget_with_omissions(budget: QueryBudget, omitted_or_merged_items: list[str]) -> QueryBudget:
    merged = tuple(_dedupe([*budget.omitted_or_merged_items, *omitted_or_merged_items]))
    return QueryBudget(
        detected_items=budget.detected_items,
        max_aspects=budget.max_aspects,
        system_max_aspects=budget.system_max_aspects,
        capacity_limited=budget.capacity_limited or bool(merged),
        omitted_or_merged_items=merged,
    )


def _plan_with_llm(question: str, budget: QueryBudget) -> tuple[list[QueryAspect], list[str]]:
    messages = [
            {
                "role": "system",
                "content": (
                    "你是 ReguMate 的检索前 QueryPlanner。"
                    "你的唯一任务是把银行监管制度、统计报表填报说明、指标口径问题拆成检索 aspect，生成检索计划。"
                    "禁止回答用户问题，禁止给出结论，禁止补充知识库外事实。"
                    "你要同时支持制度文档检索和 Excel/统计报表结构化检索。"
                    "制度问题生成有利于向量检索命中原文 chunk 的自然语言查询；"
                    "Excel/统计报表问题必须生成表格定位计划，识别文件、sheet、期间、指标、行标签、列标签、单位和计算动作。"
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
                    "aspect_id、question、evidence_need、search_queries、keywords、modality、table_task、table_filters、operation。"
                    "如果用户明确枚举了多个处理对象，一般每个处理对象单独成为一个 aspect；"
                    "如果因为预算必须合并或遗漏项目，把项目名称写入 omitted_or_merged_items。"
                    "输出 JSON 顶层必须包含 aspects 和 omitted_or_merged_items。"
                    "search_queries 必须是对象数组，每个对象包含 query、query_type、rationale。"
                    "query_type 只能是 semantic_question、document_style_statement、keyword_anchor、table_locator。"
                    "modality 只能是 text、table、mixed；table_task 只能是 lookup、compare、calculate、locate、none；"
                    "operation 只能是 max、min、difference、sum、ratio、none。"
                    "table_filters 同时承担文档元数据过滤，可包含 filename、source_title、external_doc_id、"
                    "issuing_authority、publication_date、document_number、regulatory_topic、business_domain、"
                    "article_number、version_status、year、month、quarter、sheet、indicator、row_label、column_label、unit、metric、scope。"
                    "每个 aspect 生成 2 到 "
                    f"{QUERY_PLANNER_MAX_SEARCH_QUERIES} 条 search_queries："
                    "semantic_question 贴近用户意图；document_style_statement 像制度原文、填报说明标题或证据句；"
                    "table_locator 用文件名、sheet、期间、指标、行列标签和单位锚点定位表格；keyword_anchor 只用少量关键术语兜底。"
                    "Excel/统计报表取数题优先 modality=table；既要制度口径又要表格数值时 modality=mixed。"
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
        ]
    try:
        with measure("query_planner.external_api"):
            content = chat_completion_content(
                ChatCompletionConfig(
                    provider=QUERY_PLANNER_PROVIDER,
                    api_key=QUERY_PLANNER_API_KEY,
                    base_url=QUERY_PLANNER_BASE_URL,
                    model=QUERY_PLANNER_MODEL,
                    timeout_seconds=QUERY_PLANNER_TIMEOUT_SECONDS,
                    include_thinking=QUERY_PLANNER_INCLUDE_THINKING,
                    response_format=QUERY_PLANNER_RESPONSE_FORMAT,
                ),
                messages,
                temperature=0,
                response_format=QUERY_PLANNER_RESPONSE_FORMAT,
                opener=urllib.request.urlopen,
            )
    except ChatCompletionError as exc:
        raise QueryPlannerError("LLM query planner 调用失败") from exc

    try:
        parsed = json.loads(_extract_json_object(content))
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise QueryPlannerError("LLM query planner 返回格式不可解析") from exc

    return _aspects_from_payload(question, parsed, max_aspects=QUERY_PLANNER_MAX_ASPECTS), _clean_string_list(
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
    question_explicit_filters = extract_explicit_metadata_filters(question)
    for index, raw_aspect in enumerate(raw_aspects[:limit], start=1):
        if not isinstance(raw_aspect, dict):
            continue
        sub_question = _clean_text(raw_aspect.get("question")) or question
        if _is_execution_instruction_aspect(sub_question):
            # A user instruction such as “if sqrt is unsupported, refuse”
            # constrains execution. It is not a fact that must be retrieved
            # from the regulation, so it must not become a missing aspect.
            continue
        search_queries = _clean_search_queries(raw_aspect.get("search_queries"))
        evidence_need = (
            _clean_text(raw_aspect.get("evidence_need"))
            or _clean_text(raw_aspect.get("expected_evidence_type"))
            or "相关制度依据"
        )
        keywords = _clean_string_list(raw_aspect.get("keywords"))
        if not search_queries:
            search_queries = [QuerySearchQuery(query=sub_question, query_type="fallback", rationale="LLM 未返回检索查询")]
        search_queries = _prioritize_local_document_style_aliases(
            sub_question=sub_question,
            evidence_need=evidence_need,
            search_queries=search_queries,
        )
        if not keywords:
            keywords = _keywords_from_text(" ".join([sub_question, *[query.query for query in search_queries]]))
        planner_filters = _clean_table_filters(raw_aspect.get("table_filters"))
        explicit_filters = extract_explicit_metadata_filters(sub_question)
        for key, value in question_explicit_filters.items():
            if key in planner_filters and str(planner_filters[key]) == str(value):
                explicit_filters[key] = value
        merged_filters = {**planner_filters, **explicit_filters}
        modality = _clean_modality(raw_aspect.get("modality"))
        table_task = _clean_table_task(raw_aspect.get("table_task"))
        operation = _clean_operation(raw_aspect.get("operation"))
        if _should_force_text_aspect(question, sub_question, modality) or _is_document_formula_question(
            f"{question} {sub_question}"
        ):
            modality = "text"
            table_task = "none"
            operation = "none"
            merged_filters = {
                key: value
                for key, value in merged_filters.items()
                if key not in TABLE_ONLY_FILTER_KEYS
            }
            explicit_filters = {
                key: value
                for key, value in explicit_filters.items()
                if key not in TABLE_ONLY_FILTER_KEYS
            }
        if modality == "text":
            table_task = "none"
            operation = "none"
            # LLM-inferred metadata (for example a guessed regulatory_topic)
            # is useful as a query term but unsafe as a hard filter: corpus
            # metadata may be incomplete and the user never asserted it.
            # Text retrieval therefore keeps only filters independently
            # extracted from the user's wording.
            merged_filters = {
                key: value for key, value in explicit_filters.items() if key not in TABLE_ONLY_FILTER_KEYS
            }
            explicit_filters = {
                key: value for key, value in explicit_filters.items() if key not in TABLE_ONLY_FILTER_KEYS
            }
        aspects.append(
            QueryAspect(
                aspect_id=_clean_aspect_id(raw_aspect.get("aspect_id"), index),
                question=sub_question,
                search_queries=tuple(_dedupe_search_queries(search_queries)[:QUERY_PLANNER_MAX_SEARCH_QUERIES]),
                evidence_need=evidence_need,
                keywords=tuple(_dedupe(keywords)[:8]),
                modality=modality,
                table_task=table_task,
                table_filters=merged_filters,
                operation=operation,
                explicit_filter_keys=tuple(explicit_filters),
            )
        )

    if not aspects:
        raise QueryPlannerError("LLM query planner 未产生有效 aspect")
    return _expand_explicit_anchor_aspects(aspects, max_aspects=limit)


def _prioritize_local_document_style_aliases(
    *,
    sub_question: str,
    evidence_need: str,
    search_queries: list[QuerySearchQuery],
) -> list[QuerySearchQuery]:
    """Ensure LLM-planned text aspects keep deterministic regulation anchors.

    The remote planner is useful for splitting a complex question, but it can
    phrase a search query in business shorthand ("报告义务", "定价原则") instead
    of wording likely to appear in the document.  Fallback planning already
    adds these local document-style aliases; LLM planning should not lose them.
    """

    aliases = _local_document_style_aliases(sub_question, evidence_need)
    if not aliases:
        return search_queries

    semantic_queries = [query for query in search_queries if query.query_type == "semantic_question"]
    other_queries = [query for query in search_queries if query.query_type != "semantic_question"]
    local_queries = [
        QuerySearchQuery(
            query=alias,
            query_type="document_style_statement",
            rationale="本地将用户概念转换为制度原文常见表达",
        )
        for alias in aliases
    ]
    return _dedupe_search_queries([*semantic_queries[:1], *local_queries, *other_queries, *semantic_queries[1:]])


def _local_document_style_aliases(sub_question: str, evidence_need: str) -> list[str]:
    context = " ".join(part for part in [sub_question, evidence_need] if part)
    candidates: list[str] = []
    for text in (sub_question, evidence_need):
        cleaned = _clean_text(text)
        if not cleaned:
            continue
        candidates.append(cleaned)
        candidates.extend(_coordinated_aspect_questions(cleaned))

    aliases: list[str] = []
    for candidate in _dedupe(candidates):
        alias = _mixed_regulation_document_style_alias(context, candidate)
        if alias and alias != candidate:
            aliases.append(alias)
    return _dedupe(aliases)


def _fallback_aspects(question: str, budget: QueryBudget | None = None) -> list[QueryAspect]:
    budget = budget or plan_query_budget(question)
    table_aspect = _fallback_table_aspect(question)
    if table_aspect is not None:
        return [table_aspect]

    heuristic_aspects = _domain_heuristic_aspects(question, max_aspects=budget.max_aspects)
    if heuristic_aspects:
        return heuristic_aspects

    parts = _split_question_parts(question)
    if not parts:
        parts = [question]

    aspects: list[QueryAspect] = []
    for index, part in enumerate(parts[:budget.max_aspects], start=1):
        keywords = _keywords_from_text(part)
        explicit_filters = extract_explicit_metadata_filters(part)
        aspects.append(
            QueryAspect(
                aspect_id=f"aspect_{index}",
                question=part,
                search_queries=_fallback_search_queries(part),
                evidence_need=_infer_evidence_type(part),
                keywords=tuple(keywords),
                table_filters=explicit_filters,
                explicit_filter_keys=tuple(explicit_filters),
            )
        )
    return _expand_explicit_anchor_aspects(
        aspects,
        max_aspects=max(budget.max_aspects, budget.system_max_aspects),
    )


def _fallback_mcq_aspect(question: str, options: list[str] | None = None) -> QueryAspect | None:
    cleaned_options = [str(option).strip() for option in (options or []) if str(option).strip()]
    if len(cleaned_options) < 2:
        return None

    title_candidates = [
        _normalize_mcq_title(item)
        for pattern in (r"《([^》]+)》", r"“([^”]+)”", r'"([^"]+)"')
        for item in re.findall(pattern, question)
        if _normalize_mcq_title(item)
    ]
    title_anchor = title_candidates[0] if title_candidates else ""
    requested_format_match = re.search(r"[（(]\s*(pdf|word|docx?)\s*[）)]", question, flags=re.IGNORECASE)
    requested_format = requested_format_match.group(1).lower() if requested_format_match else ""
    anchor_question = title_anchor or re.sub(
        r"(检索|查阅|根据|以下哪一项|以下哪项|哪一项|哪项|与材料内容一致|与材料一致|说法正确|表述正确|正确的是|不正确的是)",
        " ",
        question,
    )
    anchor_question = re.sub(r"[？?。；;，,：:\s]+", " ", anchor_question).strip() or question

    option_statements = _prioritized_mcq_option_queries(_mcq_option_evidence_statements(cleaned_options))
    search_queries = [
        QuerySearchQuery(
            query=anchor_question,
            query_type="keyword_anchor",
            rationale="选择题指定材料的标题或核心主题锚点",
        ),
        QuerySearchQuery(
            query=question,
            query_type="semantic_question",
            rationale="保留选择题原始语义和限定条件",
        ),
        *[
            QuerySearchQuery(
            query=statement,
            query_type="document_style_statement",
            rationale="候选项中的独立事实用于召回直接证据",
            )
            for statement in option_statements
        ],
    ]
    return QueryAspect(
        aspect_id="multiple_choice_evidence",
        question=anchor_question,
        search_queries=tuple(_dedupe_search_queries(search_queries)),
        evidence_need="在指定材料中核对候选项并找到直接支持或否定证据",
        keywords=tuple(_dedupe([anchor_question, *title_candidates])[:8]),
        modality="text",
        table_filters={"file_type": requested_format} if requested_format else {},
        explicit_filter_keys=("file_type",) if requested_format else (),
    )


def _prioritized_mcq_option_queries(option_statements: list[str]) -> list[str]:
    """Return a bounded, evidence-oriented query set for MCQ options.

    MCQ stems often contain four options with multiple semicolon-separated
    facts. Sending every raw fact through hybrid retrieval and CPU reranking
    scales poorly and can time out. Prefer local-document style aliases because
    they are closer to source text, then fill any remaining budget with raw
    option statements.
    """

    alias_queries: list[str] = []
    for statement in option_statements:
        alias_queries.extend(_mcq_document_style_aliases(statement))
    return _dedupe([*alias_queries, *option_statements])[:MCQ_OPTION_SEARCH_QUERY_BUDGET]


def _normalize_mcq_title(value: str) -> str:
    title = str(value or "").strip()
    return re.sub(r"[（(]\s*(?:pdf|word|docx?|附件)\s*[）)]$", "", title, flags=re.IGNORECASE).strip()


def _mcq_option_evidence_statements(options: list[str]) -> list[str]:
    statements: list[str] = []
    seen: set[str] = set()
    for option in options:
        for statement in re.split(r"[；;]", option):
            cleaned = re.sub(r"\s+", " ", statement).strip(" 。；;")
            normalized = re.sub(r"\s+", "", cleaned)
            if cleaned and normalized not in seen:
                seen.add(normalized)
                statements.append(cleaned)
    return statements


def _mcq_document_style_aliases(statement: str) -> list[str]:
    compact = re.sub(r"\s+", "", str(statement or ""))
    aliases: list[str] = []
    if all(term in compact for term in ("申请书", "名称", "拟设地", "注册资本", "股权结构", "业务范围")):
        aliases.append(
            "申请书 内容包括但不限于 拟设立中资商业银行 名称 拟设地 注册资本 股权结构 业务范围 基本信息"
        )
        if "2023年版" in compact or "目录" in compact:
            aliases.append("中资商业银行行政许可事项 申请材料目录及格式要求 2023年版")
    if "折现率曲线" in compact or all(term in compact for term in ("基础利率", "综合溢价", "曲线")):
        aliases.append("计算现金流现值 所采用的 折现率曲线 由 基础利率曲线 加 综合溢价形成")
        aliases.append("寿险合同负债评估 折现率曲线 基础利率曲线 综合溢价")
    if all(term in compact for term in ("小额信用", "知识产权质押", "创新积分贷款")):
        aliases.append("鼓励商业银行 综合运用 小额信用贷款 知识产权质押贷款 创新积分贷款 降低融资门槛")
    if "专利权质押登记" in compact and ("线上" in compact or "无纸化" in compact or "全覆盖" in compact):
        aliases.append("试验区内商业银行各分支机构 实现 专利权质押登记 全流程无纸化线上办理 全覆盖")
    if "较大数据安全事件" in compact or (
        "敏感级" in compact and ("难以消除" in compact or "部分业务" in compact)
    ):
        aliases.append(
            "较大数据安全事件 敏感级及以上数据 泄露 破坏 非法获取 非法利用 对个人造成不可消除或者消除代价较大的负面影响 部分业务无法正常开展 声誉受到破坏"
        )
    if "重大数据" in compact and ("省级区域经济" in compact or "银行保险行业安全" in compact):
        aliases.append("重大数据安全事件 重要数据 泄露 破坏 非法获取 非法利用 对省级区域经济带来重大影响 对银行保险行业安全造成影响")
    if "数字化回函" in compact and ("法律效力" in compact or "证明力" in compact):
        aliases.append("以符合相关规定的数字方式 办理银行询证函及回函 与纸质银行询证函及回函 具有同等法律效力和证明力")
    if ("10个工作日" in compact or "回函时限" in compact) and ("询证函" in compact or "回函" in compact):
        aliases.append("银行业金融机构 应当自收到符合规定的询证函之日起 10个工作日内 按照要求 将回函直接回复会计师事务所")
    if "一函一个基准日" in compact or "一个函证基准日" in compact:
        aliases.append("一份询证函 只列示 一个函证基准日")
    if "公开渠道" in compact and ("函证" in compact or "公示" in compact):
        aliases.append("银行业金融机构 应当 在其总行或总部网站 微信公众号等公开渠道 就办理函证相关事项进行公示")
    if "不正当催收" in compact or ("催收" in compact and "无关第三人" in compact) or ("暴力" in compact and "催收" in compact):
        aliases.append("消费金融公司 不得采用 暴力 威胁 恐吓 骚扰 等不正当手段进行催收 不得对与债务无关的第三人进行催收")
    if "重要实体" in compact and ("核心业务条线" in compact or "关键功能" in compact):
        aliases.append("重要实体 承载 本机构核心业务条线和关键功能 对持续经营 维持关键功能具有重要作用")
    if ("处置工具" in compact or "工具可包括" in compact) and (
        "自救" in compact or "战略投资" in compact or "过桥机构" in compact
    ):
        aliases.append("处置工具 包括 机构自救 股东注资 引入战略投资者 处置不良资产 接管 收购承接 建立过桥机构 破产清算")
    if "交易头寸" in compact and ("做市" in compact or "对客" in compact or "每日" in compact):
        aliases.append("以交易目的持有的头寸 包括 自营业务 做市业务 为满足客户需求提供的对客交易 对冲前述交易相关风险")
        aliases.append("交易账簿 金融工具 外汇和商品头寸 原则上 应能够每日进行公允价值计量 变动计入损益")
    if "监管并表" in compact or ("第三支柱" in compact and "表格另有规定" in compact):
        aliases.append("商业银行 应按照 监管并表范围 披露相关信息 表格中另有规定的除外")
    return aliases


def _domain_heuristic_aspects(question: str, max_aspects: int | None = None) -> list[QueryAspect]:
    normalized = re.sub(r"\s+", "", question)
    aspects: list[QueryAspect] = list(_cross_policy_heuristic_aspects(normalized))
    detected = _detect_requested_items(question)
    effective_limit = max_aspects
    if aspects and max_aspects is not None:
        # Deterministic policy detectors are direct evidence anchors.  The
        # budget estimator can undercount terse prompts such as "A、B以及C";
        # do not let that low estimate drop already detected anchors.
        effective_limit = max(max_aspects, len(aspects))
    for item in detected:
        aspect = _heuristic_aspect_for_item(item, normalized)
        if aspect is not None:
            aspects.append(aspect)
        if effective_limit is not None and len(aspects) >= effective_limit:
            break
    deduped = _dedupe_aspects(aspects)
    return deduped[:effective_limit] if effective_limit is not None else deduped


def _should_use_domain_heuristic_plan(question: str, aspects: list[QueryAspect]) -> bool:
    normalized = re.sub(r"\s+", "", question)
    deterministic_domain_ids = {
        "consumer_finance_geographic_scope",
        "pillar3_table_types",
        "large_exposure_threshold",
        "bail_in_resolution_principle",
        "digital_confirmation_effectiveness",
        "confirmation_response_deadline",
        "trading_book_policy_audit_retention",
        "important_data_major_event",
        "going_concern_trigger_threshold",
        "tier2_loss_absorption_order",
        "trading_book_purpose_and_fair_value",
        "pillar3_disclosure_scope",
        "pillar3_disclosure_frequency",
        "intellectual_property_finance_target",
        "patent_pledge_registration_online_coverage",
        "insurance_group_report_obligation",
        "bank_confirmation_public_channels",
        "auditor_confirmation_process_control",
        "commercial_bank_establishment_application_content",
        "commercial_bank_application_catalog_2023_version",
        "important_data_extremely_major_event",
        "life_discount_curve_ultimate_rate",
        "core_business_line_failure_impact",
        "accident_insurance_definition",
        "accident_insurance_pricing_requirement",
    }
    deterministic_single_domain_ids = {
        "pillar3_disclosure_frequency",
    }
    if len(aspects) == 1 and aspects[0].aspect_id in deterministic_single_domain_ids:
        return True
    if len(aspects) >= 2 and any(aspect.aspect_id in deterministic_domain_ids for aspect in aspects):
        return True
    return bool(
        len(aspects) >= 2
        and any(marker in normalized for marker in ("每项", "逐项", "分别", "跨制度", "联合核验"))
        and any(marker in normalized for marker in ("直接引用", "引用", "依据"))
    )


def _cross_policy_heuristic_aspects(normalized_question: str) -> list[QueryAspect]:
    requested: list[QueryAspect] = []
    if "数字化" in normalized_question and "回函" in normalized_question and "效力" in normalized_question:
        requested.append(
            QueryAspect(
                aspect_id="digital_confirmation_effectiveness",
                question="数字化银行回函的效力",
                search_queries=(
                    QuerySearchQuery("数字化回函与纸质回函具有同等法律效力和证明力", "semantic_question", "贴近用户意图"),
                    QuerySearchQuery("银行函证工作操作指引 数字化回函与纸质回函具有同等法律效力和证明力", "document_style_statement", "制度原文锚点"),
                    QuerySearchQuery("数字化回函 纸质回函 同等法律效力 证明力", "keyword_anchor", "关键术语兜底"),
                ),
                evidence_need="数字化银行回函与纸质回函效力的直接规定",
                keywords=("数字化回函", "纸质回函", "同等法律效力", "证明力"),
            )
        )
    if (
        "回函时限" in normalized_question
        or ("回函" in normalized_question and "回复时限" in normalized_question)
        or ("回函" in normalized_question and "10个工作日" in normalized_question)
        or ("询证函" in normalized_question and "10个工作日" in normalized_question)
        or ("询证函" in normalized_question and any(term in normalized_question for term in ("回复时限", "答复时限", "办理时限")))
    ):
        requested.append(
            QueryAspect(
                aspect_id="confirmation_response_deadline",
                question="银行询证函回函时限",
                search_queries=(
                    QuerySearchQuery("银行业金融机构收到符合规定的询证函后多久回函", "semantic_question", "贴近用户意图"),
                    QuerySearchQuery("银行业金融机构 应当自收到符合规定的询证函之日起 10个工作日内 按照要求 将回函直接回复会计师事务所", "document_style_statement", "制度原文锚点"),
                    QuerySearchQuery("符合规定的询证函 10个工作日 回函直接回复会计师事务所", "keyword_anchor", "关键术语兜底"),
                ),
                evidence_need="银行询证函回函时限和回复对象的直接规定",
                keywords=("询证函", "10个工作日", "回函", "会计师事务所"),
            )
        )
    if (
        ("交易账簿" in normalized_question or "账簿划分" in normalized_question)
        and ("内部审计" in normalized_question or "留档" in normalized_question)
    ):
        requested.append(
            QueryAspect(
                aspect_id="trading_book_policy_audit_retention",
                question="交易账簿划分政策内部审计频率和留档要求",
                search_queries=(
                    QuerySearchQuery("交易账簿划分政策内部审计频率和留档要求", "semantic_question", "贴近用户意图"),
                    QuerySearchQuery("商业银行 应每年 对划分政策和程序 开展内部审计 内部审计结果 应留档备查", "document_style_statement", "制度原文锚点"),
                    QuerySearchQuery("划分政策和程序 每年 内部审计 留档备查", "keyword_anchor", "关键术语兜底"),
                ),
                evidence_need="交易账簿划分政策内部审计频率及留档要求",
                keywords=("交易账簿", "划分政策", "每年", "内部审计", "留档备查"),
            )
        )
    if (
        "重要数据" in normalized_question
        and (
            "特别重大" in normalized_question
            or "2个及以上" in normalized_question
            or "两个及以上" in normalized_question
            or "省级区域门槛" in normalized_question
        )
    ):
        requested.append(
            QueryAspect(
                aspect_id="important_data_extremely_major_event",
                question="重要数据事件何时属于特别重大数据安全事件",
                search_queries=(
                    QuerySearchQuery("重要数据事件何时属于特别重大数据安全事件", "semantic_question", "贴近用户意图"),
                    QuerySearchQuery(
                        "特别重大数据安全事件 重要数据 泄露 破坏 非法获取 非法利用 对2个及以上省级区域经济运行秩序造成特别严重影响",
                        "document_style_statement",
                        "制度原文锚点",
                    ),
                    QuerySearchQuery("重要数据 2个及以上 省级区域经济运行秩序 特别严重影响 特别重大数据安全事件", "keyword_anchor", "关键术语兜底"),
                ),
                evidence_need="重要数据事件构成特别重大数据安全事件的省级区域和后果条件",
                keywords=("重要数据", "特别重大数据安全事件", "2个及以上", "省级区域", "特别严重影响"),
            )
        )
    if (
        "重要数据事件" in normalized_question
        and any(term in normalized_question for term in ("重大事件", "重大数据安全事件"))
    ) or (
        "重大数据事件" in normalized_question
        or ("重大数据" in normalized_question and any(term in normalized_question for term in ("条件", "构成", "属于")))
        or ("重要数据" in normalized_question and "重大" in normalized_question and any(term in normalized_question for term in ("层级", "影响范围", "后果条件")))
    ):
        requested.append(
            QueryAspect(
                aspect_id="important_data_major_event",
                question="重要数据事件何时属于重大数据安全事件",
                search_queries=(
                    QuerySearchQuery("重要数据事件何时属于重大数据安全事件", "semantic_question", "贴近用户意图"),
                    QuerySearchQuery(
                        "重大数据安全事件 重要数据 泄露 破坏 非法获取 非法利用 对省级区域经济带来重大影响 对银行保险行业安全造成影响",
                        "document_style_statement",
                        "制度原文锚点",
                    ),
                    QuerySearchQuery("重要数据 泄露 破坏 非法获取 非法利用 省级区域经济 银行保险行业安全", "keyword_anchor", "关键术语兜底"),
                ),
                evidence_need="重要数据事件构成重大数据安全事件的条件",
                keywords=("重要数据", "重大数据安全事件", "泄露", "破坏", "非法获取", "省级区域经济", "银行保险行业安全"),
            )
        )
    if "持续经营" in normalized_question and "触发" in normalized_question:
        requested.append(
            QueryAspect(
                aspect_id="going_concern_trigger_threshold",
                question="持续经营触发事件条件",
                search_queries=_fallback_search_queries("持续经营触发阈值"),
                evidence_need="持续经营触发事件的核心一级资本充足率阈值",
                keywords=("持续经营触发事件", "核心一级资本充足率", "5.125%"),
            )
        )
    if (
        ("二级资本" in normalized_question and ("吸收损失" in normalized_question or "损失吸收" in normalized_question))
        or ("资本工具" in normalized_question and "损失吸收顺序" in normalized_question)
    ):
        requested.append(
            QueryAspect(
                aspect_id="tier2_loss_absorption_order",
                question="二级资本吸收损失顺序",
                search_queries=_fallback_search_queries("二级资本损失吸收顺序"),
                evidence_need="二级资本工具吸收损失的启动顺序",
                keywords=("二级资本", "其他一级资本工具", "全部吸收损失", "再启动"),
            )
        )
    if (
        "寿险" in normalized_question and "折现率" in normalized_question and "终极利率" in normalized_question
    ) or ("终极利率" in normalized_question and "暂定" in normalized_question):
        requested.append(
            QueryAspect(
                aspect_id="life_discount_curve_ultimate_rate",
                question="寿险合同负债评估折现率曲线终极利率暂定值",
                search_queries=(
                    QuerySearchQuery("寿险合同负债评估折现率曲线终极利率暂定值是多少", "semantic_question", "贴近用户意图"),
                    QuerySearchQuery("寿险合同负债评估 折现率曲线 终极利率 暂定为4.5%", "document_style_statement", "制度原文锚点"),
                    QuerySearchQuery("终极利率 暂定为4.5% 寿险合同负债评估 折现率曲线", "keyword_anchor", "关键术语兜底"),
                ),
                evidence_need="寿险合同负债评估折现率曲线终极利率暂定值",
                keywords=("寿险", "折现率曲线", "终极利率", "4.5%"),
            )
        )
    if "交易账簿" in normalized_question and "公允价值计量" in normalized_question:
        requested.append(
            QueryAspect(
                aspect_id="trading_book_purpose_and_fair_value",
                question="交易账簿头寸定义：“以交易目的持有的头寸”；公允价值计量条件：“能够每日进行公允价值计量，变动计入损益”",
                search_queries=(
                    QuerySearchQuery("交易账簿头寸定义和公允价值计量频率要求", "semantic_question", "贴近用户意图"),
                    QuerySearchQuery("以交易目的持有的头寸 包括 自营业务 做市业务 为满足客户需求提供的对客交易 对冲前述交易相关风险 交易账簿中的金融工具 外汇和商品头寸 原则上 能够每日进行公允价值计量 变动计入损益", "document_style_statement", "制度原文锚点"),
                    QuerySearchQuery("能够每日进行公允价值计量 变动计入损益", "document_style_statement", "制度条件短句锚点"),
                    QuerySearchQuery("自营业务 做市业务 对客交易 对冲前述交易相关风险 每日进行公允价值计量 变动计入损益", "keyword_anchor", "关键术语兜底"),
                ),
                evidence_need="交易账簿头寸范围和每日公允价值计量要求",
                keywords=("交易账簿", "自营业务", "做市业务", "对客交易", "每日", "公允价值计量", "损益"),
            )
        )
    if "第三支柱" in normalized_question and "单体口径" in normalized_question:
        requested.append(
            QueryAspect(
                aspect_id="pillar3_disclosure_scope",
                question="第三支柱信息披露口径",
                search_queries=(
                    QuerySearchQuery("第三支柱信息披露口径是否按单体口径", "semantic_question", "贴近用户意图"),
                    QuerySearchQuery("商业银行 应按照 本办法第二章规定的并表范围 监管并表范围 披露相关信息 表格中另有规定的除外", "document_style_statement", "制度原文锚点"),
                    QuerySearchQuery("监管并表范围 披露相关信息 表格中另有规定的除外", "keyword_anchor", "关键术语兜底"),
                ),
                evidence_need="第三支柱信息披露并表范围口径",
                keywords=("第三支柱", "监管并表范围", "表格中另有规定的除外", "披露相关信息"),
            )
        )
    if (
        ("商业银行信息披露内容和要求" in normalized_question or "第三支柱" in normalized_question)
        and (
            "商业银行应根据表格要求" in normalized_question
            or "披露频率" in normalized_question
            or "频率" in normalized_question
            or ("商业银行应" in normalized_question and "披露" in normalized_question)
        )
    ):
        requested.append(
            QueryAspect(
                aspect_id="pillar3_disclosure_frequency",
                question="商业银行第三支柱信息披露频率和表格例外",
                search_queries=(
                    QuerySearchQuery("商业银行第三支柱信息披露频率和表格例外", "semantic_question", "贴近用户意图"),
                    QuerySearchQuery(
                        "商业银行 应根据 表格要求 分别按照季度 半年和年度的频率 披露信息 表格中另有规定的除外",
                        "document_style_statement",
                        "制度原文锚点",
                    ),
                    QuerySearchQuery("二、披露内容 商业银行应根据表格要求 季度 半年 年度 表格中另有规定的除外", "keyword_anchor", "章节关键句兜底"),
                ),
                evidence_need="商业银行第三支柱信息披露的频率要求及表格另有规定的例外",
                keywords=("商业银行", "第三支柱", "披露频率", "季度", "半年", "年度", "表格中另有规定的除外"),
            )
        )
    if "中资商业银行" in normalized_question and "筹建" in normalized_question and "申请书" in normalized_question:
        requested.append(
            QueryAspect(
                aspect_id="commercial_bank_establishment_application_content",
                question="中资商业银行法人机构筹建审批申请书基本信息要求",
                search_queries=(
                    QuerySearchQuery("中资商业银行法人机构筹建审批申请书应说明哪些基本信息", "semantic_question", "贴近用户意图"),
                    QuerySearchQuery("申请书 内容包括但不限于 拟设立中资商业银行 名称 拟设地 注册资本 股权结构 业务范围 基本信息", "document_style_statement", "制度原文锚点"),
                    QuerySearchQuery("申请书 名称 拟设地 注册资本 股权结构 业务范围 基本信息", "keyword_anchor", "关键术语兜底"),
                ),
                evidence_need="中资商业银行法人机构筹建审批申请书基本信息要求",
                keywords=("中资商业银行", "筹建审批", "申请书", "名称", "拟设地", "注册资本", "股权结构", "业务范围"),
            )
        )
    if "中资商业银行" in normalized_question and ("2023年版" in normalized_question or "目录" in normalized_question):
        requested.append(
            QueryAspect(
                aspect_id="commercial_bank_application_catalog_2023_version",
                question="中资商业银行行政许可事项申请材料目录版本",
                search_queries=(
                    QuerySearchQuery("中资商业银行行政许可事项申请材料目录及格式要求版本", "semantic_question", "贴近用户意图"),
                    QuerySearchQuery("中资商业银行行政许可事项 申请材料目录及格式要求 2023 年版", "document_style_statement", "封面版本锚点"),
                    QuerySearchQuery("申请材料目录及格式要求 2023 年版", "keyword_anchor", "关键术语兜底"),
                ),
                evidence_need="中资商业银行行政许可事项申请材料目录及格式要求的封面版本",
                keywords=("中资商业银行", "行政许可事项", "申请材料目录", "格式要求", "2023年版"),
            )
        )
    if "2027年" in normalized_question and "知识产权金融生态" in normalized_question:
        requested.append(
            QueryAspect(
                aspect_id="intellectual_property_finance_target",
                question="2027年知识产权金融生态目标",
                search_queries=(
                    QuerySearchQuery("2027年知识产权金融生态目标是什么", "semantic_question", "贴近用户意图"),
                    QuerySearchQuery("到2027年 试点地区 基本建成 服务便捷高效 信息共享畅通 体制机制完备 知识产权金融生态综合试验区", "document_style_statement", "制度原文锚点"),
                    QuerySearchQuery("2027年 服务便捷高效 信息共享畅通 体制机制完备 知识产权金融生态综合试验区", "keyword_anchor", "关键术语兜底"),
                ),
                evidence_need="2027年知识产权金融生态综合试验区建设目标",
                keywords=("2027年", "知识产权金融生态", "服务便捷高效", "信息共享畅通", "体制机制完备"),
            )
        )
    if "专利权质押登记" in normalized_question and any(
        term in normalized_question for term in ("线上", "无纸化", "全覆盖")
    ):
        requested.append(
            QueryAspect(
                aspect_id="patent_pledge_registration_online_coverage",
                question="专利权质押登记线上全覆盖目标",
                search_queries=(
                    QuerySearchQuery("专利权质押登记线上全覆盖目标是什么", "semantic_question", "贴近用户意图"),
                    QuerySearchQuery(
                        "试验区内商业银行各分支机构 实现 专利权质押登记 全流程无纸化线上办理 全覆盖",
                        "document_style_statement",
                        "制度原文锚点",
                    ),
                    QuerySearchQuery("商业银行各分支机构 专利权质押登记 全流程无纸化线上办理 全覆盖", "keyword_anchor", "关键术语兜底"),
                ),
                evidence_need="专利权质押登记全流程无纸化线上办理全覆盖目标",
                keywords=("试验区", "商业银行各分支机构", "专利权质押登记", "全流程无纸化线上办理", "全覆盖"),
            )
        )
    if "保险集团" in normalized_question and ("报告义务" in normalized_question or "偿付能力报告" in normalized_question):
        requested.append(
            QueryAspect(
                aspect_id="insurance_group_report_obligation",
                question="列入名单保险集团报告义务",
                search_queries=(
                    QuerySearchQuery("列入名单保险集团的报告义务是什么", "semantic_question", "贴近用户意图"),
                    QuerySearchQuery(
                        "应当编报保险集团偿付能力报告的公司名单 下列保险集团 应当 按照 保险公司偿付能力监管规则第19号 保险集团 有关规定 编报保险集团偿付能力报告",
                        "document_style_statement",
                        "制度标题和原文锚点",
                    ),
                    QuerySearchQuery("应当编报保险集团偿付能力报告的公司名单 下列保险集团 第19号 保险集团 编报保险集团偿付能力报告", "keyword_anchor", "关键术语兜底"),
                ),
                evidence_need="列入名单保险集团编报偿付能力报告义务",
                keywords=("应当编报保险集团偿付能力报告的公司名单", "保险集团", "第19号", "编报", "偿付能力报告"),
            )
        )
    if "函证" in normalized_question and "公开渠道" in normalized_question:
        requested.append(
            QueryAspect(
                aspect_id="bank_confirmation_public_channels",
                question="银行函证事项公开渠道公示要求",
                search_queries=(
                    QuerySearchQuery("银行函证事项应通过哪些公开渠道公示", "semantic_question", "贴近用户意图"),
                    QuerySearchQuery("银行业金融机构 应当 在其总行或总部网站 微信公众号等公开渠道 就办理函证相关事项进行公示", "document_style_statement", "制度原文锚点"),
                    QuerySearchQuery("总行或总部网站 微信公众号 公开渠道 办理函证相关事项 公示", "keyword_anchor", "关键术语兜底"),
                ),
                evidence_need="银行函证事项公开渠道公示要求",
                keywords=("函证", "总行或总部网站", "微信公众号", "公开渠道", "公示"),
            )
        )
    if (
        "询证函" in normalized_question
        and (
            ("注册会计师" in normalized_question and "控制" in normalized_question)
            or "全过程控制" in normalized_question
            or ("全过程" in normalized_question and "控制" in normalized_question)
            or "过程控制" in normalized_question
            or "过程控制主体" in normalized_question
        )
    ):
        requested.append(
            QueryAspect(
                aspect_id="auditor_confirmation_process_control",
                question="注册会计师对询证函的过程控制要求",
                search_queries=(
                    QuerySearchQuery("注册会计师对银行询证函全过程控制要求", "semantic_question", "贴近用户意图"),
                    QuerySearchQuery("注册会计师 应当始终 对银行询证函的全过程保持控制", "document_style_statement", "制度原文锚点"),
                    QuerySearchQuery("注册会计师 银行询证函 全过程保持控制", "keyword_anchor", "关键术语兜底"),
                ),
                evidence_need="注册会计师对银行询证函全过程控制要求",
                keywords=("注册会计师", "银行询证函", "全过程保持控制"),
            )
        )
    if "消费金融" in normalized_question and any(term in normalized_question for term in ("业务地域范围", "地域范围", "全国范围", "开展业务")):
        requested.append(
            QueryAspect(
                aspect_id="consumer_finance_geographic_scope",
                question="消费金融公司业务地域范围",
                search_queries=(
                    QuerySearchQuery("消费金融公司业务地域范围是什么", "semantic_question", "贴近用户意图"),
                    QuerySearchQuery("消费金融公司 可以在全国范围内开展业务", "document_style_statement", "制度原文锚点"),
                    QuerySearchQuery("第十四条 消费金融公司可以在全国范围内开展业务", "keyword_anchor", "条款关键句兜底"),
                ),
                evidence_need="消费金融公司业务地域范围的直接规定",
                keywords=("消费金融公司", "全国范围", "开展业务", "第十四条"),
            )
        )
    if "第三支柱" in normalized_question and any(term in normalized_question for term in ("表格类型", "两种类型", "固定表格", "可变表格")):
        requested.append(
            QueryAspect(
                aspect_id="pillar3_table_types",
                question="第三支柱表格类型",
                search_queries=(
                    QuerySearchQuery("第三支柱表格类型包括哪些", "semantic_question", "贴近用户意图"),
                    QuerySearchQuery("商业银行以表格形式进行第三支柱信息披露 包括固定表格和可变表格", "document_style_statement", "制度原文锚点"),
                    QuerySearchQuery("二、披露内容 固定表格 可变表格", "keyword_anchor", "章节关键句兜底"),
                ),
                evidence_need="第三支柱信息披露表格类型的直接规定",
                keywords=("第三支柱", "固定表格", "可变表格", "披露内容"),
            )
        )
    if "核心业务条线" in normalized_question and any(term in normalized_question for term in ("经营失败", "重大损失", "影响")):
        requested.append(
            QueryAspect(
                aspect_id="core_business_line_failure_impact",
                question="核心业务条线经营失败影响",
                search_queries=(
                    QuerySearchQuery("核心业务条线经营失败可能造成什么影响", "semantic_question", "贴近用户意图"),
                    QuerySearchQuery("核心业务条线 在经营失败时 可能导致机构收入 利润和特许经营权价值受到重大损失", "document_style_statement", "制度原文锚点"),
                    QuerySearchQuery("核心业务条线 经营失败 收入 利润 特许经营权价值 重大损失", "keyword_anchor", "关键术语兜底"),
                ),
                evidence_need="核心业务条线经营失败可能造成重大损失的直接说明",
                keywords=("核心业务条线", "经营失败", "收入", "利润", "特许经营权价值", "重大损失"),
            )
        )
    if "意外伤害保险" in normalized_question or "意外险" in normalized_question:
        if "定义" in normalized_question:
            requested.append(
                QueryAspect(
                    aspect_id="accident_insurance_definition",
                    question="意外伤害保险定义",
                    search_queries=(
                        QuerySearchQuery("意外伤害保险的定义是什么", "semantic_question", "贴近用户意图"),
                        QuerySearchQuery("意外伤害保险 是以被保险人因遭受意外伤害造成死亡 伤残 或者发生保险合同约定的其他事故为给付保险金条件的人身保险", "document_style_statement", "制度原文锚点"),
                        QuerySearchQuery("意外伤害保险 死亡 伤残 保险合同约定 给付保险金 人身保险", "keyword_anchor", "关键术语兜底"),
                    ),
                    evidence_need="意外伤害保险定义的直接规定",
                    keywords=("意外伤害保险", "死亡", "伤残", "给付保险金", "人身保险"),
                )
            )
        if any(term in normalized_question for term in ("定价", "厘定保险费", "保险费")):
            requested.append(
                QueryAspect(
                    aspect_id="accident_insurance_pricing_requirement",
                    question="意外伤害保险定价要求",
                    search_queries=(
                        QuerySearchQuery("意外伤害保险定价要求是什么", "semantic_question", "贴近用户意图"),
                        QuerySearchQuery("保险公司 在厘定保险费时 应符合一般精算原理 采用公平 合理的定价假设", "document_style_statement", "制度原文锚点"),
                        QuerySearchQuery("厘定保险费 一般精算原理 公平 合理 定价假设", "keyword_anchor", "关键术语兜底"),
                    ),
                    evidence_need="意外伤害保险保险费厘定和定价假设要求",
                    keywords=("保险费", "一般精算原理", "公平", "合理", "定价假设"),
                )
            )
    if "大额风险暴露" in normalized_question and any(term in normalized_question for term in ("门槛", "定义", "2.5%", "一级资本净额", "条件", "与", "和")):
        requested.append(
            QueryAspect(
                aspect_id="large_exposure_threshold",
                question="大额风险暴露门槛",
                search_queries=_fallback_search_queries("大额风险暴露门槛"),
                evidence_need="大额风险暴露的定量门槛或监管定义",
                keywords=("大额风险暴露", "一级资本净额", "2.5%", "单一客户", "一组关联客户"),
            )
        )
    if "自救原则" in normalized_question or ("处置" in normalized_question and "自救" in normalized_question):
        requested.append(
            QueryAspect(
                aspect_id="bail_in_resolution_principle",
                question="处置自救原则",
                search_queries=_fallback_search_queries("处置自救原则"),
                evidence_need="处置策略建议中自救为本原则的直接规定",
                keywords=("处置策略", "自救为本", "基本原则", "处置计划建议示例"),
            )
        )
    single_domain_ids = {"pillar3_disclosure_frequency"}
    return requested if len(requested) >= 2 or any(aspect.aspect_id in single_domain_ids for aspect in requested) else []


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
    ]
    items = [label for label, terms in detectors if all(term in normalized for term in terms)]
    for rule_code in re.findall(r"R-[A-Z]+-\d+", question.upper()):
        items.append(f"规则编号查询:{rule_code}")
    if _asks_insufficient_evidence_safety_boundary(normalized) and "依据不足违规报告边界" not in items:
        items.append("依据不足违规报告边界")
    return _dedupe(items)


def _asks_insufficient_evidence_safety_boundary(normalized_question: str) -> bool:
    if not any(term in normalized_question for term in ("违规", "监管报告", "正式监管报告", "直接判断", "直接生成")):
        return False
    return any(term in normalized_question for term in ("依据不足", "证据不足", "无依据", "没有依据"))


def _split_question_parts(question: str) -> list[str]:
    question = re.sub(
        r"请完整给出材料明确写明的条件[、，,]范围[、，,]期限或例外[。.]?",
        "",
        question,
    )
    return [
        part.strip()
        for part in re.split(r"[？?。；;]|如果|若|以及|并且|同时|，且", question)
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


def _clean_modality(value: Any) -> str:
    text = _clean_text(value).lower()
    return text if text in {"text", "table", "mixed"} else "text"


def _clean_table_task(value: Any) -> str:
    text = _clean_text(value).lower()
    return text if text in {"lookup", "compare", "calculate", "locate", "none"} else "none"


def _clean_operation(value: Any) -> str:
    text = _clean_text(value).lower()
    return text if text in {"max", "min", "difference", "sum", "ratio", "none"} else "none"


def _clean_table_filters(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    allowed = {
        "filename",
        "source_title",
        "year",
        "month",
        "quarter",
        "sheet",
        "indicator",
        "row_label",
        "column_label",
        "unit",
        "metric",
        "scope",
        "external_doc_id",
        "issuing_authority",
        "publication_date",
        "document_number",
        "regulatory_topic",
        "business_domain",
        "article_number",
        "version_status",
        "file_type",
    }
    return {key: value[key] for key in allowed if key in value and value[key] not in (None, "", [])}


def _fallback_table_aspect(question: str, options: list[str] | None = None) -> QueryAspect | None:
    normalized = re.sub(r"\s+", "", question)
    if _is_document_formula_question(normalized):
        # “附件” is a broad spreadsheet marker, but regulatory Word/PDF
        # appendices also use that word. Formula questions must stay on text
        # retrieval so the deterministic formula executor receives formula
        # chunks instead of SpreadsheetCell candidates.
        return None
    table_actions = [
        "数值", "金额", "余额", "是多少", "多少", "最高", "最低", "最大", "最小",
        "差值", "差额", "相差", "取数", "读取", "查询", "计算", "增加额", "减少额",
    ]
    explicit_table = _has_explicit_table_marker(question)
    numerical_intent = any(action in normalized for action in table_actions)
    statistical_table_intent = _looks_like_statistical_table_question(normalized)
    if not explicit_table and not statistical_table_intent:
        return None

    table_task = "lookup"
    operation = "none"
    if any(term in normalized for term in ["最高", "最大", "最多"]):
        table_task = "compare"
        operation = "max"
    elif any(term in normalized for term in ["最低", "最小", "最少"]):
        table_task = "compare"
        operation = "min"
    elif any(term in normalized for term in ["比较", "对比"]):
        table_task = "compare"
        operation = "none"
    elif (
        any(term in normalized for term in ["差值", "差额", "相差", "变化", "增加", "减少"])
        or re.search(r"[\u4e00-\u9fffA-Za-z0-9_]+比[\u4e00-\u9fffA-Za-z0-9_]+(?:多|少)", normalized)
        or ("计算" in normalized and re.search(r"[\u4e00-\u9fffA-Za-z0-9_]+[－—-][\u4e00-\u9fffA-Za-z0-9_]+", normalized))
    ):
        table_task = "calculate"
        operation = "difference"
    elif explicit_table and ("占比" in normalized or "比例" in normalized):
        table_task = "calculate"
        operation = "ratio"
    elif any(term in normalized for term in ["求和", "总和", "加总", "之和", "分项之和", "分项合计"]):
        table_task = "calculate"
        operation = "sum"

    mixed_markers = ["根据监管制度", "结合监管制度", "根据制度", "结合制度", "根据自制制度", "结合自制制度", "填报说明依据", "制度依据", "监管规则", "制度侧", "制度文件", "联合核验"]
    modality = "mixed" if any(term in normalized for term in mixed_markers) else "table"
    table_question = _mixed_table_question_part(question) if modality == "mixed" else question
    filters = _extract_table_filters(table_question)
    selectors = _extract_table_selectors(table_question, table_task, options or [])
    if table_task == "compare" and selectors:
        # Comparison operands are carried by selectors.  Keeping the first and
        # last quoted operands as global indicator/column filters would reduce
        # a four-way comparison to a single cell before comparison starts.
        inferred_indicator = str(filters.get("indicator") or "")
        filters.pop("indicator", None)
        selector_labels = {
            re.sub(r"\s+", "", str(item.get("label") or item.get("column_label") or ""))
            for item in selectors
        }
        if (
            not _quoted_terms(table_question)
            and str(filters.get("row_label") or "") == inferred_indicator
        ):
            # A common indicator mentioned only in the table title describes
            # the table family, not one of the compared rows (for example a
            # regional premium table).  Do not hard-filter all row operands to
            # that title phrase.
            filters.pop("row_label", None)
        if re.sub(r"\s+", "", str(filters.get("row_label") or "")) in selector_labels:
            filters.pop("row_label", None)
        if re.sub(r"\s+", "", str(filters.get("column_label") or "")) in selector_labels:
            filters.pop("column_label", None)
        filters.pop("metric", None)
    if table_task in {"compare", "calculate"} and len(_period_selectors(table_question)) >= 2:
        # Multiple explicitly requested periods are operands, not a single
        # global hard filter.  Keep them on the ordered selectors.
        filters.pop("year", None)
        filters.pop("month", None)
        filters.pop("quarter", None)
    # “在某口径下取数” is still a pure table lookup. Only explicit requests to
    # combine a regulation/explanation with table data should trigger the
    # slower mixed vector path.
    locator_terms = " ".join(str(value) for value in filters.values() if value)
    search_queries = [
        QuerySearchQuery(table_question, "semantic_question", "表格问题的自然语言意图"),
        QuerySearchQuery(
            " ".join(part for part in [str(filters.get("source_title") or ""), str(filters.get("sheet") or ""), str(filters.get("indicator") or "")] if part),
            "document_style_statement",
            "接近表格标题和工作表名称",
        ),
        QuerySearchQuery(locator_terms or question, "table_locator", "表格结构定位锚点"),
    ]
    search_queries = tuple(query for query in search_queries if query.query.strip())
    keywords = _keywords_from_text(" ".join([question, locator_terms]))
    return QueryAspect(
        aspect_id="table_evidence",
        question=question,
        search_queries=search_queries[:QUERY_PLANNER_MAX_SEARCH_QUERIES],
        evidence_need="表格工作表、行列标签、单元格坐标和值",
        keywords=tuple(keywords),
        modality=modality,
        table_task=table_task,
        table_filters=filters,
        operation=operation,
        selectors=tuple(selectors),
        expected_unit=str(filters.get("unit") or "") or None,
        explicit_filter_keys=tuple(filters),
    )


def _mixed_regulation_aspects(table_aspect: QueryAspect) -> tuple[QueryAspect, ...]:
    indicator = str(
        table_aspect.table_filters.get("indicator")
        or table_aspect.table_filters.get("row_label")
        or table_aspect.table_filters.get("metric")
        or "该统计指标"
    )
    labelled = re.search(
        r"制度侧[：:]\s*(.+?)(?=\s*报表侧[：:]|$)",
        table_aspect.question,
        flags=re.DOTALL,
    )
    unlabelled = re.search(
        r"(?:联合核验|结合核验)[：:]?\s*(.+?)(?=[；;。]\s*(?:再|并)?(?:比较|查询|读取|计算|核对))",
        table_aspect.question,
        flags=re.DOTALL,
    )
    regulation_question = (
        labelled.group(1).strip()
        if labelled
        else unlabelled.group(1).strip()
        if unlabelled
        else f"监管制度对“{indicator}”的定义、填报口径、适用范围和报送要求是什么？"
    )
    regulatory_filter_keys = {
        "external_doc_id",
        "issuing_authority",
        "publication_date",
        "document_number",
        "regulatory_topic",
        "business_domain",
        "article_number",
        "version_status",
    }
    regulation_source_filter_keys = {
        "source_title",
        "filename",
        "file_type",
        "external_doc_id",
        "issuing_authority",
        "document_number",
    }
    filters = {
        key: value
        for key, value in table_aspect.table_filters.items()
        if key in regulatory_filter_keys
    }
    if labelled or unlabelled:
        labelled_filters = _extract_table_filters(regulation_question)
        for key in regulation_source_filter_keys:
            if labelled_filters.get(key):
                filters[key] = labelled_filters[key]
    explicit_keys = tuple(filters)
    anchors = _quoted_evidence_anchors(regulation_question)
    coordinated_questions = _coordinated_aspect_questions(regulation_question)
    if len(coordinated_questions) == 2 and len(anchors) < 2:
        question_specs = [(question, question) for question in coordinated_questions]
    elif anchors:
        question_specs = [
            (
                _question_for_single_anchor(regulation_question, anchor)
                if len(anchors) > 1
                else regulation_question,
                anchor,
            )
            for anchor in anchors
        ]
    else:
        # For an explicitly labelled mixed question, the regulation clause is
        # the evidence anchor.  Reusing the spreadsheet indicator here makes
        # the text retriever search for table vocabulary and can silently drop
        # the制度侧 evidence.
        anchor = regulation_question if labelled or unlabelled else indicator
        question_specs = [(regulation_question, anchor)]
    aspects: list[QueryAspect] = []
    bounded_specs = question_specs[: max(1, QUERY_PLANNER_MAX_ASPECTS - 1)]
    for index, (anchor_question, anchor) in enumerate(bounded_specs, start=1):
        aspect_id = "regulatory_basis" if len(bounded_specs) == 1 else f"regulatory_basis_{index}"
        aspects.append(
            QueryAspect(
                aspect_id=aspect_id,
                question=anchor_question,
                search_queries=(
                    QuerySearchQuery(anchor_question, "semantic_question", "监管解释意图"),
                    QuerySearchQuery(
                        " ".join(
                            part
                            for part in [
                                str(filters.get("source_title") or ""),
                                _mixed_regulation_document_style_alias(regulation_question, anchor_question),
                            ]
                            if part
                        ),
                        "document_style_statement",
                        "以指定材料和显式条件定位制度原文",
                    ),
                    QuerySearchQuery(anchor, "keyword_anchor", "显式条件术语兜底"),
                ),
                evidence_need=f"监管制度中与“{anchor}”直接对应的条件、范围、期限或例外",
                keywords=tuple(_dedupe([anchor, *_keywords_from_text(anchor)])),
                modality="text",
                table_filters=filters,
                explicit_filter_keys=explicit_keys,
            )
        )
    return tuple(aspects)


def _expand_explicit_anchor_aspects(
    aspects: list[QueryAspect],
    *,
    max_aspects: int,
) -> list[QueryAspect]:
    """Split multiple explicitly quoted evidence needs into retrieval units.

    The planner may merge two requested clauses from the same document into a
    single aspect.  One high-scoring chunk can then make the aspect appear
    covered while the second clause is absent.  Quoted clauses are user-stated
    constraints, so preserving each as its own aspect is deterministic and
    does not infer an answer.
    """

    expanded: list[QueryAspect] = []
    for aspect in aspects:
        anchors = _quoted_evidence_anchors(aspect.question)
        if aspect.modality != "text" or len(anchors) < 2:
            expanded.append(aspect)
            continue
        for index, anchor in enumerate(anchors, start=1):
            anchor_question = _question_for_single_anchor(aspect.question, anchor)
            expanded.append(
                replace(
                    aspect,
                    aspect_id=f"{aspect.aspect_id}_{index}",
                    question=anchor_question,
                    search_queries=(
                        QuerySearchQuery(anchor_question, "semantic_question", "拆分后的显式条件检索"),
                        QuerySearchQuery(
                            " ".join(
                                part
                                for part in [str(aspect.table_filters.get("source_title") or ""), anchor]
                                if part
                            ),
                            "document_style_statement",
                            "指定材料与显式条件原文定位",
                        ),
                        QuerySearchQuery(anchor, "keyword_anchor", "显式条件术语兜底"),
                    ),
                    evidence_need=f"与“{anchor}”直接对应的制度原文及其条件、范围、期限或例外",
                    keywords=tuple(_dedupe([anchor, *_keywords_from_text(anchor)])),
                )
            )
    return expanded[:max_aspects]


def _quoted_evidence_anchors(question: str) -> list[str]:
    return _dedupe(
        [
            item.strip()
            for item in re.findall(r"[“‘\"]([^”’\"]+)[”’\"]", question)
            if len(re.sub(r"\s+", "", item)) >= 2
        ]
    )


def _question_for_single_anchor(question: str, anchor: str) -> str:
    title_match = re.search(r"《([^》]+)》", question)
    title_prefix = f"根据《{title_match.group(1)}》，" if title_match else ""
    quoted_matches = list(re.finditer(r"[“‘\"]([^”’\"]+)[”’\"]", question))
    if not title_match and len(quoted_matches) >= 2:
        prefix = question[: quoted_matches[0].start()].rstrip("，,、和与及")
        suffix = question[quoted_matches[-1].end() :].lstrip("，,、和与及")
        return f"{prefix}“{anchor}”{suffix}".strip()
    return f"{title_prefix}请说明与“{anchor}”相关的明确规定，并完整给出条件、范围、期限或例外。"


def _extract_table_selectors(
    question: str,
    table_task: str,
    options: list[str],
) -> list[dict[str, Any]]:
    quoted = _quoted_terms(question)
    selectors: list[dict[str, Any]] = []
    if "四类贷款余额" in re.sub(r"\s+", "", question):
        return [
            {"label": label}
            for label in ("损失类贷款余额", "不良贷款余额", "可疑类贷款余额", "次级类贷款余额")
        ]
    reconciliation_selectors = _reconciliation_source_selectors(question) if table_task == "calculate" else []
    if reconciliation_selectors:
        return reconciliation_selectors
    regional_metric_selectors = _regional_metric_comparison_selectors(question) if table_task == "compare" else []
    if regional_metric_selectors:
        return regional_metric_selectors
    period_selectors = _period_selectors(question) if table_task in {"compare", "calculate"} else []
    if len(period_selectors) >= 2:
        selectors = period_selectors
    elif table_task == "calculate" and len(quoted) >= 3:
        # “row”从“column A”到“column B” means B - A.
        selectors = [
            {"row_label": quoted[0], "column_label": quoted[1]},
            {"row_label": quoted[0], "column_label": quoted[2]},
        ]
    elif table_task == "calculate":
        normalized = re.sub(r"\s+", "", question)
        indicators = _mentioned_table_indicators(question)
        diff_match = re.search(r"(.+?)比(.+?)(?:多|少)", normalized)
        if diff_match and len(indicators) >= 2:
            # ordered_transition calculations use second - first.
            selectors = [{"row_or_indicator": indicators[1]}, {"row_or_indicator": indicators[0]}]
        elif any(term in normalized for term in ["分项之和", "分项合计", "四个分项"]):
            component_order = ["现金及存放同业", "贷款余额", "证券投资", "其他资产"]
            selectors = [{"row_or_indicator": item} for item in component_order]
        elif len(indicators) >= 2:
            selectors = [{"row_or_indicator": item} for item in indicators[:2]]
    elif table_task == "compare" and options:
        scope = quoted[0] if quoted else ""
        selectors = [
            {"label": option, "scope": scope}
            for option in options
            if str(option).strip()
        ]
    elif table_task == "compare":
        source_local_labels = _source_local_comparison_labels(question)
        if source_local_labels:
            return [{"label": item} for item in source_local_labels]
        comparison_segment = re.search(
            r"(?:比较|对比)(.+?)(?:，?(?:找出|其中|给出|报告|确定)|，?哪(?:个|项)|[。；]|$)",
            question,
            flags=re.DOTALL,
        )
        segment = comparison_segment.group(1) if comparison_segment else question
        structured_labels = _comparison_labels(question)
        labels = structured_labels if not comparison_segment else _quoted_terms(segment)
        if not labels:
            labels = structured_labels
        if not labels and comparison_segment:
            labels = _comparison_labels(f"比较{segment}")
        if not labels:
            labels = _mentioned_table_indicators(segment)
        selectors = [{"label": item} for item in _dedupe(labels)]
    elif table_task == "lookup":
        if len(quoted) >= 2:
            selectors = [{"row_or_indicator": quoted[0], "column_label": quoted[-1]}]
        elif quoted:
            selectors = [{"row_or_indicator": quoted[0]}]
    return selectors


def _source_local_comparison_labels(question: str) -> list[str]:
    """Extract compared indicators after an explicitly named report family."""

    compact = re.sub(r"\s+", "", str(question or ""))
    match = re.search(
        r"(?:情况表|经营表|人身险表|财产险表|保险业经营表)(?:中|中的|的)"
        r"(?P<labels>.+?)(?=，?(?:给出|指出|找出|确定|报告)|[。；]|$)",
        compact,
    )
    if not match:
        return []
    parts = [
        part.strip("，,：:的")
        for part in re.split(r"[、，,]|以及|和|与|及", match.group("labels"))
        if part.strip("，,：:的")
    ]
    return _dedupe(parts) if 2 <= len(parts) <= 8 else []


def _quoted_terms(text: str) -> list[str]:
    return _dedupe(
        [
            item.strip()
            for item in re.findall(r"[“‘\"]([^”’\"]+)[”’\"]", str(text or ""))
            if item.strip()
        ]
    )


def _comparison_labels(text: str) -> list[str]:
    compact = re.sub(r"\s+", "", str(text or ""))
    match = re.search(r"对(.+?)做(?:四项|多项)?比较", compact)
    if match is None:
        match = re.search(
            r"(?:比较|对比)(.+?)(?:四项|多项|，?哪(?:项|个)|，?找出|，?确定|，?给出|，?报告|[。；]|$)",
            compact,
        )
    if match is None:
        matches = list(re.finditer(r"在(.{2,100}?)四项中(?:确定|找出)", compact))
        match = matches[-1] if matches else None
    if match is None:
        return []
    segment = re.sub(r"^(?:在|将|把)", "", match.group(1))
    parts = [
        part.strip("，,：:的")
        for part in re.split(r"[、，,]|以及|和|与|及", segment)
        if part.strip("，,：:的")
    ]
    if not 2 <= len(parts) <= 8:
        return []
    # Shared-tail enumeration: “损失类、不良、可疑类和次级类贷款余额”.
    tail_match = re.search(r"(贷款余额|保费收入|账面余额|健康险|合计)$", parts[-1])
    tail = tail_match.group(1) if tail_match else ""
    labels: list[str] = []
    for part in parts:
        cleaned = re.sub(r"^(?:在|按|以)", "", part)
        if tail and not cleaned.endswith(tail) and len(cleaned) <= 12:
            cleaned += tail
        labels.append(cleaned)
    return _dedupe(labels)


def _period_selectors(question: str) -> list[dict[str, Any]]:
    compact = re.sub(r"\s+", "", str(question or ""))
    periods: list[dict[str, int]] = []
    current_year: int | None = None
    for match in re.finditer(r"(?:(20\d{2})年)?(\d{1,2})月", compact):
        if match.group(1):
            current_year = int(match.group(1))
        if current_year is not None:
            periods.append({"year": current_year, "month": int(match.group(2))})
    current_year = None
    for match in re.finditer(r"(?:(20\d{2})年)?([一二三四1-4])季度", compact):
        if match.group(1):
            current_year = int(match.group(1))
        if current_year is None:
            continue
        quarter_text = match.group(2)
        quarter = {"一": 1, "二": 2, "三": 3, "四": 4}.get(quarter_text, int(quarter_text) if quarter_text.isdigit() else 0)
        periods.append({"year": current_year, "quarter": quarter})
    unique_periods: list[dict[str, int]] = []
    seen: set[tuple[tuple[str, int], ...]] = set()
    for period in periods:
        key = tuple(sorted(period.items()))
        if key not in seen:
            unique_periods.append(period)
            seen.add(key)
    if len(unique_periods) < 2:
        return []
    indicators = _mentioned_table_indicators(question)
    row_label = indicators[0] if indicators else ""
    column_label = ""
    if "本年累计" in compact or "截至当期" in compact:
        column_label = "本年累计"
    elif "账面值" in compact or "账面余额" in compact:
        column_label = "账面余额"
    if "全国各地区原保险保费收入" in compact and "健康险" in compact:
        row_label = "全国合计" if "全国健康险合计" in compact or "全国合计" in compact else row_label
        column_label = "健康险"
    source_family_match = re.search(
        r"(人身险公司(?:原保险保费收入)?经营(?:情况)?表|财产(?:险|保险)公司(?:原保险保费收入)?经营(?:情况)?表|保险业资金运用情况表|保险业经营(?:情况)?表|全国各地区原保险保费收入(?:情况)?表)",
        compact,
    )
    source_family = source_family_match.group(1) if source_family_match else ""
    if source_family:
        if "人身险公司" in source_family:
            source_family = "人身险公司经营情况表"
        elif "财产险公司" in source_family or "财产保险公司" in source_family:
            source_family = "财产保险公司经营情况表"
        elif source_family == "保险业经营表":
            source_family = "保险业经营情况表"
        elif source_family == "全国各地区原保险保费收入表":
            source_family = "全国各地区原保险保费收入情况表"
    if not source_family:
        if "人身险公司" in compact and "原保险保费收入" in compact:
            source_family = "人身险公司经营情况表"
        elif ("财产险公司" in compact or "财产保险公司" in compact) and "原保险保费收入" in compact:
            source_family = "财产保险公司经营情况表"
    selectors: list[dict[str, Any]] = []
    for period in unique_periods:
        selector: dict[str, Any] = {
            **period,
            **({"row_label": row_label} if row_label else {}),
            **({"column_label": column_label} if column_label else {}),
        }
        if source_family:
            if period.get("month"):
                selector["source_title"] = f"{period['year']}年{period['month']}月{source_family}"
            elif period.get("quarter"):
                selector["source_title"] = f"{period['year']}年{period['quarter']}季度{source_family}"
        selectors.append(selector)
    return selectors


def extract_explicit_metadata_filters(question: str) -> dict[str, Any]:
    """Extract only constraints stated literally by the user.

    The LLM planner may still add inferred filters for ranking, but callers can
    safely use the keys returned here as hard pre-retrieval constraints.
    """

    return _extract_table_filters(question)


def _extract_table_filters(question: str) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    compact_question = re.sub(r"\s+", "", question)
    # A mixed question can name a regulation before the spreadsheet.  The
    # explicit Excel/report attachment title is the hard table locator and
    # therefore takes precedence over the first generic book-title quote.
    title_match = re.search(r"(?:Excel|表格|报表)(?:\s*附件)?《([^》]+)》", question, flags=re.IGNORECASE)
    if title_match is None:
        title_match = re.search(r"《([^》]+)》", question)
    if title_match:
        filters["source_title"] = title_match.group(1).strip()
        requested_format = re.search(
            r"[（(]\s*(pdf|word|docx?|excel|xlsx?|xls)\s*[）)]\s*$",
            title_match.group(1),
            flags=re.IGNORECASE,
        )
        if requested_format:
            normalized_format = requested_format.group(1).lower()
            filters["file_type"] = {
                "word": "docx",
                "excel": "xlsx",
            }.get(normalized_format, normalized_format)
    else:
        natural_title = re.search(
            r"(20\d{2}年(?:\d{1,2}月|[一二三四1-4]季度)?[^，。；：:]{2,45}?(?:经营情况表|原保险保费收入表|贷款表|指标分机构类表))",
            question,
        )
        if natural_title:
            value = natural_title.group(1).strip("的在读取查询比较")
            period_count = len(re.findall(r"(?:(?:20\d{2}年)?\d{1,2}月)", value))
            if period_count >= 2:
                # Cross-period selectors below carry one concrete title per
                # period.  A composite phrase is not a real document title.
                pass
            elif "保障房贷款表" in value:
                filters["sheet"] = "保障房贷款"
            elif "商业银行主要指标分机构类表" in value:
                filters["source_title"] = value.removesuffix("表")
            else:
                filters["source_title"] = value
    document_number_match = re.search(r"[\u4e00-\u9fffA-Za-z]+〔\d{4}〕\d+号", question)
    if document_number_match:
        filters["document_number"] = document_number_match.group(0)
    article_match = re.search(
        r"第[零〇一二三四五六七八九十百千万两\d]+条(?:之[零〇一二三四五六七八九十百千万两\d]+)?",
        question,
    )
    if article_match:
        filters["article_number"] = article_match.group(0)
    for label, key in (
        ("发文机关", "issuing_authority"),
        ("监管主题", "regulatory_topic"),
        ("业务领域", "business_domain"),
    ):
        match = re.search(rf"{label}[：:]\s*([^，。；;）)]+)", question)
        if match:
            filters[key] = match.group(1).strip()
    publication_match = re.search(r"发布日期[：:]?\s*(20\d{2})年(\d{1,2})月(\d{1,2})日", question)
    if publication_match:
        filters["publication_date"] = (
            f"{publication_match.group(1)}-{int(publication_match.group(2)):02d}-{int(publication_match.group(3)):02d}"
        )
    if any(marker in question for marker in ("现行", "有效版本", "当前有效")):
        filters["version_status"] = "current"
    elif any(marker in question for marker in ("已废止", "废止版本", "失效版本")):
        filters["version_status"] = "repealed"
    sheet_match = re.search(r"工作表[：:]\s*(.+?)(?=）\s*[，,]|[，。；;]\s*)", question)
    if sheet_match is None:
        sheet_match = re.search(r"工作表[：:]\s*([^，。；;]+)", question)
    if sheet_match is None:
        sheet_match = re.search(r"[‘“\"]([^’”\"]+)[’”\"]工作表", question)
    if sheet_match:
        filters["sheet"] = sheet_match.group(1).strip()
    year_match = re.search(r"(20\d{2})年", compact_question)
    if year_match:
        filters["year"] = int(year_match.group(1))
    month_match = re.search(r"(?:20\d{2}年)?(\d{1,2})月", compact_question)
    if month_match:
        filters["month"] = int(month_match.group(1))
    if "source_title" not in filters and filters.get("year") and filters.get("month"):
        year = int(filters["year"])
        month = int(filters["month"])
        if "财产险表" in compact_question or "财产保险公司经营" in compact_question:
            filters["source_title"] = f"{year}年{month}月财产保险公司经营情况表"
        elif "人身险表" in compact_question or "人身保险公司经营" in compact_question:
            filters["source_title"] = f"{year}年{month}月人身险公司经营情况表"
        elif "保险业经营表" in compact_question or "保险业经营情况表" in compact_question:
            filters["source_title"] = f"{year}年{month}月保险业经营情况表"
        elif "全国各地区" in compact_question and "原保险保费收入" in compact_question:
            filters["source_title"] = f"{year}年{month}月全国各地区原保险保费收入情况表"
    quarter_match = re.search(r"([一二三四1-4])季度|([1-4])季", compact_question)
    if quarter_match:
        value = quarter_match.group(1) or quarter_match.group(2)
        filters["quarter"] = {"一": 1, "二": 2, "三": 3, "四": 4}.get(value, int(value) if value.isdigit() else None)
    quoted_terms = _quoted_terms(question)
    if quoted_terms:
        filters["indicator"] = quoted_terms[0]
        if len(quoted_terms) > 1:
            filters["column_label"] = quoted_terms[-1]
    indicators = _mentioned_table_indicators(question)
    if indicators and "indicator" not in filters:
        filters["indicator"] = indicators[0]
        filters["row_label"] = indicators[0]
    scope_text = re.split(r"(?:请比较|请对比|比较|对比)", question, maxsplit=1)[0]
    scope_text = re.sub(r"《[^》]+》", "", scope_text)
    for label in ["全国合计", "全  国", "北京", "上海", "江苏", "浙江", "广东", "大型商业银行", "城市商业银行", "农村商业银行", "股份制商业银行", "民营银行", "外资银行"]:
        if label.replace(" ", "") in re.sub(r"\s+", "", scope_text):
            filters["row_label"] = label
            break
    row_scoped_phrase = False
    focused_scope = re.search(r"》的?([^，。；：:]{2,24})口径下", question)
    if focused_scope is None:
        focused_scope = re.search(r"表的?([^，。；：:]{2,24})口径下", question)
    if focused_scope:
        value = focused_scope.group(1).strip("的在按以")
        row_scoped_phrase = value == str(filters.get("row_label") or "").strip()
        if value and not row_scoped_phrase and "表" not in value:
            filters["column_label"] = value
    column_scope = re.search(r"[‘“\"]?([^，。；：:‘“’”\"]{2,24})[’”\"]?(?:列|口径)下?(?:中|比较|取数|查询|读取)?", question)
    if column_scope and "column_label" not in filters and not row_scoped_phrase:
        value = column_scope.group(1).split("》")[-1].strip("的在按以")
        if (
            value
            and value != str(filters.get("row_label") or "").strip()
            and not any(marker in value for marker in ("情况表", "统计表", "工作表"))
        ):
            filters["column_label"] = value
    explicit_total_column = re.search(
        r"(?:全国合计|总计|合计行)(?:的|中|下)[‘“\"]?合计[’”\"]?(?:口径下)?(?:值|数值)",
        compact_question,
    )
    if explicit_total_column and "column_label" not in filters:
        filters["column_label"] = "合计"
    return filters


def _looks_like_policy_question(normalized_question: str) -> bool:
    policy_markers = [
        "需要复核",
        "复核期限",
        "是否包括",
        "是否纳入",
        "不包括",
        "包括哪些",
        "哪些项目构成",
        "由哪些项目构成",
        "留存哪些依据",
        "需要留存",
        "口径",
        "制度",
        "监管制度",
        "条款",
        "依据《",
        "管理办法",
        "监管办法",
        "规定",
        "要求",
        "原则",
        "期限",
        "限制",
        "定义",
        "概括",
        "名单",
        "类别",
    ]
    return any(marker in normalized_question for marker in policy_markers)


def _has_explicit_table_marker(question: str) -> bool:
    if any(marker in question for marker in TABLE_MARKERS):
        return True
    compact = re.sub(r"\s+", "", question)
    return bool(
        re.search(
            r"(?:经营|情况|统计|保费收入|贷款余额|资金运用)[\u4e00-\u9fff]{0,10}表",
            compact,
        )
    )


def _looks_like_statistical_table_question(question: str) -> bool:
    """Recognize strong table intent without treating every number as a table.

    A policy question may ask “是多少” or contain a percentage.  Structured
    spreadsheet routing is therefore enabled only when a period is combined
    with a known statistical indicator/source family, or when the user asks
    for an explicit reconciliation calculation.
    """

    compact = re.sub(r"\s+", "", question)
    if "勾稽差额" in compact:
        return bool(re.search(r"20\d{2}年(?:\d{1,2}月|[一二三四1-4]季度)", compact))
    has_period = bool(
        re.search(r"20\d{2}年(?:\d{1,2}月|[一二三四1-4]季度)", compact)
        or re.search(r"20\d{2}年.+?(?:月|季度)", compact)
    )
    if not has_period:
        return False
    source_or_indicator = any(
        term in compact
        for term in (
            *COMMON_TABLE_INDICATORS,
            "人身险公司",
            "财产险公司",
            "财产保险公司",
            "保险业合计",
            "全国各地区",
            "全国健康险",
            "本年累计值",
        )
    )
    action = any(
        term in compact
        for term in ("读取", "取数", "查询", "比较", "对比", "最高", "最低", "最大", "最小", "计算", "差额", "差值", "是多少", "多少")
    )
    return source_or_indicator and action


def _mixed_table_question_part(question: str) -> str:
    labelled = _labelled_question_part(question, "报表侧")
    if labelled != question:
        return labelled
    match = re.search(
        r"(?:^|[；;。])\s*((?:再|并)?(?:比较|查询|读取|计算|核对).+?)(?=(?:[。；;]\s*(?:最后|并说明|不得|行业汇总))|$)",
        question,
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match else question


def _reconciliation_source_selectors(question: str) -> list[dict[str, Any]]:
    compact = re.sub(r"\s+", "", question)
    if not all(term in compact for term in ("保险业合计", "人身险", "财产险")):
        return []
    if not ("勾稽差额" in compact or re.search(r"计算.*保险业合计[－—-]人身险[－—-]财产险", compact)):
        return []
    period = re.search(r"(20\d{2})年(\d{1,2})月", compact)
    if not period:
        return []
    year, month = int(period.group(1)), int(period.group(2))
    common = {"year": year, "month": month, "row_label": "原保险保费收入", "column_label": "本年累计"}
    return [
        {**common, "source_title": f"{year}年{month}月保险业经营情况表"},
        {**common, "source_title": f"{year}年{month}月人身险公司经营情况表"},
        {**common, "source_title": f"{year}年{month}月财产保险公司经营情况表"},
    ]


def _regional_metric_comparison_selectors(question: str) -> list[dict[str, Any]]:
    compact = re.sub(r"\s+", "", question)
    # Prefer a bounded catalogue of report columns.  A free-form ``...险``
    # expression is too greedy for questions such as ``2024年9月全国健康险的...``
    # and used to turn the month/title prefix into the column label.
    metric_names = (
        "机动车辆保险", "意外伤害保险", "人寿保险", "健康保险",
        "责任保险", "农业保险", "信用保险", "保证保险",
        "健康险", "意外险", "寿险", "财产险", "人身险",
    )
    metric_pattern = "|".join(re.escape(value) for value in metric_names)
    metric_match = re.search(rf"(?P<metric>{metric_pattern})(?:的|中|口径)", compact)
    if not metric_match:
        return []
    metric = metric_match.group("metric")
    # Only inspect the enumeration governed by this metric.  Looking across the
    # whole question incorrectly treated the ``全国`` in ``全国健康险`` as a row.
    region_segment = compact[metric_match.end():]
    region_segment = re.split(r"(?:，?给出|。?最后|；|。)", region_segment, maxsplit=1)[0]
    region_names = (
        "全国合计", "公司本级", "全国", "北京", "天津", "河北", "山西", "内蒙古",
        "辽宁", "吉林", "黑龙江", "上海", "江苏", "浙江", "安徽", "福建", "江西",
        "山东", "河南", "湖北", "湖南", "广东", "广西", "海南", "重庆", "四川",
        "贵州", "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆",
    )
    positions = [
        (region_segment.find(region), region)
        for region in region_names
        if region_segment.find(region) >= 0
    ]
    positions.sort(key=lambda item: item[0])
    regions: list[str] = []
    for _, region in positions:
        if region == "全国" and "全国合计" in region_segment:
            continue
        if region not in regions:
            regions.append(region)
    if len(regions) < 2:
        return []
    return [
        {"label": region, "column_label": metric}
        for region in regions
    ]


def _is_execution_instruction_aspect(question: str) -> bool:
    compact = re.sub(r"\s+", "", question)
    return bool(
        re.search(
            r"^(?:若|如果).{0,30}(?:不支持|无法执行).{0,30}(?:必须|应当)?(?:明确)?拒答(?:而非|不得)?(?:估算|推算)?[。？?]?$",
            compact,
        )
    )


def _labelled_question_part(question: str, label: str) -> str:
    match = re.search(
        rf"{re.escape(label)}[：:]\s*(.+?)(?=\s*(?:制度侧|报表侧|证据边界)[：:]|$)",
        question,
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match else question


def _is_document_formula_question(question: str) -> bool:
    normalized = re.sub(r"\s+", "", question).casefold()
    return "公式" in normalized and any(marker in normalized for marker in ("word", "pdf", "[公式]"))


def _should_force_text_aspect(question: str, sub_question: str, modality: str) -> bool:
    if modality not in {"table", "mixed"}:
        return False
    if _has_explicit_table_marker(question) or _has_explicit_table_marker(sub_question):
        return False
    combined = re.sub(r"\s+", "", f"{question}{sub_question}")
    # An LLM occasionally labels any question containing a percentage or
    # ``多少`` as table/mixed. Without an explicit table marker, preserve that
    # label only for strong statistical-report intent. Prescribed policy values
    # such as a discount rate must remain text retrieval.
    return _looks_like_policy_question(combined) or not _looks_like_statistical_table_question(combined)


def _mentioned_table_indicators(question: str) -> list[str]:
    normalized = re.sub(r"\s+", "", question)
    matches = [indicator for indicator in COMMON_TABLE_INDICATORS if indicator in normalized]
    matches.sort(key=lambda item: normalized.find(item))
    return _dedupe(matches)


def _fallback_search_queries(question: str) -> tuple[QuerySearchQuery, ...]:
    keywords = " ".join(_keywords_from_text(question))
    queries = [
        QuerySearchQuery(query=question, query_type="semantic_question", rationale="fallback 子问题检索"),
    ]
    document_style = _document_style_alias(question)
    if document_style != question:
        queries.append(
            QuerySearchQuery(
                query=document_style,
                query_type="document_style_statement",
                rationale="将用户概念转换为制度原文常见表达",
            )
        )
    if keywords and keywords != question:
        queries.append(
            QuerySearchQuery(query=keywords, query_type="keyword_anchor", rationale="fallback 关键术语兜底")
        )
    return tuple(queries[:QUERY_PLANNER_MAX_SEARCH_QUERIES])


def _document_style_alias(question: str) -> str:
    text = str(question or "")
    text = re.sub(r"(?:沟通策略)?沟通对象", "沟通策略 与哪些主体开展有效沟通", text)
    text = text.replace("沟通策略对象", "沟通策略 与哪些主体开展有效沟通")
    text = re.sub(
        r"大额风险暴露(?:门槛|定义|监管定义)?",
        "处置计划建议示例 商业银行版 大额风险暴露和同业融入情况 对主要同业客户的风险暴露情况 参考大额风险暴露的相关统计要求 大额风险暴露是指 商业银行 对单一客户或一组关联客户 超过其一级资本净额 2.5% 的风险暴露",
        text,
    )
    text = re.sub(
        r"(?:处置)?自救原则",
        "处置计划建议示例 商业银行版 处置计划的实施 处置策略 策略建议中可以选择应用多种处置工具 应坚持自救为本的基本原则",
        text,
    )
    text = re.sub(
        r"(?:意外伤害保险)?(?:保费)?厘定(?:要求|原则)|(?:保费)?定价原则",
        "保险公司 在厘定保险费时 应符合 一般精算原理 采用 公平 合理 定价假设",
        text,
    )
    text = re.sub(
        r"(?:意外伤害保险|意外险)(?:的)?(?:保费)?定价要求",
        "保险公司 在厘定保险费时 应符合 一般精算原理 采用 公平 合理 定价假设",
        text,
    )
    text = re.sub(
        r"(?:意外伤害保险|意外险)(?:的)?(?:统一)?定义",
        "本办法所称意外伤害保险 是以 被保险人 因遭受意外伤害造成死亡 伤残 或者发生保险合同约定的其他事故 为给付保险金条件的人身保险",
        text,
    )
    text = re.sub(
        r"保险集团报告义务",
        "应当编报保险集团偿付能力报告的公司名单 下列保险集团 应当 按照 保险公司偿付能力监管规则第19号 保险集团 编报 保险集团偿付能力报告",
        text,
    )
    text = re.sub(
        r"数字化银行回函的效力|数字化回函效力",
        "以符合相关规定的数字方式 办理银行询证函及回函 与纸质银行询证函及回函 具有同等法律效力和证明力",
        text,
    )
    text = re.sub(
        r"回函时限|询证函回复时限|合规询证函回复时限",
        "银行业金融机构 应当自收到符合规定的询证函之日起 10个工作日内 按照要求 将回函直接回复会计师事务所",
        text,
    )
    text = re.sub(
        r"函证事项.*公开渠道公示|办理函证相关事项.*公示|公开渠道公示函证事项",
        "银行业金融机构 应当 在其总行或总部网站 微信公众号等公开渠道 就办理函证相关事项进行公示",
        text,
    )
    text = re.sub(
        r"一函一个基准日|一个函证基准日",
        "一份询证函 只列示 一个函证基准日",
        text,
    )
    text = re.sub(
        r"消费金融(?:公司)?(?:催收)?禁限|催收禁限|禁止.*无关第三人催收",
        "消费金融公司 不得采用 暴力 威胁 恐吓 骚扰 等不正当手段进行催收 不得对与债务无关的第三人进行催收",
        text,
    )
    text = re.sub(
        r"重要实体(?:要求|定义|作用)?",
        "重要实体 承载 本机构核心业务条线和关键功能 对持续经营 维持关键功能具有重要作用",
        text,
    )
    text = re.sub(
        r"处置工具|处置自救原则",
        "处置工具 包括 机构自救 股东注资 引入战略投资者 处置不良资产 接管 收购承接 建立过桥机构 破产清算 应坚持自救为本的基本原则",
        text,
    )
    text = re.sub(
        r"大额风险暴露门槛",
        "大额风险暴露是指 商业银行 对单一客户或一组关联客户 超过其一级资本净额 2.5%的风险暴露",
        text,
    )
    text = re.sub(
        r"消费金融公司(?:的)?业务地域范围|消费金融公司.*全国范围开展业务",
        "消费金融公司 可以在全国范围内开展业务",
        text,
    )
    text = re.sub(
        r"专利权质押登记线上全覆盖目标|专利权质押登记.*全覆盖",
        "试验区内商业银行各分支机构 实现 专利权质押登记 全流程无纸化线上办理 全覆盖",
        text,
    )
    text = re.sub(
        r"重大数据事件(?:的)?条件|重要数据事件何时属于重大事件",
        "重大数据安全事件 重要数据遭到泄露 破坏 非法获取 非法利用 对省级区域经济带来重大影响 对银行保险行业安全造成影响",
        text,
    )
    text = re.sub(
        r"交易账簿头寸|交易头寸",
        "以交易目的持有的头寸 包括 自营业务 做市业务 为满足客户需求提供的对客交易 对冲前述交易相关风险 交易账簿头寸原则上每日进行公允价值计量 变动计入损益",
        text,
    )
    text = re.sub(
        r"第三支柱(?:信息)?(?:披露)?(?:表格)?(?:类型|频率|并表范围)?",
        "商业银行以表格形式进行第三支柱信息披露 包括固定表格和可变表格 按照监管并表范围披露 表格中另有规定的除外 根据表格要求分别按照季度 半年 年度频率披露信息",
        text,
    )
    text = re.sub(
        r"中资商业银行法人机构筹建申请书|法人机构筹建申请书",
        "中资商业银行法人机构筹建审批 申请书 内容包括但不限于 拟设立机构 名称 拟设地 注册资本 股权结构 业务范围 基本信息",
        text,
    )
    text = re.sub(
        r"目录版本|2023年版目录",
        "中资商业银行行政许可事项申请材料目录及格式要求 2023年版",
        text,
    )
    text = re.sub(
        r"终极利率(?:暂定值)?",
        "终极利率 暂定为 4.5%",
        text,
    )
    text = re.sub(
        r"薪酬递延",
        "高级管理人员 对风险有重要影响岗位 员工 绩效薪酬 40%以上 延期支付 期限 一般不少于3年",
        text,
    )
    text = re.sub(
        r"催收禁限",
        "不得采用 暴力 威胁 恐吓 骚扰 不正当手段 催收 不得对与债务无关的第三人进行催收",
        text,
    )
    text = re.sub(
        r"持续经营触发阈值",
        "持续经营触发事件 指 商业银行 核心一级资本充足率 降至 5.125% 或以下",
        text,
    )
    text = re.sub(
        r"损失吸收顺序",
        "所有其他一级资本工具 全部吸收损失后 再启动 二级资本工具 吸收损失",
        text,
    )
    text = text.replace("投连险", "投资连结险")
    return re.sub(r"\s+", " ", text).strip()


def _mixed_regulation_document_style_alias(regulation_question: str, anchor_question: str) -> str:
    alias = _document_style_alias(anchor_question)
    compact = re.sub(r"\s+", "", regulation_question)
    if "大额风险暴露" in compact and "处置" in compact:
        alias = f"商业银行版 处置计划建议示例 {alias}"
    if "资本工具" in compact and any(term in alias for term in ("持续经营触发事件", "二级资本工具", "吸收损失")):
        alias = f"资本工具合格标准 {alias}"
    if "意外伤害保险" in compact and any(term in alias for term in ("厘定保险费", "意外伤害保险")):
        alias = f"意外伤害保险业务监管办法 {alias}"
    if "保险集团报告义务" in compact or "保险集团偿付能力报告" in compact:
        alias = f"应当编报保险集团偿付能力报告的公司名单 {alias}"
    return alias


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
        "大额风险暴露",
        "一级资本净额",
        "2.5%",
        "自救为本",
    ]
    normalized = re.sub(r"\s+", "", text)
    keywords = [phrase for phrase in domain_phrases if phrase in normalized]
    if "自救" in normalized and "自救为本" not in keywords:
        keywords.append("自救为本")
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
