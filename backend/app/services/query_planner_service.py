from dataclasses import dataclass, field, replace
import json
import re
import urllib.error
import urllib.request
from typing import Any
from backend.app.services.performance_metrics import measure, timed

from backend.app.core.config import (
    QUERY_PLANNER_DISABLE_THINKING,
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
    # Insurance-fund asset classes and income/expense lines (composite row
    # labels like "财产险公司/银行存款" need the asset term itself).
    "银行存款",
    "债券",
    "股票",
    "证券投资基金",
    "长期股权投资",
    "保户投资款新增交费",
    "投连险独立账户新增交费",
    "赔付支出",
    "总资产",
    "净资产",
    "净利润",
]

# Terms that describe a table's measurement scope rather than a queried row.
TABLE_SCOPE_TERMS = {
    "截至当期",
    "本年累计",
    "账面余额",
    "规模占比",
    "同比增长",
    "月末",
    "期末",
    "期初",
    "当月",
    "累计",
    "本月",
    "本期",
}


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
    question_mode: str = "open"

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "original_question": self.original_question,
            "planner": self.planner,
            "fallback_used": self.fallback_used,
            "error": self.error,
            "budget": self.budget or {},
            "question_mode": self.question_mode,
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
    question_mode = "choice" if any(str(option or "").strip() for option in options or []) else "open"
    if not cleaned_question:
        return QueryPlan(original_question="", aspects=(), planner="empty", fallback_used=True, question_mode=question_mode)

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
            question_mode=question_mode,
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
            question_mode=question_mode,
        )

    anchored_aspects = _explicit_document_anchor_aspects(
        cleaned_question,
        max_aspects=budget.system_max_aspects,
    )
    if anchored_aspects:
        return QueryPlan(
            original_question=cleaned_question,
            aspects=anchored_aspects,
            planner="deterministic-document-anchor",
            fallback_used=True,
            budget=budget.to_debug_dict(),
            question_mode=question_mode,
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
                        question_mode=question_mode,
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
                question_mode=question_mode,
            )

    return QueryPlan(
        original_question=cleaned_question,
        aspects=tuple(_fallback_aspects(cleaned_question, budget)),
        planner="fallback",
        fallback_used=True,
        budget=budget.to_debug_dict(),
        question_mode=question_mode,
    )


def plan_query_budget(question: str) -> QueryBudget:
    explicit_items = _explicit_requested_items(question)
    detected_items = tuple(_dedupe(explicit_items))
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
        document_titles = _dedupe(re.findall(r"《([^》]{2,120})》", aspect.question))
        if (
            len(document_titles) >= 2
            and remaining_capacity >= len(document_titles)
            and any(marker in aspect.question for marker in ("分别依据", "分别根据", "各自", "两份文件"))
        ):
            for index, title in enumerate(document_titles, start=1):
                split_question = f"请依据《{title}》相关条款，列出该文件的核心监管要求并明确来源。"
                split_filters = {
                    **aspect.table_filters,
                    "source_title": title,
                }
                result.append(
                    replace(
                        aspect,
                        aspect_id=f"{aspect.aspect_id}_document_{index}",
                        question=split_question,
                        search_queries=_fallback_search_queries(split_question),
                        evidence_need=f"《{title}》中可直接支持核心监管要求的原文",
                        keywords=tuple(_keywords_from_text(split_question)),
                        table_filters=split_filters,
                        explicit_filter_keys=tuple(
                            _dedupe([*aspect.explicit_filter_keys, "source_title"])
                        ),
                    )
                )
            continue
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
                    disable_thinking=QUERY_PLANNER_DISABLE_THINKING,
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
        if not keywords:
            keywords = _keywords_from_text(" ".join([sub_question, *[query.query for query in search_queries]]))
        planner_filters = _clean_table_filters(raw_aspect.get("table_filters"))
        explicit_filters = extract_explicit_metadata_filters(sub_question)
        # A table title, sheet and period stated in the user's question are
        # hard locators.  They must win over an LLM plan even when the plan
        # guessed a different period (for example confusing "3季度" with a
        # monthly report).  Keeping a guessed value would turn a valid lookup
        # into a false "period not found" refusal.
        # The original user question is more authoritative than an LLM's
        # decomposed sub-question, which may itself have introduced a period
        # or title not present in the request.
        explicit_filters = {**explicit_filters, **question_explicit_filters}
        original_titles = _dedupe(re.findall(r"《([^》]{2,120})》", question))
        sub_question_titles = _dedupe(re.findall(r"《([^》]{2,120})》", sub_question))
        if len(original_titles) >= 2 and len(sub_question_titles) == 1:
            sub_title = sub_question_titles[0]
            if sub_title in original_titles:
                # A decomposed cross-document aspect is narrower than the
                # original multi-title question.  Preserve that explicit title
                # instead of applying the first title globally to every aspect.
                explicit_filters["source_title"] = sub_title
        if "quarter" in explicit_filters and "month" not in explicit_filters:
            planner_filters.pop("month", None)
        if "month" in explicit_filters and "quarter" not in explicit_filters:
            planner_filters.pop("quarter", None)
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


def _fallback_aspects(question: str, budget: QueryBudget | None = None) -> list[QueryAspect]:
    budget = budget or plan_query_budget(question)
    table_aspect = _fallback_table_aspect(question)
    if table_aspect is not None:
        return [table_aspect]

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
    expanded = _expand_explicit_anchor_aspects(
        aspects,
        max_aspects=max(budget.max_aspects, budget.system_max_aspects),
    )
    return _split_merged_llm_aspects(expanded, budget)


def _explicit_document_anchor_aspects(
    question: str,
    *,
    max_aspects: int,
) -> tuple[QueryAspect, ...]:
    """Build one source-isolated aspect per explicit document/quoted anchor."""

    text = str(question or "")
    title_matches = list(re.finditer(r"《([^》]+)》", text))
    matches: list[tuple[str, str]] = []
    for title_index, title_match in enumerate(title_matches):
        end = title_matches[title_index + 1].start() if title_index + 1 < len(title_matches) else len(text)
        segment = text[title_match.end() : end]
        for anchor_match in re.finditer(
            r"[“‘\"']([^”’\"']+)[”’\"']",
            segment,
        ):
            matches.append((title_match.group(1), anchor_match.group(1)))
    if not matches:
        return ()
    aspects: list[QueryAspect] = []
    for index, (raw_title, raw_anchor) in enumerate(matches[:max_aspects], start=1):
        title = raw_title.strip()
        anchor = raw_anchor.strip()
        if not title or len(re.sub(r"\s+", "", anchor)) < 2:
            continue
        definition_request = any(
            marker in text
            for marker in ("完整定义", "定义是什么", "如何定义", "所称", "是指")
        ) and len(matches) == 1
        if definition_request:
            sub_question = f"根据《{title}》，“{anchor}”的完整定义是什么？"
            document_query = f"{title} {anchor} 是指 完整定义"
            evidence_need = f"《{title}》中“{anchor}”的完整定义原文"
        else:
            sub_question = (
                f"根据《{title}》，请说明与“{anchor}”直接对应的明确规定，"
                "并完整保留条件、范围、期限或例外。"
            )
            document_query = f"{title} {anchor}"
            evidence_need = f"《{title}》中与“{anchor}”直接对应的制度原文"
        filters = {"source_title": title, "indicator": anchor}
        requested_format = re.search(
            r"[（(]\s*(pdf|word|docx?|excel|xlsx?|xls)\s*[）)]\s*$",
            title,
            flags=re.IGNORECASE,
        )
        if requested_format:
            filters["file_type"] = {
                "word": "docx",
                "excel": "xlsx",
            }.get(requested_format.group(1).lower(), requested_format.group(1).lower())
        aspects.append(
            QueryAspect(
                aspect_id=f"document_anchor_{index}",
                question=sub_question,
                search_queries=(
                    QuerySearchQuery(sub_question, "semantic_question", "指定文档与引文联合检索"),
                    QuerySearchQuery(
                        document_query,
                        "document_style_statement",
                        "以标题和条款锚点定位原文",
                    ),
                    QuerySearchQuery(anchor, "keyword_anchor", "引文术语兜底"),
                ),
                evidence_need=evidence_need,
                keywords=tuple(_dedupe([anchor, *_keywords_from_text(anchor)])),
                modality="text",
                table_filters=filters,
                explicit_filter_keys=tuple(filters),
            )
        )
    if len(aspects) >= 3:
        title_counts: dict[str, int] = {}
        for title, _anchor in matches:
            title_counts[title.strip()] = title_counts.get(title.strip(), 0) + 1
        repeated_title = next(
            (title for title, count in title_counts.items() if count > 1),
            None,
        )
        if repeated_title:
            # Preserve a bounded document-level context aspect in addition to
            # each explicit anchor. This retains the surrounding clause when a
            # single source contains multiple quoted requirements.
            aspects.append(
                QueryAspect(
                    aspect_id=f"document_anchor_context_{len(aspects) + 1}",
                    question=f"根据《{repeated_title}》整理用户列出的各项明确规定并分别引用原文。",
                    search_queries=_fallback_search_queries(
                        f"{repeated_title} 明确规定 条件 范围 期限 例外"
                    ),
                    evidence_need=f"《{repeated_title}》中用户明确列出的制度规定",
                    keywords=tuple(_keywords_from_text(repeated_title)),
                    modality="text",
                    table_filters={"source_title": repeated_title},
                    explicit_filter_keys=("source_title",),
                )
            )
    return tuple(aspects)


def _asks_ratio_calculation(question: str) -> bool:
    """Distinguish a ratio request from a lookup whose quoted column is a ratio."""

    intent_text = re.sub(r"《[^》]*》", "", str(question or ""))
    intent_text = re.sub(r"[“‘\"'][^”’\"']*[”’\"']", "", intent_text)
    compact = re.sub(r"\s+", "", intent_text)
    return bool(
        re.search(r"数值占.*?数值", compact)
        or re.search(r"占.*?(?:百分比|比例|比重|多少)", compact)
        or re.search(r"(?:比例|比重|百分比).*?(?:多少|是多少|计算)", compact)
        # “原保险保费收入是赔付支出的多少倍” — multiplier form of a ratio.
        or re.search(r"(?:是|为|相当于)[^，。；]{0,16}?(?:多少|几)倍", compact)
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
    """Bound option-derived queries without adding topic-specific facts."""

    return _dedupe(option_statements)[:MCQ_OPTION_SEARCH_QUERY_BUDGET]


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
    operation_intent = re.sub(r"《[^》]*》", "", normalized)
    operation_intent = re.sub(r"[“‘\"'][^”’\"']*[”’\"']", "", operation_intent)
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
    if any(term in operation_intent for term in ["最高", "最大", "最多"]):
        table_task = "compare"
        operation = "max"
    elif any(term in operation_intent for term in ["最低", "最小", "最少"]):
        table_task = "compare"
        operation = "min"
    elif any(term in operation_intent for term in ["比较", "对比"]):
        table_task = "compare"
        operation = "none"
    elif (
        any(term in operation_intent for term in ["差值", "差额", "相差", "变化", "增加", "减少"])
        # “A比B多|少” is the difference form; “占…比例…多少” (the 比 of 比例
        # plus the 多 of 多少) must not be read as a difference.
        or re.search(r"[\u4e00-\u9fffA-Za-z0-9_]+比(?!例|重|百分)[\u4e00-\u9fffA-Za-z0-9_]+(?:多|少)", operation_intent)
        or ("计算" in operation_intent and re.search(r"[\u4e00-\u9fffA-Za-z0-9_]+[－—-][\u4e00-\u9fffA-Za-z0-9_]+", operation_intent))
    ):
        table_task = "calculate"
        operation = "difference"
    elif explicit_table and _asks_ratio_calculation(question):
        table_task = "calculate"
        operation = "ratio"
    elif any(term in operation_intent for term in ["求和", "总和", "加总", "之和", "分项之和", "分项合计"]):
        table_task = "calculate"
        operation = "sum"

    mixed_markers = ["根据监管制度", "结合监管制度", "根据制度", "结合制度", "根据自制制度", "结合自制制度", "填报说明依据", "制度依据", "监管规则", "制度侧", "制度文件", "联合核验"]
    modality = "mixed" if any(term in normalized for term in mixed_markers) else "table"
    table_question = _mixed_table_question_part(question) if modality == "mixed" else question
    filters = _extract_table_filters(table_question)
    selectors = _extract_table_selectors(table_question, table_task, options or [])
    if (
        filters.get("source_title")
        and filters.get("row_label")
        and filters.get("column_label")
    ):
        # A title + row + column is a complete cell locator. Period tokens may
        # legitimately live inside a hierarchical row/column label and must not
        # become independent hard filters against sparse legacy metadata.
        source_title = str(filters.get("source_title") or "")
        period_tokens = {
            "year": (f"{filters.get('year')}年",),
            "month": (f"{filters.get('month')}月",),
            "quarter": (
                f"{filters.get('quarter')}季度",
                f"{'一二三四'[int(filters['quarter']) - 1]}季度"
                if str(filters.get("quarter") or "").isdigit()
                and 1 <= int(filters["quarter"]) <= 4
                else "",
            ),
        }
        for period_key, tokens in period_tokens.items():
            if not any(token and token in source_title for token in tokens):
                filters.pop(period_key, None)
    if table_task == "calculate" and selectors:
        # Calculation operands carry their own columns.  A global column filter
        # would discard all but one operand before the executor can calculate.
        filters.pop("column_label", None)
    if table_task in {"compare", "calculate"} and len(selectors) >= 2:
        selector_columns = [
            str(selector.get("column_label") or "")
            for selector in selectors
            if selector.get("column_label")
        ]
        if len(set(selector_columns)) >= 2 and all(
            re.search(r"(?:\d{1,2}月|[一二三四1-4]季度|[1-4]季)", column)
            for column in selector_columns
        ):
            # Period labels embedded in distinct operand columns are selector
            # constraints, not one global period hard filter.
            filters.pop("month", None)
            filters.pop("quarter", None)
    selector_columns = [
        str(selector.get("column_label") or "")
        for selector in selectors
        if selector.get("column_label")
    ]
    if (
        filters.get("source_title")
        and selector_columns
        and all(
            re.search(r"(?:\d{1,2}月|[一二三四1-4]季度|[1-4]季)", column)
            for column in selector_columns
        )
    ):
        # An explicitly named report plus a period-bearing column is already a
        # complete locator.  Legacy indexes do not always duplicate that column
        # period into cell-level month/quarter metadata, so it must not become a
        # second hard filter.
        filters.pop("month", None)
        filters.pop("quarter", None)
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
    if table_task == "calculate" and selectors:
        # Calculation operands carry their own columns; a global column filter
        # would discard all but one operand before the executor can calculate.
        filters.pop("column_label", None)
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
    named = re.search(
        r"(?:先)?依据《([^》]+)》(.+?)(?=[，,；;。]\s*再从《)",
        table_aspect.question,
        flags=re.DOTALL,
    )
    regulation_question = (
        labelled.group(1).strip()
        if labelled
        else unlabelled.group(1).strip()
        if unlabelled
        else f"根据《{named.group(1).strip()}》{named.group(2).strip()}"
        if named
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
    if labelled or unlabelled or named:
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
        anchor = regulation_question if labelled or unlabelled or named else indicator
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
    row_match = re.search(r"行项目\s*[“\"‘']([^”\"’']+)", question)
    row_label = row_match.group(1).strip() if row_match else ""
    column_labels = _dedupe(
        [
            value.strip()
            for value in re.findall(r"(?:在)?列\s*[“\"‘']([^”\"’']+)", question)
            if value.strip()
        ]
    )
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
    if table_task == "calculate" and row_label and len(column_labels) >= 2:
        # Explicit row/column grammar is unambiguous and must take precedence
        # over generic quoted-term and period extraction.  Sheet names are also
        # quoted, but are never calculation operands.
        selectors = [
            {"row_label": row_label, "column_label": column_labels[0]},
            {"row_label": row_label, "column_label": column_labels[1]},
        ]
    elif table_task == "calculate" and row_label:
        ratio_columns = re.search(
            r"行项目\s*[“\"‘'][^”\"’']+[”\"’']的"
            r"\s*[“\"‘']([^”\"’']+)[”\"’']数值占"
            r"\s*[“\"‘']([^”\"’']+)[”\"’']数值",
            question,
        )
        if ratio_columns:
            selectors = [
                {"row_label": row_label, "column_label": ratio_columns.group(1).strip()},
                {"row_label": row_label, "column_label": ratio_columns.group(2).strip()},
            ]
    if selectors:
        pass
    elif len(period_selectors) >= 2:
        # Cross-period differences (“一季度与四季度…的差值”) follow the
        # question order (first − second): the period selectors are already in
        # text order, so no ordered-transition marker is applied.  Only the
        # explicit “从A到B” and “A比B多” forms mark a reversed direction.
        # Transition-amount phrases (“2月当月新增”, “12月值比10月值增加”)
        # name the period that owns the change; when that period is the
        # second selector, the difference must be evaluated second − first.
        selectors = period_selectors
        subject_index = _period_difference_subject_index(question, selectors)
        if subject_index == 1:
            selectors = [
                {**selector, "ordered_transition": True} for selector in selectors
            ]
    elif table_task == "calculate" and len(quoted) >= 3:
        # “row”从“column A”到“column B” means B - A.
        selectors = [
            {"row_label": quoted[0], "column_label": quoted[1], "ordered_transition": True},
            {"row_label": quoted[0], "column_label": quoted[2]},
        ]
    elif table_task == "calculate":
        normalized = re.sub(r"\s+", "", question)
        indicators = _mentioned_table_indicators(question)
        source_titles = [m.group(1).strip() for m in re.finditer(r"《([^》]+)》", question)][:2]
        if len(source_titles) >= 2 and indicators:
            # Cross-table questions (“《表1》和《表2》…的差值”) pin each operand
            # to its own quoted title.  The operand is usually the same metric
            # in both tables.
            metrics = (
                indicators[:2]
                if len(indicators) >= 2
                else [indicators[0], indicators[0]]
            )
            selectors = [
                {"row_or_indicator": metric, "source_title": title}
                for metric, title in zip(metrics, source_titles)
            ]
        elif any(term in normalized for term in ["分项之和", "分项合计", "四个分项"]):
            component_order = ["现金及存放同业", "贷款余额", "证券投资", "其他资产"]
            selectors = [{"row_or_indicator": item} for item in component_order]
        elif len(indicators) >= 2:
            # “占…比例/比重”与“…的多少倍”是 ratio 句式；其中的“比”字不能与
            # “A比B多|少”的差值句式混淆（“比例”的“比”+“多少”的“多”），
            # 故排除 例/重/百分 开头。
            diff_match = re.search(r"(.+?)比(?!例|重|百分)(.+?)(?:多|少)", normalized)
            if diff_match:
                # "A比B多" means A - B.  The selectors are deliberately reversed
                # (B first) and marked as an ordered transition, so the calculator
                # evaluates second - first.
                selectors = [
                    {"row_or_indicator": indicators[1], "ordered_transition": True},
                    {"row_or_indicator": indicators[0]},
                ]
            else:
                selectors = [{"row_or_indicator": item} for item in indicators[:2]]
        else:
            institution_terms = _institution_row_terms(question, indicators[0] if indicators else "")
            if len(institution_terms) >= 2:
                # “人身险公司与财产险公司资金运用余额的差值”：两个机构词各自
                # 限定同一指标的一行（问题顺序）。
                selectors = [
                    {"row_or_indicator": f"{term} {indicators[0]}"}
                    for term in institution_terms[:2]
                ]
        if selectors and len(source_titles) < 2 and indicators:
            # 单表差值/比例：操作数都限定在同一机构行（“财产险公司 银行存款”
            # 与“财产险公司 资金运用余额”），防止分母落到无机构前缀的合计行。
            # 多机构词场景（“人身险公司与财产险公司…的差值”）的操作数已由
            # 上面的分支各自带好机构词，不再重复并入。
            institution_terms = _institution_row_terms(question, indicators[0])
            if len(institution_terms) == 1:
                term = institution_terms[0]
                selectors = [
                    (
                        {**sel, "row_or_indicator": f"{term} {sel['row_or_indicator']}"}
                        if "row_or_indicator" in sel and term not in str(sel["row_or_indicator"])
                        else sel
                    )
                    for sel in selectors
                ]
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
        if row_label and column_labels:
            selectors = [{
                "row_label": row_label,
                "column_label": column_labels[0],
            }]
        else:
            lookup_match = re.search(r"请给出\s*[“\"]([^”\"]+)[”\"]\s*在\s*[“\"]([^”\"]+)[”\"]\s*列", question)
        if not selectors and lookup_match:
            selectors = [{
                "row_or_indicator": lookup_match.group(1).strip(),
                "column_label": lookup_match.group(2).strip(),
            }]
        elif not selectors:
            sheet_match = re.search(r"工作表\s*[“\"‘']([^”\"’']+)", question)
            sheet_name = sheet_match.group(1).strip() if sheet_match else ""
            data_terms = [term for term in quoted if term != sheet_name]
            if len(data_terms) >= 2:
                selectors = [{"row_or_indicator": data_terms[0], "column_label": data_terms[-1]}]
            elif data_terms:
                selectors = [{"row_or_indicator": data_terms[0]}]
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
    # A year inside a book title (“《2025年…表》…一季度…”) is not adjacent to
    # the quarter words, so a per-match year prefix never fires.  Fall back to
    # the first year mentioned anywhere in the question.
    global_year_match = re.search(r"(20\d{2})年", compact)
    global_year = int(global_year_match.group(1)) if global_year_match else None
    current_year = None
    for match in re.finditer(r"(?:(20\d{2})年)?([一二三四1-4])季度", compact):
        if match.group(1):
            current_year = int(match.group(1))
        elif current_year is None and global_year is not None:
            current_year = global_year
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


_PERIOD_TRANSITION_TERMS = ("当月新增", "本月新增", "新增加", "环比", "较上月", "新增", "增加", "增长")


def _period_difference_subject_index(
    question: str,
    selectors: list[dict[str, Any]],
) -> int | None:
    """Selector index of the period a transition amount belongs to, if any.

    Cross-period differences default to text order (first − second).  A
    transition phrase (“2月当月新增”, “12月值比10月值增加”, “环比”) states
    how much the subject period gained over the base period, which is not
    necessarily the first selector — monthly tables are usually named in
    ascending order (“《1月表》和《2月表》…当月新增” still asks for
    2月 − 1月).  Returns the minuend selector index, or None when the
    question is not a transition-amount difference (plain “差值/差额”).
    """
    compact = re.sub(r"\s+", "", str(question or ""))
    if len(selectors) != 2:
        return None
    # “同比/增速/增长率” name column labels, not a period transition.
    if "同比" in compact or "增速" in compact or "增长率" in compact:
        return None
    if not any(term in compact for term in _PERIOD_TRANSITION_TERMS):
        return None

    def _subject_month() -> int | None:
        # “12月值比10月值增加” / “2月原保险保费收入较1月增加”：变化主体是
        # “较/比”之前的月份（X − Y）。主体与“较/比”之间可间隔指标名称。
        match = re.search(
            r"(\d{1,2})月(?:值)?(?:比|较)(\d{1,2})月(?:值)?(?:增加|增长|新增)",
            compact,
        )
        if match:
            return int(match.group(1))
        match = re.search(
            r"(\d{1,2})月(?:值)?[^，。；;]{0,20}?(?:较|比)(\d{1,2})月(?:值)?(?:增加|增长|新增)",
            compact,
        )
        if match:
            return int(match.group(1))
        match = re.search(r"(\d{1,2})月(?:当月|本月)?(?:新增|新增加|增加|增长)", compact)
        if match:
            return int(match.group(1))
        return None

    subject_month = _subject_month()
    if subject_month is not None:
        for index, selector in enumerate(selectors):
            if selector.get("month") == subject_month:
                return index
        return None
    # “当月新增/本月新增/环比/较上月/从A到B” without an explicit subject
    # period: the change belongs to the later period.
    def _period_key(selector: dict[str, Any]) -> tuple[int, int, int]:
        year = int(selector.get("year") or 0)
        if "month" in selector:
            return (year, 0, int(selector["month"]))
        if "quarter" in selector:
            return (year, 1, int(selector["quarter"]))
        return (year, 2, 0)

    if _period_key(selectors[0]) < _period_key(selectors[1]):
        return 1
    return None


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
    work_sheet_position = question.find("工作表")
    if work_sheet_position >= 0:
        preceding_titles = list(re.finditer(r"《([^》]+)》", question[:work_sheet_position]))
        if preceding_titles:
            # In a joint policy/report request, the report is the last named
            # document immediately before the worksheet locator.
            title_match = preceding_titles[-1]
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
    # Cross-table questions (“《表1》和《表2》…的差值”) carry one concrete
    # title per operand.  Keep up to two quoted titles so the selectors can
    # pin each operand to its own table; a single title keeps the historical
    # single-source behaviour.
    all_titles = [m.group(1).strip() for m in re.finditer(r"《([^》]+)》", question)]
    if len(all_titles) >= 2:
        filters["source_titles"] = all_titles[:2]
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
    # Prefer the bounded quote form.  It remains stable when the sheet name is
    # followed by a Chinese closing quote and then ordinary prose (the common
    # ``工作表“Sheet1”，请…`` form).
    sheet_match = re.search(r"工作表\s*[“\"‘']([^”\"’']+)", question)
    if sheet_match is None:
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
    explicit_row = re.search(r"行项目\s*[“\"‘']([^”\"’']+)", question)
    explicit_columns = [
        value.strip()
        for value in re.findall(r"(?:在)?列\s*[“\"‘']([^”\"’']+)", question)
        if value.strip()
    ]
    lookup_match = re.search(r"请给出\s*[“\"]([^”\"]+)[”\"]\s*在\s*[“\"]([^”\"]+)[”\"]\s*列", question)
    if lookup_match:
        # This is an explicit row/column lookup syntax, so the preceding
        # quoted sheet name must not be mistaken for the requested indicator.
        filters["indicator"] = lookup_match.group(1).strip()
        filters["row_label"] = lookup_match.group(1).strip()
        filters["column_label"] = lookup_match.group(2).strip()
    elif explicit_row:
        filters["indicator"] = explicit_row.group(1).strip()
        filters["row_label"] = explicit_row.group(1).strip()
        if explicit_columns:
            filters["column_label"] = explicit_columns[-1]
    elif quoted_terms:
        sheet_name = str(filters.get("sheet") or "").strip()
        # Quoted measurement-scope phrases ("本年累计", "截至当期", "账面余额")
        # describe the column, not a queried row, so they must not become the
        # indicator.  Only the remaining quoted terms name the requested row.
        non_scope_terms = [
            term
            for term in quoted_terms
            if term.strip() != sheet_name and not any(scope in term for scope in TABLE_SCOPE_TERMS)
        ]
        if non_scope_terms:
            filters["indicator"] = non_scope_terms[0]
            if len(non_scope_terms) > 1:
                filters["column_label"] = non_scope_terms[-1]
    indicators = _mentioned_table_indicators(question)
    if indicators and "indicator" not in filters:
        filters["indicator"] = indicators[0]
        # The row label stays the bare indicator: composite corpus rows carry
        # the institution as a separate segment ("其中：财产险公司 / 其中：
        # 银行存款"), which is not a contiguous substring, and the institution
        # qualifier is already rewarded by the token-overlap scoring.
        filters["row_label"] = indicators[0]
    scope_text = re.split(r"(?:请比较|请对比|比较|对比)", question, maxsplit=1)[0]
    # A position word right after the title (“《…情况表》中保险公司…”) belongs
    # to the title fragment, not to the institution name.
    scope_text = re.sub(r"《[^》]+》[中里内下于的]?", "", scope_text)
    if explicit_row is None:
        matched_label = False
        for label in ["全国合计", "全  国", "北京", "上海", "江苏", "浙江", "广东", "大型商业银行", "城市商业银行", "农村商业银行", "股份制商业银行", "民营银行", "外资银行"]:
            if label.replace(" ", "") in re.sub(r"\s+", "", scope_text):
                filters["row_label"] = label
                matched_label = True
                break
        if not matched_label:
            # Generic institution row-locator words (X公司/X银行 forms such as
            # 财产险公司/人身险公司/保险公司, plus named aggregate rows).  The
            # term must co-occur with the indicator in the same clause so a
            # title-embedded institution word ("…保险公司资金运用情况表中…")
            # is never mistaken for a row locator.  Words are joined with a
            # space; the spreadsheet matcher requires every word to be covered
            # by the candidate row's label (AND semantics).
            institution_terms = _institution_row_terms(
                scope_text,
                str(filters.get("indicator") or ""),
            )
            if institution_terms:
                current_row_label = str(filters.get("row_label") or "").strip()
                filters["row_label"] = f"{institution_terms[0]} {current_row_label}".strip()
    row_scoped_phrase = False
    # Quoted column scope first: the quotes are an exact boundary, so
    # '截至当期-账面余额'口径下 extracts cleanly while an institution term in
    # front of the quote ("财产险公司资金运用余额在'截至当期-账面余额'口径下")
    # is never swallowed into the column label.  Handles ASCII single quotes,
    # which _quoted_terms does not.
    quoted_scope = re.search(r"[‘“\"']([^’”\"']{1,24})[’”\"']?(?:列|口径)下", question)
    if quoted_scope:
        value = quoted_scope.group(1).strip("的在按以")
        row_scoped_phrase = value == str(filters.get("row_label") or "").strip()
        if value and not row_scoped_phrase and "表" not in value:
            filters["column_label"] = value
    else:
        focused_scope = re.search(r"》的?([^，。；：:]{2,24})口径下", question)
        if focused_scope is None:
            focused_scope = re.search(r"表的?([^，。；：:]{2,24})口径下", question)
        if focused_scope:
            value = focused_scope.group(1).strip("的在按以")
            row_scoped_phrase = value == str(filters.get("row_label") or "").strip()
            if (
                value
                and not row_scoped_phrase
                and "表" not in value
                and not re.search(r"(?:公司|银行)", value)
            ):
                filters["column_label"] = value
    column_scope = re.search(r"[‘“\"]?([^，。；：:‘“’”\"]{2,24})[’”\"]?(?:列|口径)下?(?:中|比较|取数|查询|读取)?", question)
    if column_scope and "column_label" not in filters and not row_scoped_phrase:
        value = column_scope.group(1).split("》")[-1].strip("的在按以'‘’\"”")
        if (
            value
            and value != str(filters.get("row_label") or "").strip()
            and not any(marker in value for marker in ("情况表", "统计表", "工作表"))
            and not re.search(r"(?:公司|银行)", value)
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
    named_report = re.search(
        r"(再从《[^》]+》工作表.+?)(?=(?:[。；;]\s*(?:最后|并说明|不得|行业汇总|请分别))|$)",
        question,
        flags=re.DOTALL,
    )
    if named_report:
        return named_report.group(1).strip()
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
    # Substring overlap: “不良贷款余额” also contains “贷款余额”.  A longer
    # indicator already covers its own shorter substring, so drop any match
    # that is a substring of an earlier (longer) one to avoid duplicated
    # operands such as [不良贷款余额, 贷款余额].
    filtered: list[str] = []
    for match in matches:
        if any(match in existing and match != existing for existing in filtered):
            continue
        filtered.append(match)
    return filtered


def _institution_row_terms(text: str, indicator: str) -> list[str]:
    """Extract generic institution/category terms usable as row-location words.

    Matches 公司/银行-suffixed institution names (财产险公司/人身险公司/
    保险公司/大型商业银行…) and the corpus's named aggregate rows
    (保险业合计/全国合计).  ``text`` must already have book titles removed;
    an indicator is required (an institution term only locates a row together
    with an indicator — compare/max-style questions never name a row here),
    and the term must co-occur with the indicator in the same clause, so a
    title-embedded institution word inside a natural title ("…保险公司资金
    运用情况表中…") is never treated as a row locator.
    """

    compact = re.sub(r"\s+", "", str(text or ""))
    # Drop natural-title fragments ("…保险公司资金运用情况表中…") so their
    # institution words are never treated as row locators; 中/里/内/下 are the
    # position words that follow a title ("…情况表中").
    compact = re.sub(
        r"[\u4e00-\u9fff]{0,12}?(?:情况表|统计表|经营表)(?:中|里|内|下)?",
        "",
        compact,
    )
    # A sheet name ("工作表：人身保险公司（月度）") is its own filter and its
    # institution words locate the sheet, not a row — drop it before matching.
    compact = re.sub(r"工作表\s*[：:]\s*[^，。；）)]+[）)]?", "", compact)
    # An institution term only locates a row together with an indicator
    # ("财产险公司 资金运用余额").  Without an indicator (compare/max-style
    # questions) the term would constrain rows the question never names.
    if not indicator:
        return []
    candidates: list[str] = []
    for match in re.finditer(r"(?:其中：)?[\u4e00-\u9fff]{2,6}?(?:公司|银行)", compact):
        # 剥掉“其中：”与并列连词（“人身险公司与财产险公司”中的“与”属于
        # 句法连接，不是机构名的一部分）。
        word = re.sub(r"^(?:其中：|以及|与|和|及|或)+", "", match.group(0))
        # 指标词残片可能被吞进机构词（“保险公司资金运用余额与财产险公司”
        # 中匹配到“余额与财产险公司”）：取最后一个连词之后的部分，把
        # “余额与财产险公司”还原为“财产险公司”。机构名称本身不含连词。
        tail = re.split(r"(?:其中：|以及|与|和|及|或)", word)[-1]
        if tail:
            word = tail
        # 标题/时间粘连（如 "…情况表中保险公司…"）可能把位置/时间词吞进机构名；
        # 剥掉通用前缀。 "中国" 是机构名合法前缀（中国银行/中国人寿），保留。
        word = re.sub(
            r"^(?!中国)(?:[\d一二三四五六七八九十]{1,2}|年|月|季度|上|下|当|本|今|去|各|全|其|中|里|内|于|的|在|情况|统计|经营)+",
            "",
            word,
        )
        if word and word not in candidates:
            candidates.append(word)
    for label in ("保险业合计", "全国合计"):
        if label in compact and label not in candidates:
            candidates.append(label)
    # 机构词必须与指标词同分句且紧邻（指标词起点在机构词后 12 字符内），
    # 否则标题粘连的机构词（"…保险公司资金运用情况表中…"）会被误当行定位词。
    segments = re.split(r"[，。；;、？?]", compact)
    same_clause: list[str] = []
    for term in candidates:
        for segment in segments:
            if indicator in segment and term in segment:
                position = segment.find(term)
                after = segment[position + len(term):]
                if indicator in after and after.find(indicator) <= 12:
                    same_clause.append(term)
                break
    return same_clause


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
    """Create a query-shaped alias only from the user's own wording.

    This transformation intentionally contains no regulatory conclusions,
    thresholds, document titles, or benchmark-specific vocabulary.
    """

    text = re.sub(r"\s+", " ", str(question or "")).strip()
    text = re.sub(r"[\uFF1F?\u3002\uFF1B;\uFF0C,\uFF1A:]+", " ", text)
    text = re.sub(
        r"(?:\u8BF7\u95EE|\u8BF7\u8BF4\u660E|\u8BF7\u89E3\u91CA|"
        r"\u662F\u4EC0\u4E48|\u6709\u54EA\u4E9B|\u5982\u4F55|"
        r"\u4E3A\u4F55|\u591A\u5C11)$",
        "",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


def _mixed_regulation_document_style_alias(regulation_question: str, anchor_question: str) -> str:
    del regulation_question
    return _document_style_alias(anchor_question)


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
    tokens = re.findall(r"[A-Za-z0-9_./%-]{2,}|[\u4e00-\u9fff]{2,12}", str(text or ""))
    stopwords = {"请问", "请说明", "请解释", "是什么", "有哪些", "如何", "为何"}
    return [token for token in _dedupe(tokens) if token not in stopwords][:8]


def _infer_evidence_type(text: str) -> str:
    normalized = re.sub(r"\s+", "", text)
    if any(term in normalized for term in ["保留", "依据", "留痕", "材料"]):
        return "留痕依据或支持材料"
    if any(term in normalized for term in ["怎么", "如何", "处理", "排查", "检查"]):
        return "处理流程或排查要求"
    if any(term in normalized for term in ["哪些", "情形", "条件", "口径"]):
        return "分类条件或填报口径"
    return "相关制度依据"
