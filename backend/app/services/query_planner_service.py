import json
import re
import urllib.error
import urllib.request
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any

from backend.app.core.config import (
    QA_REQUEST_TOTAL_BUDGET_SECONDS,
    QUERY_PLANNER_API_KEY,
    QUERY_PLANNER_BASE_URL,
    QUERY_PLANNER_DISABLE_THINKING,
    QUERY_PLANNER_ENABLED,
    QUERY_PLANNER_MAX_ASPECTS,
    QUERY_PLANNER_MAX_SEARCH_QUERIES,
    QUERY_PLANNER_MODEL,
    QUERY_PLANNER_PROVIDER,
    QUERY_PLANNER_RESPONSE_FORMAT,
)
from backend.app.schemas.qa import EvidenceRequirement
from backend.app.services.clarification_text import (
    DEFAULT_CLARIFICATION_TEXT,
    normalize_clarification_text,
)
from backend.app.services.evidence_contract_service import (
    document_style_phrases,
    requirements_for_plan,
    target_terms,
)
from backend.app.services.llm_client import (
    ChatCompletionConfig,
    ChatCompletionError,
    chat_completion_content,
)
from backend.app.services.performance_metrics import measure, timed


class QueryPlannerError(RuntimeError):
    pass


class QueryNeedsClarification(QueryPlannerError):
    pass


VALID_QUERY_TYPES = {
    "semantic_question",
    "document_style_statement",
    "keyword_anchor",
    "table_locator",
    "legacy",
    "fallback",
}
MCQ_OPTION_SEARCH_QUERY_BUDGET = 6
TABLE_MARKERS = [
    "Excel",
    "excel",
    "工作表",
    "单元格",
    "情况表",
    "统计表",
    "经营表",
    "月报",
    "报表",
    "CSV",
    "csv",
]
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

_TABLE_VOCABULARY: ContextVar[tuple[str, ...]] = ContextVar(
    "table_vocabulary", default=()
)
_TABLE_SOURCES: ContextVar[tuple[str, ...]] = ContextVar("table_sources", default=())
_TABLE_ROWS: ContextVar[tuple[str, ...]] = ContextVar("table_rows", default=())
_TABLE_COLUMNS: ContextVar[tuple[str, ...]] = ContextVar("table_columns", default=())


@contextmanager
def table_schema_context(db, question: str):
    """Read actual indexed labels once, instead of maintaining indicator lists."""
    from sqlalchemy import select

    from backend.app.models.document import SpreadsheetCell

    labels = ()
    sources = ()
    row_labels = ()
    column_labels = ()
    if _has_explicit_table_marker(question) or re.search(r"20\d{2}年", question):
        from backend.app.models.document import Document
        from backend.app.services.evidence_contract_service import title_key

        documents = db.execute(
            select(Document.document_id, Document.filename).where(
                Document.document_id.in_(select(SpreadsheetCell.document_id).distinct())
            )
        ).all()
        sources = tuple(row[1] for row in documents)
        statement = select(
            SpreadsheetCell.row_label, SpreadsheetCell.column_label
        ).distinct()
        titles = re.findall(r"《([^》]+)》", question)
        if titles:
            document_ids = [
                doc_id
                for doc_id, filename in documents
                if any(title_key(title) in title_key(filename) for title in titles)
            ]
            statement = statement.where(SpreadsheetCell.document_id.in_(document_ids))
        values = db.execute(statement).all()

        def axis_parts(value):
            return [
                re.sub(
                    r"^(?:其中\s*[:：]?|[0-9一二三四五六七八九十]+[、.．])\s*",
                    "",
                    part.strip(),
                )
                for part in re.split(r"[/／]", str(value or ""))
                if 2 <= len(part.strip()) <= 80
            ]

        row_labels = tuple(
            dict.fromkeys(part for row in values for part in axis_parts(row[0]))
        )
        column_labels = tuple(
            dict.fromkeys(part for row in values for part in axis_parts(row[1]))
        )
        labels = tuple(dict.fromkeys([*row_labels, *column_labels]))
    token = _TABLE_VOCABULARY.set(labels)
    source_token = _TABLE_SOURCES.set(sources)
    row_token = _TABLE_ROWS.set(row_labels)
    column_token = _TABLE_COLUMNS.set(column_labels)
    try:
        yield
    finally:
        _TABLE_VOCABULARY.reset(token)
        _TABLE_SOURCES.reset(source_token)
        _TABLE_ROWS.reset(row_token)
        _TABLE_COLUMNS.reset(column_token)


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
    subject_scope: tuple[str, ...] = ()
    mandatory: bool = True
    # User-facing answer obligation.  Retrieval/generation may use a rewritten
    # aspect question, but validation must not demand entities or sub-details
    # the user never asked for.
    requirement_question: str | None = None
    intent: str = field(default="", init=False)
    targets: tuple[str, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        if not self.intent:
            intent_text = re.sub(r"《[^》]+》", "", self.question)
            intent = (
                "calculate"
                if self.operation != "none"
                else "define"
                if re.search(r"定义|是什么|含义|界定", intent_text)
                else "list"
                if re.search(r"哪些|列出|包括", intent_text)
                else "lookup"
            )
            object.__setattr__(self, "intent", intent)
        if not self.targets:
            quoted = re.findall(r"[“\"‘\']([^”\"’\']{2,80})[”\"’\']", self.question)
            target_values = list(quoted or target_terms(self.question))
            if not quoted:
                plain_question = re.sub(r"《[^》]+》", " ", self.question)
                target_values.extend(
                    re.findall(
                        r"[\u4e00-\u9fff]{2,10}(?:策略|原则|主体|对象|方式|类型|方面|条件|范围|期限|"
                        r"标准|程序|职责|内容|信息|数据|机构|公司|工具|资本|风险|比例|限额|情形|类别|档次|版本|标识)",
                        plain_question,
                    )
                )
            if self.intent == "define":
                # “《制度》中‘章节’的内容，风险价值应如何定义？” asks for the
                # definition of 风险价值, not of the section label.  Prefer the
                # explicit defined term over the quoted location.
                plain = re.sub(r"《[^》]+》", " ", self.question)
                defined = re.search(
                    r'[“"]?([^“”"，。；：：？?]{2,30}?)[”"]?\s*(?:应|应当)?'
                    r"(?:如何|怎样|怎么)\s*(?:定义|界定|理解)",
                    plain,
                )
                if defined:
                    term = defined.group(1).strip(" 的")
                    term = re.sub(r"[（(][^）)]*[）)]", "", term).strip()
                    if 2 <= len(term) <= 30 and not re.search(
                        r"规定|要求|内容|方面|情况|相关|前述|上述|该|其", term
                    ):
                        target_values = [term]
            object.__setattr__(
                self, "targets", tuple(dict.fromkeys(target_values))[:24]
            )

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
            "subject_scope": list(self.subject_scope),
            "mandatory": self.mandatory,
            "requirement_question": self.requirement_question,
            "intent": self.intent,
            "targets": list(self.targets),
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
    requirements: tuple[EvidenceRequirement, ...] = ()
    clarification_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.requirements:
            object.__setattr__(self, "requirements", tuple(requirements_for_plan(self)))

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "original_question": self.original_question,
            "planner": self.planner,
            "fallback_used": self.fallback_used,
            "error": self.error,
            "budget": self.budget or {},
            "question_mode": self.question_mode,
            "aspects": [aspect.to_debug_dict() for aspect in self.aspects],
            "requirements": [item.model_dump() for item in self.requirements],
            "clarification_reason": self.clarification_reason,
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
def _scope_text(value: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]", "", str(value or "")).casefold()


def _scope_overlap(left: str, right: str) -> float:
    left_text = _scope_text(left)
    right_text = _scope_text(right)
    if not left_text or not right_text:
        return 0.0
    left_grams = {
        left_text[index : index + 2] for index in range(max(len(left_text) - 1, 1))
    }
    right_grams = {
        right_text[index : index + 2] for index in range(max(len(right_text) - 1, 1))
    }
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / len(left_grams | right_grams)


def _scope_focus(question: str) -> str:
    text = re.sub(r"\s+", "", str(question or ""))
    after = re.search(r"(?:有什么|有哪些|哪些)(.+?)(?:[？?]|$)", text)
    if after:
        return after.group(1)
    before = re.search(r"(.+?)(?:是什么|如何|怎样|怎么)(?:[？?]|$)", text)
    return before.group(1) if before else ""


def _scope_focus_coverage(focus: str, aspect: str) -> float:
    focus_chars = set(_scope_text(focus))
    if not focus_chars:
        return 0.0
    return len(focus_chars & set(_scope_text(aspect))) / len(focus_chars)


def _scope_priority(aspect: QueryAspect, question: str, index: int) -> float:
    aspect_text = re.sub(r"\s+", "", aspect.question)
    score = _scope_overlap(aspect.question, question)
    focus = _scope_focus(question)
    if focus and _scope_focus_coverage(focus, aspect_text) >= 0.75:
        score += 0.2
    if "哪些" in question and re.search(r"哪些|清单|列出|包括", aspect_text):
        score += 0.15
    if "哪些" in question and re.match(
        r"^(?:集团内|特定类型|除|门槛|其中|此外|同时|资金|规模|交易对手)", aspect_text
    ):
        score -= 0.15
    if index == 0:
        score += 0.1
    return score


def _mark_aspect_scope(
    question: str, aspects: list[QueryAspect], max_mandatory: int
) -> list[QueryAspect]:
    """Keep user-requested aspects mandatory and demote planner expansions.

    The planner may add useful research aspects (approval process, retention,
    exceptions, reporting).  Those can improve recall but must not become
    answer obligations unless the user asked for them or the planner budget
    explicitly detected a multi-part question.
    """

    if not aspects:
        return []
    limit = max(1, min(int(max_mandatory or 1), len(aspects)))
    if len(aspects) <= limit:
        return [replace(aspect, mandatory=True) for aspect in aspects]
    titles = list(dict.fromkeys(re.findall(r"《([^》]+)》", question)))
    # Source names may also be written as plain document-style phrases (for
    # example "商业银行版恢复计划示例").  They are used only to keep one
    # source-isolated aspect per named source; they never filter evidence.
    source_names = _source_names_in_question(question)
    if len(titles) > 1 or len(source_names) > 1:
        limit = max(limit, len(source_names), len(titles))
    parts = _split_question_parts(question)
    scores = [
        _scope_priority(aspect, question, index) for index, aspect in enumerate(aspects)
    ]
    mandatory: set[int] = set()
    if len(titles) > 1:
        for index, aspect in enumerate(aspects):
            if any(title in aspect.question for title in titles):
                mandatory.add(index)
    if len(source_names) > 1:
        for index, aspect in enumerate(aspects):
            if _source_match_count(aspect, source_names) == 1:
                mandatory.add(index)
    if len(parts) > 1:
        for part in parts:
            scores_for_part = [
                _scope_overlap(aspect.question, part) for aspect in aspects
            ]
            best = max(range(len(aspects)), key=lambda index: scores_for_part[index])
            if scores_for_part[best] >= 0.15:
                mandatory.add(best)
    for index in sorted(
        range(len(aspects)), key=lambda item: scores[item], reverse=True
    ):
        if len(mandatory) >= limit:
            break
        mandatory.add(index)
    if not mandatory:
        mandatory.add(0)
    return [
        replace(aspect, mandatory=index in mandatory)
        for index, aspect in enumerate(aspects)
    ]


def _source_names_in_question(question: str) -> list[str]:
    text = str(question or "")
    names = [title.strip() for title in re.findall(r"《([^》]{2,80})》", text)]
    stripped = re.sub(r"《[^》]*》", " ", text)
    for part in re.split(r"[和与及、，,。；;：:？?]", stripped):
        candidate = re.sub(
            r"^[ “”\"'‘’《》,，。；;：:]+|[ “”\"'‘’《》,，。；;：:]+$", "", part
        )
        if not candidate:
            continue
        # A source name is usually followed by a scope/predicate phrase such
        # as "中关于…" or "对…"; keep the name itself rather than the tail.
        candidate = re.split(
            r"(?:中对|中关于|中的|对|关于|针对|是否|比较|对比|分别|有何|差异)",
            candidate,
            maxsplit=1,
        )[0]
        candidate = re.sub(
            r"^(?:根据|依据|按照|从|在|关于|对于|针对|请|说明|比较|对比|分别)+",
            "",
            candidate,
        )
        candidate = candidate.strip(" 的了中里内")
        if 4 <= len(candidate) <= 80 and re.search(
            r"示例|办法|规则|通知|附件|制度|规定|指南|指引|规范|意见|条例|细则|版",
            candidate,
        ):
            names.append(candidate)
    return list(dict.fromkeys(name for name in names if name))


def _source_match_count(aspect: QueryAspect, source_names: list[str]) -> int:
    """How many user-named sources one aspect explicitly targets."""

    scope_sources = document_style_phrases(getattr(aspect, "subject_scope", ()))
    filter_source = str((aspect.table_filters or {}).get("source_title") or "")
    matched = {
        name
        for name in source_names
        if name
        and (name in aspect.question or (filter_source and name in filter_source))
    }
    if not matched and len(scope_sources) == 1:
        matched = {scope_sources[0]}
    return len(matched)


def _matched_source_names(aspect: QueryAspect, source_names: list[str]) -> list[str]:
    filter_source = str((aspect.table_filters or {}).get("source_title") or "")
    return [
        name
        for name in source_names
        if name
        and (name in aspect.question or (filter_source and name in filter_source))
    ]


def _named_source_match_count(aspect: QueryAspect, source_names: list[str]) -> int:
    return len(_matched_source_names(aspect, source_names))


def _target_key_set(aspect: QueryAspect) -> set[str]:
    return {
        re.sub(r"[\s“”\"'‘’《》，,。；;：:]+", "", str(term)).casefold()
        for term in (aspect.targets or ())
        if len(re.sub(r"\s+", "", str(term))) >= 2
    }


def _aspect_targets_overlap(left: QueryAspect, right: QueryAspect) -> float:
    left_keys = _target_key_set(left)
    right_keys = _target_key_set(right)
    if not left_keys or not right_keys:
        return 0.0
    return len(left_keys & right_keys) / min(len(left_keys), len(right_keys))


def _promote_cross_document_aspects(
    question: str, aspects: list[QueryAspect], question_mode: str
) -> list[QueryAspect]:
    """Keep each user-named source as its own evidence obligation.

    This is source-oriented, not sentence-oriented: any aspect that targets
    exactly one named source is required, while an umbrella "compare both"
    aspect is optional because it repeats the answer task rather than adding
    a separate piece of evidence.  Duplicate source aspects produced by a
    planner split are optional once every named source already has a real
    source-isolated aspect.  No punctuation or comparison phrase is required
    to trigger the rule.
    """

    if question_mode != "open" or not 2 <= len(aspects) <= 8:
        return aspects
    source_names = _source_names_in_question(question)
    if len(source_names) < 2:
        return aspects
    named_counts = {
        aspect.aspect_id: _named_source_match_count(aspect, source_names)
        for aspect in aspects
    }
    complete_named_aspects = [
        aspect for aspect in aspects if named_counts[aspect.aspect_id] == 1
    ]
    match_counts = [
        (aspect, _source_match_count(aspect, source_names)) for aspect in aspects
    ]
    source_aspects = [(aspect, count) for aspect, count in match_counts if count == 1]
    if len(source_aspects) < 2:
        return aspects
    use_named_only = len(complete_named_aspects) >= 2
    kept_by_source: dict[str, list[QueryAspect]] = {}
    promoted: list[QueryAspect] = []
    for aspect, count in match_counts:
        named_count = named_counts[aspect.aspect_id]
        if named_count == 1:
            matched = _matched_source_names(aspect, source_names)
            source_key = matched[0] if matched else aspect.aspect_id
            existing = kept_by_source.setdefault(source_key, [])
            if any(_aspect_targets_overlap(aspect, kept) >= 0.6 for kept in existing):
                # A planner split repeats the same source and the same target
                # property.  Drop the duplicate retrieval aspect instead of
                # spending prompt space on a second copy of the same evidence
                # obligation.  A different quoted condition on the same source
                # is not a duplicate.
                continue
            existing.append(aspect)
            promoted.append(replace(aspect, mandatory=True))
        elif count >= 2:
            # The umbrella comparison aspect repeats the answer task, not an
            # additional source obligation.
            promoted.append(replace(aspect, mandatory=False))
        elif use_named_only and count >= 1:
            # A scope-only duplicate must not add a second required obligation
            # after every named source is already covered.  Drop it only when
            # it repeats the same target property; a different property stays
            # available as optional context.
            if any(
                _aspect_targets_overlap(aspect, kept) >= 0.6
                for group in kept_by_source.values()
                for kept in group
            ):
                continue
            promoted.append(replace(aspect, mandatory=False))
        else:
            promoted.append(aspect)
    return promoted


def plan_query(question: str, options: list[str] | None = None) -> QueryPlan:
    plan = _plan_query_impl(question, options)
    if (plan.budget or {}).get("capacity_limited"):
        if plan.question_mode == "open":
            # Open-mode callers want an answer, not a request to split the
            # question.  Keep the planner's aspects (or a bounded fallback)
            # and record the omitted items instead of returning an empty plan.
            planned = list(plan.aspects) or _fallback_aspects(
                plan.original_question, plan_query_budget(plan.original_question)
            )
            planned = planned[:QUERY_PLANNER_MAX_ASPECTS]
            if planned:
                budget = dict(plan.budget or {})
                budget["capacity_limited"] = False
                budget["omitted_or_merged_items"] = list(
                    (plan.budget or {}).get("omitted_or_merged_items") or ()
                )
                return replace(
                    plan,
                    aspects=tuple(planned),
                    requirements=(),
                    clarification_reason=None,
                    budget=budget,
                )
        return replace(
            plan,
            aspects=(),
            requirements=(),
            clarification_reason="问题包含较多独立事项，请分批提问或明确本次优先回答的事项。",
        )
    if (
        plan.clarification_reason
        and not [option for option in options or [] if str(option or "").strip()]
        and _should_answer_without_options(plan.original_question)
    ):
        sanitized = _sanitize_open_choice_question(plan.original_question)
        fallback = _fallback_aspects(sanitized, plan_query_budget(sanitized))
        plan = replace(
            plan,
            aspects=tuple(fallback),
            planner="fallback-open-no-options",
            fallback_used=True,
            clarification_reason=None,
        )
    aspects = []
    for aspect in plan.aspects:
        titles = list(dict.fromkeys(re.findall(r"《([^》]+)》", aspect.question)))
        if aspect.modality == "text" and len(titles) > 1:
            body = _clean_source_body(aspect.question)
            for index, title in enumerate(titles, 1):
                aspects.append(
                    replace(
                        aspect,
                        aspect_id=f"{aspect.aspect_id}_source{index}",
                        question=f"根据《{title}》：{body}",
                        table_filters={**aspect.table_filters, "source_title": title},
                        explicit_filter_keys=tuple(
                            set(aspect.explicit_filter_keys) | {"source_title"}
                        ),
                    )
                )
        else:
            aspects.append(aspect)
    aspects = [_apply_format_filter(aspect) for aspect in aspects]
    aspects = _merge_attachment_family_aspects(plan.original_question, aspects)
    aspects = _prune_optionless_research_aspects(plan.original_question, aspects)
    if not plan.fallback_used:
        aspects = _mark_aspect_scope(
            plan.original_question,
            aspects,
            int((plan.budget or {}).get("max_aspects") or 1),
        )
    aspects = _promote_coordinated_aspects(plan.original_question, aspects)
    aspects = _ensure_question_part_coverage(
        plan.original_question, aspects, plan_query_budget(plan.original_question)
    )
    aspects = _bind_requirement_questions(plan.original_question, aspects)
    aspects = _inject_quoted_anchor_queries(plan.original_question, aspects)
    aspects = _promote_cross_document_aspects(
        plan.original_question, aspects, plan.question_mode
    )
    aspects = _demote_planner_expansions(plan.original_question, aspects)
    return replace(plan, aspects=tuple(aspects), requirements=())


def _plan_query_impl(question: str, options: list[str] | None = None) -> QueryPlan:
    cleaned_question = question.strip()
    question_mode = (
        "choice"
        if any(str(option or "").strip() for option in options or [])
        else "open"
    )
    if not cleaned_question:
        return QueryPlan(
            original_question="",
            aspects=(),
            planner="empty",
            fallback_used=True,
            question_mode=question_mode,
        )

    budget = plan_query_budget(cleaned_question)

    # Questions that name an indexed spreadsheet plus a separate regulation
    # are split by source before the table locator runs.  This prevents the
    # first quoted regulation title from being mistaken for the table source.
    multi_source_aspects = _multi_source_table_aspects(cleaned_question, options or [])
    if multi_source_aspects:
        return QueryPlan(
            original_question=cleaned_question,
            aspects=multi_source_aspects,
            planner="deterministic-multi-source",
            fallback_used=True,
            budget=budget.to_debug_dict(),
            question_mode=question_mode,
        )

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

    # A quoted anchor is only a safe deterministic shortcut when the user
    # asked one question about it.  For a multi-part question, collapsing to
    # the anchor would silently drop the other parts.
    anchored_aspects = (
        ()
        if len(_explicit_question_parts(cleaned_question)) > 1
        else _explicit_document_anchor_aspects(
            cleaned_question,
            max_aspects=budget.system_max_aspects,
        )
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
                budget = _budget_with_omissions(budget, omitted_or_merged_items)
                return QueryPlan(
                    original_question=cleaned_question,
                    aspects=tuple(aspects),
                    planner=f"{QUERY_PLANNER_PROVIDER}:{QUERY_PLANNER_MODEL}",
                    fallback_used=False,
                    budget=budget.to_debug_dict(),
                    question_mode=question_mode,
                )
        except QueryNeedsClarification as exc:
            # 先检索后澄清：LLM 判定歧义不再直接终态。开放问题先构造一次
            # 通用检索并携带澄清理由——检索命中（例如问题词汇来自用户刚上传
            # 的文档）则正常作答；检索无可靠证据时由 supervisor 回到澄清。
            # 选择题模式保留直接澄清（选项消歧必须先行）。
            fallback = _fallback_aspects(cleaned_question, budget)
            if fallback and not any(
                str(option or "").strip() for option in options or []
            ):
                return QueryPlan(
                    original_question=cleaned_question,
                    aspects=tuple(fallback),
                    planner="clarification-retrieval",
                    fallback_used=True,
                    question_mode=question_mode,
                    clarification_reason=str(exc),
                    budget=budget.to_debug_dict(),
                )
            return QueryPlan(
                cleaned_question,
                (),
                "clarification",
                question_mode=question_mode,
                clarification_reason=str(exc),
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
        max(1, len(_coordinated_aspect_questions(part))) for part in question_parts
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
        "两": 2,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    count_match = re.search(
        r"([二三四五六七八九十两]|\d+)(?:项|类)(?:信息|要求|内容|控制|事实)?", compact
    )
    if not count_match:
        return []
    expected_count = (
        int(count_match.group(1))
        if count_match.group(1).isdigit()
        else chinese_counts.get(count_match.group(1), 0)
    )
    if expected_count <= 1:
        return []
    original_match = re.search(
        r"([二三四五六七八九十两]|\d+)(?:项|类)(?:信息|要求|内容|控制|事实)?", text
    )
    if not original_match:
        return []
    if (
        original_match.end() < len(text)
        and "：" in text[original_match.end() : original_match.end() + 4]
    ):
        segment = text[text.find("：", original_match.end()) + 1 :]
    else:
        left = text[: original_match.start()]
        segment = re.split(r"[：:]", left)[-1]
    segment = re.split(r"[。！？?]", segment, maxsplit=1)[0]
    parts = [
        re.sub(
            r"^(?:形成审查摘要|联合核验|核验|说明|概括|回答|列出|请)+",
            "",
            part.strip(" ，,；;。:："),
        )
        for part in re.split(r"[、；;,，]|以及|和|与|及", segment)
        if part.strip(" ，,；;。:：")
    ]
    parts = [
        part
        for part in parts
        if part and not re.fullmatch(r"(?:再)?(?:比较|查询|读取|计算|说明|最后)", part)
    ]
    return _dedupe(parts)[:expected_count] if len(parts) >= expected_count else []


def _explicit_requested_aspect_count(question: str) -> int:
    """Estimate only an upper budget for visibly enumerated user requests."""

    compact = re.sub(r"\s+", "", str(question or ""))
    chinese_counts = {
        "两": 2,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    explicit = 1
    for match in re.finditer(
        r"([二三四五六七八九十两]|\d+)(?:项|类)(?:信息|要求|内容|控制|事实)?", compact
    ):
        token = match.group(1)
        explicit = max(
            explicit, int(token) if token.isdigit() else chinese_counts.get(token, 1)
        )
    if any(
        marker in compact
        for marker in ("分别", "逐项", "汇总", "串联", "同时", "跨制度", "审查摘要")
    ):
        # Enumeration commas are ambiguous in ordinary prose, so only use
        # them when the user explicitly asks for a multi-item response. This
        # raises the planner ceiling; it does not force artificial aspects.
        explicit = max(explicit, 1 + compact.count("、") + compact.count("；"))
    return min(explicit, QUERY_PLANNER_MAX_ASPECTS)


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
        r"^(.{2,30}?)(?:与|和)(.{2,20}?原则)(.*)$",
        text,
    )
    if principle_pair:
        left, right, suffix = principle_pair.groups()
        return [f"{left}{suffix}", f"{right}{suffix}"]
    return [text]


def _summary_enumerated_aspect_questions(question: str) -> list[str]:
    text = str(question or "").strip()
    if "、" not in text or not any(
        marker in text for marker in ("串联", "汇总", "逐项", "分别")
    ):
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
        r"^(.{2,30}?)(?:与|和)(.{2,20}?原则)(.*)$",
        part,
    )
    if principle_pair:
        left, right, suffix = principle_pair.groups()
        return [f"{left}{suffix}", f"{right}{suffix}"]
    return [part]


def _inherit_left_subject(left: str, right: str) -> str:
    if "的" in right:
        return right
    if not re.search(r"(?:定义|概念|含义)$", left):
        return right
    if not re.match(
        r"^.{0,6}(?:范围|条件|要求|期限|程序|职责|例外|原则|流程|边界)", right
    ):
        return right
    bare_subject = re.sub(r"(?:的)?(?:定义|概念|含义)$", "", left).strip()
    bare_subject = re.sub(
        r"^(?:说明|概括|回答|列出|判断|核验|比较)", "", bare_subject
    ).strip()
    if bare_subject and len(re.sub(r"\s+", "", bare_subject)) >= 2:
        return f"{bare_subject}的{right}"
    subject_match = re.match(r"^(.{2,40}?的).+", left)
    if not subject_match:
        return right
    subject = subject_match.group(1)
    if right.startswith(subject):
        return right
    return f"{subject}{right}"


def _budget_with_omissions(
    budget: QueryBudget, omitted_or_merged_items: list[str]
) -> QueryBudget:
    merged = tuple(_dedupe([*budget.omitted_or_merged_items, *omitted_or_merged_items]))
    return QueryBudget(
        detected_items=budget.detected_items,
        max_aspects=budget.max_aspects,
        system_max_aspects=budget.system_max_aspects,
        capacity_limited=budget.capacity_limited or bool(merged),
        omitted_or_merged_items=merged,
    )


def _plan_with_llm(
    question: str, budget: QueryBudget
) -> tuple[list[QueryAspect], list[str]]:
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
                "询问“包括哪些方式/类型/类别”时，只保留用户原问的范围，"
                "不得把推测的办理平台、流程、主体或类别成员新增为 aspect 或 evidence_need，"
                "不得预设某一类清单的成员。"
                "不得丢弃依据不足、违规判断、正式监管报告生成等安全边界问题。"
                "只输出 JSON 对象。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请按本次预算把下面问题拆成 1 到 "
                f"{budget.system_max_aspects} 个 aspect。按实际需要拆分，不必填满上限。每个 aspect 包含："
                "aspect_id、question、evidence_need、search_queries、keywords、modality、table_task、table_filters、operation。"
                "如果用户明确枚举了多个处理对象，一般每个处理对象单独成为一个 aspect；"
                "跨文档比较时，每个命名来源单独一个 aspect；若该 aspect 只针对某个具体文件/版本/附件，"
                "把用户问题中的精确来源名写入 table_filters.source_title；另加一个对比 aspect；不要把两个来源合并。"
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
                f"本次规则识别到的用户处理对象：{json.dumps(list(budget.detected_items), ensure_ascii=False)}\n"
                "每个aspect可填写subject_scope数组，仅允许逐字来自用户问题的主体或业务范围限定。"
                "若用户目标存在无法消除的歧义，返回clarification说明需要补充的信息，禁止猜测。"
                "判断歧义要克制：只要知识库中可能存在能定位该问题的文档、指标或字段"
                "（包括用户刚上传的材料），就按正常 aspect 检索，不要因为问题简短、"
                "缺少报表名或期间而要求澄清；只有当任何检索都无法定位时才返回clarification。"
                "clarification 必须是面向用户的中文字符串，不要返回对象、布尔值，"
                "也不要写 needed、missing 等内部字段名。"
                '输出格式：{"aspects":[...],"clarification":null,"omitted_or_merged_items":[]}\n\n'
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
                    timeout_seconds=QA_REQUEST_TOTAL_BUDGET_SECONDS,
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

    return _aspects_from_payload(
        question, parsed, max_aspects=QUERY_PLANNER_MAX_ASPECTS
    ), _clean_string_list(parsed.get("omitted_or_merged_items"))


def _aspects_from_payload(
    question: str,
    payload: dict[str, Any],
    max_aspects: int | None = None,
) -> list[QueryAspect]:
    clarification = normalize_clarification_text(payload.get("clarification"))
    if payload.get("clarification"):
        # The provider may return a string, an object such as
        # {"needed": true, "question": "..."}, or even a Python literal.
        # Only the normalised sentence may reach the user-facing answer.
        raise QueryNeedsClarification(clarification or DEFAULT_CLARIFICATION_TEXT)
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
            search_queries = [
                QuerySearchQuery(
                    query=sub_question,
                    query_type="fallback",
                    rationale="LLM 未返回检索查询",
                )
            ]
        if not keywords:
            keywords = _keywords_from_text(
                " ".join([sub_question, *[query.query for query in search_queries]])
            )
        explicit_filters = dict(question_explicit_filters)
        # A table title, sheet and period stated in the user's question are
        # hard locators.  They must win over an LLM plan even when the plan
        # guessed a different period (for example confusing "3季度" with a
        # monthly report).  Keeping a guessed value would turn a valid lookup
        # into a false "period not found" refusal.
        # The original user question is more authoritative than an LLM's
        # decomposed sub-question, which may itself have introduced a period
        # or title not present in the request.
        original_titles = _dedupe(re.findall(r"《([^》]{2,120})》", question))
        sub_question_titles = _dedupe(re.findall(r"《([^》]{2,120})》", sub_question))
        if len(original_titles) >= 2 and len(sub_question_titles) == 1:
            sub_title = sub_question_titles[0]
            if sub_title in original_titles:
                # A decomposed cross-document aspect is narrower than the
                # original multi-title question.  Preserve that explicit title
                # instead of applying the first title globally to every aspect.
                explicit_filters["source_title"] = sub_title
        merged_filters = dict(explicit_filters)
        modality = _clean_modality(raw_aspect.get("modality"))
        table_task = _clean_table_task(raw_aspect.get("table_task"))
        operation = _clean_operation(raw_aspect.get("operation"))
        if _should_force_text_aspect(
            question, sub_question, modality
        ) or _is_document_formula_question(f"{question} {sub_question}"):
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
                key: value
                for key, value in explicit_filters.items()
                if key not in TABLE_ONLY_FILTER_KEYS
            }
            explicit_filters = {
                key: value
                for key, value in explicit_filters.items()
                if key not in TABLE_ONLY_FILTER_KEYS
            }
        aspects.append(
            QueryAspect(
                aspect_id=_clean_aspect_id(raw_aspect.get("aspect_id"), index),
                question=sub_question,
                search_queries=tuple(
                    _dedupe_search_queries(search_queries)[
                        :QUERY_PLANNER_MAX_SEARCH_QUERIES
                    ]
                ),
                evidence_need=evidence_need,
                keywords=tuple(_dedupe(keywords)[:8]),
                modality=modality,
                table_task=table_task,
                table_filters=merged_filters,
                operation=operation,
                explicit_filter_keys=tuple(explicit_filters),
                subject_scope=tuple(
                    x
                    for x in _clean_string_list(raw_aspect.get("subject_scope"))
                    if x in question
                ),
                mandatory=raw_aspect.get("mandatory")
                if isinstance(raw_aspect.get("mandatory"), bool)
                else True,
            )
        )

    if not aspects:
        raise QueryPlannerError("LLM query planner 未产生有效 aspect")
    return _expand_explicit_anchor_aspects(aspects, max_aspects=limit)


def _fallback_aspects(
    question: str, budget: QueryBudget | None = None
) -> list[QueryAspect]:
    budget = budget or plan_query_budget(question)
    table_aspect = _fallback_table_aspect(question)
    if table_aspect is not None:
        return [table_aspect]

    parts = _explicit_question_parts(question)
    if not parts:
        parts = [question]

    aspects: list[QueryAspect] = []
    for index, part in enumerate(parts[: budget.max_aspects], start=1):
        keywords = _keywords_from_text(part)
        explicit_filters = extract_explicit_metadata_filters(part)
        # When the user names one source and then asks several sub-questions,
        # the source title applies to every part.  Previously only the first
        # part carried the hard source constraint, so later parts could be
        # answered from an unrelated document.
        if not explicit_filters.get("source_title"):
            inherited_title = _source_title_for_part(question, part)
            if inherited_title:
                explicit_filters["source_title"] = inherited_title
        aspects.append(
            QueryAspect(
                aspect_id=f"aspect_{index}",
                question=part,
                search_queries=_fallback_search_queries(part),
                evidence_need=_infer_evidence_type(part),
                keywords=tuple(keywords),
                table_filters=explicit_filters,
                explicit_filter_keys=tuple(explicit_filters),
                requirement_question=part,
            )
        )
    expanded = _expand_explicit_anchor_aspects(
        aspects,
        max_aspects=max(budget.max_aspects, budget.system_max_aspects),
    )
    return expanded


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
        end = (
            title_matches[title_index + 1].start()
            if title_index + 1 < len(title_matches)
            else len(text)
        )
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
        definition_request = (
            any(
                marker in text
                for marker in ("完整定义", "定义是什么", "如何定义", "所称", "是指")
            )
            and len(matches) == 1
        )
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
                    QuerySearchQuery(
                        sub_question, "semantic_question", "指定文档与引文联合检索"
                    ),
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


def _fallback_mcq_aspect(
    question: str, options: list[str] | None = None
) -> QueryAspect | None:
    cleaned_options = [
        str(option).strip() for option in (options or []) if str(option).strip()
    ]
    if len(cleaned_options) < 2:
        return None

    title_candidates = [
        _normalize_mcq_title(item)
        for pattern in (r"《([^》]+)》", r"“([^”]+)”", r'"([^"]+)"')
        for item in re.findall(pattern, question)
        if _normalize_mcq_title(item)
    ]
    title_anchor = title_candidates[0] if title_candidates else ""
    requested_format_match = re.search(
        r"[（(]\s*(pdf|word|docx?)\s*[）)]", question, flags=re.IGNORECASE
    )
    requested_format = (
        requested_format_match.group(1).lower() if requested_format_match else ""
    )
    anchor_question = title_anchor or re.sub(
        r"(检索|查阅|根据|以下哪一项|以下哪项|哪一项|哪项|与材料内容一致|与材料一致|说法正确|表述正确|正确的是|不正确的是)",
        " ",
        question,
    )
    anchor_question = (
        re.sub(r"[？?。；;，,：:\s]+", " ", anchor_question).strip() or question
    )

    option_statements = _prioritized_mcq_option_queries(
        _mcq_option_evidence_statements(cleaned_options)
    )
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
        question=question,
        search_queries=tuple(_dedupe_search_queries(search_queries)),
        evidence_need="在指定材料中核对候选项并找到直接支持或否定证据",
        keywords=tuple(_dedupe([anchor_question, *title_candidates])[:8]),
        modality="text",
        table_filters={
            **({"file_type": requested_format} if requested_format else {}),
            **(
                {"source_title": title_candidates[0]}
                if len(re.findall(r"《([^》]+)》", question)) == 1
                else {}
            ),
        },
        explicit_filter_keys=tuple(
            (["file_type"] if requested_format else [])
            + (
                ["source_title"]
                if len(re.findall(r"《([^》]+)》", question)) == 1
                else []
            )
        ),
    )


def _prioritized_mcq_option_queries(option_statements: list[str]) -> list[str]:
    """Bound option-derived queries without adding topic-specific facts."""

    return _dedupe(option_statements)[:MCQ_OPTION_SEARCH_QUERY_BUDGET]


def _normalize_mcq_title(value: str) -> str:
    title = str(value or "").strip()
    return re.sub(
        r"[（(]\s*(?:pdf|word|docx?|附件)\s*[）)]$", "", title, flags=re.IGNORECASE
    ).strip()


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
    return [
        part.strip()
        for part in re.split(r"[？?。；;]|如果|若|以及|并且|同时|，且", question)
        if part.strip()
    ]


_QUESTION_PART_MARKER_RE = re.compile(
    r"如何|怎样|怎么|哪些|什么|是否|多少|分别|包括|列举|列出|界定|定义|"
    r"含义|处理|要求|规定|安排|适用|期限|对象|原则|条件|范围|情形|标准|"
    r"程序|职责|内容|归属|类别|版本|方式|类型|主体"
)
_SKIP_PART_PREFIX_RE = re.compile(r"^(?:若|如果|如|当|对于|在|其中|同时|并且|以及)")


def _explicit_question_parts(question: str) -> list[str]:
    """Return substantive sub-questions the user explicitly asked.

    The splitter is deliberately wording-level.  It is used only to make sure
    a planner rewrite did not silently drop one of several questions; it does
    not add domain facts or source-specific branches.
    """

    parts: list[str] = []
    for part in _split_question_parts(question):
        compact = re.sub(r"\s+", "", part)
        if len(compact) < 6:
            continue
        if re.fullmatch(
            r"(?:请直接给出规定内容|不要要求提供选项|请直接回答).*", compact
        ):
            continue
        if re.match(r"^(?:并)?(?:完整)?保留", compact):
            continue
        if re.match(r"^(?:请)?(?:完整)?(?:给出|写明)", compact) and re.search(
            r"条件|范围|期限|例外", compact
        ):
            continue
        if not _QUESTION_PART_MARKER_RE.search(compact):
            continue
        # Conditions are context, not separate answer obligations.  Keep a
        # conditional fragment only when it itself asks for an outcome.
        if _SKIP_PART_PREFIX_RE.search(compact) and not re.search(
            r"哪些|什么|如何|怎样|怎么|是否|多少|分别|包括|列举|列出|界定|定义|"
            r"含义|处理|要求|规定|安排|适用|期限|对象|原则|条件|范围|情形|标准|"
            r"程序|职责|内容|归属|类别|版本|方式|类型|主体",
            compact,
        ):
            continue
        parts.append(part)
    return list(dict.fromkeys(parts))


def _clean_source_body(question: str) -> str:
    """Remove title marks and dangling “所附/中的” after a source split."""

    body = re.sub(r"《[^》]+》", "", str(question or ""))
    body = re.sub(r"^(?:根据|依据|按照|对于|在|关于|回答|检索)[，,：:\s]*", "", body)
    body = re.sub(r"^(?:所附|中的|中)[，,：:\s]*", "", body)
    body = re.sub(r"^(?:相关)?内容[，,：:\s]*", "", body)
    return body.strip("，,：: ").strip()


def _source_title_for_part(question: str, part: str) -> str:
    """Inherit the single user-named source for a split sub-question."""

    part_titles = list(dict.fromkeys(re.findall(r"《([^》]+)》", part)))
    if len(part_titles) == 1:
        return part_titles[0]
    titles = list(dict.fromkeys(re.findall(r"《([^》]+)》", question)))
    if len(titles) == 1:
        return titles[0]
    if not titles:
        return ""
    position = question.find(part)
    if position < 0:
        return titles[0]
    before = [
        match.group(1)
        for match in re.finditer(r"《([^》]+)》", question)
        if match.end() <= position
    ]
    return before[-1] if before else titles[0]


def _part_covered_by_aspect(part: str, aspect: QueryAspect) -> bool:
    part_text = _scope_text(re.sub(r"《[^》]+》", " ", part))
    if len(part_text) < 4:
        return True
    aspect_text = _scope_text(
        " ".join(
            [
                aspect.question,
                *aspect.keywords,
                *[query.query for query in aspect.search_queries],
            ]
        )
    )
    if not aspect_text:
        return False
    if part_text in aspect_text or aspect_text in part_text:
        return True
    part_grams = {
        part_text[index : index + 2] for index in range(max(len(part_text) - 1, 1))
    }
    if not part_grams:
        return True
    recall = len(
        part_grams
        & {
            aspect_text[index : index + 2]
            for index in range(max(len(aspect_text) - 1, 1))
        }
    )
    if recall / len(part_grams) >= 0.62:
        return True
    part_terms = {term for term in _keywords_from_text(part) if len(term) >= 2}
    aspect_terms = set(aspect.keywords) | set(_keywords_from_text(aspect.question))
    return bool(part_terms) and len(part_terms & aspect_terms) >= min(
        2, len(part_terms)
    )


def _attachment_family_titles(question: str) -> tuple[str, str] | None:
    """Return (parent, child) when the question points at an attachment.

    The corpus stores many parent notifications as one file per attachment;
    the parent itself is often not a separate document.  Treating the parent
    title as a second mandatory source produces a false "missing source"
    refusal even though the attachment contains the answer.
    """

    text = str(question or "")
    patterns = (
        r"《([^》]+)》\s*所附\s*《([^》]+)》",
        r"《([^》]+)》\s*所涉及\s*《([^》]+)》",
        r"《([^》]+)》\s*所引\s*《([^》]+)》",
        r"《([^》]+)》\s*中(?:的)?\s*《([^》]+)》",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1), match.group(2)
    titles = list(dict.fromkeys(re.findall(r"《([^》]+)》", text)))
    if len(titles) == 2:
        first, second = titles
        regulation = re.search(r"通知|办法|规则|规定|意见|条例|细则", first)
        attachment = re.search(
            r"名单|示例|曲线|附件|表|目录|解释|模板|指引|要求|标准", second
        )
        pointer = re.search(
            r"所附|附件|检索|中的|中《|中关于|有关|名单|列示|材料|内容和要求|相关规定",
            text,
        )
        if regulation and attachment and pointer:
            return first, second
    return None


def _apply_format_filter(aspect: QueryAspect) -> QueryAspect:
    title = str(aspect.table_filters.get("source_title") or "")
    match = re.search(
        r"[（(]\s*(PDF|Word|Excel|DOCX|XLSX)\s*[）)]", title, re.IGNORECASE
    )
    if not match:
        return aspect
    value = match.group(1).lower()
    file_type = {"word": "docx", "excel": "xlsx"}.get(value, value)
    return replace(
        aspect,
        table_filters={**aspect.table_filters, "file_type": file_type},
        explicit_filter_keys=tuple(set(aspect.explicit_filter_keys) | {"file_type"}),
    )


def _merge_attachment_family_aspects(
    question: str, aspects: list[QueryAspect]
) -> list[QueryAspect]:
    """Map parent-title aspects to the concrete attachment document.

    This is a corpus-family resolution, not a document-specific branch: the
    relation is read from the user's own “《parent》所附《child》” wording.
    """

    family = _attachment_family_titles(question)
    if family is None:
        return aspects
    parent, child = family
    mapped: list[QueryAspect] = []
    for aspect in aspects:
        source = str(aspect.table_filters.get("source_title") or "")
        if source == parent:
            body = _clean_source_body(aspect.question)
            new_question = f"根据《{child}》：{body}" if body else aspect.question
            filters = {**aspect.table_filters, "source_title": child}
            aspect = replace(
                aspect,
                question=new_question,
                table_filters=filters,
                explicit_filter_keys=tuple(
                    set(aspect.explicit_filter_keys) | {"source_title"}
                ),
            )
        mapped.append(aspect)
    deduped: list[QueryAspect] = []
    seen: set[tuple[str, str]] = set()
    for aspect in mapped:
        source = str(aspect.table_filters.get("source_title") or "")
        key = (
            (re.sub(r"《[^》]+》", "", aspect.question), source) if source else ("", "")
        )
        if source and key in seen:
            continue
        if source:
            seen.add(key)
        deduped.append(aspect)
    return deduped


_OPTIONLESS_RESEARCH_RE = re.compile(
    r"位于何处|何种形式|正文及附件结构|差异或衔接|执行时间|过渡期|例外条件|"
    r"实施要求|执行安排|引用|附件定位|执行要求|正文"
)


def _prune_optionless_research_aspects(
    question: str, aspects: list[QueryAspect]
) -> list[QueryAspect]:
    """Drop research-only expansions for an option-less choice question."""

    if not _should_answer_without_options(question) or not aspects:
        return aspects
    kept = [
        aspect
        for aspect in aspects
        if not _OPTIONLESS_RESEARCH_RE.search(aspect.question)
    ]
    if not kept:
        kept = list(aspects)
    sanitized = _sanitize_open_choice_question(question)
    kept.sort(
        key=lambda aspect: _scope_overlap(aspect.question, sanitized), reverse=True
    )
    kept = kept[:2]
    return [replace(aspect, mandatory=True) for aspect in kept]


_EXPANSION_MARKERS = (
    "分别",
    "各自",
    "具体职责",
    "具体类别",
    "载体",
    "实施要求",
    "执行安排",
    "过渡期",
    "例外条件",
    "逐档",
    "逐项",
    "定义",
    "概念",
    "含义",
    "划分标准",
    "适用前提",
    "核实方式",
    "确认要求",
    "异常处理",
    "签收登记",
    "留痕要求",
    "排除",
    "例外情形",
    "划分条件",
    "判断标准",
    "不得划入",
)


def _demote_planner_expansions(
    question: str, aspects: list[QueryAspect]
) -> list[QueryAspect]:
    """Demote an aspect that adds answer obligations the user did not state."""

    if len(aspects) <= 1:
        return aspects
    if "名单" in question:
        # A user-named list attachment is the answer scope.  Planner aspects
        # about the parent notice's other clauses are retrieval hints, not
        # additional answer obligations.
        result = [
            aspect
            if ("名单" in aspect.question or "编报" in aspect.question)
            else replace(aspect, mandatory=False)
            for aspect in aspects
        ]
        if not any(aspect.mandatory for aspect in result):
            result[0] = replace(result[0], mandatory=True)
        return result
    demoted = []
    for aspect in aspects:
        added = [
            marker
            for marker in _EXPANSION_MARKERS
            if marker in aspect.question and marker not in question
        ]
        demoted.append(replace(aspect, mandatory=False) if added else aspect)
    if not any(aspect.mandatory for aspect in demoted):
        demoted[0] = replace(demoted[0], mandatory=True)
    return demoted


def _promote_coordinated_aspects(
    question: str, aspects: list[QueryAspect]
) -> list[QueryAspect]:
    """Keep both sides of an explicit “A 与 B 分别……” question.

    The planner often emits one aspect per item plus an umbrella comparison
    aspect.  The scope limiter then keeps only one item mandatory, so half of
    the user's question is never validated.  Mark the item-level aspects
    mandatory and leave the umbrella aspect optional.
    """

    if not aspects or "分别" not in question:
        return aspects
    match = re.search(
        r"([^，。；：？]{2,40}?)(?:与|和|及|、)([^，。；：？]{2,40}?)(?:分别|均|都)",
        question,
    )
    terms: list[str] = []
    if match:
        for value in match.groups():
            value = re.split(r"的", value)[-1]
            value = re.sub(r'[“”"\'《》\s]', "", value)
            if 2 <= len(value) <= 20:
                terms.append(value)
    promoted: set[int] = set()
    if len(terms) == 2:
        for index, aspect in enumerate(aspects):
            hits = sum(1 for term in terms if term in aspect.question)
            if hits == 1:
                promoted.add(index)
    if len(promoted) < 2:
        promoted.update(range(min(2, len(aspects))))
    return [
        replace(aspect, mandatory=True) if index in promoted else aspect
        for index, aspect in enumerate(aspects)
    ]


def _ensure_question_part_coverage(
    question: str,
    aspects: list[QueryAspect],
    budget: QueryBudget,
) -> list[QueryAspect]:
    """Keep every explicit sub-question in the plan.

    An LLM or deterministic anchor rewrite may collapse "A how? and B how?"
    into a single aspect about A.  The missing B is then never retrieved or
    validated.  Add the missing part as a normal requirement, inheriting the
    user-stated source when there is one; no answer text or gold data is used.
    """

    if not isinstance(budget, QueryBudget):
        return aspects
    parts = _explicit_question_parts(question)
    if len(parts) < 2:
        return aspects
    current = list(aspects)
    uncovered = [
        part
        for part in parts
        if not any(_part_covered_by_aspect(part, aspect) for aspect in current)
    ]
    if not uncovered:
        return aspects

    # A single generic aspect that covers none of the user's sub-questions is
    # a research rewrite, not an answer obligation.  Drop it before adding the
    # explicit parts so it cannot cause a false "missing requirement" refusal.
    if len(current) == 1 and not any(
        _part_covered_by_aspect(part, current[0]) for part in parts
    ):
        current = []

    limit = max(int(budget.max_aspects or 1), len(parts), 2)
    limit = min(limit, QUERY_PLANNER_MAX_ASPECTS)
    existing_ids = {aspect.aspect_id for aspect in current}
    for index, part in enumerate(uncovered, start=1):
        if len(current) >= limit:
            break
        title = _source_title_for_part(question, part)
        sub_question = part
        if title and f"《{title}》" not in sub_question:
            sub_question = f"根据《{title}》：{sub_question}"
        queries = list(_fallback_search_queries(sub_question))
        if title and not any(title in query.query for query in queries):
            queries.insert(
                0,
                QuerySearchQuery(
                    query=f"{title} {re.sub(r'[“”\"\'《》]', ' ', part)}",
                    query_type="document_title",
                    rationale="inherit the user-stated source for this sub-question",
                ),
            )
        filters = {"source_title": title} if title else {}
        aspect_id = f"part_{index}"
        while aspect_id in existing_ids:
            aspect_id = f"part_{index}_{len(current)}"
        existing_ids.add(aspect_id)
        current.append(
            QueryAspect(
                aspect_id=aspect_id,
                question=sub_question,
                search_queries=tuple(
                    _dedupe_search_queries(queries)[:QUERY_PLANNER_MAX_SEARCH_QUERIES]
                ),
                evidence_need=_infer_evidence_type(part),
                keywords=tuple(_dedupe(_keywords_from_text(sub_question))[:8]),
                table_filters=filters,
                explicit_filter_keys=tuple(filters),
                mandatory=True,
                requirement_question=part,
            )
        )
    return current


_PLANNER_ADDED_ENTITY_RE = re.compile(
    r"董事会|监事会|高级管理层|高管层|牵头部门|相关部门|业务部门|"
    r"高级管理人员|地方政府|股东|客户|员工|社会公众|监管部门|"
    r"母公司|子公司|分支机构|第一档|第二档|第三档|各档"
)


def _bind_requirement_questions(
    question: str, aspects: list[QueryAspect]
) -> list[QueryAspect]:
    """Keep validation tied to what the user asked, not planner additions.

    Retrieval may expand "相关主体职责" into a list of example organs; that is
    useful for search, but the answer must not be rejected because the source
    does not separately describe an organ the user never named.
    """

    if re.search(r"以下哪一项|下列哪项|哪一项|哪项", question):
        return aspects
    parts = _explicit_question_parts(question)
    if not parts:
        return aspects
    result: list[QueryAspect] = []
    for aspect in aspects:
        if aspect.requirement_question:
            result.append(aspect)
            continue
        parentheticals = re.findall(r"[（(]([^）)]*)[）)]", aspect.question)
        added_parentheticals = [
            value for value in parentheticals if value and value not in question
        ]
        if added_parentheticals:
            base = re.sub(r"[（(][^）)]*[）)]", "", aspect.question).strip()
            if base:
                result.append(replace(aspect, requirement_question=base))
                continue
        scores = [_part_covered_by_aspect(part, aspect) for part in parts]
        best = max(
            range(len(parts)),
            key=lambda index: (
                _scope_overlap(parts[index], aspect.question),
                scores[index],
            ),
        )
        part = parts[best]
        added = [
            term
            for term in _PLANNER_ADDED_ENTITY_RE.findall(aspect.question)
            if term not in part
        ]
        if added:
            result.append(replace(aspect, requirement_question=part))
        else:
            result.append(aspect)
    return result


def _inject_quoted_anchor_queries(
    question: str, aspects: list[QueryAspect]
) -> list[QueryAspect]:
    """Keep a user-quoted section/term in the aspect's search queries.

    For "《商业银行资本管理办法》中“账簿划分和名词解释”的内容，A？B？",
    the source title alone matches every attachment.  The quoted section name
    is the discriminator that selects the right attachment; if the planner
    drops it from the search queries, an unrelated attachment can rank first.
    """

    anchors = [
        value.strip()
        for value in re.findall(r'[“"]([^”"]{2,60})[”"]', question)
        if value.strip()
    ]
    if not anchors:
        return aspects
    result: list[QueryAspect] = []
    for aspect in aspects:
        applicable = [anchor for anchor in anchors if anchor in aspect.question]
        if not applicable:
            result.append(aspect)
            continue
        queries = list(aspect.search_queries)
        missing = [
            anchor
            for anchor in applicable
            if not any(anchor in query.query for query in queries)
        ]
        if missing:
            anchor_text = " ".join(missing)
            base = f"{anchor_text} {aspect.question}"
            queries.insert(
                0,
                QuerySearchQuery(
                    query=base,
                    query_type="document_style_statement",
                    rationale="retain the user-quoted section/term for exact location",
                ),
            )
        keywords = list(aspect.keywords)
        for anchor in applicable:
            if anchor not in keywords:
                keywords.insert(0, anchor)
        result.append(
            replace(
                aspect,
                search_queries=tuple(
                    _dedupe_search_queries(queries)[:QUERY_PLANNER_MAX_SEARCH_QUERIES]
                ),
                keywords=tuple(_dedupe(keywords)[:8]),
            )
        )
    return result


def _sanitize_open_choice_question(question: str) -> str:
    """Turn an option-less "which one" question into a content request.

    The literal run intentionally removes options, so a few rewritten
    questions still contain "以下哪一项".  Asking for options would dead-end;
    the useful open-mode fallback is to ask for the material's operative
    content.  This is only used after the planner has requested clarification.
    """

    text = str(question or "")
    text = re.sub(
        r"[，,。；;]?\s*(?:检索[^，。；;]*后[，,]?)?(?:以下|下列)哪(?:一)?项[^？?]*[？?]?",
        "，请概括材料的主要规定内容。",
        text,
    )
    # The literal runner appends this meta-instruction.  It is not a second
    # substantive question and must not become a separate mandatory aspect.
    text = re.sub(
        r"(?:请直接给出规定内容|不要要求提供选项|请直接回答|不要要求提供选项)[^。？?]*[。？?]?",
        "",
        text,
    )
    return re.sub(r"。{2,}", "。", text).strip("，,。；; ")


def _should_answer_without_options(question: str) -> bool:
    """Whether a multiple-choice-looking open question can still be answered.

    When the caller passes no options, "which one" is not answerable as a
    choice, but the user may still want the material's operative content.
    This guard is only used for open mode and requires a named source or an
    explicit "give the content" instruction.
    """

    text = re.sub(r"\s+", "", str(question or ""))
    if not re.search(r"以下哪一项|下列哪项|哪一项|哪项", text):
        return False
    return bool(
        re.search(r"《[^》]+》", text)
        or re.search(r"请直接给出|不要要求提供选项|直接说明|材料内容", text)
    )


def _extract_json_object(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
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
        return (
            [
                QuerySearchQuery(
                    query=cleaned, query_type="legacy", rationale="兼容旧字符串格式"
                )
            ]
            if cleaned
            else []
        )
    if not isinstance(value, list):
        return []

    queries: list[QuerySearchQuery] = []
    for item in value:
        if isinstance(item, str):
            cleaned = _clean_text(item)
            if cleaned:
                queries.append(
                    QuerySearchQuery(
                        query=cleaned, query_type="legacy", rationale="兼容旧字符串格式"
                    )
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
    return (
        text if text in {"lookup", "compare", "calculate", "locate", "none"} else "none"
    )


def _clean_operation(value: Any) -> str:
    text = _clean_text(value).lower()
    return (
        text if text in {"max", "min", "difference", "sum", "ratio", "none"} else "none"
    )


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
    return {
        key: value[key]
        for key in allowed
        if key in value and value[key] not in (None, "", [])
    }


def _source_title_matches(candidate: str, indexed: str) -> bool:
    from backend.app.services.retrieval_metadata_filter_service import _normalize_title

    left = _normalize_title(candidate)
    right = _normalize_title(re.sub(r"^(?:\d+_)|\.(?:xlsx?|csv)$", "", indexed))
    return bool(left and right and (left in right or right in left))


def _question_source_segments(question: str) -> list[str]:
    segments = [
        part.strip(" ，,。；;")
        for part in re.split(r"(?:同时|并且|以及|另|；|;|。)+", str(question or ""))
    ]
    return [part for part in segments if len(part) >= 4]


def _multi_source_table_aspects(
    question: str, options: list[str]
) -> tuple[QueryAspect, ...]:
    """Split a question that names a table plus at least one text source.

    The deterministic table planner used to treat the first quoted title as the
    table source, so a regulation title placed first could suppress the named
    spreadsheet.  This split is source-oriented: the clause naming an indexed
    spreadsheet becomes the table aspect, and every other explicitly named
    source becomes a text aspect with its own local question.
    """

    table_sources = _TABLE_SOURCES.get()
    if not table_sources:
        return ()
    segments = _question_source_segments(question)
    titles = list(dict.fromkeys(re.findall(r"《([^》]+)》", question)))
    if len(titles) < 2:
        return ()
    table_titles = [
        title
        for title in titles
        if any(_source_title_matches(title, source) for source in table_sources)
    ]
    text_titles = [title for title in titles if title not in table_titles]
    if len(table_titles) != 1 or not text_titles:
        return ()
    table_title = table_titles[0]
    table_segment = next(
        (segment for segment in segments if table_title in segment), question
    )
    table_aspect = _fallback_table_aspect(table_segment, options=options)
    if table_aspect is None:
        return ()
    table_aspect = replace(
        table_aspect, aspect_id="table_evidence", question=table_segment
    )
    aspects: list[QueryAspect] = [table_aspect]
    for index, title in enumerate(text_titles, start=1):
        segment = next(
            (item for item in segments if title in item),
            re.sub(r"《[^》]*》", "", question),
        )
        body = re.sub(r"《[^》]*》", "", segment)
        body = re.sub(
            r"^(?:根据|依据|按照|从|在|对|关于|同时|并|以及|另|其中|另据)+[，,：:\s]*",
            "",
            body,
        )
        body = re.sub(r"^[中里内][，,：:\s]*", "", body).strip(" ，,。；;")
        sub_question = (
            f"根据《{title}》，{body}"
            if body
            else f"根据《{title}》，说明与当前问题直接相关的明确规定。"
        )
        filters = {"source_title": title}
        aspects.append(
            QueryAspect(
                aspect_id=f"source_text_{index}",
                question=sub_question,
                search_queries=_fallback_search_queries(sub_question),
                evidence_need=f"《{title}》中与用户问题直接对应的制度原文",
                keywords=tuple(_keywords_from_text(sub_question)),
                modality="text",
                table_filters=filters,
                explicit_filter_keys=("source_title",),
            )
        )
    return tuple(aspects)


def _fallback_table_aspect(
    question: str, options: list[str] | None = None
) -> QueryAspect | None:
    normalized = re.sub(r"\s+", "", question)
    operation_intent = re.sub(r"《[^》]*》", "", normalized)
    operation_intent = re.sub(r"[“‘\"'][^”’\"']*[”’\"']", "", operation_intent)
    if _is_document_formula_question(normalized):
        # “附件” is a broad spreadsheet marker, but regulatory Word/PDF
        # appendices also use that word. Formula questions must stay on text
        # retrieval so the deterministic formula executor receives formula
        # chunks instead of SpreadsheetCell candidates.
        return None
    explicit_table = _has_explicit_table_marker(question)
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
        any(
            term in operation_intent
            for term in ["差值", "差额", "相差", "变化", "增加", "减少"]
        )
        # “A比B多|少” is the difference form; “占…比例…多少” (the 比 of 比例
        # plus the 多 of 多少) must not be read as a difference.
        or re.search(
            r"[\u4e00-\u9fffA-Za-z0-9_]+比(?!例|重|百分)[\u4e00-\u9fffA-Za-z0-9_]+(?:多|少)",
            operation_intent,
        )
        or (
            "计算" in operation_intent
            and re.search(
                r"[\u4e00-\u9fffA-Za-z0-9_]+[－—-][\u4e00-\u9fffA-Za-z0-9_]+",
                operation_intent,
            )
        )
    ) and not re.search(r"百分比|百分数|比重|占比", operation_intent):
        table_task = "calculate"
        operation = "difference"
    elif (
        # “A占B的比重” is a ratio regardless of how the table was anchored:
        # statistical-intent questions (“2021年12月X占Y的比重”) have no book
        # title, and gating this branch on explicit_table turned them into
        # single-cell lookups that can never answer.
        _asks_ratio_calculation(question)
    ):
        table_task = "calculate"
        operation = "ratio"
    elif any(
        term in operation_intent
        for term in ["求和", "总和", "加总", "之和", "分项之和", "分项合计"]
    ):
        table_task = "calculate"
        operation = "sum"

    mixed_markers = [
        "根据监管制度",
        "结合监管制度",
        "根据制度",
        "结合制度",
        "根据自制制度",
        "结合自制制度",
        "填报说明依据",
        "制度依据",
        "监管规则",
        "制度侧",
        "制度文件",
        "联合核验",
    ]
    modality = "mixed" if any(term in normalized for term in mixed_markers) else "table"
    table_question = (
        _mixed_table_question_part(question) if modality == "mixed" else question
    )
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
        axis_labels = (
            str(filters.get("row_label") or "")
            + " "
            + str(filters.get("column_label") or "")
        )
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
            if any(token and token in axis_labels for token in tokens):
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
        if (
            re.sub(r"\s+", "", str(filters.get("column_label") or ""))
            in selector_labels
        ):
            filters.pop("column_label", None)
        filters.pop("metric", None)
    if table_task == "calculate" and selectors:
        # Calculation operands carry their own columns; a global column filter
        # would discard all but one operand before the executor can calculate.
        filters.pop("column_label", None)
    if (
        table_task in {"compare", "calculate"}
        and len(_period_selectors(table_question)) >= 2
    ):
        # Multiple explicitly requested periods are operands, not a single
        # global hard filter.  Keep them on the ordered selectors.
        filters.pop("year", None)
        filters.pop("month", None)
        filters.pop("quarter", None)
        if (
            len({s.get("source_title") for s in selectors if s.get("source_title")})
            >= 2
        ):
            filters.pop("source_title", None)
    common_columns = {s.get("column_label") for s in selectors if s.get("column_label")}
    if (
        table_task == "compare"
        and len(common_columns) == 1
        and all(s.get("column_label") for s in selectors)
    ):
        filters["column_label"] = next(iter(common_columns))
    # “在某口径下取数” is still a pure table lookup. Only explicit requests to
    # combine a regulation/explanation with table data should trigger the
    # slower mixed vector path.
    locator_terms = " ".join(str(value) for value in filters.values() if value)
    search_queries = [
        QuerySearchQuery(table_question, "semantic_question", "表格问题的自然语言意图"),
        QuerySearchQuery(
            " ".join(
                part
                for part in [
                    str(filters.get("source_title") or ""),
                    str(filters.get("sheet") or ""),
                    str(filters.get("indicator") or ""),
                ]
                if part
            ),
            "document_style_statement",
            "接近表格标题和工作表名称",
        ),
        QuerySearchQuery(
            locator_terms or question, "table_locator", "表格结构定位锚点"
        ),
    ]
    search_queries = tuple(query for query in search_queries if query.query.strip())
    keywords = _keywords_from_text(f"{question} {locator_terms}")
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
        aspect_id = (
            "regulatory_basis"
            if len(bounded_specs) == 1
            else f"regulatory_basis_{index}"
        )
        aspects.append(
            QueryAspect(
                aspect_id=aspect_id,
                question=anchor_question,
                search_queries=(
                    QuerySearchQuery(
                        anchor_question, "semantic_question", "监管解释意图"
                    ),
                    QuerySearchQuery(
                        " ".join(
                            part
                            for part in [
                                str(filters.get("source_title") or ""),
                                _mixed_regulation_document_style_alias(
                                    regulation_question, anchor_question
                                ),
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
                        QuerySearchQuery(
                            anchor_question, "semantic_question", "拆分后的显式条件检索"
                        ),
                        QuerySearchQuery(
                            " ".join(
                                part
                                for part in [
                                    str(aspect.table_filters.get("source_title") or ""),
                                    anchor,
                                ]
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
    if quoted_matches:
        prefix = question[: quoted_matches[0].start()].rstrip("，,、和与及")
        suffix = question[quoted_matches[-1].end() :].lstrip("，,、和与及")
        return f"{prefix}“{anchor}”{suffix}".strip()
    return f"{title_prefix}{question}（目标：{anchor}）"


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
    reconciliation_selectors = (
        _reconciliation_source_selectors(question) if table_task == "calculate" else []
    )
    if reconciliation_selectors:
        return reconciliation_selectors
    regional_metric_selectors = (
        _regional_metric_comparison_selectors(question)
        if table_task == "compare"
        else []
    )
    if regional_metric_selectors:
        return regional_metric_selectors
    period_selectors = (
        _period_selectors(question) if table_task in {"compare", "calculate"} else []
    )
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
                {
                    "row_label": row_label,
                    "column_label": ratio_columns.group(1).strip(),
                },
                {
                    "row_label": row_label,
                    "column_label": ratio_columns.group(2).strip(),
                },
            ]
    if not selectors and table_task == "calculate":
        # “A占B的比重”：two rows of one table, one shared metric column.
        ratio_rows = _ratio_operand_labels(question)
        if len(ratio_rows) == 2 and ratio_rows[0] != ratio_rows[1]:
            selectors = [
                {"row_label": ratio_rows[0]},
                {"row_label": ratio_rows[1]},
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
            {
                "row_label": quoted[0],
                "column_label": quoted[1],
                "ordered_transition": True,
            },
            {"row_label": quoted[0], "column_label": quoted[2]},
        ]
    elif table_task == "calculate":
        normalized = re.sub(r"\s+", "", question)
        indicators = _mentioned_table_indicators(question)
        source_titles = [
            m.group(1).strip() for m in re.finditer(r"《([^》]+)》", question)
        ][:2]
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
            institution_terms = _institution_row_terms(
                question, indicators[0] if indicators else ""
            )
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
                        if "row_or_indicator" in sel
                        and term not in str(sel["row_or_indicator"])
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
            selectors = [
                {
                    "row_label": row_label,
                    "column_label": column_labels[0],
                }
            ]
        else:
            lookup_match = re.search(
                r"请给出\s*[“\"]([^”\"]+)[”\"]\s*在\s*[“\"]([^”\"]+)[”\"]\s*列",
                question,
            )
        if not selectors and lookup_match:
            selectors = [
                {
                    "row_or_indicator": lookup_match.group(1).strip(),
                    "column_label": lookup_match.group(2).strip(),
                }
            ]
        elif not selectors:
            sheet_match = re.search(r"工作表\s*[“\"‘']([^”\"’']+)", question)
            sheet_name = sheet_match.group(1).strip() if sheet_match else ""
            data_terms = [term for term in quoted if term != sheet_name]
            if len(data_terms) >= 2:
                selectors = [
                    {"row_or_indicator": data_terms[0], "column_label": data_terms[-1]}
                ]
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
    # Rewritten comparison stems enumerate the candidates verbatim
    # (“原保险保费收入、总资产、人身险、财产险四项的哪一项数值最高”)；the
    # enumeration itself is the compared domain, so prefer it over the
    # indicator-vocabulary fallback that also sweeps in words like 数值.
    part = r"[^，。；？！;?！、]{2,30}"
    enumeration = re.search(
        rf"({part}、(?:{part}、){{0,5}}{part})四项的哪一项",
        compact,
    )
    if enumeration:
        parts = [
            item for item in enumeration.group(1).split("、") if item.strip("，,：:的")
        ]
        if 2 <= len(parts) <= 7:
            return _dedupe(parts)
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
    tails = [
        label
        for label in _TABLE_VOCABULARY.get()
        if parts[-1].endswith(label) and label != parts[-1]
    ]
    tail = max(tails, key=len, default="")
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
        quarter = {"一": 1, "二": 2, "三": 3, "四": 4}.get(
            quarter_text, int(quarter_text) if quarter_text.isdigit() else 0
        )
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
    value_text = re.sub(r"《[^》]+》", "", compact)
    # Column headers shaped like periods (“一季度末”, “12月”) are selected by
    # the selector's own period key, never as a shared column target: a
    # “三季度末到四季度末” question would otherwise pin every operand to one
    # quarter column and empty the other selector.
    period_shaped = re.compile(
        r"^(?:20\d{2}年)?(?:第?[一二三四1-4]季度末?|\d{1,2}月末?)$"
    )
    actual_columns = [
        x
        for x in _TABLE_COLUMNS.get()
        if x in value_text
        and x not in TABLE_SCOPE_TERMS
        and not re.fullmatch(r"20\d{2}年|\d{1,2}月", x)
        and not period_shaped.fullmatch(x)
    ]
    actual_columns.sort(key=len, reverse=True)
    if actual_columns:
        column_label = actual_columns[0]
        scope = compact.replace(column_label, "")
        scoped_rows = [
            r for r in _TABLE_ROWS.get() if r not in actual_columns and r in scope
        ]
        if len(scoped_rows) == 1:
            row_label = scoped_rows[0]
    if not column_label and ("本年累计" in compact or "截至当期" in compact):
        column_label = "本年累计"
    elif not column_label and ("账面值" in compact or "账面余额" in compact):
        column_label = "账面余额"
    if row_label:
        # The institution scope (“人身险公司 综合偿付能力充足率”) must ride on
        # every period selector: the global row filter is dropped for
        # calculations, so without it an aggregate row (保险公司) can win over
        # the institution-specific row the question names.  Only a subject
        # DIRECTLY adjacent to the indicator belongs to the row — an
        # institution word separated from it by period words (“农村商业银行
        # 一季度与四季度不良贷款余额”) is the cross-period subject itself and
        # stays with the global row filter.
        institution_terms = _institution_row_terms(value_text, row_label)
        if institution_terms and institution_terms[0] not in row_label:
            label_pos = value_text.find(row_label)
            term_pos = value_text.find(institution_terms[0])
            if label_pos > term_pos >= 0:
                between = value_text[term_pos + len(institution_terms[0]) : label_pos]
                if not re.search(r"[一二三四1-4]季度|\d{1,2}月", between):
                    row_label = f"{institution_terms[0]} {row_label}"
    selectors: list[dict[str, Any]] = []
    for period in unique_periods:
        selector: dict[str, Any] = {
            **period,
            **({"row_label": row_label} if row_label else {}),
            **({"column_label": column_label} if column_label else {}),
        }
        title = _indexed_source_for_question(question, period)
        if title:
            selector["source_title"] = title
        selectors.append(selector)
    return selectors


_PERIOD_TRANSITION_TERMS = (
    "当月新增",
    "本月新增",
    "新增加",
    "环比",
    "较上月",
    "新增",
    "增加",
    "增长",
)


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
        match = re.search(
            r"(\d{1,2})月(?:当月|本月)?(?:新增|新增加|增加|增长)", compact
        )
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
    title_match = re.search(
        r"(?:Excel|表格|报表)(?:\s*附件)?《([^》]+)》", question, flags=re.IGNORECASE
    )
    work_sheet_position = question.find("工作表")
    if work_sheet_position >= 0:
        preceding_titles = list(
            re.finditer(r"《([^》]+)》", question[:work_sheet_position])
        )
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
    elif title_match is None:
        natural_title = re.search(
            r"(20\d{2}年(?:\d{1,2}月|[一二三四1-4]季度)?[^，。；：:]{2,45}?(?:情况表|统计表|报表|表))",
            question,
        )
        if natural_title:
            value = natural_title.group(1).strip("的在读取查询比较")
            period_count = len(re.findall(r"(?:(?:20\d{2}年)?\d{1,2}月)", value))
            if period_count >= 2:
                # Cross-period selectors below carry one concrete title per
                # period.  A composite phrase is not a real document title.
                pass
            else:
                filters["source_title"] = (
                    _indexed_source_for_question(question, {}) or value
                )
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
    publication_match = re.search(
        r"发布日期[：:]?\s*(20\d{2})年(\d{1,2})月(\d{1,2})日", question
    )
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
        sheet_match = re.search(
            r"工作表[：:]\s*(.+?)(?=）\s*[，,]|[，。；;]\s*)", question
        )
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
    if "source_title" not in filters:
        title = _indexed_source_for_question(
            question,
            {k: filters[k] for k in ("year", "month", "quarter") if k in filters},
        )
        if title:
            filters["source_title"] = title
    # “第1-4季度哪个最低” spans every quarter: a single quarter filter would
    # truncate the comparison set to the interval's last member.  Leave the
    # quarter unconstrained so the comparison scans all period columns.
    quarter_interval = re.search(
        r"[1-4一二三四]\s*[-—~至到]\s*[1-4一二三四]\s*季度", compact_question
    )
    quarter_match = (
        None
        if quarter_interval
        else re.search(r"([一二三四1-4])季度|([1-4])季", compact_question)
    )
    if quarter_match:
        value = quarter_match.group(1) or quarter_match.group(2)
        filters["quarter"] = {"一": 1, "二": 2, "三": 3, "四": 4}.get(
            value, int(value) if value.isdigit() else None
        )
    quoted_terms = _quoted_terms(question)
    explicit_row = re.search(r"行项目\s*[“\"‘']([^”\"’']+)", question)
    explicit_columns = [
        value.strip()
        for value in re.findall(r"(?:在)?列\s*[“\"‘']([^”\"’']+)", question)
        if value.strip()
    ]
    lookup_match = re.search(
        r"请给出\s*[“\"]([^”\"]+)[”\"]\s*在\s*[“\"]([^”\"]+)[”\"]\s*列", question
    )
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
            if term.strip() != sheet_name
            and not any(scope in term for scope in TABLE_SCOPE_TERMS)
        ]
        if non_scope_terms:
            filters["indicator"] = non_scope_terms[0]
            if len(non_scope_terms) > 1:
                filters["column_label"] = non_scope_terms[-1]
    # Source and worksheet labels locate the table. Their words do not request
    # an indicator or institution row in that table.
    value_question = re.sub(r"《[^》]+》", "", question)
    if sheet_match:
        value_question = value_question.replace(sheet_match.group(0), "")
    indicators = _mentioned_table_indicators(value_question)
    if indicators and "indicator" not in filters:
        scope, core = _split_indicator_scope(indicators[0])
        if scope:
            filters["scope"] = scope
        # “X中，Y是多少”（资金运用余额中，债券投资）：Y 是所求子行，X 只是
        # 父项上下文，用于锁定表族，不能替代 Y 成为行目标。
        if len(indicators) >= 2 and re.search(
            rf"{re.escape(core)}\s*中[，,]", value_question
        ):
            filters["context"] = core
            core = _split_indicator_scope(indicators[1])[1]
        filters["indicator"] = core
        # The row label stays the bare indicator: composite corpus rows carry
        # the institution as a separate segment ("其中：财产险公司 / 其中：
        # 银行存款"), which is not a contiguous substring, and the institution
        # qualifier is already rewarded by the token-overlap scoring.
        if core in _TABLE_COLUMNS.get() and core not in _TABLE_ROWS.get():
            filters.setdefault("column_label", core)
        else:
            filters["row_label"] = core
    # A family word anywhere in the question (“保险业…”, “商业银行…”) scores
    # the source table family; longest match wins so 大型商业银行 beats 商业银行.
    if "scope" not in filters:
        for scope_word in sorted(_TABLE_FAMILY_SCOPE_WORDS, key=len, reverse=True):
            if scope_word in value_question:
                filters["scope"] = scope_word
                break
    scope_text = re.split(r"(?:请比较|请对比|比较|对比)", value_question, maxsplit=1)[0]
    # A position word right after the title (“《…情况表》中保险公司…”) belongs
    # to the title fragment, not to the institution name.
    scope_text = re.sub(r"《[^》]+》[中里内下于的]?", "", scope_text)
    if explicit_row is None and "context" not in filters:
        matched_label = False
        for label in sorted(_TABLE_ROWS.get(), key=len, reverse=True):
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
                filters["row_label"] = (
                    f"{institution_terms[0]} {current_row_label}".strip()
                )
    if "column_label" not in filters:
        # Structural column words (“机构”, “名称”, “项目”) describe table
        # layout, not a measurement column; “银行业金融机构总资产” must not
        # grow a column filter 机构 that no value column contains.
        structural_column_words = {
            "机构",
            "名称",
            "项目",
            "指标",
            "地区",
            "机构类别",
            "机构类型",
        }
        column_terms = [
            c
            for c in _TABLE_COLUMNS.get()
            if c in value_question
            and c not in TABLE_SCOPE_TERMS
            and c not in structural_column_words
            and not re.fullmatch(r"20\d{2}年|\d{1,2}月", c)
        ]
        if not column_terms:
            # “X 的账面余额是多少” names the measurement column without an
            # explicit “在…列” phrase.  Scope words are valid columns when the
            # sentence directly asks for that value; they remain excluded from
            # row/indicator extraction.
            column_terms = [
                c
                for c in _TABLE_COLUMNS.get()
                if c in TABLE_SCOPE_TERMS
                and c in value_question
                and re.search(
                    rf"{re.escape(c)}\s*(?:是多少|为|的数值|的余额)", value_question
                )
            ]
        column_terms.sort(key=len, reverse=True)
        if column_terms:
            filters["column_label"] = column_terms[0]
    # An institution and a metric may be separate segments of one indexed row.
    # Preserve both rather than replacing the metric with its parent category.
    # With several institution terms the FIRST (question order) anchors the
    # row, matching the single-table selector split above.
    institution_terms = _institution_row_terms(
        value_question, str(filters.get("indicator") or "")
    )
    if institution_terms and institution_terms[0] != filters.get("column_label"):
        metric = str(filters.get("indicator") or "")
        if metric and institution_terms[0] not in metric:
            filters["row_label"] = institution_terms[0] + " " + metric
    # Rating-class locators (“风险综合评级为A类…有多少家”): the class word is
    # part of the indexed row tail (“… / A类公司”), never an institution term.
    # A generic “(保险)公司” mention in such questions is the subject being
    # counted, not a row word — “保险公司” is not a substring of the row tail
    # “A类公司”, so keeping it would fail every candidate under AND semantics.
    class_match = re.search(r"([1-4一二三四A-Da-d])类", scope_text)
    if class_match:
        class_word = f"{class_match.group(1)}类"
        current_row = str(filters.get("row_label") or "").strip()
        words = [
            word
            for word in re.split(r"\s+", current_row)
            if word and word not in {"保险公司", "保险机构", "机构", "公司"}
        ]
        if class_word not in words:
            words.append(class_word)
        filters["row_label"] = " ".join(dict.fromkeys(words))
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
    column_scope = re.search(
        r"[‘“\"]?([^，。；：:‘“’”\"]{2,24})[’”\"]?(?:列|口径)下?(?:中|比较|取数|查询|读取)?",
        question,
    )
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
    source_or_indicator = bool(_mentioned_table_indicators(question))
    action = any(
        term in compact
        for term in (
            "读取",
            "取数",
            "查询",
            "比较",
            "对比",
            "最高",
            "最低",
            "最大",
            "最小",
            "计算",
            "差额",
            "差值",
            "是多少",
            "多少",
        )
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
    from backend.app.services.retrieval_metadata_filter_service import _normalize_title

    compact = re.sub(r"\s+", "", question)
    expression = re.search(r"(?:计算|差额)[：:]?([^，。；]+[－—-][^，。；]+)", compact)
    if not expression:
        return []
    subjects = re.split(r"[－—-]", re.split(r"[:：]", expression.group(1))[-1])
    subjects = [
        re.sub(r"^(?:勾稽|差额|计算|合计)[：:]?|合计$", "", x) for x in subjects
    ]
    indicators = _mentioned_table_indicators(question)
    if not indicators or len(subjects) < 2:
        return []
    period = re.search(r"(20\d{2})年(\d{1,2})月", compact)
    year, month = (int(period[1]), int(period[2])) if period else (None, None)
    result = []
    for subject in subjects:
        titles = [
            title
            for title in _TABLE_SOURCES.get()
            if subject
            and _normalize_title(subject) in _normalize_title(title)
            and (year is None or f"{year}年{month}月" in title)
        ]
        if len(titles) != 1:
            return []
        selector = {
            "source_title": re.sub(r"^(?:\d+_)|\.xlsx?$", "", titles[0]),
            "row_label": indicators[-1],
        }
        if year is not None:
            selector.update(year=year, month=month)
        if "累计" in compact:
            columns = [x for x in _TABLE_VOCABULARY.get() if x.endswith("累计")]
            if len(columns) == 1:
                selector["column_label"] = columns[0]
        result.append(selector)
    return result


def _regional_metric_comparison_selectors(question: str) -> list[dict[str, Any]]:
    compact = re.sub(r"\s+", "", question)
    labels = sorted(_TABLE_VOCABULARY.get(), key=len, reverse=True)
    for metric in labels:
        marker = re.search(re.escape(metric) + r"(?:的|中|口径)", compact)
        if not marker:
            continue
        segment = re.split(
            r"(?:，?给出|。?最后|；|。)", compact[marker.end() :], maxsplit=1
        )[0]
        pieces = re.split(r"[、,，]|以及|与|和|及", segment)
        rows = []
        for piece in pieces:
            matches = [label for label in labels if label in piece and label != metric]
            if matches:
                rows.append(matches[0])
        if len(rows) >= 2:
            return [
                {"label": row, "column_label": metric} for row in dict.fromkeys(rows)
            ]
    return []


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
    return "公式" in normalized and any(
        marker in normalized for marker in ("word", "pdf", "[公式]")
    )


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
    return _looks_like_policy_question(
        combined
    ) or not _looks_like_statistical_table_question(combined)


def _vocabulary_match_terms() -> list[tuple[str, str]]:
    """Vocabulary entries as (display, match_form) pairs.

    Indexed row labels carry layout artefacts — inner spaces
    ("风险综合评级    （家）"), section suffixes ("综合偿付能力充足率（%）"),
    numbering prefixes.  Questions never quote those artefacts, so matching
    uses a stripped form while the display form keeps the full indexed label
    for downstream row matching.
    """

    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for label in _TABLE_VOCABULARY.get():
        display = str(label or "").strip()
        if not display:
            continue
        match_form = re.sub(r"\s+", "", display)
        # Leading numbering ("（3）人身意外伤害险", "其中：银行存款") and a
        # trailing parenthetical ("（家）", "（%）") are layout artefacts a
        # question never quotes.
        match_form = re.sub(
            r"^[（(][0-9一二三四五六七八九十]{1,3}[）)]", "", match_form
        )
        match_form = re.sub(r"^其中\s*[:：]?", "", match_form)
        match_form = re.sub(r"[（(][^）)]*[）)]$", "", match_form)
        if len(match_form) < 2 or match_form in seen:
            continue
        seen.add(match_form)
        pairs.append((display, match_form))
    return pairs


def _mentioned_table_indicators(question: str) -> list[str]:
    normalized = re.sub(r"《[^》]+》", "", question)
    normalized = re.sub(
        r"工作表\s*[：:]\s*.+?(?=）\s*[，,]|[，。；;]|$)", "", normalized
    )
    normalized = re.sub(r"工作表\s*[“\"‘']([^”\"’']+)[”\"’']", "", normalized)
    # Spaces first: users write “2027 年 3 月”, and the period patterns only
    # match the compact form — an unstripped “月” would glue onto the
    # indicator phrase (“月绿色信贷余额”).
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"20\d{2}年|[一二三四1-4]季度|\d{1,2}月", "", normalized)
    vocab = _vocabulary_match_terms()
    matches = [match_form for _display, match_form in vocab if match_form in normalized]
    matches = [value for value in matches if value not in TABLE_SCOPE_TERMS]
    # A category name embedded in an institution name is a scope, not an
    # independent operand (for example, a row named X in "X公司收入").
    matches = [
        value
        for value in matches
        if not re.search(re.escape(value) + r"(?:公司|银行|机构)", normalized)
    ]
    if any(not re.search(r"(?:公司|银行|机构)$", value) for value in matches):
        matches = [
            value for value in matches if not re.search(r"(?:公司|银行|机构)$", value)
        ]
    # A user may name only the distinguishing part of an indexed label.
    # Keep the user's quantity phrase when an actual label contains it — in
    # either direction: "债券投资" narrows to the indexed row "债券", while a
    # longer indexed label keeps its own substring match.
    phrases = re.findall(
        r"[\u4e00-\u9fff]{0,12}(?:余额|金额|收入|支出|投资|总额|费用|合计|总量|数量|净额|比例|比率|占比|率)",
        normalized,
    )
    phrases = [
        re.split(r"公司|银行|以及|与|和|及|中|的|查询|计算|给出|读取", value)[-1]
        for value in phrases
    ]
    match_forms = {match_form for _display, match_form in vocab}
    # Reverse containment stays tight: a user phrase may narrow an indexed
    # label by at most a two-character qualifier ("债券投资" → row "债券").
    # Anything longer ("原保险保费收入是赔付支出" vs "赔付支出") is a
    # different clause, not a narrowing.
    matches.extend(
        value
        for value in phrases
        if len(value) >= 2
        and (
            not vocab
            or any(
                value in match_form
                or (match_form in value and len(value) - len(match_form) <= 2)
                for match_form in match_forms
            )
        )
    )
    matches = list(dict.fromkeys(matches))
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


# Table-family scope words.  A question like “保险业原保险保费收入” glues the
# family word onto the row label; the family disambiguates the SOURCE table
# (保险业经营情况表 vs 人身险公司经营情况表 vs 财产保险公司经营情况表) and must
# therefore score the source, not disappear into an unmatchable row target.
# Matched softly (scoring bonus only) so a question that names a family while
# its data lives in a differently titled table still answers.
_TABLE_FAMILY_SCOPE_WORDS = (
    "银行业金融机构",
    "大型商业银行",
    "股份制商业银行",
    "城市商业银行",
    "民营银行",
    "农村商业银行",
    "农村金融机构",
    "财产保险公司",
    "人身保险公司",
    "财产险公司",
    "人身险公司",
    "再保险公司",
    "保险公司",
    "保险业",
    "银行业",
    "商业银行",
    "保险集团",
    "各地区",
    "全国",
)


def _split_indicator_scope(phrase: str) -> tuple[str, str]:
    """Split a glued compound phrase into ``(scope, indicator)``.

    “保险业原保险保费收入” → (“保险业”, “原保险保费收入”): the row label is
    only matchable without the family prefix, and the prefix instead scores
    the source table.  Returns (“”, phrase) when nothing splits.
    """

    compact = re.sub(r"\s+", "", str(phrase or ""))
    vocab_forms = {match_form for _display, match_form in _vocabulary_match_terms()}
    for scope in sorted(_TABLE_FAMILY_SCOPE_WORDS, key=len, reverse=True):
        if compact.startswith(scope) and len(compact) > len(scope) + 1:
            remainder = compact[len(scope) :]
            if not vocab_forms or remainder in vocab_forms:
                return scope, remainder
    return "", compact


def _ratio_operand_labels(question: str) -> list[str]:
    """Row operands of an “A占B的比重/百分比” question, in question order.

    A compound operand (“人身意外伤害险原保险保费收入”) splits at the longest
    vocabulary prefix whose remainder is itself a label: the prefix is the
    queried row and the remainder the shared metric.  A scope-glued operand
    (“保险业原保险保费收入”) strips the family word instead.  Returns fewer
    than two labels when either side fails to resolve.
    """

    compact = re.sub(r"\s+", "", re.sub(r"《[^》]+》", "", str(question or "")))
    match = re.search(
        r"(.{2,30}?)占(.{2,30}?)(?:的)?(?:比重|百分比|比例|占比)", compact
    )
    if not match:
        return []
    vocab_forms = {match_form for _display, match_form in _vocabulary_match_terms()}

    def operand(raw: str) -> str:
        text = re.sub(r"\s+", "", raw)
        text = re.sub(r"^(?:20\d{2}年)?(?:\d{1,2}月)?[，,：:]?", "", text)
        for form in sorted(vocab_forms, key=len, reverse=True):
            if len(form) >= 3 and text.startswith(form) and len(text) > len(form) + 1:
                remainder = text[len(form) :]
                if remainder in vocab_forms:
                    return form
        scope, core = _split_indicator_scope(text)
        # A family word the question also uses as a ROW locator (“财产险公司
        # 银行存款”) stays with the row: stripping it would drop the
        # institution half of a composite row (“财产险公司 银行存款”).
        if scope and scope not in _institution_row_terms(text, core):
            return core or text
        return f"{scope + ' ' if scope else ''}{core or text}".strip()

    labels = [operand(match.group(1)), operand(match.group(2))]
    if any(not label for label in labels):
        return []
    if vocab_forms and any(label not in vocab_forms for label in labels):
        return []
    return labels


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
    # The subject part may not itself contain 机构后缀字 (银/公/司/机/构), so
    # “财产险公司银行存款” cuts at “财产险公司” instead of greedy-backtracking
    # into “财产险公司银行”, and “大型商业银行/农村金融机构” still match whole.
    _INSTITUTION_RE = re.compile(
        r"(?:(?![银公司机构])[\u4e00-\u9fff]){2,8}(?:公司|银行|机构)"
    )
    for match in _INSTITUTION_RE.finditer(compact):
        # 剥掉“其中：”与并列连词（“人身险公司与财产险公司”中的“与”属于
        # 句法连接，不是机构名的一部分）。
        word = re.sub(r"^(?:其中：|以及|与|和|及|或)+", "", match.group(0))
        # 指标词残片可能被吞进机构词（“保险公司资金运用余额与财产险公司”
        # 中匹配到“余额与财产险公司”）：取最后一个连词之后的部分，把
        # “余额与财产险公司”还原为“财产险公司”。机构名称本身不含连词。
        tail = re.split(r"(?:其中：|以及|与|和|及|或)", word)[-1]
        if tail:
            word = tail
        word = re.sub(r"^.*?表[的中里内下]", "", word)
        # 标题/时间粘连（如 "…情况表中保险公司…"、"四季度末财产险公司…"）可能把
        # 位置/时间词吞进机构名；剥掉通用前缀。 "中国" 是机构名合法前缀
        # （中国银行/中国人寿），保留。 末/底 是 "季度末/年末" 的尾字，
        # 同样属于时间残片，必须剥离，否则 "末财产险公司" 永远匹配不到行。
        word = re.sub(
            r"^(?!中国)(?:[\d一二三四五六七八九十]{1,2}|年|月|季度|上|下|末|底|当|本|今|去|各|全|其|中|里|内|于|的|在|情况|统计|经营)+",
            "",
            word,
        )
        if word and word not in candidates:
            candidates.append(word)
    for label in _TABLE_VOCABULARY.get():
        if not label.endswith("合计"):
            continue
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
                after = segment[position + len(term) :]
                if indicator in after and after.find(indicator) <= 12:
                    same_clause.append(term)
                break
    return _indexed_row_locator_terms(same_clause)


def _indexed_row_locator_terms(terms: list[str]) -> list[str]:
    """Drop institution terms that the index does not carry as row segments.

    Row matching applies AND semantics over every row_label word
    ("财产险公司 资金运用余额"), so a glued word that is not any indexed row
    segment ("…商业银行主要监管指标情况表" keeps 商业银行 only in the title)
    invalidates every candidate row.  Keep such words out of the row locator;
    the institution still disambiguates through title/token scoring.  Without
    a loaded row vocabulary (no table context) keep every term to preserve the
    historical behavior.
    """
    row_parts = {re.sub(r"\s+", "", str(part)) for part in _TABLE_ROWS.get()}
    if not row_parts:
        return terms
    return [term for term in terms if re.sub(r"\s+", "", str(term)) in row_parts]


def _fallback_search_queries(question: str) -> tuple[QuerySearchQuery, ...]:
    keywords = " ".join(_keywords_from_text(question))
    queries = [
        QuerySearchQuery(
            query=question,
            query_type="semantic_question",
            rationale="fallback 子问题检索",
        ),
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
            QuerySearchQuery(
                query=keywords,
                query_type="keyword_anchor",
                rationale="fallback 关键术语兜底",
            )
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


def _mixed_regulation_document_style_alias(
    regulation_question: str, anchor_question: str
) -> str:
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


def _indexed_source_for_question(question: str, period: dict[str, Any]) -> str | None:
    from backend.app.services.retrieval_metadata_filter_service import _normalize_title

    def key(value):
        value = _normalize_title(value)
        value = re.sub(
            r"^\d+_|20\d{2}年|\d{1,2}月|[一二三四1-4]季度|\.(?:xlsx?|csv)$", "", value
        )
        return re.sub(r"情况|公司经营|[《》\s]", "", value)

    if not period:
        periods = re.findall(r"(20\d{2})年(\d{1,2})月", question)
        months = re.findall(r"(\d{1,2})月", question)
        if len(set(periods)) == 1 and len(set(months)) == 1:
            period = {"year": int(periods[0][0]), "month": int(periods[0][1])}
    normalized_question = key(question)
    matches = []
    for title in _TABLE_SOURCES.get():
        if period.get("year") and f"{period['year']}年" not in title:
            continue
        if period.get("month") and not re.search(rf"(?<!\d){period['month']}月", title):
            continue
        if period.get("quarter") and not re.search(rf"{period['quarter']}季度", title):
            continue
        stem = key(title)
        if stem and stem in normalized_question:
            matches.append(title)
    if len(matches) != 1:
        return None
    return re.sub(r"^(?:\d+_)|\.(?:xlsx?|csv)$", "", matches[0])
