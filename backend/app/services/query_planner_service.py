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

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "original_question": self.original_question,
            "planner": self.planner,
            "fallback_used": self.fallback_used,
            "error": self.error,
            "aspects": [aspect.to_debug_dict() for aspect in self.aspects],
        }


def plan_query(question: str) -> QueryPlan:
    cleaned_question = question.strip()
    if not cleaned_question:
        return QueryPlan(original_question="", aspects=(), planner="empty", fallback_used=True)

    if QUERY_PLANNER_ENABLED and QUERY_PLANNER_API_KEY:
        try:
            aspects = _plan_with_deepseek(cleaned_question)
            if aspects:
                return QueryPlan(
                    original_question=cleaned_question,
                    aspects=tuple(aspects),
                    planner=f"deepseek:{QUERY_PLANNER_MODEL}",
                    fallback_used=False,
                )
        except QueryPlannerError as exc:
            fallback = _fallback_aspects(cleaned_question)
            return QueryPlan(
                original_question=cleaned_question,
                aspects=tuple(fallback),
                planner="fallback",
                fallback_used=True,
                error=str(exc),
            )

    return QueryPlan(
        original_question=cleaned_question,
        aspects=tuple(_fallback_aspects(cleaned_question)),
        planner="fallback",
        fallback_used=True,
    )


def _plan_with_deepseek(question: str) -> list[QueryAspect]:
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
                    "只输出 JSON 对象。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "请把下面问题拆成 1 到 "
                    f"{QUERY_PLANNER_MAX_ASPECTS} 个 aspect。每个 aspect 包含："
                    "aspect_id、question、evidence_need、search_queries、keywords。"
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
                    "输出格式：{\"aspects\":[...]}\n\n"
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

    return _aspects_from_payload(question, parsed)


def _aspects_from_payload(question: str, payload: dict[str, Any]) -> list[QueryAspect]:
    raw_aspects = payload.get("aspects")
    if not isinstance(raw_aspects, list):
        raise QueryPlannerError("LLM query planner 未返回 aspects")

    aspects: list[QueryAspect] = []
    for index, raw_aspect in enumerate(raw_aspects[:QUERY_PLANNER_MAX_ASPECTS], start=1):
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


def _fallback_aspects(question: str) -> list[QueryAspect]:
    heuristic_aspects = _domain_heuristic_aspects(question)
    if heuristic_aspects:
        return heuristic_aspects

    parts = [
        part.strip()
        for part in re.split(r"[？?。；;]|如果|若|以及|并且|同时|，且", question)
        if part.strip()
    ]
    if not parts:
        parts = [question]

    aspects: list[QueryAspect] = []
    for index, part in enumerate(parts[:QUERY_PLANNER_MAX_ASPECTS], start=1):
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


def _domain_heuristic_aspects(question: str) -> list[QueryAspect]:
    normalized = re.sub(r"\s+", "", question)
    aspects: list[QueryAspect] = []
    if "资产合计" in normalized and "差异" in normalized:
        aspects.append(
            QueryAspect(
                aspect_id="asset_total_difference_check",
                question="资产合计差异应该优先排查哪些问题",
                search_queries=(
                    QuerySearchQuery(
                        query="资产合计与分项合计存在差异时应优先检查哪些原因",
                        query_type="semantic_question",
                        rationale="贴近用户问题的处理口径",
                    ),
                    QuerySearchQuery(
                        query="资产合计校验差异处理流程 币种折算 四舍五入 科目映射 重复汇总",
                        query_type="document_style_statement",
                        rationale="贴近填报说明中的证据句",
                    ),
                    QuerySearchQuery(
                        query="资产合计 差异 币种折算 四舍五入 科目映射 重复汇总",
                        query_type="keyword_anchor",
                        rationale="关键术语兜底",
                    ),
                ),
                evidence_need="资产合计差异处理或优先排查章节",
                keywords=("资产合计", "差异", "优先检查", "优先排查", "币种折算", "四舍五入", "科目映射", "重复汇总"),
            )
        )
    if "外币折算" in normalized and any(term in normalized for term in ["保留", "依据", "留痕"]):
        aspects.append(
            QueryAspect(
                aspect_id="foreign_currency_evidence",
                question="外币折算差异需要保留什么依据",
                search_queries=(
                    QuerySearchQuery(
                        query="外币折算导致资产合计差异时需要保留哪些支持材料",
                        query_type="semantic_question",
                        rationale="贴近用户问题的留痕依据",
                    ),
                    QuerySearchQuery(
                        query="外币折算差异 留痕依据 汇率日期 折算规则 原币金额来源",
                        query_type="document_style_statement",
                        rationale="贴近制度材料表述",
                    ),
                    QuerySearchQuery(
                        query="外币折算 汇率日期 折算规则 原币金额来源",
                        query_type="keyword_anchor",
                        rationale="关键术语兜底",
                    ),
                ),
                evidence_need="外币折算差异的留痕依据或支持材料",
                keywords=("外币折算", "保留", "依据", "汇率日期", "折算规则", "原币金额", "来源"),
            )
        )
    return aspects


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
