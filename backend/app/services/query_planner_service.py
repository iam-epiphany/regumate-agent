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


@dataclass(frozen=True)
class QueryAspect:
    aspect_id: str
    question: str
    search_queries: tuple[str, ...]
    expected_evidence_type: str
    keywords: tuple[str, ...]

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "aspect_id": self.aspect_id,
            "question": self.question,
            "search_queries": list(self.search_queries),
            "expected_evidence_type": self.expected_evidence_type,
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
                    "你的唯一任务是把用户问题拆成检索 aspect，生成检索计划。"
                    "禁止回答用户问题，禁止给出结论，禁止补充知识库外事实。"
                    "只输出 JSON 对象。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "请把下面问题拆成 1 到 "
                    f"{QUERY_PLANNER_MAX_ASPECTS} 个 aspect。每个 aspect 包含："
                    "aspect_id、question、search_queries、expected_evidence_type、keywords。"
                    "search_queries 用于向量检索，应短而具体；expected_evidence_type 描述要找章节定义、填报口径、处理流程、例外条件或留痕依据。"
                    "如果问题只有一个主题，也返回一个 aspect。"
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
        search_queries = _clean_string_list(raw_aspect.get("search_queries"))
        keywords = _clean_string_list(raw_aspect.get("keywords"))
        if not search_queries:
            search_queries = [sub_question]
        if not keywords:
            keywords = _keywords_from_text(" ".join([sub_question, *search_queries]))
        aspects.append(
            QueryAspect(
                aspect_id=_clean_aspect_id(raw_aspect.get("aspect_id"), index),
                question=sub_question,
                search_queries=tuple(_dedupe(search_queries)[:QUERY_PLANNER_MAX_SEARCH_QUERIES]),
                expected_evidence_type=_clean_text(raw_aspect.get("expected_evidence_type")) or "相关制度依据",
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
                search_queries=(part,),
                expected_evidence_type=_infer_evidence_type(part),
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
                search_queries=("资产合计 差异 优先检查 币种折算 四舍五入 科目映射 重复汇总",),
                expected_evidence_type="资产合计差异处理或优先排查章节",
                keywords=("资产合计", "差异", "优先检查", "优先排查", "币种折算", "四舍五入", "科目映射", "重复汇总"),
            )
        )
    if "外币折算" in normalized and any(term in normalized for term in ["保留", "依据", "留痕"]):
        aspects.append(
            QueryAspect(
                aspect_id="foreign_currency_evidence",
                question="外币折算差异需要保留什么依据",
                search_queries=("外币折算 保留 汇率日期 折算规则 原币金额来源",),
                expected_evidence_type="外币折算差异的留痕依据或支持材料",
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


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


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
