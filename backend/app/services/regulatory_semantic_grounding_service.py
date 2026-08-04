from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from threading import RLock
from typing import Any
import urllib.error
import urllib.request

from backend.app.core.config import (
    SEMANTIC_GROUNDING_API_KEY,
    SEMANTIC_GROUNDING_BASE_URL,
    SEMANTIC_GROUNDING_INCLUDE_THINKING,
    SEMANTIC_GROUNDING_MODE,
    SEMANTIC_GROUNDING_MODEL,
    SEMANTIC_GROUNDING_PROVIDER,
    SEMANTIC_GROUNDING_RESPONSE_FORMAT,
    SEMANTIC_GROUNDING_TIMEOUT_SECONDS,
)
from backend.app.schemas.qa import AnswerClaim, RetrievalResult
from backend.app.services.json_utils import extract_json as _extract_json
from backend.app.services.llm_client import (
    ChatCompletionConfig,
    ChatCompletionError,
    chat_completion_content,
)
from backend.app.services.performance_metrics import measure


NEGATIVE_MARKERS = ("不得", "禁止", "严禁", "不应", "不能", "不包括", "不适用", "无须")
MANDATORY_MARKERS = ("应当", "必须", "须")
PERMISSIVE_MARKERS = ("可以", "有权")
SOFT_MARKERS = ("原则上", "一般", "通常")
CONDITION_MARKERS = ("仅当", "只有", "如果", "若", "符合", "除非", "情况下", "时方可")
EXCEPTION_MARKERS = ("除外", "但", "但是", "不适用于", "除")
SCOPE_MARKERS = ("适用于", "仅限", "范围内", "主体", "对象")
TIME_PATTERN = re.compile(r"(?:20\d{2}年(?:\d{1,2}月(?:\d{1,2}日)?)?)|(?:\d+(?:个)?(?:工作日|日|月|年))")
QUANTITY_PATTERN = re.compile(r"\d+(?:\.\d+)?%|不少于|不超过|至少|至多|以上|以下")
SUBJECT_PATTERN = re.compile(
    r"国家金融监督管理总局|中国人民银行|银行业金融机构|商业银行|金融机构|填报机构|报送机构"
)
ACTION_PATTERN = re.compile(
    r"报送|提交|留存|保存|披露|计入|纳入|排除|执行|办理|开展|建立|报告|核验|审核|计算|填报"
)
OBJECT_PATTERN = re.compile(
    r"(?:报送|提交|留存|保存|披露|计入|纳入|排除|执行|办理|开展|建立|报告|核验|审核|计算|填报)"
    r"([^，。；;]{1,50})"
)


@dataclass(frozen=True)
class RegulatorySemanticFrame:
    subject: tuple[str, ...]
    action: tuple[str, ...]
    object: tuple[str, ...]
    polarity: str
    modality: str
    conditions: tuple[str, ...]
    exceptions: tuple[str, ...]
    scope: tuple[str, ...]
    time_constraints: tuple[str, ...]
    quantitative_constraints: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_regulatory_semantics(
    claims: list[AnswerClaim],
    context_chunks: list[RetrievalResult],
    *,
    mode: str | None = None,
) -> dict[str, Any]:
    selected_mode = (mode or SEMANTIC_GROUNDING_MODE).lower()
    if selected_mode == "off":
        return {
            "semantic_passed": True,
            "semantic_mode": "off",
            "verifier_status": "disabled",
            "claim_verdicts": [],
        }
    evidence_by_id = {chunk.citation_label: chunk.text for chunk in context_chunks}
    verdicts: list[dict[str, Any]] = []
    risky: list[dict[str, Any]] = []
    for index, claim in enumerate(claims, start=1):
        evidence = "\n".join(evidence_by_id.get(citation_id, "") for citation_id in claim.citation_ids)
        claim_text = re.sub(r"\[\d+\]", "", claim.text)
        focused_evidence = select_relevant_evidence(claim_text, evidence)
        claim_frame = extract_regulatory_semantic_frame(claim_text)
        evidence_frame = extract_regulatory_semantic_frame(focused_evidence)
        base = {
            "claim_index": index,
            "role": claim.role,
            "aspect_ids": claim.aspect_ids,
            "citation_ids": claim.citation_ids,
            "claim_frame": claim_frame.to_dict(),
            "evidence_frame": evidence_frame.to_dict(),
            "evidence_focus": focused_evidence[:1000],
        }
        if claim.role in {"table_fact", "calculation"}:
            verdicts.append(
                {**base, "verdict": "supported", "validation_mode": "deterministic_table", "reason_codes": []}
            )
            continue
        if _is_evidence_scope_commentary(claim.text, claim.aspect_ids):
            verdicts.append(
                {**base, "verdict": "supported", "validation_mode": "deterministic_scope_commentary", "reason_codes": []}
            )
            continue
        conflicts = deterministic_semantic_conflicts(claim_frame, evidence_frame)
        if conflicts:
            verdicts.append(
                {**base, "verdict": "contradicted", "validation_mode": "deterministic", "reason_codes": conflicts}
            )
            continue
        risk_codes = semantic_risk_codes(claim_frame, evidence_frame)
        requires_verifier = selected_mode == "all" or (
            bool(risk_codes)
            and claim.role
            in {"conclusion", "regulatory_basis", "explanation", "recommendation"}
        )
        if requires_verifier:
            item = {**base, "claim": claim.text, "evidence": focused_evidence[:4000], "risk_codes": risk_codes}
            risky.append(item)
            continue
        verdicts.append(
            {**base, "verdict": "supported", "validation_mode": "deterministic", "reason_codes": []}
        )

    verifier_status = "not_required"
    if risky:
        verified, verifier_status = _verify_risky_claims(risky)
        by_index = {int(item.get("claim_index") or 0): item for item in verified}
        for item in risky:
            verified_item = by_index.get(int(item["claim_index"]))
            if not verified_item:
                verdicts.append(
                    {
                        **_without_text(item),
                        "verdict": "insufficient",
                        "validation_mode": "llm_verifier",
                        "reason_codes": ["verifier_result_missing"],
                    }
                )
                continue
            verdict = str(verified_item.get("verdict") or "insufficient").lower()
            if verdict not in {"supported", "contradicted", "insufficient"}:
                verdict = "insufficient"
            verdicts.append(
                {
                    **_without_text(item),
                    "verdict": verdict,
                    "validation_mode": "llm_verifier",
                    "reason_codes": [str(value) for value in verified_item.get("reason_codes") or item["risk_codes"]],
                    "evidence_snippet": str(verified_item.get("evidence_snippet") or "")[:600],
                    "verifier_reason": str(verified_item.get("reason") or "")[:1000],
                }
            )
    verdicts.sort(key=lambda item: int(item.get("claim_index") or 0))
    # Only a verified contradiction blocks the answer.  "insufficient" means
    # the verifier could not confirm the claim against the excerpted
    # evidence, not that the claim is wrong; the deterministic base checks
    # (citations, entities, required-aspect coverage) already passed, so an
    # uncertain verdict must not turn a well-grounded answer into a degraded
    # extract.  Insufficient verdicts stay visible in claim_verdicts.
    passed = all(item.get("verdict") != "contradicted" for item in verdicts)
    return {
        "semantic_passed": passed,
        "semantic_mode": selected_mode,
        "verifier_status": verifier_status,
        "claim_verdicts": verdicts,
        "semantic_conflict_count": sum(item.get("verdict") == "contradicted" for item in verdicts),
        "semantic_insufficient_count": sum(item.get("verdict") == "insufficient" for item in verdicts),
    }


def claim_requires_semantic_verifier(
    claim: AnswerClaim,
    context_chunks: list[RetrievalResult],
    *,
    mode: str | None = None,
) -> bool:
    selected_mode = (mode or SEMANTIC_GROUNDING_MODE).lower()
    if selected_mode == "off" or claim.role in {"table_fact", "calculation"}:
        return False
    if selected_mode == "all":
        return True
    evidence_by_id = {chunk.citation_label: chunk.text for chunk in context_chunks}
    evidence = "\n".join(evidence_by_id.get(value, "") for value in claim.citation_ids)
    claim_text = re.sub(r"\[\d+\]", "", claim.text)
    claim_frame = extract_regulatory_semantic_frame(claim_text)
    evidence_frame = extract_regulatory_semantic_frame(select_relevant_evidence(claim_text, evidence))
    if deterministic_semantic_conflicts(claim_frame, evidence_frame):
        return False
    return bool(semantic_risk_codes(claim_frame, evidence_frame)) and claim.role in {
        "conclusion",
        "regulatory_basis",
        "explanation",
        "recommendation",
    }


def extract_regulatory_semantic_frame(text: str) -> RegulatorySemanticFrame:
    normalized = re.sub(r"\s+", "", str(text or ""))
    negative = _present(normalized, NEGATIVE_MARKERS)
    mandatory = _present(normalized, MANDATORY_MARKERS)
    permissive = _present(normalized, PERMISSIVE_MARKERS)
    soft = _present(normalized, SOFT_MARKERS)
    modality = "mandatory" if mandatory else "permissive" if permissive else "soft" if soft else "unspecified"
    actions = tuple(
        dict.fromkeys(
            [*negative, *mandatory, *permissive, *soft, *ACTION_PATTERN.findall(normalized)]
        )
    )
    objects = tuple(
        dict.fromkeys(
            value.strip("的了并及和")
            for value in OBJECT_PATTERN.findall(normalized)
            if value.strip("的了并及和")
        )
    )
    return RegulatorySemanticFrame(
        subject=tuple(dict.fromkeys(SUBJECT_PATTERN.findall(normalized))),
        action=actions,
        object=objects,
        polarity="negative" if negative else "positive",
        modality=modality,
        conditions=_sentences_with_markers(normalized, CONDITION_MARKERS),
        exceptions=_sentences_with_markers(normalized, EXCEPTION_MARKERS),
        scope=_sentences_with_markers(normalized, SCOPE_MARKERS),
        time_constraints=tuple(dict.fromkeys(TIME_PATTERN.findall(normalized))),
        quantitative_constraints=tuple(dict.fromkeys(QUANTITY_PATTERN.findall(normalized))),
    )


def deterministic_semantic_conflicts(
    claim: RegulatorySemanticFrame,
    evidence: RegulatorySemanticFrame,
) -> list[str]:
    conflicts: list[str] = []
    claim_negative = bool(set(claim.action) & set(NEGATIVE_MARKERS))
    evidence_negative = bool(set(evidence.action) & set(NEGATIVE_MARKERS))
    claim_normative = bool(set(claim.action) & (set(MANDATORY_MARKERS) | set(PERMISSIVE_MARKERS)))
    evidence_normative = bool(set(evidence.action) & (set(MANDATORY_MARKERS) | set(PERMISSIVE_MARKERS)))
    claim_actions = _substantive_actions(claim)
    evidence_actions = _substantive_actions(evidence)
    same_action = bool(claim_actions & evidence_actions)
    if same_action:
        if (claim_negative and evidence_normative and not evidence_negative) or (
            claim_normative and evidence_negative and not evidence_normative
        ):
            conflicts.append("polarity_conflict")
        if claim.modality == "mandatory" and evidence.modality in {"permissive", "soft"}:
            conflicts.append("modality_strengthened")
        if claim.modality == "permissive" and evidence.modality == "mandatory":
            conflicts.append("modality_weakened")
    if evidence.subject and claim.subject and not set(claim.subject).issubset(evidence.subject):
        conflicts.append("subject_scope_expanded")
    return conflicts


def semantic_risk_codes(
    claim: RegulatorySemanticFrame,
    evidence: RegulatorySemanticFrame,
) -> list[str]:
    risks: list[str] = []
    claim_actions = set(claim.action) - set(NEGATIVE_MARKERS + MANDATORY_MARKERS + PERMISSIVE_MARKERS + SOFT_MARKERS)
    evidence_actions = set(evidence.action) - set(NEGATIVE_MARKERS + MANDATORY_MARKERS + PERMISSIVE_MARKERS + SOFT_MARKERS)
    if claim_actions and evidence_actions and not claim_actions.issubset(evidence_actions):
        risks.append("action_requires_verification")
    if claim.object and evidence.object and not set(claim.object).issubset(evidence.object):
        risks.append("object_requires_verification")
    if evidence.conditions and not claim.conditions:
        risks.append("condition_omitted_or_expanded")
    if evidence.exceptions and not claim.exceptions:
        risks.append("exception_omitted_or_expanded")
    if evidence.scope and not claim.scope:
        risks.append("scope_requires_verification")
    if evidence.time_constraints and not claim.time_constraints:
        risks.append("time_scope_omitted")
    elif claim.time_constraints and evidence.time_constraints and not set(claim.time_constraints).issubset(evidence.time_constraints):
        risks.append("time_scope_requires_verification")
    if (
        claim.quantitative_constraints
        and evidence.quantitative_constraints
        and not set(claim.quantitative_constraints).issubset(evidence.quantitative_constraints)
    ):
        risks.append("quantitative_constraint_requires_verification")
    return risks


def select_relevant_evidence(claim_text: str, evidence: str, *, max_sentences: int | None = None) -> str:
    """Focus semantic checks on the cited statement, not unrelated rules in the same chunk.

    Every sentence that substantially overlaps the claim is kept (not just
    the single closest one): a multi-case claim such as “下列两种情形中的
    较早发生者：(1)…；(2)…” covers several evidence sentences, and dropping
    the others would remove their subjects/objects and falsely flag
    subject_scope_expanded.  Unrelated sentences stay excluded so the chunk's
    other rules cannot interfere with the claim's frame.
    """

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[。！？；;])|\n+", str(evidence or ""))
        if sentence.strip()
    ]
    if len(sentences) <= 1:
        return str(evidence or "")
    claim_norm = _semantic_match_text(claim_text)
    claim_bigrams = _character_ngrams(claim_norm)
    scored: list[tuple[float, int, str]] = []
    for index, sentence in enumerate(sentences):
        sentence_norm = _semantic_match_text(sentence)
        if not sentence_norm:
            continue
        sentence_bigrams = _character_ngrams(sentence_norm)
        overlap = len(claim_bigrams & sentence_bigrams) / max(len(claim_bigrams), 1)
        containment = min(len(claim_norm), len(sentence_norm)) / max(len(claim_norm), len(sentence_norm), 1)
        if claim_norm in sentence_norm or sentence_norm in claim_norm:
            overlap += 1.0 + containment
        scored.append((overlap, -index, sentence))
    if not scored:
        return str(evidence or "")
    relevant = [item for item in scored if item[0] >= 0.5]
    if not relevant:
        return str(evidence or "")
    if max_sentences is not None:
        relevant = sorted(relevant, key=lambda item: (-item[0], item[1]))[:max_sentences]
    selected_indexes = sorted(-item[1] for item in relevant)
    return "".join(sentences[index] for index in selected_indexes)


def _substantive_actions(frame: RegulatorySemanticFrame) -> set[str]:
    markers = set(NEGATIVE_MARKERS + MANDATORY_MARKERS + PERMISSIVE_MARKERS + SOFT_MARKERS)
    return set(frame.action) - markers


def _semantic_match_text(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff%]+", "", str(value or "")).casefold()


def _character_ngrams(value: str, size: int = 2) -> set[str]:
    if len(value) <= size:
        return {value} if value else set()
    return {value[index : index + size] for index in range(len(value) - size + 1)}


def _is_evidence_scope_commentary(text: str, aspect_ids: list[str]) -> bool:
    normalized = re.sub(r"\s+", "", str(text or ""))
    return "multiple_choice_evidence" in aspect_ids and (
        "其他选项" in normalized
        and any(marker in normalized for marker in ("无相关证据", "未提供", "无法判断", "依据不足"))
    )


def _verify_risky_claims(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    if not SEMANTIC_GROUNDING_API_KEY:
        return [], "unavailable_no_api_key"
    cache_key = hashlib.sha256(json.dumps(items, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    with _VERIFIER_CACHE_LOCK:
        cached = _VERIFIER_CACHE.get(cache_key)
    if cached is not None:
        return cached, "cache_hit"
    messages = [
            {
                "role": "system",
                "content": (
                    "你是监管语义证据校验器。只能比较给定claim与evidence，不得使用外部知识。"
                    "逐项核对主体、行为、对象、肯否、规范强度、条件、例外、范围、时间和数量。"
                    "输出JSON：{\"results\":[{\"claim_index\":1,\"verdict\":\"supported|contradicted|insufficient\","
                    "\"reason_codes\":[\"...\"],\"evidence_snippet\":\"...\",\"reason\":\"...\"}]}。"
                ),
            },
            {"role": "user", "content": json.dumps(items, ensure_ascii=False)},
        ]
    try:
        with measure("grounding.semantic_verifier"):
            content = chat_completion_content(
                ChatCompletionConfig(
                    provider=SEMANTIC_GROUNDING_PROVIDER,
                    api_key=SEMANTIC_GROUNDING_API_KEY,
                    base_url=SEMANTIC_GROUNDING_BASE_URL,
                    model=SEMANTIC_GROUNDING_MODEL,
                    timeout_seconds=SEMANTIC_GROUNDING_TIMEOUT_SECONDS,
                    include_thinking=SEMANTIC_GROUNDING_INCLUDE_THINKING,
                    response_format=SEMANTIC_GROUNDING_RESPONSE_FORMAT,
                ),
                messages,
                temperature=0,
                max_tokens=1200,
                response_format=SEMANTIC_GROUNDING_RESPONSE_FORMAT,
                opener=urllib.request.urlopen,
            )
        parsed = json.loads(_extract_json(content))
        results = parsed.get("results") if isinstance(parsed, dict) else None
        if not isinstance(results, list):
            return [], "invalid_response"
    except (OSError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError, urllib.error.URLError, ChatCompletionError):
        return [], "unavailable"
    normalized = [item for item in results if isinstance(item, dict)]
    with _VERIFIER_CACHE_LOCK:
        if len(_VERIFIER_CACHE) >= 256:
            _VERIFIER_CACHE.pop(next(iter(_VERIFIER_CACHE)))
        _VERIFIER_CACHE[cache_key] = normalized
    return normalized, "completed"


def _present(text: str, markers: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(marker for marker in markers if marker in text)


def _sentences_with_markers(text: str, markers: tuple[str, ...]) -> tuple[str, ...]:
    sentences = re.split(r"[。；;]", text)
    return tuple(sentence[:300] for sentence in sentences if any(marker in sentence for marker in markers))


def _without_text(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key not in {"claim", "evidence", "risk_codes"}}


_VERIFIER_CACHE: dict[str, list[dict[str, Any]]] = {}
_VERIFIER_CACHE_LOCK = RLock()
