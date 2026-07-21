from __future__ import annotations

import math
import re
from typing import Any

from backend.app.schemas.qa import AnswerClaim, RetrievalResult
from backend.app.services.performance_metrics import timed
from backend.app.services.regulatory_semantic_grounding_service import (
    validate_regulatory_semantics,
)


ENTITY_PATTERN = re.compile(
    r"(?:20\d{2}年(?:\d{1,2}月(?:\d{1,2}日)?)?)|(?:\d+(?:\.\d+)?%?)|"
    r"(?:[A-Za-z一-龥]+〔\d{4}〕\d+号)"
)
ORGANIZATION_PATTERN = re.compile(
    "|".join(
        re.escape(value)
        for value in (
            "国家金融监督管理总局",
            "中国银行保险监督管理委员会",
            "中国证券监督管理委员会",
            "中国人民银行",
            "中国银保监会",
            "会计师事务所",
            "银行业金融机构",
            "消费金融公司",
            "商业银行",
            "金融机构",
            "财政部",
        )
    )
)


@timed("grounding.validation")
def validate_grounded_answer(
    answer: str,
    claims: list[AnswerClaim],
    context_chunks: list[RetrievalResult],
) -> dict[str, Any]:
    valid_ids = {chunk.citation_label for chunk in context_chunks}
    invalid_citation_ids = sorted(
        {
            citation_id
            for claim in claims
            for citation_id in claim.citation_ids
            if citation_id not in valid_ids
        }
    )
    answer_citation_ids = set(re.findall(r"\[\d+\]", answer))
    invalid_answer_citation_ids = sorted(answer_citation_ids - valid_ids)
    claimed_citation_ids = {
        citation_id for claim in claims for citation_id in claim.citation_ids
    }
    missing_inline_citation_ids = sorted(claimed_citation_ids - answer_citation_ids)
    evidence_by_id = {
        chunk.citation_label: _normalize(chunk.text) for chunk in context_chunks
    }
    evidence = _normalize("\n".join(chunk.text for chunk in context_chunks))
    answer_without_citations = _strip_presentation_markers(
        re.sub(r"\[\d+\]", "", answer)
    )
    unsupported_entities = sorted(
        {
            entity
            for entity in _extract_entities(answer_without_citations)
            if not _entity_is_supported(entity, evidence)
        }
    )
    missing_claim_citations = [index + 1 for index, claim in enumerate(claims) if not claim.citation_ids]
    missing_claims = not claims
    unsupported_claim_entities: list[dict[str, Any]] = []
    for index, claim in enumerate(claims, start=1):
        cited_evidence = "\n".join(
            evidence_by_id[citation_id]
            for citation_id in claim.citation_ids
            if citation_id in evidence_by_id
        )
        unsupported = sorted(
            {
                entity
                for entity in _extract_entities(re.sub(r"\[\d+\]", "", claim.text))
                if not _entity_is_supported(entity, cited_evidence)
            }
        )
        if unsupported:
            unsupported_claim_entities.append(
                {
                    "claim_index": index,
                    "citation_ids": claim.citation_ids,
                    "entities": unsupported,
                }
            )
    base_passed = (
        not invalid_citation_ids
        and not invalid_answer_citation_ids
        and not missing_inline_citation_ids
        and not unsupported_entities
        and not missing_claims
        and not missing_claim_citations
        and not unsupported_claim_entities
    )
    semantic_validation = validate_regulatory_semantics(claims, context_chunks)
    passed = base_passed and bool(semantic_validation.get("semantic_passed"))
    return {
        "passed": passed,
        "checked_claims": len(claims),
        "invalid_citation_ids": invalid_citation_ids,
        "invalid_answer_citation_ids": invalid_answer_citation_ids,
        "missing_inline_citation_ids": missing_inline_citation_ids,
        "unsupported_entities": unsupported_entities,
        "unsupported_claim_entities": unsupported_claim_entities,
        "missing_claims": missing_claims,
        "missing_claim_citations": missing_claim_citations,
        **semantic_validation,
    }


def _normalize(value: Any) -> str:
    normalized = str(value or "").translate(
        str.maketrans({"％": "%", "－": "-", "−": "-", "﹣": "-"})
    )
    return re.sub(r"[\s,，]", "", normalized).lower()


def _strip_presentation_markers(value: str) -> str:
    """Remove list ordinals that carry layout, not regulatory facts.

    Numeric validation must still inspect numbers inside a sentence.  Only a
    marker at the beginning of a rendered line is ignored, so ``1. 摘录`` is
    not treated as a factual claim while ``期限为1年`` remains validated.
    """

    return re.sub(
        r"(?m)^\s*(?:(?:\d+|[一二三四五六七八九十]+)[.、)]|[-*•])\s+",
        "",
        value,
    )


def _entity_is_supported(entity: str, normalized_evidence: str) -> bool:
    normalized = _normalize(entity)
    if normalized in normalized_evidence:
        return True
    numeric = re.fullmatch(r"(-?\d+(?:\.(\d+))?)(%?)", normalized)
    if not numeric:
        return False
    value = float(numeric.group(1))
    decimal_places = len(numeric.group(2) or "")
    percent = numeric.group(3)
    tolerance = 0.0 if decimal_places == 0 else 0.5 * (10 ** -decimal_places) + 1e-9
    for candidate, _, candidate_percent in re.findall(
        r"(-?\d+(?:\.(\d+))?)(%?)", normalized_evidence
    ):
        if candidate_percent != percent:
            continue
        if math.isclose(value, float(candidate), rel_tol=0.0, abs_tol=tolerance):
            return True
    return False


def _extract_entities(value: str) -> set[str]:
    return set(ENTITY_PATTERN.findall(value)) | set(ORGANIZATION_PATTERN.findall(value))
