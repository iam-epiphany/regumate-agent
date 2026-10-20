"""Projection of existing QA objects into display-only explanation data.

This module never calls a model and never reads prompts, credentials or
internal reasoning.  It joins the QueryPlan requirements, EvidenceBundle
coverage, cited chunks, structured claims and the final ValidationReport so
the UI can link a conclusion to its evidence and calculation record.
"""
from __future__ import annotations

from time import perf_counter
from typing import Any

from backend.app.schemas.qa import (
    AnswerExplanation,
    ExplanationBudget,
    ExplanationCalculation,
    ExplanationClaimLink,
    ExplanationOperand,
    ExplanationRequirement,
    ExplanationSource,
    ExplanationSourceGroup,
    ExplanationStage,
    ExplanationValidation,
)
from backend.app.services.answer_validation_service import validation_passed
from backend.app.services.performance_metrics import (
    model_call_count,
    request_budget_snapshot,
)

_EVIDENCE_MISSING_REASONS = {
    "insufficient_context",
    "insufficient_evidence",
    "missing_required_aspect",
    "related_context_only",
    "table_evidence_not_found",
    "table_evidence_not_valid",
    "metadata_filter_no_match",
}
_CALCULATION_REASONS = {
    "missing_variables",
    "ambiguous_variables",
    "unit_conflict",
    "unsupported_operation",
    "invalid_calculation_trace",
    "calculation_result_not_in_options",
    "comparison_winner_not_in_options",
}


def _compact(value: Any, limit: int = 500) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def _label_map(package: Any) -> dict[str, str]:
    return {
        str(chunk.chunk_id): str(chunk.citation_label)
        for chunk in (getattr(package, "context_chunks", None) or [])
        if getattr(chunk, "chunk_id", None) and getattr(chunk, "citation_label", None)
    }


def _related_requirement_ids(chunk: Any, reverse_coverage: dict[str, list[str]]) -> list[str]:
    metadata = getattr(chunk, "metadata", None) or {}
    ids: list[str] = []
    for key in ("prompt_matched_aspects", "aspect_ids", "supported_requirement_ids"):
        ids.extend(str(item) for item in (metadata.get(key) or []) if item)
    if metadata.get("aspect_id"):
        ids.append(str(metadata["aspect_id"]))
    ids.extend(reverse_coverage.get(str(getattr(chunk, "chunk_id", "")), []))
    return list(dict.fromkeys(ids))


def _evidence_id(label_map: dict[str, str], chunk_id: str) -> str:
    return label_map.get(str(chunk_id), str(chunk_id))


def _missing_reason(requirement_id: str, decisions: list[dict[str, Any]]) -> str | None:
    for decision in reversed(decisions):
        if str(decision.get("requirement_id")) != requirement_id:
            continue
        reason = decision.get("reason")
        if reason and reason != "source_and_requirement_match":
            return str(reason)
    return None


def _build_requirements(
    summary: dict[str, Any],
    package: Any,
    reverse_coverage: dict[str, list[str]],
) -> list[ExplanationRequirement]:
    bundle = summary.get("evidence_bundle") or {}
    raw_requirements = list(bundle.get("requirements") or [])
    coverage = {
        str(key): [str(item) for item in (value or [])]
        for key, value in (bundle.get("coverage") or {}).items()
    }
    missing = {str(item) for item in (bundle.get("missing_requirements") or [])}
    missing_optional = {str(item) for item in (bundle.get("missing_optional_requirements") or [])}
    decisions = list(bundle.get("decisions") or [])
    label_map = _label_map(package)
    requirements: list[ExplanationRequirement] = []
    for raw in raw_requirements:
        requirement_id = str(raw.get("requirement_id") or "")
        if not requirement_id:
            continue
        evidence_ids = [
            _evidence_id(label_map, chunk_id)
            for chunk_id in coverage.get(requirement_id, [])
        ]
        mandatory = bool(raw.get("mandatory", True))
        if requirement_id in missing:
            status = "missing"
        elif requirement_id in missing_optional:
            status = "optional_missing"
        elif evidence_ids:
            status = "covered"
        else:
            status = "missing" if mandatory else "optional_missing"
        requirements.append(ExplanationRequirement(
            requirement_id=requirement_id,
            question=str(raw.get("question") or ""),
            source_titles=[str(item) for item in (raw.get("source_titles") or [])],
            mandatory=mandatory,
            status=status,
            evidence_ids=list(dict.fromkeys(evidence_ids)),
            missing_reason=_missing_reason(requirement_id, decisions),
        ))
    if not requirements:
        # Old or lightweight context packages may not carry an EvidenceBundle.
        # The QueryPlan still gives a truthful "recorded/not recorded" view.
        query_plan = summary.get("query_plan") or {}
        for raw in query_plan.get("aspects") or []:
            aspect_id = str(raw.get("aspect_id") or "")
            if not aspect_id:
                continue
            evidence_chunk_ids = reverse_coverage.get(aspect_id, [])
            requirements.append(ExplanationRequirement(
                requirement_id=aspect_id,
                question=str(raw.get("question") or ""),
                source_titles=[
                    str((raw.get("table_filters") or {}).get("source_title"))
                ] if (raw.get("table_filters") or {}).get("source_title") else [],
                mandatory=bool(raw.get("mandatory", True)),
                status="covered" if evidence_chunk_ids else "missing",
                evidence_ids=[_evidence_id(label_map, item) for item in evidence_chunk_ids],
                missing_reason=None if evidence_chunk_ids else "not_recorded",
            ))
    return requirements


def _build_claims(claims: list[Any], label_map: dict[str, str],
                  reverse_coverage: dict[str, list[str]],
                  requirements: list[ExplanationRequirement]) -> list[ExplanationClaimLink]:
    known_requirements = {item.requirement_id for item in requirements}
    links: list[ExplanationClaimLink] = []
    for index, claim in enumerate(claims or [], start=1):
        citation_ids = [str(item) for item in (getattr(claim, "citation_ids", ()) or ())]
        requirement_ids: list[str] = []
        for citation_id in citation_ids:
            for chunk_id, label in label_map.items():
                if label == citation_id:
                    requirement_ids.extend(reverse_coverage.get(chunk_id, []))
        aspect_ids = [str(item) for item in (getattr(claim, "aspect_ids", ()) or ())]
        requirement_ids.extend(item for item in aspect_ids if item in known_requirements)
        links.append(ExplanationClaimLink(
            claim_index=index,
            text=str(getattr(claim, "text", "") or ""),
            role=str(getattr(claim, "role", "other") or "other"),
            evidence_ids=list(dict.fromkeys(citation_ids)),
            requirement_ids=list(dict.fromkeys(
                item for item in requirement_ids if item in known_requirements
            )),
            aspect_ids=aspect_ids,
        ))
    return links


def _build_sources(cited_results: list[Any], label_map: dict[str, str],
                   claims: list[ExplanationClaimLink],
                   reverse_coverage: dict[str, list[str]]) -> list[ExplanationSource]:
    sources: list[ExplanationSource] = []
    for chunk in cited_results or []:
        chunk_id = str(getattr(chunk, "chunk_id", ""))
        metadata = getattr(chunk, "metadata", None) or {}
        evidence_id = _evidence_id(label_map, chunk_id)
        claim_indices = [
            claim.claim_index for claim in claims if evidence_id in claim.evidence_ids
        ]
        sources.append(ExplanationSource(
            evidence_id=evidence_id,
            chunk_id=chunk_id,
            document_id=str(metadata.get("document_id") or ""),
            source=str(getattr(chunk, "source_doc", "") or ""),
            source_title=(
                str(metadata.get("source_title")) if metadata.get("source_title")
                else str(getattr(chunk, "source_doc", "") or "")
            ),
            section_title=getattr(chunk, "section_title", None),
            section_path=list(getattr(chunk, "section_path", ()) or ()),
            page_number=(int(metadata["page_number"]) if str(metadata.get("page_number") or "").isdigit() else None),
            sheet_name=str(metadata.get("sheet_name")) if metadata.get("sheet_name") else None,
            cell=str(metadata.get("cell")) if metadata.get("cell") else None,
            excerpt=_compact(getattr(chunk, "text", ""), 500),
            requirement_ids=_related_requirement_ids(chunk, reverse_coverage),
            claim_indices=claim_indices,
            citation_located=bool(evidence_id and evidence_id == label_map.get(chunk_id)),
        ))
    return sources


def _build_source_groups(requirements: list[ExplanationRequirement],
                         sources: list[ExplanationSource],
                         claims: list[ExplanationClaimLink]) -> list[ExplanationSourceGroup]:
    groups: list[ExplanationSourceGroup] = []
    seen_titles: set[str] = set()
    for requirement in requirements:
        titles = requirement.source_titles or [""]
        for title in titles:
            key = title or requirement.requirement_id
            if key in seen_titles:
                group = next(item for item in groups if item.source_title == key)
            else:
                seen_titles.add(key)
                group = ExplanationSourceGroup(source_title=title)
                groups.append(group)
            group.requirement_ids = list(dict.fromkeys(
                [*group.requirement_ids, requirement.requirement_id]
            ))
            matching_sources = [
                source for source in sources
                if requirement.requirement_id in source.requirement_ids
                or (title and title in (source.source_title or ""))
            ]
            if not matching_sources and requirement.evidence_ids:
                matching_sources = [
                    source for source in sources
                    if source.evidence_id in requirement.evidence_ids
                ]
            group.evidence_ids = list(dict.fromkeys(
                [*group.evidence_ids, *[item.evidence_id for item in matching_sources]]
            ))
            if not group.source_name and matching_sources:
                group.source_name = matching_sources[0].source
            claim_indices = [
                claim.claim_index for claim in claims
                if requirement.requirement_id in claim.requirement_ids
                or any(item in claim.evidence_ids for item in group.evidence_ids)
            ]
            group.claim_indices = list(dict.fromkeys([*group.claim_indices, *claim_indices]))
            group.status = (
                "covered" if group.evidence_ids else
                "missing" if requirement.mandatory else "partial"
            )
    # A context package without source titles still gets source-derived groups.
    if not groups and sources:
        by_source: dict[str, ExplanationSourceGroup] = {}
        for source in sources:
            key = source.source_title or source.source
            group = by_source.setdefault(key, ExplanationSourceGroup(
                source_title=source.source_title or source.source,
                source_name=source.source,
            ))
            group.evidence_ids.append(source.evidence_id)
            group.claim_indices = list(dict.fromkeys([*group.claim_indices, *source.claim_indices]))
            group.status = "covered"
        groups = list(by_source.values())
    return groups


def _calculation_from_chunk(chunk: Any, evidence_id: str,
                            validated: bool) -> ExplanationCalculation | None:
    metadata = getattr(chunk, "metadata", None) or {}
    if not (
        metadata.get("dynamic_table_evidence")
        or metadata.get("calculation_result") is not None
        or metadata.get("calculation_formula")
    ):
        return None
    operands: list[ExplanationOperand] = []
    for index, cell in enumerate(metadata.get("calculation_cells") or [], start=1):
        if not isinstance(cell, dict):
            continue
        value = cell.get("normalized_value")
        if value is None:
            value = cell.get("value")
        operands.append(ExplanationOperand(
            label=str(
                cell.get("row_label") or cell.get("column_label")
                or cell.get("cell") or f"操作数 {index}"
            ),
            value=str(value) if value is not None else None,
            unit=str(cell.get("unit")) if cell.get("unit") else None,
            sheet_name=str(cell.get("sheet_name")) if cell.get("sheet_name") else None,
            cell=str(cell.get("cell")) if cell.get("cell") else None,
            source=str(cell.get("filename") or getattr(chunk, "source_doc", "") or ""),
            evidence_id=evidence_id,
        ))
    if not operands and metadata.get("value") is not None:
        operands.append(ExplanationOperand(
            label=str(metadata.get("row_label") or metadata.get("column_label") or "取值单元格"),
            value=str(metadata.get("value")),
            unit=str(metadata.get("unit")) if metadata.get("unit") else None,
            sheet_name=str(metadata.get("sheet_name")) if metadata.get("sheet_name") else None,
            cell=str(metadata.get("cell")) if metadata.get("cell") else None,
            source=str(getattr(chunk, "source_doc", "") or ""),
            evidence_id=evidence_id,
        ))
    result = metadata.get("calculation_result_text")
    if result is None:
        result = metadata.get("calculation_result")
    if result is None:
        result = metadata.get("value")
    formula = metadata.get("calculation_display_formula") or metadata.get("calculation_formula")
    trace_valid = metadata.get("calculation_trace_valid")
    verified = bool(validated and trace_valid is not False)
    return ExplanationCalculation(
        operation=str(metadata.get("operation") or metadata.get("comparison_operation") or "") or None,
        formula=str(metadata["calculation_formula"]) if metadata.get("calculation_formula") else None,
        display_formula=str(formula) if formula else None,
        result=str(result) if result is not None else None,
        unit=str(metadata.get("unit")) if metadata.get("unit") else None,
        operands=operands,
        evidence_ids=[evidence_id],
        source=str(getattr(chunk, "source_doc", "") or ""),
        verified=verified,
        verification_note=(
            "操作数、公式与结果已通过确定性校验" if verified
            else "计算记录未通过完整校验或未记录校验结果"
        ),
    )


def _outcome(answer_type: str, refused: bool, reason: str | None,
             validation: dict[str, Any]) -> str:
    if answer_type == "clarification":
        return "clarification"
    if answer_type == "service_error":
        return "service_error"
    if not refused:
        return "answered"
    if answer_type == "formula_refusal" or reason in _CALCULATION_REASONS:
        return "calculation_gap"
    if reason in {"model_refusal", "generation_unavailable"}:
        return "service_error" if reason == "generation_unavailable" else "model_refusal"
    if reason in {"grounding_validation_failed", "validation_not_run"}:
        return "validation_failed"
    if validation.get("status") == "error" or validation.get("failure_stage") == "validation":
        if reason in {"validation_transport_failure", "validation_schema_failure"}:
            return "service_error"
        return "validation_failed"
    if validation.get("status") == "failed":
        return "validation_failed"
    if reason in _EVIDENCE_MISSING_REASONS or validation.get("failure_stage") == "evidence":
        return "evidence_missing"
    return "refused"


def build_answer_explanation(
    *,
    question: str,
    generated: Any,
    package: Any,
    cited_results: list[Any],
    progress_events: list[dict[str, Any]] | None = None,
    started_at: float | None = None,
    first_feedback_ms: float | None = None,
) -> AnswerExplanation:
    """Build the optional display projection without any model call."""

    summary = dict(getattr(package, "retrieval_summary", None) or {}) if package is not None else {}
    label_map = _label_map(package)
    bundle = summary.get("evidence_bundle") or {}
    coverage = bundle.get("coverage") or {}
    reverse_coverage: dict[str, list[str]] = {}
    for requirement_id, chunk_ids in coverage.items():
        for chunk_id in chunk_ids or []:
            reverse_coverage.setdefault(str(chunk_id), []).append(str(requirement_id))

    requirements = _build_requirements(summary, package, reverse_coverage)
    claims = _build_claims(
        list(getattr(generated, "claims", None) or []), label_map, reverse_coverage, requirements
    )
    sources = _build_sources(cited_results, label_map, claims, reverse_coverage)
    source_groups = _build_source_groups(requirements, sources, claims)

    validation_raw = dict(getattr(generated, "grounding_validation", None) or {})
    passed = validation_passed(validation_raw)
    status = str(validation_raw.get("status") or "unknown")
    if status not in {"passed", "failed", "not_run", "error"}:
        status = "unknown"
    validation = ExplanationValidation(
        status=status,  # type: ignore[arg-type]
        passed=passed,
        citation_integrity=bool(validation_raw.get("citation_integrity")),
        facts_supported=bool(validation_raw.get("facts_supported")),
        task_complete=bool(validation_raw.get("task_complete")),
        method=str(validation_raw.get("method") or "none"),
        reasons=[str(item) for item in (validation_raw.get("reasons") or [])],
        missing_requirements=[
            str(item) for item in (validation_raw.get("missing_requirements") or [])
        ],
    )

    calculations = [
        calculation for source in sources
        for chunk in [next(
            (item for item in (getattr(package, "context_chunks", None) or [])
             if str(getattr(item, "chunk_id", "")) == source.chunk_id),
            None,
        )]
        if chunk is not None
        for calculation in [_calculation_from_chunk(chunk, source.evidence_id, passed)]
        if calculation is not None
    ]

    now = perf_counter()
    elapsed_ms = round(max(0.0, now - started_at) * 1000, 2) if started_at else 0.0
    budget_raw = request_budget_snapshot(model_calls=model_call_count())
    category = budget_raw.get("category")
    budget = ExplanationBudget(
        category=category if category in {"standard", "complex"} else "unknown",  # type: ignore[arg-type]
        total_seconds=budget_raw.get("total_seconds"),
        elapsed_ms=elapsed_ms,
        model_calls=int(budget_raw.get("model_calls") or 0),
        model_call_limit=int(budget_raw.get("model_call_limit") or 5),
        first_feedback_ms=first_feedback_ms,
        upgraded=bool(budget_raw.get("upgraded")),
    )

    stages = [
        ExplanationStage(
            stage=str(event.get("stage") or ""),
            status=str(event.get("status") or ""),
            title=str(event.get("title") or ""),
            elapsed_ms=(
                float(event["elapsed_ms"]) if isinstance(event.get("elapsed_ms"), (int, float))
                else None
            ),
            agent=str(event.get("agent")) if event.get("agent") else None,
            summary={
                str(key): value
                for key, value in (event.get("summary") or {}).items()
                if isinstance(value, (str, int, float, bool)) and not str(key).startswith("_")
            },
        )
        for event in (progress_events or [])
        if isinstance(event, dict) and event.get("stage")
    ][-100:]

    answer_type = str(getattr(generated, "answer_type", "") or "refusal")
    refused = bool(getattr(generated, "refused", True))
    reason = getattr(generated, "refusal_reason", None)
    cross_document = len([
        group for group in source_groups if group.source_title
    ]) >= 2 or len({
        source.source for source in sources if source.source
    }) >= 2
    return AnswerExplanation(
        question=question,
        outcome=_outcome(answer_type, refused, reason, validation_raw),
        cross_document=cross_document,
        requirements=requirements,
        claims=claims,
        sources=sources,
        source_groups=source_groups,
        calculations=calculations,
        validation=validation,
        budget=budget,
        stages=stages,
        missing_requirements=[
            *validation.missing_requirements,
            *[
                requirement.question or requirement.requirement_id
                for requirement in requirements
                if requirement.status == "missing"
            ],
        ],
        elapsed_ms=elapsed_ms,
        first_feedback_ms=first_feedback_ms,
    )
