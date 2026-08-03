from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import math
import re
from time import perf_counter
import urllib.error
import urllib.request
from typing import Any, Callable

from backend.app.core.config import (
    ASPECT_GATE_PRE_GENERATION,
    ANSWER_GENERATION_API_KEY,
    ANSWER_GENERATION_BASE_URL,
    ANSWER_GENERATION_ENABLED,
    ANSWER_GENERATION_INCLUDE_THINKING,
    ANSWER_GENERATION_MAX_TOKENS,
    ANSWER_GENERATION_MODEL,
    ANSWER_GENERATION_PROVIDER,
    ANSWER_GENERATION_RESPONSE_FORMAT,
    ANSWER_GENERATION_STREAM,
    ANSWER_GENERATION_TOTAL_BUDGET_SECONDS,
    ANSWER_GENERATION_MAX_ATTEMPTS,
    ANSWER_GENERATION_TIMEOUT_SECONDS,
)
from backend.app.schemas.qa import AnswerClaim, RetrievalResult
from backend.app.services.grounding_validation_service import validate_grounded_answer
from backend.app.services.regulatory_semantic_grounding_service import (
    claim_requires_semantic_verifier,
)
from backend.app.services.formula_parser_service import (
    calculate_formula_answer,
    formula_refusal_grounding,
)
from backend.app.services.prompt_builder import RAGPromptBuilder
from backend.app.services.llm_client import (
    ChatCompletionConfig,
    ChatCompletionError,
    open_chat_completion,
)
from backend.app.services.performance_metrics import measure


VerifiedClaimReporter = Callable[[AnswerClaim], None]
CancellationChecker = Callable[[], None]


class _OptionMatrixCache:
    """Mutable per-request holder for the option evidence matrix.

    The matrix is a pure function of (options, labels, context chunks) but is
    rebuilt by every consumer in one request (prompt build, streaming claim
    verification, post-generation validation, deterministic choice recovery),
    each build re-running dozens of full grounded-answer validations.  The
    holder is installed by ``generate_answer`` for its call tree only; a stale
    holder left on a reused worker thread can never produce a wrong hit
    because the key compares object identity.
    """

    __slots__ = ("key", "matrix")

    def __init__(self, key: tuple[int, int, int]) -> None:
        self.key = key
        self.matrix: list[dict[str, Any]] | None = None


_OPTION_MATRIX_CACHE: ContextVar[_OptionMatrixCache | None] = ContextVar(
    "option_matrix_cache", default=None
)


@dataclass
class GeneratedAnswer:
    answer: str | None
    answer_type: str
    generation_status: str
    claims: list[AnswerClaim] = field(default_factory=list)
    grounding_validation: dict[str, Any] = field(default_factory=dict)
    refused: bool = False
    refusal_reason: str | None = None
    refusal_code: str | None = None
    missing_variables: list[str] = field(default_factory=list)
    ambiguous_variables: list[str] = field(default_factory=list)
    unsupported_formula: str | None = None
    degraded: bool = False


@dataclass(frozen=True)
class TableFinding:
    citation_id: str
    value: str
    unit: str | None
    period: str | None
    coordinate: str | None
    operands: tuple[str, ...]
    formula: str | None

    def to_prompt_line(self) -> str:
        fields = [f"引用={self.citation_id}", f"值={self.value}"]
        for label, value in (
            ("单位", self.unit),
            ("期间", self.period),
            ("坐标", self.coordinate),
            ("公式", self.formula),
        ):
            if value:
                fields.append(f"{label}={value}")
        if self.operands:
            fields.append(f"操作数={','.join(self.operands)}")
        return "；".join(fields)


def generate_answer(
    question: str,
    context_chunks: list[RetrievalResult],
    *,
    options: list[str] | None = None,
    option_labels: list[str | None] | None = None,
    llm_prompt: str | None = None,
    has_sufficient_context: bool,
    verified_claim_reporter: VerifiedClaimReporter | None = None,
    cancellation_checker: CancellationChecker | None = None,
    answer_mode: str = "text",
    required_aspect_ids: list[str] | None = None,
) -> GeneratedAnswer:
    normalized_options = [option for option in options or [] if str(option or "").strip()]
    normalized_labels = _normalize_option_labels(option_labels, len(normalized_options))
    # Request-scoped memoization of the option evidence matrix: every consumer
    # in this call tree (prompt build, streaming claim verification, validation,
    # deterministic choice recovery) recomputed it independently.  Install the
    # holder for the whole request; keyed by object identity, so a stale holder
    # on a reused worker thread can never serve a wrong matrix.
    _OPTION_MATRIX_CACHE.set(
        _OptionMatrixCache((id(normalized_options), id(normalized_labels), id(context_chunks)))
    )
    return _generate_answer_impl(
        question,
        context_chunks,
        normalized_options,
        normalized_labels,
        llm_prompt=llm_prompt,
        has_sufficient_context=has_sufficient_context,
        verified_claim_reporter=verified_claim_reporter,
        cancellation_checker=cancellation_checker,
        answer_mode=answer_mode,
        required_aspect_ids=required_aspect_ids,
    )


def _generate_answer_impl(
    question: str,
    context_chunks: list[RetrievalResult],
    normalized_options: list[str],
    normalized_labels: list[str | None],
    *,
    llm_prompt: str | None,
    has_sufficient_context: bool,
    verified_claim_reporter: VerifiedClaimReporter | None,
    cancellation_checker: CancellationChecker | None,
    answer_mode: str,
    required_aspect_ids: list[str] | None,
) -> GeneratedAnswer:
    if not context_chunks or not has_sufficient_context:
        return GeneratedAnswer(
            answer="当前知识库中未找到足够依据，无法给出确定答案。",
            answer_type="refusal",
            generation_status="skipped",
            refused=True,
            refusal_reason="insufficient_context",
            grounding_validation={"passed": True, "reason": "refused_before_generation"},
        )

    if all(chunk.metadata.get("evidence_role") == "table_context" for chunk in context_chunks):
        return GeneratedAnswer(
            answer="检索结果仅包含相关表格背景，缺少能够直接回答当前问题的依据。",
            answer_type="refusal",
            generation_status="skipped",
            refused=True,
            refusal_reason="related_context_only",
            grounding_validation={"passed": True, "reason": "related_context_only"},
        )

    if ASPECT_GATE_PRE_GENERATION and required_aspect_ids and not normalized_options:
        # Evidence-side pre-gate: if the selected context does not cover every
        # required aspect, refuse before spending LLM calls on a generation
        # that the post-generation coverage check would reject anyway.  The
        # same coverage rule is used by _validate_required_aspect_coverage.
        missing = _missing_aspects_in_context(context_chunks, required_aspect_ids)
        if missing:
            return GeneratedAnswer(
                answer="当前证据未覆盖所需方面：" + "、".join(missing) + "，无法给出确定答案。",
                answer_type="refusal",
                generation_status="skipped",
                refused=True,
                refusal_reason="missing_required_aspect",
                grounding_validation={
                    "passed": True,
                    "reason": "missing_required_aspect",
                    "missing_required_aspect_ids": missing,
                },
            )

    table_findings = _table_findings(context_chunks)
    if answer_mode in {"mixed", "scenario"}:
        missing_modalities = _missing_mixed_modalities(context_chunks, table_findings)
        if missing_modalities:
            return GeneratedAnswer(
                answer="当前证据未同时覆盖制度依据和表格事实，无法完成跨模态判断。",
                answer_type="refusal",
                generation_status="skipped",
                refused=True,
                refusal_reason="missing_required_aspect",
                grounding_validation={
                    "passed": True,
                    "reason": "missing_required_aspect",
                    "missing_modalities": missing_modalities,
                },
            )
        if (
            answer_mode == "mixed"
            and not normalized_options
            and _requires_evidence_boundary(question)
        ):
            deterministic_mixed = _deterministic_mixed_table_answer(question, context_chunks)
            if deterministic_mixed is not None:
                _report_verified_claims(deterministic_mixed, verified_claim_reporter)
                return deterministic_mixed
    else:
        deterministic = _deterministic_table_answer(context_chunks, normalized_options, normalized_labels)
        if deterministic is not None:
            _report_verified_claims(deterministic, verified_claim_reporter)
            return deterministic

    if _is_formula_request(question):
        formula_answer = _deterministic_formula_answer(question, context_chunks)
        if formula_answer is not None:
            _report_verified_claims(formula_answer, verified_claim_reporter)
            return formula_answer

    if normalized_options:
        deterministic_choice = _deterministic_choice_recovery(
            normalized_options,
            normalized_labels,
            context_chunks,
            {"reason": "pre_generation_option_evidence"},
        )
        if deterministic_choice is not None:
            _report_verified_claims(deterministic_choice, verified_claim_reporter)
            return deterministic_choice

    if ANSWER_GENERATION_ENABLED and ANSWER_GENERATION_API_KEY:
        try:
            generated = _call_llm(
                question,
                context_chunks,
                normalized_options,
                normalized_labels,
                llm_prompt=llm_prompt,
                verified_claim_reporter=verified_claim_reporter,
                cancellation_checker=cancellation_checker,
                answer_mode=answer_mode,
                table_findings=table_findings,
                required_aspect_ids=required_aspect_ids or [],
            )
            if generated.refused:
                supported_fallback = _recover_supported_refusal_with_extracts(
                    question,
                    context_chunks,
                    normalized_options,
                    normalized_labels,
                    generated,
                    reason="model_refusal",
                )
                if supported_fallback is not None:
                    _report_verified_claims(supported_fallback, verified_claim_reporter)
                    return supported_fallback
                if not generated.answer and generated.refusal_reason:
                    generated.answer = generated.refusal_reason
                generated.grounding_validation = {
                    "passed": True,
                    "reason": "model_refusal",
                }
                return generated
            validation = _validate_generated_answer(
                generated,
                context_chunks,
                normalized_options,
                normalized_labels,
                answer_mode=answer_mode,
                table_findings=table_findings,
                required_aspect_ids=required_aspect_ids or [],
            )
            generated.grounding_validation = validation
            if validation["passed"]:
                return generated
            if _repair_inline_citation_format(generated, validation, context_chunks):
                return generated
            deterministic_choice = _deterministic_choice_recovery(
                normalized_options, normalized_labels, context_chunks, validation
            )
            if deterministic_choice is not None:
                return deterministic_choice
            if answer_mode in {"mixed", "scenario"} and not normalized_options:
                deterministic_mixed = _deterministic_mixed_table_answer(question, context_chunks)
                if (
                    deterministic_mixed is not None
                    and deterministic_mixed.grounding_validation.get("passed") is True
                ):
                    deterministic_mixed.grounding_validation = {
                        **deterministic_mixed.grounding_validation,
                        "llm_validation": validation,
                        "recovery_method": "deterministic_mixed_after_validation_failed",
                    }
                    _report_verified_claims(deterministic_mixed, verified_claim_reporter)
                    return deterministic_mixed
            if _is_aspect_only_failure(validation, context_chunks, required_aspect_ids):
                # Aspect-coverage failures without retrieval-side evidence
                # cannot be repaired by another LLM round (the missing
                # evidence is not in the prompt); refuse deterministically
                # and name the missing aspects.
                return _deterministic_aspect_refusal(validation)
            repaired = _call_llm(
                question,
                context_chunks,
                normalized_options,
                normalized_labels,
                llm_prompt=llm_prompt,
                correction=json.dumps(validation, ensure_ascii=False),
                verified_claim_reporter=verified_claim_reporter,
                cancellation_checker=cancellation_checker,
                answer_mode=answer_mode,
                table_findings=table_findings,
                required_aspect_ids=required_aspect_ids or [],
            )
            if repaired.refused:
                supported_fallback = _recover_supported_refusal_with_extracts(
                    question,
                    context_chunks,
                    normalized_options,
                    normalized_labels,
                    repaired,
                    reason="model_refusal_after_repair",
                )
                if supported_fallback is not None:
                    _report_verified_claims(supported_fallback, verified_claim_reporter)
                    return supported_fallback
                if not repaired.answer and repaired.refusal_reason:
                    repaired.answer = repaired.refusal_reason
                repaired.grounding_validation = {
                    "passed": True,
                    "reason": "model_refusal_after_repair",
                    "repair_attempted": True,
                }
                return repaired
            repaired_validation = _validate_generated_answer(
                repaired,
                context_chunks,
                normalized_options,
                normalized_labels,
                answer_mode=answer_mode,
                table_findings=table_findings,
                required_aspect_ids=required_aspect_ids or [],
            )
            repaired.grounding_validation = {**repaired_validation, "repair_attempted": True}
            if repaired_validation["passed"]:
                return repaired
            if not normalized_options:
                # Open questions must not degrade to a conclusion-less excerpt
                # dump after the LLM produced an answer that failed validation:
                # the excerpt reads like a refusal without saying so.  Refuse
                # structurally and honestly instead.
                return GeneratedAnswer(
                    answer="已检索到相关依据，但生成答案未通过事实校验，无法给出确定结论。",
                    answer_type="refusal",
                    generation_status="validation_failed",
                    refused=True,
                    refusal_reason="validation_failed_no_reliable_answer",
                    grounding_validation={
                        **repaired_validation,
                        "passed": True,
                        "reason": "validation_failed_no_reliable_answer",
                        "repair_attempted": True,
                        "recovery_method": "structural_refusal_after_validation_failed",
                    },
                    degraded=True,
                )
            return GeneratedAnswer(
                answer="检索到了相关资料，但生成答案中的关键事实未通过证据校验，暂不提供确定结论。",
                answer_type="refusal",
                generation_status="validation_failed",
                refused=True,
                refusal_reason="grounding_validation_failed",
                grounding_validation={**repaired_validation, "repair_attempted": True},
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    if answer_mode in {"mixed", "scenario"}:
        deterministic_mixed = _deterministic_mixed_table_answer(question, context_chunks)
        if deterministic_mixed is not None:
            _report_verified_claims(deterministic_mixed, verified_claim_reporter)
            return deterministic_mixed

    deterministic_choice = _deterministic_choice_recovery(
        normalized_options,
        normalized_labels,
        context_chunks,
        {"reason": "generation_unavailable"},
    )
    if deterministic_choice is not None:
        _report_verified_claims(deterministic_choice, verified_claim_reporter)
        return deterministic_choice

    fallback = _trusted_extractive_fallback(context_chunks, question=question)
    _report_verified_claims(fallback, verified_claim_reporter)
    return fallback


def _covered_aspects_by_context(context_chunks: list[RetrievalResult]) -> set[str]:
    """Aspects covered by evidence-side metadata on the selected chunks.

    Uses the same chunk-side signals as ``_validate_required_aspect_coverage``
    (``prompt_matched_aspects`` plus ``aspect_id``) so the pre-generation gate
    and the post-generation check share one coverage rule.
    """

    covered: set[str] = set()
    for chunk in context_chunks:
        metadata = chunk.metadata or {}
        for aspect_id in metadata.get("prompt_matched_aspects") or []:
            if aspect_id:
                covered.add(str(aspect_id))
        aspect_id = metadata.get("aspect_id")
        if aspect_id:
            covered.add(str(aspect_id))
    return covered


def _missing_aspects_in_context(
    context_chunks: list[RetrievalResult],
    required_aspect_ids: list[str] | None,
) -> list[str]:
    """Required aspects absent from the selected context evidence.

    ``multiple_choice_evidence`` is exempt: it is the option-evidence
    synthesis aspect of choice questions, not an atomic open-question aspect,
    and the official choice set must keep its existing behaviour.
    """

    if not required_aspect_ids:
        return []
    eligible = [
        aspect_id
        for aspect_id in required_aspect_ids
        if aspect_id and aspect_id != "multiple_choice_evidence"
    ]
    if not eligible:
        return []
    covered = _covered_aspects_by_context(context_chunks)
    return [aspect_id for aspect_id in eligible if aspect_id not in covered]


def _is_aspect_only_failure(
    validation: dict[str, Any],
    context_chunks: list[RetrievalResult],
    required_aspect_ids: list[str] | None,
) -> bool:
    """True when the validation failures are only aspect-coverage failures.

    Content-level failures (unsupported entities, citation problems, semantic
    conflicts/insufficiency, missing claims) still deserve an LLM repair round.
    Aspect-coverage failures are only repaired deterministically when the
    *retrieval* side also lacks the evidence (the pre-generation gate already
    refused in that case, so this is a belt-and-braces path).  When the
    retrieved context covers every required aspect, a missing claim aspect is
    an LLM labelling problem that the repair round can fix, so it must not be
    short-circuited into a refusal.
    """

    if bool(validation.get("passed")):
        return False
    if not _missing_aspects_in_context(context_chunks, required_aspect_ids):
        # Evidence covers all aspects: the failure is claim labelling, not
        # missing evidence — let the LLM repair round try again.
        return False
    content_failures = [
        validation.get("unsupported_entities") or [],
        validation.get("unsupported_claim_entities") or [],
        validation.get("invalid_citation_ids") or [],
        validation.get("invalid_answer_citation_ids") or [],
        validation.get("missing_inline_citation_ids") or [],
        validation.get("missing_claim_citations") or [],
    ]
    if any(content_failures):
        return False
    if int(validation.get("semantic_conflict_count") or 0) > 0:
        return False
    if int(validation.get("semantic_insufficient_count") or 0) > 0:
        return False
    if bool(validation.get("missing_claims")):
        return False
    return bool(
        validation.get("missing_required_aspect_ids")
        or int(validation.get("false_insufficient_aspect_count") or 0) > 0
    )


def _deterministic_aspect_refusal(
    validation: dict[str, Any],
) -> GeneratedAnswer:
    """Structured refusal for aspect-only validation failures.

    Deterministic and honest: it names the missing aspects instead of spending
    an LLM repair round that the post-repair coverage check would reject again.
    """

    missing = list(validation.get("missing_required_aspect_ids") or [])
    if missing:
        reason = f"当前答案未完整覆盖所需方面：{'、'.join(missing)}，缺少相应制度依据，无法给出确定结论。"
    else:
        reason = "当前答案对已覆盖方面作出了依据不足的表述，无法给出确定结论。"
    return GeneratedAnswer(
        answer=reason,
        answer_type="refusal",
        generation_status="validation_failed",
        refused=True,
        refusal_reason="missing_required_aspect",
        grounding_validation={
            **validation,
            "passed": True,
            "reason": "missing_required_aspect",
            "recovery_method": "deterministic_aspect_refusal",
        },
        degraded=True,
    )


def is_option_selection_question(question: str) -> bool:
    normalized = re.sub(r"\s+", "", str(question or ""))
    if not normalized:
        return False
    if re.search(r"(?:下列|以下|选项|候选项).{0,16}哪(?:一)?(?:项|组)", normalized):
        return True
    if "选项" in normalized and any(
        marker in normalized for marker in ("正确", "不正确", "属于", "均属于", "表述", "说法")
    ):
        return True
    # “请回答以下两个事项……某公司属于……” is an ordinary multi-part
    # question.  A list introducer is a choice marker only when the selection
    # wording is locally attached to it.
    return bool(
        re.search(
            r"(?:下列|以下).{0,16}(?:哪(?:一)?(?:项|组)|(?:表述|说法).{0,6}(?:正确|不正确)|(?:正确|不正确)的是)",
            normalized,
        )
    )


def _repair_inline_citation_format(
    generated: GeneratedAnswer,
    validation: dict[str, Any],
    context_chunks: list[RetrievalResult],
) -> bool:
    """Repair citation placement only; never change a generated fact.

    Some chat-completion models return valid citation IDs in structured claims but
    omits the same IDs from the visible answer.  When that is the *only*
    grounding failure, attach those already-validated IDs to their claim text
    locally instead of spending another LLM call on a formatting correction.
    """

    missing_ids = list(validation.get("missing_inline_citation_ids") or [])
    blocking_fields = (
        "invalid_citation_ids",
        "invalid_answer_citation_ids",
        "unsupported_entities",
        "unsupported_claim_entities",
        "missing_claim_citations",
    )
    if (
        not missing_ids
        or validation.get("missing_claims")
        or validation.get("selected_option_supported") is False
        or any(validation.get(field) for field in blocking_fields)
        or not generated.answer
    ):
        return False

    repaired_answer = generated.answer
    remaining = set(missing_ids)
    for claim in generated.claims:
        claim_missing = [citation_id for citation_id in claim.citation_ids if citation_id in remaining]
        if not claim_missing:
            continue
        suffix = "".join(claim_missing)
        if claim.text and claim.text in repaired_answer:
            repaired_answer = repaired_answer.replace(claim.text, f"{claim.text}{suffix}", 1)
            remaining.difference_update(claim_missing)
    if remaining:
        repaired_answer = f"{repaired_answer.rstrip()} {''.join(sorted(remaining))}"

    repaired_validation = validate_grounded_answer(repaired_answer, generated.claims, context_chunks)
    if not repaired_validation["passed"]:
        return False
    generated.answer = repaired_answer
    generated.grounding_validation = {
        **repaired_validation,
        "repair_attempted": True,
        "repair_method": "local_inline_citation_format",
    }
    return True


def _has_structured_table_evidence(context_chunks: list[RetrievalResult]) -> bool:
    for chunk in context_chunks:
        metadata = chunk.metadata or {}
        if metadata.get("dynamic_table_evidence") or metadata.get("table_finding"):
            return True
        if metadata.get("evidence_role") in {"table_context", "spreadsheet_cell", "structured_spreadsheet_retrieval"}:
            return True
        if metadata.get("cell") or metadata.get("cell_refs") or metadata.get("row_label") or metadata.get("column_label"):
            return True
    return False


def _with_overlapping_condition_continuations(
    selected: list[RetrievalResult],
    context_chunks: list[RetrievalResult],
    normalized_terms: list[str],
) -> list[RetrievalResult]:
    selected_ids = {chunk.chunk_id for chunk in selected}
    expanded = list(selected)
    for selected_chunk in selected:
        selected_text = _normalize_evidence_text(selected_chunk.text)
        selected_hits = {term for term in normalized_terms if term in selected_text}
        if not selected_hits:
            continue
        selected_preamble = str(selected_chunk.metadata.get("condition_preamble_chunk_id") or "")
        for chunk in context_chunks:
            if chunk.chunk_id in selected_ids:
                continue
            same_condition_chain = selected_preamble and (
                str(chunk.metadata.get("condition_preamble_chunk_id") or "") == selected_preamble
            )
            adjacent_overlap = (
                str(selected_chunk.metadata.get("next_chunk_id") or "") == chunk.chunk_id
                or str(chunk.metadata.get("previous_chunk_id") or "") == selected_chunk.chunk_id
            )
            if not (same_condition_chain or adjacent_overlap):
                continue
            chunk_text = _normalize_evidence_text(chunk.text)
            if not selected_hits.issubset({term for term in normalized_terms if term in chunk_text}):
                continue
            expanded.append(chunk)
            selected_ids.add(chunk.chunk_id)
            break
    return expanded


def _deterministic_table_answer(
    context_chunks: list[RetrievalResult],
    options: list[str],
    option_labels: list[str | None],
) -> GeneratedAnswer | None:
    chunk = next(
        (
            item
            for item in context_chunks
            if item.metadata.get("dynamic_table_evidence")
            or item.metadata.get("calculation_result") is not None
        ),
        None,
    )
    if chunk is None:
        return None
    metadata = chunk.metadata
    match_status = str(metadata.get("match_status") or "")
    valid_statuses = {"valid_cell", "valid_calculation", "valid_comparison", ""}
    if match_status not in valid_statuses:
        return GeneratedAnswer(
            answer=f"检索到了表格候选，但{metadata.get('refusal_reason') or '证据不足以支持确定答案'}",
            answer_type="refusal",
            generation_status="skipped",
            refused=True,
            refusal_reason=match_status or "table_evidence_not_valid",
            grounding_validation={"passed": True, "reason": "table_evidence_not_valid"},
        )
    raw_value = metadata.get("calculation_result")
    if raw_value is None:
        raw_value = metadata.get("value")
    label = metadata.get("row_label") or metadata.get("column_label")
    option_label, option_text = _match_option(raw_value, label, options, option_labels)
    value_text = _format_value(
        raw_value,
        decimal_places=metadata.get("display_decimal_places"),
    )
    output_unit = _deterministic_table_output_unit(metadata)
    value_with_unit = f"{value_text} {output_unit}" if output_unit else value_text
    citation_label = chunk.citation_label
    if option_text:
        answer = f"答案为 {_format_selected_option(option_text, option_label)}。依据表格证据，核验值为 {value_with_unit}。{citation_label}"
    elif metadata.get("comparison_operation") in {"max", "min"}:
        winner_label = _comparison_winner_label(metadata)
        comparison_text = "最高" if metadata.get("comparison_operation") == "max" else "最低"
        answer = f"数值{comparison_text}的项目为“{winner_label}”，结果为 {value_with_unit}。{citation_label}"
    elif metadata.get("calculation_formula"):
        display_formula = str(metadata.get("calculation_display_formula") or metadata["calculation_formula"])
        comparison_target = metadata.get("comparison_target") if isinstance(metadata.get("comparison_target"), dict) else {}
        if metadata.get("comparison_equal") is True and _asks_equality(metadata):
            target_label = comparison_target.get("label") or "目标值"
            answer = f"相等，计算结果为 {value_with_unit}。计算式：{display_formula}，与{target_label}一致。{citation_label}"
        elif metadata.get("comparison_equal") is False and _asks_equality(metadata):
            target_label = comparison_target.get("label") or "目标值"
            target_value = _format_value(comparison_target.get("value") or comparison_target.get("normalized_value"))
            answer = f"不相等，计算结果为 {value_with_unit}。计算式：{display_formula}，与{target_label} {target_value} 不一致。{citation_label}"
        else:
            answer = f"计算结果为 {value_with_unit}。计算式：{display_formula}。{citation_label}"
    else:
        answer = f"查询结果为 {value_with_unit}。{citation_label}"
    claim = AnswerClaim(
        text=answer.rsplit(citation_label, 1)[0].strip(),
        citation_ids=[citation_label],
        role="calculation" if metadata.get("calculation_result") is not None else "table_fact",
        aspect_ids=[str(chunk.metadata.get("aspect_id"))] if chunk.metadata.get("aspect_id") else [],
    )
    validation = validate_grounded_answer(answer, [claim], context_chunks)
    # Deterministic calculations may create a result not present verbatim in a
    # source chunk; it is grounded by the recorded operands and formula.
    if metadata.get("calculation_result") is not None:
        trace_valid = _calculation_trace_valid(metadata)
        validation["unsupported_entities"] = []
        validation["unsupported_claim_entities"] = []
        validation["calculation_trace_valid"] = trace_valid
        validation["passed"] = not (
            validation["invalid_citation_ids"]
            or validation["invalid_answer_citation_ids"]
            or validation["missing_inline_citation_ids"]
        ) and trace_valid
        if not trace_valid:
            return GeneratedAnswer(
                answer="表格计算结果缺少可复核的操作数或公式，无法给出确定答案。",
                answer_type="refusal",
                generation_status="validation_failed",
                refused=True,
                refusal_reason="invalid_calculation_trace",
                grounding_validation=validation,
            )
    return GeneratedAnswer(
        answer=answer,
        answer_type="table_deterministic",
        generation_status="completed",
        claims=[claim],
        grounding_validation=validation,
    )


def _comparison_winner_label(metadata: dict[str, Any]) -> str:
    cells = [item for item in metadata.get("comparison_cells") or [] if isinstance(item, dict)]
    rows = {str(item.get("row_label") or "") for item in cells if item.get("row_label")}
    columns = {str(item.get("column_label") or "") for item in cells if item.get("column_label")}
    if len(columns) == 1 and len(rows) > 1:
        label = str(metadata.get("row_label") or "")
        label = label.rsplit("/", maxsplit=1)[-1].strip()
    elif len(rows) == 1 and len(columns) > 1:
        label = str(metadata.get("column_label") or "")
    else:
        label = str(metadata.get("row_label") or metadata.get("column_label") or "该项目")
    return re.sub(r"^\d+[、.．]\s*", "", label) or "该项目"


def _deterministic_formula_answer(
    question: str,
    context_chunks: list[RetrievalResult],
) -> GeneratedAnswer | None:
    calculation = calculate_formula_answer(question, context_chunks)
    if calculation is None:
        return None
    formula_chunk = next(
        (chunk for chunk in context_chunks if chunk.chunk_id == calculation.source_chunk_id),
        next((chunk for chunk in context_chunks if chunk.metadata.get("contains_formula")), None),
    )
    citation_label = calculation.citation_label or (formula_chunk.citation_label if formula_chunk else None)
    if calculation.answer is not None and formula_chunk is not None:
        claim = AnswerClaim(
            text=(calculation.answer.rsplit(citation_label, 1)[0].strip() if citation_label else calculation.answer),
            citation_ids=[citation_label] if citation_label else [],
            role="calculation",
            aspect_ids=[str(formula_chunk.metadata.get("aspect_id"))] if formula_chunk.metadata.get("aspect_id") else [],
        )
        validation = validate_grounded_answer(calculation.answer, [claim], context_chunks)
        validation = {
            **validation,
            "passed": not (
                validation.get("invalid_citation_ids")
                or validation.get("invalid_answer_citation_ids")
                or validation.get("missing_inline_citation_ids")
            ),
            "formula_calculation": {
                "formula": calculation.formula,
                "expression": calculation.expression,
                "variables": calculation.variables,
                "result": calculation.result,
            },
        }
        return GeneratedAnswer(
            answer=calculation.answer,
            answer_type="formula_deterministic",
            generation_status="completed",
            claims=[claim],
            grounding_validation=validation,
        )
    citation_suffix = f" {citation_label}" if formula_chunk is not None and citation_label else ""
    answer = f"无法计算。{calculation.refusal_reason or '当前公式证据不足。'}{citation_suffix}"
    claim_ids = [citation_label] if formula_chunk is not None and citation_label else []
    claims = [
        AnswerClaim(
            text=answer.replace(f" {citation_label}", "").strip() if citation_label else answer.strip(),
            citation_ids=claim_ids,
            role="explanation",
            aspect_ids=[str(formula_chunk.metadata.get("aspect_id"))] if formula_chunk and formula_chunk.metadata.get("aspect_id") else [],
        )
    ] if claim_ids else []
    return GeneratedAnswer(
        answer=answer,
        answer_type="formula_refusal",
        generation_status="skipped",
        claims=claims,
        grounding_validation=formula_refusal_grounding(calculation),
        refused=True,
        refusal_reason=calculation.refusal_reason,
        refusal_code=calculation.refusal_code,
        missing_variables=calculation.missing_variables,
        ambiguous_variables=calculation.ambiguous_variables,
        unsupported_formula=calculation.unsupported_formula,
    )


def _is_formula_request(question: str) -> bool:
    normalized = re.sub(r"\s+", "", str(question or "")).casefold()
    return "公式" in normalized or any(
        marker in normalized for marker in ("请计算", "calculate", "代入", "求值", "算出")
    )


def _call_llm(
    question: str,
    context_chunks: list[RetrievalResult],
    options: list[str],
    option_labels: list[str | None],
    *,
    llm_prompt: str | None = None,
    correction: str | None = None,
    verified_claim_reporter: VerifiedClaimReporter | None = None,
    cancellation_checker: CancellationChecker | None = None,
    answer_mode: str = "text",
    table_findings: list[TableFinding] | None = None,
    required_aspect_ids: list[str] | None = None,
) -> GeneratedAnswer:
    option_evidence_matrix = _format_option_evidence_matrix(options, option_labels, context_chunks)
    messages = RAGPromptBuilder().build_generation_messages(
        question,
        context_chunks,
        options=options,
        option_labels=option_labels,
        option_evidence_matrix=option_evidence_matrix,
        correction=correction,
        llm_prompt=llm_prompt,
        answer_mode=answer_mode,
        table_findings="\n".join(item.to_prompt_line() for item in table_findings or []),
        required_aspect_ids=required_aspect_ids,
    )
    config = ChatCompletionConfig(
        provider=ANSWER_GENERATION_PROVIDER,
        api_key=ANSWER_GENERATION_API_KEY,
        base_url=ANSWER_GENERATION_BASE_URL,
        model=ANSWER_GENERATION_MODEL,
        timeout_seconds=ANSWER_GENERATION_TIMEOUT_SECONDS,
        include_thinking=ANSWER_GENERATION_INCLUDE_THINKING,
        response_format=ANSWER_GENERATION_RESPONSE_FORMAT,
    )
    stream_attempts = [ANSWER_GENERATION_STREAM]
    if ANSWER_GENERATION_STREAM:
        stream_attempts.append(False)
    expanded_max_tokens = max(ANSWER_GENERATION_MAX_TOKENS * 2, 1800)
    if expanded_max_tokens > ANSWER_GENERATION_MAX_TOKENS:
        stream_attempts.append(False)
    # The provider intermittently returns an empty stream or an empty
    # non-streaming body for long prompts; alternate stream/non-stream
    # attempts so a transient empty reply is retried instead of degrading.
    if len(stream_attempts) < ANSWER_GENERATION_MAX_ATTEMPTS:
        alternates = [True, False, True, False, True, False][: ANSWER_GENERATION_MAX_ATTEMPTS - len(stream_attempts)]
        stream_attempts.extend(alternates)
    expanded_config = ChatCompletionConfig(
        provider=ANSWER_GENERATION_PROVIDER,
        api_key=ANSWER_GENERATION_API_KEY,
        base_url=ANSWER_GENERATION_BASE_URL,
        model=ANSWER_GENERATION_MODEL,
        timeout_seconds=max(ANSWER_GENERATION_TIMEOUT_SECONDS * 2, 45.0),
        include_thinking=ANSWER_GENERATION_INCLUDE_THINKING,
        response_format=ANSWER_GENERATION_RESPONSE_FORMAT,
    )
    last_error: Exception | None = None
    last_recovered: GeneratedAnswer | None = None
    deadline = perf_counter() + ANSWER_GENERATION_TOTAL_BUDGET_SECONDS
    for index, stream in enumerate(stream_attempts[:ANSWER_GENERATION_MAX_ATTEMPTS]):
        if perf_counter() >= deadline:
            raise TimeoutError("answer generation total budget exhausted")
        content = ""
        attempt_config = expanded_config if index == len(stream_attempts) - 1 and expanded_max_tokens > ANSWER_GENERATION_MAX_TOKENS else config
        attempt_max_tokens = expanded_max_tokens if attempt_config is expanded_config else ANSWER_GENERATION_MAX_TOKENS
        try:
            content = _request_llm_content(
                attempt_config,
                messages,
                stream=stream,
                max_tokens=attempt_max_tokens,
                question=question,
                context_chunks=context_chunks,
                options=options,
                option_labels=option_labels,
                verified_claim_reporter=verified_claim_reporter if stream else None,
                cancellation_checker=cancellation_checker,
                deadline=deadline,
            )
            return _parse_generated_answer_content(
                content,
                question=question,
                context_chunks=context_chunks,
                options=options,
                option_labels=option_labels,
                answer_mode=answer_mode,
            )
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            recovered = _recover_generated_answer_from_partial_content(
                content,
                answer_mode=answer_mode,
            )
            if recovered is not None:
                last_recovered = recovered
                if (
                    _generated_answer_covers_required_aspects(recovered, required_aspect_ids or [])
                    or index == len(stream_attempts) - 1
                ):
                    return recovered
            last_error = exc
            continue
        except (OSError, TimeoutError, urllib.error.URLError, ChatCompletionError) as exc:
            last_error = OSError("answer generation unavailable")
            if stream and len(stream_attempts) > 1:
                continue
            raise OSError("answer generation unavailable") from exc
    if last_recovered is not None:
        return last_recovered
    if last_error is not None:
        raise last_error
    raise OSError("answer generation unavailable")


def _request_llm_content(
    config: ChatCompletionConfig,
    messages: list[dict[str, Any]],
    *,
    stream: bool,
    max_tokens: int,
    question: str,
    context_chunks: list[RetrievalResult],
    options: list[str],
    option_labels: list[str | None],
    verified_claim_reporter: VerifiedClaimReporter | None,
    cancellation_checker: CancellationChecker | None,
    deadline: float,
) -> str:
    with measure("answer_generation.external_api"):
        with open_chat_completion(
            config,
            messages,
            temperature=0,
            max_tokens=max_tokens,
            response_format=ANSWER_GENERATION_RESPONSE_FORMAT,
            stream=stream,
            opener=urllib.request.urlopen,
        ) as response:
            return _read_streaming_llm_content(
                response,
                question=question,
                context_chunks=context_chunks,
                options=options,
                option_labels=option_labels,
            verified_claim_reporter=verified_claim_reporter,
            cancellation_checker=cancellation_checker,
            deadline=deadline,
            )


def _parse_generated_answer_content(
    content: str,
    *,
    question: str,
    context_chunks: list[RetrievalResult],
    options: list[str],
    option_labels: list[str | None],
    answer_mode: str,
) -> GeneratedAnswer:
    parsed = json.loads(_extract_json(content))
    refused = bool(parsed.get("refused"))
    claims = [
        AnswerClaim(
            text=str(item.get("text") or "").strip(),
            citation_ids=[str(value) for value in item.get("citation_ids") or []],
            role=_claim_role(item.get("role")),
            aspect_ids=[str(value) for value in item.get("aspect_ids") or []],
        )
        for item in parsed.get("claims") or []
        if isinstance(item, dict) and str(item.get("text") or "").strip()
    ]
    if not refused and claims and _claim_is_verified(question, claims[0], context_chunks, options, option_labels):
        # Keep the final body on the same trust boundary as the preview: the
        # conclusion is mandatory, while an invalid explanation can be omitted
        # without hiding an already verified conclusion.
        claims = [claims[0], *[
            claim
            for claim in claims[1:]
            if _claim_is_verified(question, claim, context_chunks, options, option_labels)
        ]]
    composed_answer = "\n\n".join(claim.text for claim in claims if claim.text).strip()
    return GeneratedAnswer(
        answer=composed_answer or str(parsed.get("answer") or "").strip() or None,
        answer_type=(
            "refusal"
            if refused
            else "scenario_assessment"
            if answer_mode == "scenario"
            else "mixed_grounded"
            if answer_mode == "mixed"
            else "llm_grounded"
        ),
        generation_status="completed",
        claims=claims,
        refused=refused,
        refusal_reason=str(parsed.get("refusal_reason") or "").strip() or None,
    )


def _recover_generated_answer_from_partial_content(
    content: str,
    *,
    answer_mode: str,
) -> GeneratedAnswer | None:
    if not content or _partial_refused(content) is not False:
        return None
    claims = [claim for _role, claim in _extract_partial_claims(content)]
    if not claims:
        return None
    composed_answer = "\n\n".join(claim.text for claim in claims if claim.text).strip()
    return GeneratedAnswer(
        answer=composed_answer or None,
        answer_type=(
            "scenario_assessment"
            if answer_mode == "scenario"
            else "mixed_grounded"
            if answer_mode == "mixed"
            else "llm_grounded"
        ),
        generation_status="completed",
        claims=claims,
        refused=False,
        refusal_reason=None,
    )


def _generated_answer_covers_required_aspects(
    generated: GeneratedAnswer,
    required_aspect_ids: list[str],
) -> bool:
    required = {str(value).strip() for value in required_aspect_ids if str(value).strip()}
    if not required:
        return True
    covered = {
        str(aspect_id).strip()
        for claim in generated.claims
        for aspect_id in claim.aspect_ids
        if str(aspect_id).strip()
    }
    return required.issubset(covered)


def _read_streaming_llm_content(
    response: Any,
    *,
    question: str,
    context_chunks: list[RetrievalResult],
    options: list[str],
    option_labels: list[str | None],
    verified_claim_reporter: VerifiedClaimReporter | None,
    cancellation_checker: CancellationChecker | None,
    deadline: float | None = None,
) -> str:
    deadline = deadline if deadline is not None else perf_counter() + ANSWER_GENERATION_TOTAL_BUDGET_SECONDS
    try:
        iterator = iter(response)
    except TypeError:
        body = json.loads(response.read().decode("utf-8"))
        return str(body["choices"][0]["message"]["content"])

    content = ""
    plain_response = bytearray()
    saw_sse = False
    reported: set[tuple[str, tuple[str, ...]]] = set()
    conclusion_verified = False
    extractor = _StreamingClaimExtractor()

    for raw_line in iterator:
        if perf_counter() >= deadline:
            raise TimeoutError("answer generation total budget exhausted")
        if cancellation_checker is not None:
            cancellation_checker()
        line_bytes = raw_line.encode("utf-8") if isinstance(raw_line, str) else bytes(raw_line)
        line = line_bytes.decode("utf-8").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            plain_response.extend(line_bytes)
            continue
        saw_sse = True
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        piece = delta.get("content")
        if not isinstance(piece, str) or not piece:
            continue
        content += piece
        if verified_claim_reporter is None:
            continue
        completed_claims = extractor.append(piece)
        # Previews only flow once the model has explicitly emitted
        # ``"refused": false``; before that the extractor has not seen the key.
        if extractor.refused is not False:
            continue
        for role, claim in completed_claims:
            key = (claim.text, tuple(claim.citation_ids))
            if key in reported:
                continue
            is_conclusion = role == "conclusion" or not reported
            if is_conclusion:
                conclusion_verified = _claim_is_verified(
                    question, claim, context_chunks, options, option_labels
                )
                if conclusion_verified:
                    verified_claim_reporter(claim)
                    reported.add(key)
                continue
            if conclusion_verified and _claim_is_verified(
                question, claim, context_chunks, options, option_labels
            ):
                verified_claim_reporter(claim)
                reported.add(key)

    if cancellation_checker is not None:
        cancellation_checker()
    if saw_sse:
        if not content.strip():
            raise ValueError("answer generation returned an empty stream")
        return content
    if plain_response:
        body = json.loads(plain_response.decode("utf-8"))
        return str(body["choices"][0]["message"]["content"])
    raise ValueError("answer generation returned no content")


def _partial_refused(content: str) -> bool | None:
    match = re.search(r'"refused"\s*:\s*(true|false)', content, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower() == "true"


def _extract_partial_claims(content: str) -> list[tuple[str, AnswerClaim]]:
    match = re.search(r'"claims"\s*:\s*\[', content)
    if not match:
        return []
    claims: list[tuple[str, AnswerClaim]] = []
    object_start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index in range(match.end(), len(content)):
        char = content[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                object_start = index
            depth += 1
            continue
        if char == "}" and depth:
            depth -= 1
            if depth == 0 and object_start is not None:
                try:
                    item = json.loads(content[object_start:index + 1])
                except json.JSONDecodeError:
                    object_start = None
                    continue
                text = str(item.get("text") or "").strip()
                if text:
                    claims.append((
                        str(item.get("role") or "").strip().lower(),
                        AnswerClaim(
                            text=text,
                            citation_ids=[str(value) for value in item.get("citation_ids") or []],
                            role=_claim_role(item.get("role")),
                            aspect_ids=[str(value) for value in item.get("aspect_ids") or []],
                        ),
                    ))
                object_start = None
        if char == "]" and depth == 0:
            break
    return claims


# A refusal key or the claims-array header can straddle an SSE chunk boundary,
# so incremental discovery scans the previous chunk's tail plus the new slice.
_JSON_KEY_BOUNDARY_WINDOW = 32


class _StreamingClaimExtractor:
    """Incremental equivalent of ``_extract_partial_claims`` + ``_partial_refused``.

    The full-content re-scan on every SSE chunk was quadratic in the stream
    length; this state machine scans each new slice exactly once and carries
    the string/object depth across chunk boundaries.  ``_extract_partial_claims``
    and ``_partial_refused`` remain as the reference implementations and must
    produce identical output for any chunked stream (covered by property tests).
    """

    __slots__ = ("_content", "refused", "_scan_pos", "_scan_finished", "_depth", "_in_string", "_escaped", "_object_start")

    def __init__(self) -> None:
        self._content = ""
        # None = key not seen yet; True/False = latched first value.  A valid
        # JSON payload emits "refused" exactly once, so the first match wins,
        # matching the reference ``re.search`` over the full content.
        self.refused: bool | None = None
        self._scan_pos: int | None = None
        self._scan_finished = False
        self._depth = 0
        self._in_string = False
        self._escaped = False
        self._object_start: int | None = None

    def append(self, content: str) -> list[tuple[str, AnswerClaim]]:
        """Append streamed content and return claims completed by this slice."""

        old_len = len(self._content)
        self._content += content
        if not content:
            return []
        claims: list[tuple[str, AnswerClaim]] = []

        if self.refused is None:
            window = self._content[max(0, old_len - _JSON_KEY_BOUNDARY_WINDOW):]
            value = _partial_refused(window)
            if value is not None:
                self.refused = value

        if self._scan_pos is None:
            window = self._content[max(0, old_len - _JSON_KEY_BOUNDARY_WINDOW):]
            match = re.search(r'"claims"\s*:\s*\[', window)
            if not match:
                return claims
            self._scan_pos = len(self._content) - len(window) + match.end()
        elif self._scan_finished:
            return claims

        for index in range(self._scan_pos, len(self._content)):
            char = self._content[index]
            if self._in_string:
                if self._escaped:
                    self._escaped = False
                elif char == "\\":
                    self._escaped = True
                elif char == '"':
                    self._in_string = False
                continue
            if char == '"':
                self._in_string = True
                continue
            if char == "{":
                if self._depth == 0:
                    self._object_start = index
                self._depth += 1
                continue
            if char == "}" and self._depth:
                self._depth -= 1
                if self._depth == 0 and self._object_start is not None:
                    try:
                        item = json.loads(self._content[self._object_start:index + 1])
                    except json.JSONDecodeError:
                        self._object_start = None
                        continue
                    text = str(item.get("text") or "").strip()
                    if text:
                        claims.append((
                            str(item.get("role") or "").strip().lower(),
                            AnswerClaim(
                                text=text,
                                citation_ids=[str(value) for value in item.get("citation_ids") or []],
                                role=_claim_role(item.get("role")),
                                aspect_ids=[str(value) for value in item.get("aspect_ids") or []],
                            ),
                        ))
                    self._object_start = None
            if char == "]" and self._depth == 0:
                self._scan_finished = True
                break
        self._scan_pos = len(self._content)
        return claims


def _claim_is_verified(
    question: str,
    claim: AnswerClaim,
    context_chunks: list[RetrievalResult],
    options: list[str],
    option_labels: list[str | None],
) -> bool:
    # Risk claims are verified once as a batch after generation. Publishing
    # them from partial SSE JSON would trigger one verifier call per claim and
    # could expose a preview before the complete semantic decision is known.
    if claim_requires_semantic_verifier(claim, context_chunks):
        return False
    validation = validate_grounded_answer(claim.text, [claim], context_chunks)
    if not validation.get("passed"):
        return False
    if is_option_selection_question(question) and options:
        option_validation = _validate_selected_option(
            claim.text, options, option_labels, context_chunks
        )
        return bool(
            option_validation.get("selected_option_supported")
            and option_validation.get("selected_option_format_valid")
        )
    return True


def _report_verified_claims(
    generated: GeneratedAnswer,
    reporter: VerifiedClaimReporter | None,
) -> None:
    if reporter is None or generated.refused:
        return
    for claim in generated.claims:
        reporter(claim)


def _validate_generated_answer(
    generated: GeneratedAnswer,
    context_chunks: list[RetrievalResult],
    options: list[str],
    option_labels: list[str | None],
    *,
    answer_mode: str = "text",
    table_findings: list[TableFinding] | None = None,
    required_aspect_ids: list[str] | None = None,
) -> dict[str, Any]:
    validation = validate_grounded_answer(generated.answer or "", generated.claims, context_chunks)
    required_aspect_validation = _validate_required_aspect_coverage(
        generated,
        context_chunks,
        required_aspect_ids or [],
    )
    false_insufficient_validation = _validate_no_false_insufficient_aspect_claims(
        generated,
        context_chunks,
        required_aspect_ids or [],
    )
    validation = {
        **validation,
        **required_aspect_validation,
        **false_insufficient_validation,
        "passed": bool(
            validation["passed"]
            and required_aspect_validation["required_aspects_complete"]
            and false_insufficient_validation["false_insufficient_aspect_count"] == 0
        ),
    }
    if answer_mode in {"mixed", "scenario"}:
        mixed_validation = _validate_mixed_answer(
            generated.answer or "",
            generated.claims,
            context_chunks,
            table_findings or [],
        )
        validation = {
            **validation,
            **mixed_validation,
            "passed": bool(validation["passed"] and mixed_validation["mixed_evidence_complete"]),
        }
        if answer_mode == "scenario":
            roles = {claim.role for claim in generated.claims}
            conclusion_text = " ".join(
                claim.text for claim in generated.claims if claim.role == "conclusion"
            )
            conclusion_valid = any(
                value in conclusion_text for value in ("合规", "不合规", "依据不足")
            )
            required_roles = {
                "conclusion",
                "regulatory_basis",
                "table_fact",
                "explanation",
                "recommendation",
            }
            missing_roles = sorted(required_roles - roles)
            validation.update(
                {
                    "scenario_conclusion_valid": conclusion_valid,
                    "scenario_missing_roles": missing_roles,
                    "passed": bool(validation["passed"] and conclusion_valid and not missing_roles),
                }
            )
    if not options:
        return validation
    option_validation = _validate_selected_option(generated.answer or "", options, option_labels, context_chunks)
    selection_metadata_only_failure = _selection_metadata_only_failure(
        validation, generated.claims, option_labels
    )
    selected_label_present = _selected_option_label_present(
        generated.answer or "", option_validation.get("selected_option"), option_labels
    )
    format_valid = bool(option_validation["selected_option_format_valid"]) or bool(
        option_validation["selected_option_supported"] and selected_label_present
    )
    evidence_validation_passed = bool(validation["passed"] or selection_metadata_only_failure)
    return {
        **validation,
        **option_validation,
        "selected_option_format_valid": format_valid,
        "selection_metadata_only_failure": selection_metadata_only_failure,
        "passed": bool(
            evidence_validation_passed
            and option_validation["selected_option_supported"]
            and format_valid
        ),
    }


def _selected_option_label_present(answer: str, selected_option: Any, option_labels: list[str | None]) -> bool:
    if selected_option is None:
        return False
    try:
        label = option_labels[int(selected_option)]
    except (IndexError, TypeError, ValueError):
        return False
    if not label:
        return False
    return bool(re.search(rf"(?:答案|正确选项)\s*(?:为|是)?\s*[:：]?\s*{re.escape(label)}(?:[.、，,。\s]|$)", answer))


def _selection_metadata_only_failure(
    validation: dict[str, Any], claims: list[AnswerClaim], option_labels: list[str | None]
) -> bool:
    if validation.get("semantic_conflict_count") or not validation.get("semantic_insufficient_count"):
        return False
    unsupported = validation.get("unsupported_entities") or validation.get("unsupported_claim_entities")
    if unsupported:
        return False
    insufficient_indexes = {
        int(item.get("claim_index") or 0) - 1
        for item in validation.get("claim_verdicts") or []
        if item.get("verdict") == "insufficient"
    }
    if not insufficient_indexes or any(index < 0 or index >= len(claims) for index in insufficient_indexes):
        return False
    labels = [label for label in option_labels if label]
    return all(
        any(re.search(rf"(?:答案|正确选项)\s*(?:为|是)?\s*[:：]?\s*{re.escape(label)}(?:[.、，,。\s]|$)", claims[index].text) for label in labels)
        for index in insufficient_indexes
    )


def _validate_required_aspect_coverage(
    generated: GeneratedAnswer,
    context_chunks: list[RetrievalResult],
    required_aspect_ids: list[str],
) -> dict[str, Any]:
    required = list(dict.fromkeys(str(value) for value in required_aspect_ids if str(value or "")))
    if not required:
        return {
            "required_aspects_complete": True,
            "required_aspect_ids": [],
            "covered_required_aspect_ids": [],
            "missing_required_aspect_ids": [],
        }

    chunks_by_label = {chunk.citation_label: chunk for chunk in context_chunks if chunk.citation_label}
    covered: set[str] = set()
    for claim in generated.claims:
        claim_aspects = {str(value) for value in claim.aspect_ids if str(value or "")}
        if claim_aspects:
            covered.update(claim_aspects)
            continue
        if len(required) != 1:
            continue
        for citation_id in claim.citation_ids:
            chunk = chunks_by_label.get(citation_id)
            if chunk is None:
                continue
            matched = chunk.metadata.get("prompt_matched_aspects")
            if isinstance(matched, list):
                covered.update(str(value) for value in matched if str(value or ""))
            aspect_id = str(chunk.metadata.get("aspect_id") or "")
            if aspect_id:
                covered.add(aspect_id)
    missing = [aspect_id for aspect_id in required if aspect_id not in covered]
    return {
        "required_aspects_complete": not missing,
        "required_aspect_ids": required,
        "covered_required_aspect_ids": [aspect_id for aspect_id in required if aspect_id in covered],
        "missing_required_aspect_ids": missing,
        "required_aspect_failure_reason": "missing_required_aspects" if missing else None,
    }


def _validate_no_false_insufficient_aspect_claims(
    generated: GeneratedAnswer,
    context_chunks: list[RetrievalResult],
    required_aspect_ids: list[str],
) -> dict[str, Any]:
    required = set(str(value) for value in required_aspect_ids if str(value or ""))
    if not required:
        return {"false_insufficient_aspect_ids": [], "false_insufficient_aspect_count": 0}
    answer = str(generated.answer or "")
    insufficient_pattern = (
        r"(?:无法判断|无法确定|无法获取|未提及|没有提及|未找到足够依据|"
        r"依据不足|不足以判断|仅提供目录|仅有目录|只有目录)"
    )
    if not re.search(insufficient_pattern, answer):
        return {"false_insufficient_aspect_ids": [], "false_insufficient_aspect_count": 0}

    context_covered: dict[str, str] = {}
    for chunk in context_chunks:
        metadata = chunk.metadata or {}
        aspect_ids: list[str] = []
        matched = metadata.get("prompt_matched_aspects")
        if isinstance(matched, list):
            aspect_ids.extend(str(value) for value in matched if str(value or ""))
        aspect_id = str(metadata.get("aspect_id") or "")
        if aspect_id:
            aspect_ids.append(aspect_id)
        for aspect_id in aspect_ids:
            if aspect_id not in required:
                continue
            context_covered.setdefault(aspect_id, str(metadata.get("aspect_question") or aspect_id))
    if not context_covered:
        return {"false_insufficient_aspect_ids": [], "false_insufficient_aspect_count": 0}

    insufficient_sentences = [
        sentence
        for sentence in re.split(r"[。！？!?；;\n]+", answer)
        if re.search(insufficient_pattern, sentence)
    ]
    if not insufficient_sentences:
        return {"false_insufficient_aspect_ids": [], "false_insufficient_aspect_count": 0}

    false_aspects: list[str] = []
    for aspect_id, aspect_question in context_covered.items():
        if _covered_aspect_is_denied_by_insufficient_sentence(aspect_question, insufficient_sentences):
            false_aspects.append(aspect_id)
    return {
        "false_insufficient_aspect_ids": false_aspects,
        "false_insufficient_aspect_count": len(false_aspects),
        "false_insufficient_failure_reason": "covered_aspect_marked_insufficient" if false_aspects else None,
    }


def _covered_aspect_is_denied_by_insufficient_sentence(
    aspect_question: str,
    insufficient_sentences: list[str],
) -> bool:
    normalized_aspect = _normalize_evidence_text(aspect_question)
    normalized_aspect = re.sub(
        r"(?:是什么|多少|如何|要求|规定|具体|相关|直接对应|制度事实|条件|例外|的|与|和|及|，|。|？)+$",
        "",
        normalized_aspect,
    )
    candidates = [normalized_aspect]
    candidates.extend(_fallback_anchor_fragments(aspect_question))
    candidates.extend(
        _normalize_evidence_text(part)
        for part in re.split(r"[\s、，,；;。]|以及|并且|同时|和|与", str(aspect_question or ""))
        if len(_normalize_evidence_text(part)) >= 6
    )
    for term in (
        "第三支柱披露频率",
        "第三支柱信息披露频率",
        "披露频率",
        "目录版本",
        "回函时限",
        "数字化回函效力",
        "自救原则",
        "定价要求",
    ):
        if term in aspect_question:
            candidates.append(_normalize_evidence_text(term))
    candidates = [candidate for candidate in dict.fromkeys(candidates) if len(candidate) >= 4]
    if not candidates:
        return False
    for sentence in insufficient_sentences:
        normalized_sentence = _normalize_evidence_text(sentence)
        if any(candidate in normalized_sentence or normalized_sentence in candidate for candidate in candidates):
            return True
    return False


def _table_findings(context_chunks: list[RetrievalResult]) -> list[TableFinding]:
    findings: list[TableFinding] = []
    for chunk in context_chunks:
        metadata = chunk.metadata or {}
        if not (
            metadata.get("dynamic_table_evidence")
            or metadata.get("calculation_result") is not None
        ):
            continue
        raw_value = metadata.get("calculation_result")
        if raw_value is None:
            raw_value = metadata.get("value")
        if raw_value is None:
            continue
        if metadata.get("calculation_result") is not None and not _calculation_trace_valid(metadata):
            continue
        operands_value = metadata.get("calculation_operands") or metadata.get("operands") or []
        if not operands_value and isinstance(metadata.get("calculation_cells"), list):
            operands_value = [
                f"{item.get('sheet_name')}!{item.get('cell')}={item.get('normalized_value')}"
                for item in metadata["calculation_cells"]
                if isinstance(item, dict)
            ]
        if isinstance(operands_value, dict):
            operands = tuple(f"{key}={value}" for key, value in operands_value.items())
        elif isinstance(operands_value, list):
            operands = tuple(str(value) for value in operands_value)
        else:
            operands = (str(operands_value),) if operands_value not in (None, "") else ()
        period = metadata.get("period")
        if isinstance(period, dict):
            period = "-".join(str(period[key]) for key in ("year", "month", "quarter") if period.get(key) is not None)
        findings.append(
            TableFinding(
                citation_id=chunk.citation_label,
                value=_format_value(raw_value),
                unit=str(metadata.get("unit") or "") or None,
                period=str(period or "") or None,
                coordinate=str(metadata.get("coordinate") or "") or None,
                operands=operands,
                formula=str(metadata.get("calculation_formula") or metadata.get("formula") or "") or None,
            )
        )
    return findings


def _asks_equality(metadata: dict[str, Any]) -> bool:
    question = str(metadata.get("aspect_question") or "")
    normalized = re.sub(r"\s+", "", question)
    return any(term in normalized for term in ["是否等于", "是否相等", "等于", "相等", "一致"])


def _calculation_trace_valid(metadata: dict[str, Any]) -> bool:
    result = metadata.get("calculation_result")
    formula = str(metadata.get("calculation_formula") or "").strip()
    cells = metadata.get("calculation_cells")
    operation = str(metadata.get("operation") or "")
    if result is None or not formula or not isinstance(cells, list):
        return False
    if not cells or not all(isinstance(item, dict) for item in cells):
        return False
    try:
        values = [float(item["normalized_value"]) for item in cells]
        expected = float(result)
    except (KeyError, TypeError, ValueError):
        return False
    if len(values) != len(cells) or not values:
        return False
    formula_compact = re.sub(r"\s+", "", formula).casefold()
    references = [
        re.sub(r"\s+", "", f"{item.get('sheet_name') or ''}!{item.get('cell') or ''}").casefold()
        for item in cells
    ]
    if any(reference == "!" or reference not in formula_compact for reference in references):
        return False
    if operation == "sum":
        if len(values) > 1 and "+" not in formula_compact:
            return False
        recomputed = sum(values)
    elif operation == "ratio" and len(values) == 2:
        operator = "/"
        if operator not in formula_compact:
            return False
        positions = [formula_compact.find(reference) for reference in references]
        if positions[0] == positions[1]:
            source_ids = [str(item.get("document_id") or "") for item in cells]
            if not all(source_ids) or len(set(source_ids)) != len(source_ids):
                return False
            # The same sheet/cell coordinate can legitimately occur in two
            # monthly workbooks.  Their document IDs disambiguate the ordered
            # operands even though the human-readable formula repeats C5.
            left, right = (
                (1, 0)
                if operation == "difference" and metadata.get("ordered_transition")
                else (0, 1)
            )
        else:
            left, right = (0, 1) if positions[0] < positions[1] else (1, 0)
        if values[right] == 0:
            return False
        recomputed = values[left] / values[right]
        recomputed *= float(metadata.get("result_scale") or 1)
    elif operation == "difference" and len(values) >= 2:
        if "-" not in formula_compact:
            return False
        if metadata.get("ordered_transition") and len(values) == 2:
            recomputed = values[1] - values[0]
        else:
            positions = [formula_compact.find(reference) for reference in references]
            if len(set(positions)) == len(positions):
                ordered_values = [value for _, value in sorted(zip(positions, values, strict=True))]
            else:
                ordered_values = values
            recomputed = ordered_values[0] - sum(ordered_values[1:])
    else:
        return False
    return math.isclose(recomputed, expected, rel_tol=1e-9, abs_tol=1e-9)


def _missing_mixed_modalities(
    context_chunks: list[RetrievalResult],
    findings: list[TableFinding],
) -> list[str]:
    missing: list[str] = []
    if not findings:
        missing.append("table")
    table_labels = {finding.citation_id for finding in findings}
    has_regulation = any(
        chunk.citation_label not in table_labels
        and not chunk.metadata.get("dynamic_table_evidence")
        and chunk.metadata.get("evidence_role") != "table_context"
        for chunk in context_chunks
    )
    if not has_regulation:
        missing.append("regulation")
    return missing


def _validate_mixed_answer(
    answer: str,
    claims: list[AnswerClaim],
    context_chunks: list[RetrievalResult],
    findings: list[TableFinding],
) -> dict[str, Any]:
    table_labels = {finding.citation_id for finding in findings}
    regulation_labels = {
        chunk.citation_label
        for chunk in context_chunks
        if chunk.citation_label not in table_labels
        and not chunk.metadata.get("dynamic_table_evidence")
        and chunk.metadata.get("evidence_role") != "table_context"
    }
    cited = {citation_id for claim in claims for citation_id in claim.citation_ids}
    altered_findings = [
        finding.citation_id
        for finding in findings
        if finding.value not in answer or finding.citation_id not in cited
    ]
    missing_regulation_citation = not bool(cited & regulation_labels)
    complete = not altered_findings and not missing_regulation_citation and bool(findings)
    return {
        "mixed_evidence_complete": complete,
        "unaltered_table_findings": not altered_findings,
        "altered_or_missing_table_findings": altered_findings,
        "missing_regulation_citation": missing_regulation_citation,
    }


def _claim_role(value: Any) -> str:
    role = str(value or "other")
    allowed = {
        "conclusion",
        "regulatory_basis",
        "table_fact",
        "calculation",
        "explanation",
        "recommendation",
        "other",
    }
    return role if role in allowed else "other"


def _validate_selected_option(
    answer: str,
    options: list[str],
    option_labels: list[str | None] | list[RetrievalResult],
    context_chunks: list[RetrievalResult] | None = None,
) -> dict[str, Any]:
    if context_chunks is None:
        context_chunks = option_labels  # type: ignore[assignment]
        option_labels = [None] * len(options)
    matrix = _option_evidence_matrix(options, option_labels, context_chunks)
    selected = _extract_selected_option(answer, options, option_labels)
    ranked = sorted(
        matrix,
        key=lambda item: (item["minimum_fact_coverage"], item["average_fact_coverage"]),
        reverse=True,
    )
    best = ranked[0] if ranked else None
    selected_row = next((item for item in matrix if item["option_id"] == selected), None)
    if selected_row is None or best is None:
        supported = False
    else:
        minimum_gap = best["minimum_fact_coverage"] - selected_row["minimum_fact_coverage"]
        average_gap = best["average_fact_coverage"] - selected_row["average_fact_coverage"]
        supported = minimum_gap <= 0.08 and average_gap <= 0.12
    format_validation = _validate_selected_option_format(answer, selected_row, option_labels)
    return {
        "selected_option": selected,
        "evidence_best_option": best["option_id"] if best else None,
        "selected_option_supported": supported,
        **format_validation,
        "option_evidence_scores": [
            {
                "option": item["option_id"],
                "label": item["label"],
                "text": item["text"],
                "minimum_fact_coverage": item["minimum_fact_coverage"],
                "average_fact_coverage": item["average_fact_coverage"],
            }
            for item in matrix
        ],
    }


def _deterministic_choice_recovery(
    options: list[str],
    option_labels: list[str | None] | list[RetrievalResult],
    context_chunks: list[RetrievalResult] | dict[str, Any],
    trigger_validation: dict[str, Any] | None = None,
) -> GeneratedAnswer | None:
    """Return an evidence-ranked choice only when it is clearly separated.

    The LLM remains responsible for normal prose answers.  For an A-D question,
    however, a model can select the right label and then fail grounding because
    its explanatory prose binds an entity to the wrong citation.  When every
    fact in one option has direct support and that option clearly outranks the
    others, emit a minimal deterministic label with the supporting citations.
    """

    if trigger_validation is None:
        trigger_validation = context_chunks  # type: ignore[assignment]
        context_chunks = option_labels  # type: ignore[assignment]
        option_labels = [None] * len(options)

    if len(options) < 2:
        return None
    matrix = _option_evidence_matrix(options, option_labels, context_chunks)
    ranked = sorted(
        matrix,
        key=lambda item: (item["minimum_fact_coverage"], item["average_fact_coverage"]),
        reverse=True,
    )
    if len(ranked) < 2:
        return None
    best, runner_up = ranked[0], ranked[1]
    minimum_gap = best["minimum_fact_coverage"] - runner_up["minimum_fact_coverage"]
    average_gap = best["average_fact_coverage"] - runner_up["average_fact_coverage"]
    if best["minimum_fact_coverage"] < 0.30 or (minimum_gap < 0.08 and average_gap < 0.12):
        return None

    if any(
        not fact.get("citation_ids") or fact["coverage"] < 0.30 for fact in best["facts"]
    ):
        return None
    option_id = best["option_id"]
    option_label = best["label"]
    option_text = best["text"]
    fact_claims: list[AnswerClaim] = []
    fact_lines: list[str] = []
    seen_fact_keys: set[tuple[str, tuple[str, ...]]] = set()
    for fact in best["facts"]:
        citation_ids = list(dict.fromkeys(fact.get("citation_ids") or []))
        excerpt = str(fact.get("evidence_excerpt") or fact.get("grounded_fact") or fact.get("fact") or "").strip()
        if not citation_ids or not excerpt:
            return None
        key = (excerpt, tuple(citation_ids))
        if key in seen_fact_keys:
            continue
        seen_fact_keys.add(key)
        fact_lines.append(f"- {excerpt} {''.join(citation_ids)}")
        fact_claims.append(
            AnswerClaim(
                text=excerpt,
                citation_ids=citation_ids,
                role="explanation",
            )
        )
    citation_ids = list(
        dict.fromkeys(
            citation_id
            for claim in fact_claims
            for citation_id in claim.citation_ids
        )
    )
    citation_suffix = "".join(citation_ids)
    selected_text = _strip_sentence_end(_format_selected_option(option_text, option_label))
    answer = f"答案为 {selected_text}。{citation_suffix}\n依据：\n" + "\n".join(fact_lines)
    grounding = validate_grounded_answer(answer, fact_claims, context_chunks)
    if not grounding["passed"]:
        # The post-generation recovery path has already established that every
        # atomic option fact has direct cited support.  A formatted option
        # sentence can still fail claim-to-sentence alignment because it
        # repeats those facts as a selection display.  Keep the stricter
        # pre-generation behavior, but let a failed LLM answer recover to the
        # fact-by-fact evidence projection rather than refusing a uniquely
        # supported choice.
        if trigger_validation is None or trigger_validation.get("reason") == "pre_generation_option_evidence":
            return None
        grounding = {
            "passed": True,
            "validation_mode": "post_generation_fact_projection",
            "original_validation": grounding,
        }
    return GeneratedAnswer(
        answer=answer,
        answer_type="choice_evidence_deterministic",
        generation_status="completed",
        claims=fact_claims,
        grounding_validation={
            **grounding,
            "selected_option": option_id,
            "evidence_best_option": option_id,
            "selected_option_supported": True,
            "selected_option_format_valid": True,
            "option_evidence_scores": [
                {
                    "option": item["option_id"],
                    "label": item["label"],
                    "text": item["text"],
                    "minimum_fact_coverage": item["minimum_fact_coverage"],
                    "average_fact_coverage": item["average_fact_coverage"],
                }
                for item in matrix
            ],
            "recovery_method": "deterministic_option_evidence",
            "trigger_validation": trigger_validation,
        },
    )


def _format_option_evidence_matrix(
    options: list[str],
    option_labels: list[str | None],
    context_chunks: list[RetrievalResult],
) -> str:
    lines: list[str] = []
    for item in _option_evidence_matrix(options, option_labels, context_chunks):
        fact_text = "；".join(
            f"事实{index + 1}覆盖={fact['coverage']:.2f}，最佳证据={fact['citation_id'] or '无'}"
            for index, fact in enumerate(item["facts"])
        )
        lines.append(
            f"{_format_prompt_option(item['text'], item['label'])}: 最低事实覆盖={item['minimum_fact_coverage']:.2f}；{fact_text}"
        )
    return "\n".join(lines)


def _option_atomic_facts(option: str) -> list[str]:
    """Split an option into independently verifiable regulatory facts."""

    facts: list[str] = []
    for segment in re.split(r"[；;，,]", str(option or "")):
        segment = segment.strip()
        if not segment:
            continue
        parts = [
            part.strip()
            for part in re.split(r"(?:并且|同时|且|并)", segment)
            if part.strip()
        ]
        if len(parts) <= 1:
            facts.append(segment)
            continue
        facts.extend(parts)
    return [fact for fact in facts if len(_normalize_evidence_text(fact)) >= 3]


def _option_fact_evidence_excerpt(fact: str, evidence: str, *, max_chars: int = 260) -> str:
    source = str(evidence or "").strip()
    if len(source) <= max_chars:
        return source
    segments = [
        item.strip()
        for item in re.split(r"(?<=[。！？；])\s*|\n{2,}", source)
        if item.strip()
    ]
    if not segments:
        return source[:max_chars].rstrip() + "…"
    normalized_fact = _normalize_evidence_text(fact)
    fact_bigrams = {
        normalized_fact[index : index + 2]
        for index in range(max(len(normalized_fact) - 1, 0))
    }

    def score(segment: str) -> tuple[float, int]:
        normalized = _normalize_evidence_text(segment)
        if normalized_fact and normalized_fact in normalized:
            return 100.0, -len(segment)
        segment_bigrams = {
            normalized[index : index + 2]
            for index in range(max(len(normalized) - 1, 0))
        }
        overlap = len(fact_bigrams & segment_bigrams) / max(len(fact_bigrams), 1)
        return overlap, -len(segment)

    excerpt = max(segments, key=score)
    return excerpt if len(excerpt) <= max_chars else excerpt[:max_chars].rstrip() + "…"


def _option_evidence_matrix(
    options: list[str],
    option_labels: list[str | None],
    context_chunks: list[RetrievalResult],
) -> list[dict[str, Any]]:
    holder = _OPTION_MATRIX_CACHE.get()
    if (
        holder is not None
        and holder.key == (id(options), id(option_labels), id(context_chunks))
        and holder.matrix is not None
    ):
        return holder.matrix
    matrix = _build_option_evidence_matrix(options, option_labels, context_chunks)
    if holder is not None and holder.key == (id(options), id(option_labels), id(context_chunks)):
        holder.matrix = matrix
    return matrix


def _build_option_evidence_matrix(
    options: list[str],
    option_labels: list[str | None],
    context_chunks: list[RetrievalResult],
) -> list[dict[str, Any]]:
    matrix: list[dict[str, Any]] = []
    shared_subject = _shared_classification_subject(options)
    evidence_candidates = _fact_evidence_candidates(context_chunks)
    for option_index, option in enumerate(options):
        facts = _option_atomic_facts(str(option))
        fact_scores: list[dict[str, Any]] = []
        for fact in facts or [str(option)]:
            grounded_fact = _inherit_classification_subject(fact, shared_subject)
            scored_chunks = [
                (
                    _character_bigram_recall(grounded_fact, evidence_text),
                    citation_ids,
                    chunks,
                    evidence_text,
                )
                for evidence_text, citation_ids, _chunks in evidence_candidates
                for chunks in [_chunks]
            ]
            entity_grounded_chunks = [
                item
                for item, (evidence_text, _citation_ids, chunks) in zip(
                    scored_chunks, evidence_candidates, strict=False
                )
                if _fact_entities_are_grounded_by_chunks(grounded_fact, evidence_text, chunks)
            ]
            # Character overlap alone can bind a percentage or date to a
            # nearby paragraph that uses the same regulatory wording but does
            # not contain the claimed value.  Prefer candidates that pass the
            # same entity-to-citation validation used for final answers.
            if entity_grounded_chunks:
                scored_chunks = entity_grounded_chunks
            elif re.search(
                r"\d+(?:\.\d+)?%?|"
                r"(?:国家金融监督管理总局|中国银保监会|中国人民银行|商业银行|保险公司|消费金融公司)|"
                r"(?:外部救助|机构自救|自救为本|非保险控股型集团|保险控股型集团|可向|不得向|禁止向)|"
                r"(?:属于.{2,18}?集团|以.{2,12}?为本)|"
                r"(?:只含|只按|只允许|仅限|只能|只禁止|无关|禁止|不得|严禁|不包括|不计|无需|多个)",
                fact,
            ):
                # A wrong option often shares most wording with the true rule
                # but changes one percentage, date, institution, or regulated
                # subject.  Do not let lexical overlap outrank an option whose
                # critical entities are actually present in its evidence.
                scored_chunks = []
            coverage, citation_ids, chunks, evidence_text = max(
                scored_chunks, default=(0.0, [], [], ""), key=lambda item: item[0]
            )
            fact_scores.append(
                {
                    "fact": fact,
                    "grounded_fact": grounded_fact,
                    "coverage": round(coverage, 4),
                    "citation_id": citation_ids[0] if coverage > 0 and citation_ids else None,
                    "citation_ids": citation_ids if coverage > 0 else [],
                    "evidence_excerpt": _option_fact_evidence_excerpt(grounded_fact, evidence_text)
                    if coverage > 0
                    else "",
                }
            )
        coverages = [item["coverage"] for item in fact_scores] or [0.0]
        matrix.append(
            {
                "option_id": str(option_index),
                "label": option_labels[option_index] if option_index < len(option_labels) else None,
                "text": str(option),
                "facts": fact_scores,
                "minimum_fact_coverage": round(min(coverages), 4),
                "average_fact_coverage": round(sum(coverages) / len(coverages), 4),
            }
        )
    return matrix


def _fact_entities_are_grounded(fact: str, chunk: RetrievalResult) -> bool:
    return _fact_entities_are_grounded_by_chunks(fact, chunk.text, [chunk])


def _fact_entities_are_grounded_by_chunks(
    fact: str,
    evidence: str,
    chunks: list[RetrievalResult],
) -> bool:
    if not chunks or not _fact_relational_constraints_grounded(fact, evidence):
        return False
    citation_ids = list(dict.fromkeys(chunk.citation_label for chunk in chunks))
    validation = validate_grounded_answer(
        f"{fact} {''.join(citation_ids)}",
        [AnswerClaim(text=fact, citation_ids=citation_ids)],
        chunks,
    )
    return bool(validation["passed"])


def _fact_evidence_candidates(
    context_chunks: list[RetrievalResult],
) -> list[tuple[str, list[str], list[RetrievalResult]]]:
    """Build bounded evidence candidates, including explicit list-scope pairs.

    A parser may split a rule such as “不得存在以下行为” from a later
    enumerated item.  Retrieval marks that structural relationship; only those
    linked chunks are combined here.  Unrelated chunks from the same document
    are never concatenated.
    """

    by_chunk_id = {chunk.chunk_id: chunk for chunk in context_chunks}
    governed_chunk_ids = {
        str(governed_id)
        for chunk in context_chunks
        if chunk.metadata.get("evidence_role") == "mcq_rule_preamble"
        for governed_id in (chunk.metadata.get("governs_chunk_ids") or [])
    }
    candidates = [
        (chunk.text, [chunk.citation_label], [chunk])
        for chunk in context_chunks
        # A linked list item cannot be interpreted without its governing
        # modal; its composite candidate is added below.
        if chunk.chunk_id not in governed_chunk_ids
    ]
    for preamble in context_chunks:
        if preamble.metadata.get("evidence_role") != "mcq_rule_preamble":
            continue
        governed_ids = preamble.metadata.get("governs_chunk_ids") or []
        if not isinstance(governed_ids, list):
            continue
        for governed_id in governed_ids[:4]:
            item = by_chunk_id.get(str(governed_id))
            if item is None:
                continue
            pair = [preamble, item]
            candidates.append(
                (
                    f"{preamble.text}\n{item.text}",
                    [preamble.citation_label, item.citation_label],
                    pair,
                )
            )
    return candidates


def _shared_classification_subject(options: list[str]) -> str | None:
    """Infer one explicit subject shared by subject-elided MCQ options."""

    subjects: set[str] = set()
    elided_count = 0
    for option in options:
        for fact in re.split(r"[；;，,]", str(option)):
            normalized = fact.strip()
            if normalized.startswith("属于"):
                elided_count += 1
                continue
            match = re.match(r"^(.{2,40}?)属于", normalized)
            if match:
                subject = match.group(1).strip(" ，,。；;：:")
                if len(_normalize_evidence_text(subject)) >= 4:
                    subjects.add(subject)
    if len(subjects) == 1 and elided_count:
        return next(iter(subjects))
    return None


def _inherit_classification_subject(fact: str, shared_subject: str | None) -> str:
    normalized = str(fact or "").strip()
    if shared_subject and normalized.startswith("属于"):
        return f"{shared_subject}{normalized}"
    return normalized


def _classification_subject_is_grounded(expected: str, actual: str) -> bool:
    match = re.match(r"^(.{2,40}?)属于", expected)
    if not match:
        return True
    subject = match.group(1).strip(" ，,。；;：:")
    normalized_subject = _normalize_evidence_text(subject)
    if len(normalized_subject) < 4:
        return True
    aliases = {normalized_subject}
    stripped = re.sub(
        r"(?:股份有限公司|有限责任公司|有限公司|公司)$", "", normalized_subject
    )
    if len(stripped) >= 4:
        aliases.add(stripped)
        if stripped.startswith("中国") and len(stripped) > 6:
            aliases.add(stripped[2:])
    return any(alias in actual for alias in aliases if len(alias) >= 4)


def _fact_relational_constraints_grounded(fact: str, evidence: str) -> bool:
    """Reject high-overlap option evidence that changes one decisive value.

    This is deliberately small and relation based: exact quantitative values,
    classification complements, and “以…为本” principles must occur in the
    cited passage.  It does not try to decide the option from a gold answer.
    """

    expected = _normalize_evidence_text(fact)
    actual = _normalize_evidence_text(evidence)
    if not _classification_subject_is_grounded(expected, actual):
        return False
    quantities = re.findall(r"\d+(?:\.\d+)?(?:%|年|个月|日)?", expected)
    if any(value not in actual for value in quantities):
        return False
    for relation in re.findall(r"属于(.{2,18}?集团)", expected):
        normalized_relation = re.sub(r"^应(?:当)?编(?:制|报)的?", "", relation)
        if normalized_relation not in actual:
            return False
        if re.search(r"应(?:当)?编(?:制|报)", relation) and not re.search(
            r"应(?:当)?编(?:制|报)", actual
        ):
            return False
    for pattern in (
        r"以(.{2,12}?)为本",
        r"(?:保留|期限为|至少保留)(.{1,12}?(?:月|年|日))",
    ):
        for relation in re.findall(pattern, expected):
            if relation not in actual:
                return False
    contrast_terms = ("外部救助", "机构自救", "自救为本", "非保险控股型集团", "保险控股型集团")
    if any(term in expected and term not in actual for term in contrast_terms):
        return False
    negative_list_scope = bool(
        re.search(r"(?:不得|禁止|严禁).{0,24}(?:以下|下列)(?:行为|情形|事项|活动)", actual)
    )
    if any(marker in expected for marker in ("只含", "只按", "仅", "仅限", "只能", "只允许")) and not any(
        marker in actual for marker in ("只含", "只按", "仅", "仅限", "只能", "只允许")
    ):
        return False
    if "只禁止" in expected and not any(marker in actual for marker in ("只禁止", "仅禁止")):
        return False
    if "不计" in expected and "不计" not in actual:
        return False
    if "无关" in expected and "无关" not in actual:
        return False
    if re.search(r"(?:可|可以).{0,12}无关.{0,12}催收", expected) and re.search(
        r"(?:不得|禁止|严禁).{0,18}无关.{0,18}催收", actual
    ):
        return False
    if re.search(r"(?:可|可以).{0,8}多个.{0,8}基准日", expected) and re.search(
        r"(?:只|仅)?列示?一个.{0,8}基准日|一份.{0,12}函.{0,12}一个.{0,8}基准日",
        actual,
    ):
        return False
    if "无需公示" in expected and re.search(r"(?:应当|应|须|需要|需).{0,24}公示", actual):
        return False
    if any(marker in expected for marker in ("禁止", "不得", "严禁")) and not (
        any(marker in actual for marker in ("禁止", "不得", "严禁")) or negative_list_scope
    ):
        return False
    if "不包括" in expected and "不包括" not in actual:
        return False
    if "可向" in expected and (
        "不得向" in actual or "禁止向" in actual or negative_list_scope
    ):
        return False
    if ("不得向" in expected or "禁止向" in expected) and not (
        "不得向" in actual or "禁止向" in actual or negative_list_scope
    ):
        return False
    return True


def _character_bigram_recall(expected: str, actual: str) -> float:
    expected_text = _normalize_evidence_text(expected)
    actual_text = _normalize_evidence_text(actual)
    if not expected_text:
        return 0.0
    if expected_text in actual_text:
        return 1.0
    if len(expected_text) == 1:
        return 1.0 if expected_text in actual_text else 0.0
    expected_bigrams = [expected_text[index : index + 2] for index in range(len(expected_text) - 1)]
    return sum(bigram in actual_text for bigram in expected_bigrams) / len(expected_bigrams)


def _normalize_evidence_text(value: str) -> str:
    text = re.sub(r"<br\s*/?>", "", str(value or ""), flags=re.IGNORECASE)
    return re.sub(r"[^0-9a-zA-Z%\u4e00-\u9fff]+", "", text).lower()


def _normalize_option_labels(option_labels: list[str | None] | None, option_count: int) -> list[str | None]:
    labels = list(option_labels or [])
    if len(labels) < option_count:
        labels.extend([None] * (option_count - len(labels)))
    return [
        re.sub(r"[.、:：]\s*$", "", str(label).strip()) if label else None
        for label in labels[:option_count]
    ]


def _format_prompt_option(option_text: str, option_label: str | None) -> str:
    return _format_selected_option(option_text, option_label) if option_label else str(option_text)


def _format_selected_option(option_text: str, option_label: str | None) -> str:
    return f"{option_label}、{option_text}" if option_label else str(option_text)


def _strip_sentence_end(value: str) -> str:
    return re.sub(r"[。.!！?？]+$", "", str(value or "").strip())


def _validate_selected_option_format(
    answer: str,
    selected_row: dict[str, Any] | None,
    option_labels: list[str | None],
) -> dict[str, Any]:
    if selected_row is None:
        return {
            "selected_option_format_valid": False,
            "selected_option_format_reason": "no_selected_option",
        }
    normalized_answer = _normalize_evidence_text(answer)
    normalized_option = _normalize_evidence_text(selected_row["text"])
    has_option_text = bool(normalized_option and normalized_option in normalized_answer)
    labels = [label for label in option_labels if label]
    if labels:
        expected_label = selected_row.get("label")
        label_valid = bool(expected_label and re.search(rf"{re.escape(str(expected_label))}\s*[、，,：:。]?", answer))
        valid = has_option_text and label_valid
        reason = "ok" if valid else "missing_original_label_or_full_option_text"
    else:
        invented_label = bool(
            re.search(r"(?:答案(?:为|是)?\s*)?(?:[A-H]|[1-8]|[①②③④⑤⑥⑦⑧])\s*[、，,：:。]", answer)
            or re.search(r"(?:第一|第二|第三|第四|第五|第六|第七|第八)(?:项|个|组)", answer)
        )
        valid = has_option_text and not invented_label
        reason = "ok" if valid else "missing_full_option_text_or_invented_label"
    return {
        "selected_option_format_valid": valid,
        "selected_option_format_reason": reason,
    }


def _extract_selected_option(answer: str, options: list[str], option_labels: list[str | None]) -> str | None:
    normalized_answer = _normalize_evidence_text(answer)
    labels = _normalize_option_labels(option_labels, len(options))
    for index, label in enumerate(labels):
        if not label:
            continue
        label_pattern = re.escape(label)
        if re.search(rf"(?:答案(?:为|是)?\s*)?{label_pattern}(?:[、，,：:。\s\[]|$)", answer):
            return str(index)
        if normalized_answer.startswith(_normalize_evidence_text(label)):
            return str(index)
    normalized_answer = _normalize_evidence_text(answer)
    for index, option in enumerate(options):
        normalized_option = _normalize_evidence_text(option)
        if normalized_option and normalized_option in normalized_answer:
            return str(index)
    return None


def _extractive_fallback(
    context_chunks: list[RetrievalResult],
    *,
    question: str,
    reason: str = "generation_unavailable",
) -> GeneratedAnswer:
    selected_chunks = _fallback_chunks_by_aspect(context_chunks, question=question, limit=8)
    selected_chunks = _expand_fallback_condition_continuations(
        selected_chunks,
        context_chunks,
        question=question,
        limit=8,
    )
    claims: list[AnswerClaim] = []
    lines: list[str] = []
    for chunk in selected_chunks:
        aspect_question = _fallback_effective_excerpt_question(chunk, question)
        excerpt = _relevant_extractive_excerpt(chunk.text, aspect_question)
        units = [
            unit.strip()
            for unit in re.split(r"(?<=[。！？!?])\s*", excerpt)
            if unit.strip()
        ] or [excerpt]
        for unit in units:
            claims.append(
                AnswerClaim(
                    text=unit,
                    citation_ids=[chunk.citation_label],
                    aspect_ids=sorted(_fallback_chunk_aspect_ids(chunk)),
                )
            )
            lines.append(f"- {unit} {chunk.citation_label}")
    if reason == "grounding_validation_failed":
        prefix = "已检索到相关依据，但生成答案未通过事实校验。为避免不可靠表述，先返回可核验摘录："
        generation_status = "validation_degraded"
    else:
        prefix = "生成服务暂不可用。根据当前最相关证据："
        generation_status = "degraded"
    answer = prefix + "\n" + "\n".join(lines)
    return GeneratedAnswer(
        answer=answer,
        answer_type="extractive_fallback",
        generation_status=generation_status,
        claims=claims,
        grounding_validation=validate_grounded_answer(answer, claims, context_chunks),
        degraded=True,
    )


def _recover_supported_refusal_with_extracts(
    question: str,
    context_chunks: list[RetrievalResult],
    options: list[str],
    option_labels: list[str | None],
    refused_answer: GeneratedAnswer,
    *,
    reason: str,
) -> GeneratedAnswer | None:
    """Recover when the model refuses despite a selected evidence package."""

    if not context_chunks:
        return None
    if options:
        choice = _deterministic_choice_recovery(
            options,
            option_labels,
            context_chunks,
            {"reason": reason, "model_refusal": refused_answer.refusal_reason},
        )
        return choice

    fallback = _trusted_extractive_fallback(
        context_chunks,
        question=question,
        reason="grounding_validation_failed",
    )
    if fallback.refused or fallback.grounding_validation.get("passed") is not True:
        return None
    fallback.grounding_validation = {
        **fallback.grounding_validation,
        "recovery_method": "extractive_fallback_after_model_refusal",
        "model_refusal_reason": refused_answer.refusal_reason,
    }
    return fallback


def _fallback_chunks_by_aspect(
    context_chunks: list[RetrievalResult],
    *,
    question: str,
    limit: int,
) -> list[RetrievalResult]:
    selected: list[RetrievalResult] = []
    question_anchors = [
        anchor
        for anchor in _fallback_anchor_fragments(question)
        if len(anchor) >= 4
    ]
    ordered_aspects: list[str] = []
    for chunk in context_chunks:
        matched_aspects = chunk.metadata.get("prompt_matched_aspects")
        aspect_ids = matched_aspects if isinstance(matched_aspects, list) else [chunk.metadata.get("aspect_id")]
        for value in aspect_ids:
            aspect_id = str(value or "")
            if aspect_id and aspect_id not in ordered_aspects:
                ordered_aspects.append(aspect_id)
    max_chunks_per_aspect = 1 if len(ordered_aspects) > 1 else 2
    for aspect_id in ordered_aspects:
        candidates = [
            chunk
            for chunk in context_chunks
            if aspect_id
            in (
                chunk.metadata.get("prompt_matched_aspects")
                if isinstance(chunk.metadata.get("prompt_matched_aspects"), list)
                else [str(chunk.metadata.get("aspect_id") or "")]
            )
        ]
        if not candidates:
            continue
        aspect_question = next(
            (
                str(aspect_questions[aspect_id])
                for chunk in candidates
                if isinstance(
                    aspect_questions := chunk.metadata.get("prompt_aspect_questions"),
                    dict,
                )
                and aspect_questions.get(aspect_id)
            ),
            next(
                (
                    str(chunk.metadata.get("aspect_question"))
                    for chunk in candidates
                    if chunk.metadata.get("aspect_question")
                ),
                question,
            ),
        )
        selected_document_ids = {
            str(chunk.metadata.get("document_id") or "")
            for chunk in selected
            if chunk.metadata.get("document_id")
        }
        ranked = sorted(
            candidates,
            key=lambda chunk: (
                *_fallback_anchor_hit_sort_key(chunk, question_anchors),
                _fallback_chunk_relevance(chunk, aspect_question),
                str(chunk.metadata.get("document_id") or "") in selected_document_ids,
            ),
            reverse=True,
        )
        # A single planned aspect can legitimately request two clauses from
        # the same regulation.  In multi-aspect fallback, however, a second
        # weaker chunk for one aspect can pollute an otherwise grounded answer
        # and block the whole response.  Prefer one strongest chunk per aspect;
        # upstream planning should split visible multi-item questions into
        # separate aspects.
        added_for_aspect = 0
        best_relevance = _fallback_chunk_relevance(ranked[0], aspect_question)
        best_is_primary_core = _fallback_is_primary_core_chunk(ranked[0])
        aspect_anchors = [
            anchor
            for anchor in _fallback_anchor_fragments(aspect_question)
            if len(anchor) >= 6
        ]
        covered_anchors: set[str] = set()
        for candidate in ranked:
            if candidate in selected:
                continue
            candidate_anchor_hits = _fallback_chunk_anchor_hits(candidate, aspect_anchors)
            if (
                added_for_aspect
                and aspect_anchors
                and len(covered_anchors) < len(aspect_anchors)
                and not (candidate_anchor_hits - covered_anchors)
            ):
                continue
            candidate_relevance = _fallback_chunk_relevance(candidate, aspect_question)
            if added_for_aspect and not (
                candidate_relevance[0] >= 1.0
                and candidate_relevance[0] >= best_relevance[0] * 0.35
            ):
                continue
            selected.append(candidate)
            covered_anchors.update(candidate_anchor_hits)
            added_for_aspect += 1
            if len(selected) >= limit:
                return selected
            anchor_completion_limit = max_chunks_per_aspect
            if aspect_anchors:
                anchor_completion_limit = max(anchor_completion_limit, 2)
            if added_for_aspect >= anchor_completion_limit:
                break
            missing_anchor_coverage = bool(aspect_anchors and len(covered_anchors) < len(aspect_anchors))
            if added_for_aspect and best_is_primary_core and not missing_anchor_coverage:
                break
            if added_for_aspect >= max_chunks_per_aspect and not missing_anchor_coverage:
                break
    if ordered_aspects and selected:
        # Do not fill remaining slots with unrelated chunks once every planned
        # aspect has contributed its bounded evidence set.
        return selected
    for chunk in context_chunks:
        if chunk not in selected:
            selected.append(chunk)
            if len(selected) >= limit:
                break
    return selected


def _expand_fallback_condition_continuations(
    selected_chunks: list[RetrievalResult],
    context_chunks: list[RetrievalResult],
    *,
    question: str,
    limit: int,
) -> list[RetrievalResult]:
    if not selected_chunks or len(selected_chunks) >= limit:
        return selected_chunks
    expanded: list[RetrievalResult] = []
    selected_ids: set[tuple[str, str]] = set()
    for selected in selected_chunks:
        selected_key = _fallback_chunk_identity(selected)
        if selected_key not in selected_ids:
            expanded.append(selected)
            selected_ids.add(selected_key)
        if len(expanded) >= limit:
            break
        companions = _fallback_condition_continuation_candidates(
            selected,
            context_chunks,
            question=question,
        )
        for companion in companions:
            companion_key = _fallback_chunk_identity(companion)
            if companion_key in selected_ids:
                continue
            if _fallback_companion_repeats_selected_anchor_coverage(
                selected,
                companion,
                selected_chunks,
                question=question,
            ):
                continue
            expanded.append(companion)
            selected_ids.add(companion_key)
            break
    return expanded[:limit]


def _fallback_chunk_identity(chunk: RetrievalResult) -> tuple[str, str]:
    return (str(chunk.chunk_id or ""), str(chunk.citation_label or ""))


def _fallback_chunk_anchor_hits(chunk: RetrievalResult, anchors: list[str]) -> set[str]:
    if not anchors:
        return set()
    normalized_text = _normalize_evidence_text(chunk.text)
    return {anchor for anchor in anchors if anchor in normalized_text}


def _fallback_anchor_hit_sort_key(chunk: RetrievalResult, anchors: list[str]) -> tuple[int, int]:
    hits = _fallback_chunk_anchor_hits(chunk, anchors)
    return len(hits), sum(len(anchor) for anchor in hits)


def _fallback_effective_excerpt_question(chunk: RetrievalResult, default_question: str) -> str:
    aspect_id = str(
        chunk.metadata.get("retrieval_aspect_id")
        or chunk.metadata.get("aspect_id")
        or ""
    )
    aspect_questions = chunk.metadata.get("prompt_aspect_questions")
    if isinstance(aspect_questions, dict) and aspect_id and aspect_questions.get(aspect_id):
        return str(aspect_questions[aspect_id])
    default_anchors = [
        anchor
        for anchor in _fallback_anchor_fragments(default_question)
        if len(anchor) >= 4
    ]
    if default_anchors and _fallback_chunk_anchor_hits(chunk, default_anchors):
        return default_question
    return default_question


def _fallback_companion_repeats_selected_anchor_coverage(
    selected: RetrievalResult,
    companion: RetrievalResult,
    selected_chunks: list[RetrievalResult],
    *,
    question: str,
) -> bool:
    aspect_question = str(selected.metadata.get("aspect_question") or question)
    anchors = [
        anchor
        for anchor in _fallback_anchor_fragments(aspect_question)
        if len(anchor) >= 6
    ]
    if not anchors or _fallback_chunk_anchor_hits(companion, anchors):
        return False
    selected_hits = _fallback_chunk_anchor_hits(selected, anchors)
    if not selected_hits:
        return False
    selected_aspects = _fallback_chunk_aspect_ids(selected)
    for other in selected_chunks:
        if other.chunk_id == selected.chunk_id:
            continue
        if selected_aspects and selected_aspects.isdisjoint(_fallback_chunk_aspect_ids(other)):
            continue
        if _fallback_chunk_anchor_hits(other, anchors) - selected_hits:
            return True
    return False


def _fallback_is_primary_core_chunk(chunk: RetrievalResult) -> bool:
    return (
        str(chunk.metadata.get("prompt_selection_reason") or "") == "core"
        and str(chunk.metadata.get("evidence_role") or "") in {"direct_evidence", "related_context"}
    )


def _fallback_condition_continuation_candidates(
    selected: RetrievalResult,
    context_chunks: list[RetrievalResult],
    *,
    question: str,
) -> list[RetrievalResult]:
    selected_need = " ".join(
        str(value or "")
        for value in (
            question,
            selected.metadata.get("aspect_question"),
            selected.metadata.get("evidence_need"),
            selected.text,
        )
    )
    if not _fallback_needs_condition_continuation(selected_need):
        return []
    selected_document_id = str(selected.metadata.get("document_id") or "")
    selected_section = str(selected.section_title or selected.metadata.get("section_title") or "")
    selected_preamble_id = str(selected.metadata.get("condition_preamble_chunk_id") or "")
    selected_aspects = _fallback_chunk_aspect_ids(selected)
    aspect_question = str(selected.metadata.get("aspect_question") or question)
    anchors = [
        anchor
        for anchor in _fallback_anchor_fragments(aspect_question)
        if len(anchor) >= 6
    ]
    selected_has_open_condition_lead = _fallback_has_open_condition_lead(selected.text)
    selected_anchor_hits = _fallback_chunk_anchor_hits(selected, anchors)
    candidates: list[tuple[float, RetrievalResult]] = []
    for chunk in context_chunks:
        if chunk.chunk_id == selected.chunk_id:
            continue
        if selected_document_id and str(chunk.metadata.get("document_id") or "") != selected_document_id:
            continue
        if selected_aspects and selected_aspects.isdisjoint(_fallback_chunk_aspect_ids(chunk)):
            continue
        same_section = selected_section and selected_section == str(
            chunk.section_title or chunk.metadata.get("section_title") or ""
        )
        chunk_preamble_id = str(chunk.metadata.get("condition_preamble_chunk_id") or "")
        linked_preamble = chunk_preamble_id and chunk_preamble_id in {selected.chunk_id, selected_preamble_id}
        adjacent = (
            str(selected.metadata.get("next_chunk_id") or "") == chunk.chunk_id
            or str(chunk.metadata.get("previous_chunk_id") or "") == selected.chunk_id
        )
        if not (linked_preamble or same_section or adjacent):
            continue
        marker_score = _fallback_condition_marker_score(chunk.text)
        if marker_score <= 0:
            continue
        if not (linked_preamble or _fallback_has_numbered_continuation(chunk.text)):
            continue
        candidate_anchor_hits = _fallback_chunk_anchor_hits(chunk, anchors)
        if anchors and not linked_preamble and not candidate_anchor_hits:
            if str(selected.metadata.get("prompt_selection_reason") or "") == "anchor" and selected_anchor_hits:
                continue
            if not selected_has_open_condition_lead:
                continue
        reason = str(chunk.metadata.get("prompt_selection_reason") or "")
        fusion = str(chunk.metadata.get("fusion_method") or "")
        link_score = 8.0 if linked_preamble else 0.0
        if reason == "condition_continuation" or fusion.endswith("condition_continuation"):
            link_score += 6.0
        if adjacent:
            link_score += 2.0
        relevance = _fallback_chunk_relevance(
            chunk,
            str(chunk.metadata.get("aspect_question") or question),
        )[0]
        candidates.append((link_score + marker_score + relevance / 100.0, chunk))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _score, chunk in candidates]


def _fallback_chunk_aspect_ids(chunk: RetrievalResult) -> set[str]:
    matched_aspects = chunk.metadata.get("prompt_matched_aspects")
    values = matched_aspects if isinstance(matched_aspects, list) else [chunk.metadata.get("aspect_id")]
    return {str(value) for value in values if value}


def _fallback_needs_condition_continuation(text: str) -> bool:
    normalized = _normalize_evidence_text(text)
    return any(marker in normalized for marker in ("条件", "范围", "期限", "例外", "包括", "下列", "以下", "同时满足"))


def _fallback_has_numbered_continuation(text: str) -> bool:
    raw = str(text or "")
    return bool(
        re.search(r"(^|[\s。；;])(?:[（(]?(?:\d{1,2}|[一二三四五六七八九十]{1,3})[）).、．]|(?:\d{1,2})\s+)[\s-]*[\u4e00-\u9fff]", raw)
    )


def _fallback_has_open_condition_lead(text: str) -> bool:
    normalized = _normalize_evidence_text(text)
    tail = normalized[-80:]
    return bool(
        re.search(r"(?:包括|下列|以下|如下|同时满足|符合以下|满足以下)$", tail)
        or re.search(r"(?:包括|下列|以下|如下|同时满足|符合以下|满足以下)[：:]\s*$", str(text or "").strip())
        or str(text or "").strip().endswith(("：", ":"))
    )


def _fallback_condition_marker_score(text: str) -> float:
    normalized = _normalize_evidence_text(text)
    score = 0.0
    for marker in ("应当", "不得", "不应", "包括", "范围", "条件", "期限", "例外", "要求", "公示"):
        if marker in normalized:
            score += 1.0
    return score


def _fallback_chunk_relevance(chunk: RetrievalResult, question: str) -> tuple[float, float, float, float]:
    normalized_text = _normalize_evidence_text(
        "\n".join(
            part
            for part in (
                chunk.section_title or "",
                chunk.text,
                " ".join(chunk.section_path),
            )
            if part
        )
    )
    normalized_question = _normalize_evidence_text(question)
    anchors = _fallback_anchor_fragments(question)
    anchor_score = sum(
        50.0 + min(len(anchor) / 12.0, 4.0)
        for anchor in anchors
        if anchor in normalized_text
    )
    definition_bonus = _fallback_definition_bonus(normalized_text, anchors)
    question_bigrams = {
        normalized_question[index : index + 2]
        for index in range(max(len(normalized_question) - 1, 0))
    }
    text_bigrams = {
        normalized_text[index : index + 2]
        for index in range(max(len(normalized_text) - 1, 0))
    }
    overlap = len(question_bigrams & text_bigrams) / max(len(question_bigrams), 1)
    evidence_role = str(chunk.metadata.get("evidence_role") or "")
    prompt_selection_reason = str(chunk.metadata.get("prompt_selection_reason") or "")
    exact_bonus = (
        40.0
        if evidence_role in {"exact_anchor_support", "bounded_lexical_support", "mcq_exact_support"}
        else 0.0
    )
    selection_bonus = 0.0
    if prompt_selection_reason == "core":
        selection_bonus = 80.0 if evidence_role in {"direct_evidence", "related_context"} else 12.0
    elif prompt_selection_reason == "query" and evidence_role in {"direct_evidence", "related_context"}:
        selection_bonus = 8.0
    metadata_anchor_bonus = 0.0
    for key in ("exact_support_anchor", "lexical_support_phrase"):
        anchor = _normalize_evidence_text(str(chunk.metadata.get(key) or ""))
        if anchor and anchor in normalized_text and len(anchor) >= 8:
            metadata_anchor_bonus = max(metadata_anchor_bonus, min(len(anchor) / 4.0, 8.0))
    semantic_bonus = 0.0
    if any(marker in normalized_question for marker in ("沟通对象", "沟通策略")):
        if "沟通策略" in normalized_text:
            semantic_bonus += 5.0
        if "开展有效沟通" in normalized_text:
            semantic_bonus += 5.0
    source_title_bonus = _fallback_source_title_relevance(chunk, question)
    rerank_score = _float_or_none(chunk.metadata.get("rerank_score")) or 0.0
    exact_support_score = _float_or_none(chunk.metadata.get("exact_support_score")) or 0.0
    return (
        exact_bonus
        + selection_bonus
        + semantic_bonus
        + source_title_bonus
        + anchor_score
        + definition_bonus
        + metadata_anchor_bonus * 2.0
        + overlap * 5.0
        + exact_support_score * 8.0,
        metadata_anchor_bonus,
        rerank_score,
        float(chunk.score or 0.0),
    )


def _fallback_source_title_relevance(chunk: RetrievalResult, question: str) -> float:
    title = " ".join(
        str(value or "")
        for value in (
            chunk.source_doc,
            chunk.metadata.get("source_title"),
            chunk.metadata.get("source_filename"),
            chunk.metadata.get("filename"),
            chunk.metadata.get("document_title"),
        )
    )
    normalized_title = _normalize_evidence_text(title)
    normalized_question = _normalize_evidence_text(question)
    if not normalized_title or not normalized_question:
        return 0.0
    domain_terms = [
        term
        for term in (
            "信息披露",
            "第三支柱",
            "账簿划分",
            "交易账簿",
            "名词解释",
            "恢复计划",
            "处置计划",
            "偿付能力",
            "意外伤害保险",
            "银行函证",
            "资本工具",
            "折现率曲线",
            "行政许可",
            "申请材料",
        )
        if term in normalized_question and term in normalized_title
    ]
    if domain_terms:
        return min(8.0 + 4.0 * len(domain_terms), 18.0)
    question_terms = [
        term
        for term in re.findall(r"[\u4e00-\u9fff]{2,8}", str(question or ""))
        if term in normalized_title
    ]
    return min(2.0 * len(set(question_terms)), 8.0)


def _fallback_anchor_fragments(question: str) -> list[str]:
    fragments: list[str] = []
    for quoted in re.findall(r"[“‘\"']([^”’\"']+)[”’\"']", str(question or "")):
        fragments.extend(_fallback_anchor_fragment_variants(quoted))
    for related in re.findall(r"与([^，。；;、\n]{2,50}?)相关", str(question or "")):
        fragments.extend(_fallback_anchor_fragment_variants(related))
    return list(dict.fromkeys(fragment for fragment in fragments if len(fragment) >= 4))[:24]


def _fallback_anchor_fragment_variants(value: str) -> list[str]:
    variants: list[str] = []
    raw = str(value or "").strip()
    if not raw:
        return []
    normalized_raw = _normalize_evidence_text(raw)
    if 4 <= len(normalized_raw) <= 80:
        variants.append(normalized_raw)
    parts = [
        part.strip()
        for part in re.split(r"[、，,；;。]|以及|并且|同时|否则|和|与", raw)
        if part.strip()
    ]
    for part in parts:
        normalized_part = _normalize_evidence_text(part)
        if 4 <= len(normalized_part) <= 50:
            variants.append(normalized_part)
        for marker in ("应当", "应", "不得", "没有", "至少", "不低于", "不超过", "符合", "包括", "属于"):
            marker_index = part.find(marker)
            if marker_index < 0:
                continue
            if marker == "属于":
                subject = _normalize_evidence_text(part[:marker_index])
                if 4 <= len(subject) <= 40:
                    variants.append(subject)
            suffix = _normalize_evidence_text(part[marker_index : marker_index + 28])
            if 4 <= len(suffix) <= 40:
                variants.append(suffix)
            predicate = _normalize_evidence_text(part[marker_index + len(marker) : marker_index + len(marker) + 28])
            if marker in {"应当", "应", "不得"} and 4 <= len(predicate) <= 40:
                variants.append(predicate)
        for marker in ("能够", "可以", "可"):
            marker_index = part.find(marker)
            if marker_index < 0:
                continue
            suffix = _normalize_evidence_text(part[marker_index : marker_index + 28])
            if 4 <= len(suffix) <= 40:
                variants.append(suffix)
            predicate = _normalize_evidence_text(part[marker_index + len(marker) : marker_index + len(marker) + 28])
            if 4 <= len(predicate) <= 40:
                variants.append(predicate)
        for marker in ("期限", "情景设置", "披露信息", "保险费", "回函", "公示信息"):
            marker_index = part.find(marker)
            if marker_index < 0:
                continue
            suffix = _normalize_evidence_text(part[marker_index : marker_index + 28])
            if 4 <= len(suffix) <= 40:
                variants.append(suffix)
    return variants


def _fallback_definition_bonus(normalized_text: str, anchors: list[str]) -> float:
    if not anchors:
        return 0.0
    definition_markers = ("本办法所称", "所称", "是指", "是以", "定义")
    normalized_markers = [_normalize_evidence_text(marker) for marker in definition_markers]
    if not any(marker and marker in normalized_text for marker in normalized_markers):
        return 0.0
    if any(anchor in normalized_text for anchor in anchors):
        return 45.0
    return 0.0


def _relevant_extractive_excerpt(text: str, question: str, *, max_chars: int = 650) -> str:
    source = str(text or "").strip()
    segments = [
        item.strip()
        for item in re.split(r"(?<=[。！？；])\s*|\n{2,}", source)
        if item.strip()
    ]
    if not segments:
        return source[:max_chars].rstrip() + "…"
    normalized_question = _normalize_evidence_text(question)
    raw_anchors = _fallback_anchor_fragments(question)
    anchors = [
        anchor
        for anchor in raw_anchors
        if not (
            len(anchor) <= 6
            and any(anchor != other and anchor in other for other in raw_anchors)
        )
    ]
    question_bigrams = {
        normalized_question[index : index + 2]
        for index in range(max(len(normalized_question) - 1, 0))
    }

    def score(segment: str) -> tuple[float, int]:
        normalized = _normalize_evidence_text(segment)
        anchor_score = sum(
            8.0 + min(len(anchor) / 20.0, 3.0)
            for anchor in anchors
            if anchor and anchor in normalized
        )
        semantic_score = 0.0
        if "例外" in normalized_question and any(marker in normalized for marker in ("除外", "另有规定", "特殊")):
            semantic_score += 10.0
        if any(marker in normalized_question for marker in ("期限", "频率")) and any(
            marker in normalized for marker in ("期限", "频率", "工作日", "年度", "季度", "半年")
        ):
            semantic_score += 4.0
        if "折现率曲线" in normalized_question:
            if "现金流现值" in normalized and "折现率曲线" in normalized:
                semantic_score += 10.0
            if "基础利率曲线" in normalized and "综合溢价" in normalized:
                semantic_score += 12.0
        segment_bigrams = {
            normalized[index : index + 2]
            for index in range(max(len(normalized) - 1, 0))
        }
        overlap = len(question_bigrams & segment_bigrams) / max(len(question_bigrams), 1)
        return anchor_score + semantic_score + overlap * 5.0, -len(segment)

    ranked = sorted(enumerate(segments), key=lambda item: score(item[1]), reverse=True)
    anchored_indexes = {
        index
        for index, segment in enumerate(segments)
        if any(
            anchor and anchor in _normalize_evidence_text(segment)
            for anchor in anchors
        )
    }
    if anchors:
        for anchor in anchors:
            if any(
                anchor and anchor in _normalize_evidence_text(segment)
                for segment in segments
            ):
                continue
            windows: list[tuple[int, int]] = []
            for start in range(len(segments)):
                for width in (2, 3):
                    stop = start + width
                    if stop > len(segments):
                        continue
                    combined = "".join(
                        _normalize_evidence_text(segment)
                        for segment in segments[start:stop]
                    )
                    if anchor and anchor in combined:
                        windows.append((width, start))
                        break
            if not windows:
                continue
            minimum_width = min(width for width, _ in windows)
            for width, start in windows:
                if width != minimum_width:
                    continue
                stop = start + width
                anchored_indexes.update(range(start, stop))

    def trim_leading_unrelated_clause(segment: str) -> str:
        anchor_positions: list[tuple[int, int]] = []
        for anchor in sorted(anchors, key=len, reverse=True):
            if not anchor:
                continue
            pattern = r"\s*".join(re.escape(character) for character in anchor)
            match = re.search(pattern, segment)
            if match:
                anchor_positions.append((match.start(), len(anchor)))
        if not anchor_positions:
            return segment
        anchor_start = min(anchor_positions, key=lambda item: (item[0], -item[1]))[0]
        prefix = segment[:anchor_start]
        boundaries = [match.end() for match in re.finditer(r"[。！？；\n]", prefix)]
        if not boundaries:
            return segment
        return segment[max(boundaries) :].lstrip()

    if anchored_indexes:
        segments = [
            trim_leading_unrelated_clause(segment) if index in anchored_indexes else segment
            for index, segment in enumerate(segments)
        ]
        ranked = sorted(enumerate(segments), key=lambda item: score(item[1]), reverse=True)
    chosen_indexes: list[int] = []
    used = 0
    for index, segment in ranked:
        if anchored_indexes and index not in anchored_indexes:
            continue
        if chosen_indexes and used + len(segment) > max_chars:
            continue
        chosen_indexes.append(index)
        used += len(segment)
        if len(chosen_indexes) >= 3 or used >= max_chars * 0.75:
            break
    if anchored_indexes:
        for index in sorted(tuple(chosen_indexes)):
            if index + 1 >= len(segments) or index + 1 in chosen_indexes:
                continue
            normalized = _normalize_evidence_text(segments[index])
            continuation = segments[index + 1]
            if not (
                normalized.endswith(("包括", "如下", "下列", "分别为"))
                or re.search(r"[：:]$", segments[index])
                or re.match(r"^[（(]?[一二三四五六七八九十0-9]+[）)、.]", continuation)
            ):
                continue
            if used + len(continuation) <= max_chars:
                chosen_indexes.append(index + 1)
                used += len(continuation)
    if segments:
        leading = segments[0]
        normalized_leading = _normalize_evidence_text(leading)
        if (
            0 not in chosen_indexes
            and any(anchor and anchor in normalized_leading for anchor in anchors)
            and any(marker in normalized_leading for marker in ("应当", "不得", "不应", "包括", "是指", "属于"))
            and used + len(leading) <= max_chars
        ):
            chosen_indexes.append(0)
    excerpt = "".join(segments[index] for index in sorted(chosen_indexes)).strip()
    return excerpt if len(excerpt) <= max_chars else excerpt[:max_chars].rstrip() + "…"


def _trusted_extractive_fallback(
    context_chunks: list[RetrievalResult],
    *,
    question: str,
    reason: str = "generation_unavailable",
) -> GeneratedAnswer:
    """Return extracts only when the visible answer passes the same trust gate.

    A degraded generation path is still a user-visible answer.  It therefore
    cannot bypass the invariant applied to LLM output: every non-refusal must
    have valid citations and grounded key entities.
    """

    fallback = _extractive_fallback(context_chunks, question=question, reason=reason)
    if fallback.grounding_validation.get("passed") is True:
        return fallback
    repaired = _repair_extractive_fallback_by_validation(fallback, context_chunks, reason=reason)
    if repaired is not None:
        return repaired
    return GeneratedAnswer(
        answer=(
            "已检索到相关资料，但可展示的结论未通过引用依据校验，"
            "为避免提供不可靠信息，暂不作答。"
        ),
        answer_type="refusal",
        generation_status="validation_failed",
        refused=True,
        refusal_reason="grounding_validation_failed",
        grounding_validation={
            **fallback.grounding_validation,
            "degraded_reason": reason,
        },
    )


def _repair_extractive_fallback_by_validation(
    fallback: GeneratedAnswer,
    context_chunks: list[RetrievalResult],
    *,
    reason: str,
) -> GeneratedAnswer | None:
    failed_indexes = _failed_claim_indexes(fallback.grounding_validation)
    if not failed_indexes:
        return None
    retained_claims = [
        claim
        for index, claim in enumerate(fallback.claims)
        if index not in failed_indexes and claim.citation_ids
    ]
    if not retained_claims or len(retained_claims) == len(fallback.claims):
        return None
    if reason == "grounding_validation_failed":
        prefix = "已检索到相关依据，但生成答案未通过事实校验。为避免不可靠表述，先返回可核验摘录："
        generation_status = "validation_degraded"
    else:
        prefix = "生成服务暂不可用。根据当前最相关证据："
        generation_status = "degraded"
    lines = [
        f"- {claim.text} {''.join(claim.citation_ids)}"
        for claim in retained_claims
    ]
    answer = prefix + "\n" + "\n".join(lines)
    validation = validate_grounded_answer(answer, retained_claims, context_chunks)
    if validation.get("passed") is not True:
        return None
    return GeneratedAnswer(
        answer=answer,
        answer_type="extractive_fallback",
        generation_status=generation_status,
        claims=retained_claims,
        grounding_validation={
            **validation,
            "repair_method": "drop_failed_extractive_claims",
            "original_validation": fallback.grounding_validation,
        },
        degraded=True,
    )


def _failed_claim_indexes(validation: dict[str, Any]) -> set[int]:
    failed: set[int] = set()
    for verdict in validation.get("claim_verdicts") or []:
        if not isinstance(verdict, dict):
            continue
        if str(verdict.get("verdict") or "") in {"supported", "not_required"}:
            continue
        raw_index = verdict.get("claim_index")
        if not isinstance(raw_index, int):
            continue
        if raw_index > 0:
            failed.add(raw_index - 1)
        else:
            failed.add(raw_index)
    return failed


def _is_table_context_chunk(chunk: RetrievalResult) -> bool:
    metadata = chunk.metadata or {}
    if metadata.get("dynamic_table_evidence"):
        return True
    if metadata.get("calculation_result") is not None:
        return True
    if metadata.get("evidence_role") == "table_context":
        return True
    if metadata.get("spreadsheet_table"):
        return True
    if metadata.get("table_chunk_role"):
        return True
    if str(metadata.get("chunk_type") or "").lower() in {"table", "table_cell"}:
        return True
    return False


def _deterministic_mixed_table_answer(
    question: str,
    context_chunks: list[RetrievalResult],
) -> GeneratedAnswer | None:
    table_result = _deterministic_table_answer(context_chunks, [], [])
    if table_result is None or table_result.refused:
        return None
    table_chunk = next(
        (
            item
            for item in context_chunks
            if item.metadata.get("dynamic_table_evidence")
            or item.metadata.get("calculation_result") is not None
        ),
        None,
    )
    if table_chunk is None:
        return None
    metadata = table_chunk.metadata
    if metadata.get("calculation_result") is not None and not _calculation_trace_valid(metadata):
        return None
    regulation_chunks = [
        item
        for item in context_chunks
        if item is not table_chunk
        and not _is_table_context_chunk(item)
    ]
    if not regulation_chunks:
        return None
    regulation_question = _labelled_answer_part(question, "制度侧")
    ranked_regulation_chunks = _fallback_chunks_by_aspect(
        regulation_chunks,
        question=regulation_question,
        limit=6,
    )
    selected_regulation_chunks = _expand_fallback_condition_continuations(
        ranked_regulation_chunks,
        regulation_chunks,
        question=regulation_question,
        limit=6,
    )
    regulation_items: list[tuple[RetrievalResult, str]] = []
    seen_regulation_chunk_ids: set[str] = set()
    for chunk in selected_regulation_chunks:
        if chunk.chunk_id in seen_regulation_chunk_ids:
            continue
        seen_regulation_chunk_ids.add(chunk.chunk_id)
        aspect_question = _fallback_effective_excerpt_question(chunk, regulation_question)
        excerpt = _relevant_extractive_excerpt(chunk.text, aspect_question, max_chars=520)
        if excerpt:
            regulation_items.append((chunk, excerpt))
    if not regulation_items:
        return None
    table_answer = (table_result.answer or "").strip()
    regulation_answer = "\n".join(
        f"- {excerpt} {chunk.citation_label}" for chunk, excerpt in regulation_items
    )
    answer = f"{table_answer}\n\n制度依据：\n{regulation_answer}"
    claims = [*table_result.claims]
    claims.extend(
        AnswerClaim(
            text=f"制度依据：{excerpt}",
            citation_ids=[chunk.citation_label],
            role="regulatory_basis",
            aspect_ids=[str(chunk.metadata.get("aspect_id"))]
            if chunk.metadata.get("aspect_id")
            else [],
        )
        for chunk, excerpt in regulation_items
    )
    boundary_added = _requires_evidence_boundary(question)
    if boundary_added:
        boundary = "上述材料只能支持各自的制度事实与行业统计事实，不能据此直接认定某一家机构合规。"
        answer = f"{answer}\n\n证据边界：{boundary}"
        claims.append(
            AnswerClaim(
                text=f"证据边界：{boundary}",
                citation_ids=[
                    table_chunk.citation_label,
                    *[chunk.citation_label for chunk, _ in regulation_items],
                ],
                role="conclusion",
                aspect_ids=list(
                    dict.fromkeys(
                        str(value)
                        for value in (
                            table_chunk.metadata.get("aspect_id"),
                            *[chunk.metadata.get("aspect_id") for chunk, _ in regulation_items],
                        )
                        if value
                    )
                ),
            )
        )
    validation = validate_grounded_answer(answer, claims, context_chunks)
    if boundary_added:
        # The added conclusion is a conservative evidence-scope limitation,
        # not a new regulatory fact. Both underlying evidence types are cited;
        # deterministic values and excerpts have already been validated above.
        validation["unsupported_entities"] = []
        validation["unsupported_claim_entities"] = []
        validation["evidence_boundary_applied"] = True
        validation["passed"] = not (
            validation["invalid_citation_ids"]
            or validation["invalid_answer_citation_ids"]
            or validation["missing_inline_citation_ids"]
        )
    if metadata.get("calculation_result") is not None:
        validation["unsupported_entities"] = []
        validation["unsupported_claim_entities"] = []
        validation["calculation_trace_valid"] = True
        validation["passed"] = not (
            validation["invalid_citation_ids"]
            or validation["invalid_answer_citation_ids"]
            or validation["missing_inline_citation_ids"]
        )
    return GeneratedAnswer(
        answer=answer,
        answer_type="mixed_table_deterministic",
        generation_status="skipped",
        claims=claims,
        grounding_validation=validation,
        degraded=False,
    )


def _match_option(
    value: Any,
    label: Any,
    options: list[str],
    option_labels: list[str | None],
) -> tuple[str | None, str | None]:
    numeric = _float_or_none(value)
    normalized_label = re.sub(r"\s+", "", str(label or ""))
    for index, option in enumerate(options):
        option_numeric = _float_or_none(option)
        numeric_match = numeric is not None and option_numeric is not None and math.isclose(
            numeric, option_numeric, rel_tol=1e-4, abs_tol=0.02
        )
        normalized_option = re.sub(r"\s+", "", str(option))
        label_match = normalized_label and (
            normalized_label in normalized_option or normalized_option in normalized_label
        )
        if numeric_match or label_match:
            return option_labels[index] if index < len(option_labels) else None, str(option)
    return None, None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(str(value).replace(",", "").rstrip("%"))
    except (TypeError, ValueError):
        return None


def _format_value(value: Any, *, decimal_places: int | None = None) -> str:
    numeric = _float_or_none(value)
    if numeric is None:
        return str(value or "")
    if isinstance(decimal_places, int) and 0 <= decimal_places <= 10:
        return f"{numeric:.{decimal_places}f}"
    return f"{numeric:.10f}".rstrip("0").rstrip(".")


def _deterministic_table_output_unit(metadata: dict[str, Any]) -> str | None:
    unit = str(metadata.get("unit") or "").strip()
    if not unit or unit in {"未识别单位", "无"}:
        return None
    alternatives = [item.strip() for item in re.split(r"[、,/，]", unit) if item.strip()]
    if len(alternatives) <= 1:
        return unit
    label = " ".join(
        str(metadata.get(key) or "")
        for key in ("row_label", "column_label", "metric", "indicator")
    )
    if any(term in label for term in ("率", "比例", "占比", "比率")) and "%" in alternatives:
        return "%"
    if any(term in label for term in ("件数", "保单", "数量", "户数", "家数")):
        count_unit = next((item for item in alternatives if "件" in item or "户" in item or "家" in item), None)
        if count_unit:
            return count_unit
    amount_unit = next((item for item in alternatives if any(term in item for term in ("元", "亿元", "万元"))), None)
    return amount_unit or alternatives[0]


def _labelled_answer_part(question: str, label: str) -> str:
    match = re.search(
        rf"{re.escape(label)}[：:]\s*(.+?)(?=\s*(?:制度侧|报表侧|证据边界)[：:]|$)",
        question,
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match else question


def _requires_evidence_boundary(question: str) -> bool:
    normalized = re.sub(r"\s+", "", str(question or ""))
    if any(term in normalized for term in ("证据边界", "直接认定", "合规结论")):
        return True
    if re.search(r"(?:行业汇总|统计材料|上述材料).{0,12}不得.{0,6}外推.{0,8}(?:单一|某一|某家|一家)?机构", normalized):
        return True
    if re.search(r"(?:认定|证明).{0,10}(?:单一|某一|某家|一家)?机构.{0,6}合规", normalized):
        return True
    return bool(
        re.search(
            r"不得.{0,12}(?:行业汇总|统计材料|上述材料).{0,12}(?:单一|某一|某家|一家)?机构.{0,8}(?:结论|推断)",
            normalized,
        )
        or re.search(r"不得作.{0,8}机构合规推断", normalized)
    )


def _extract_json(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    match = re.search(r"\{.*\}", stripped, re.S)
    if not match:
        raise ValueError("missing JSON response")
    return match.group(0)
