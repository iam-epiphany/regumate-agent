from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
import urllib.error
import urllib.request
from typing import Any, Callable

from backend.app.core.config import (
    ANSWER_GENERATION_API_KEY,
    ANSWER_GENERATION_BASE_URL,
    ANSWER_GENERATION_ENABLED,
    ANSWER_GENERATION_INCLUDE_THINKING,
    ANSWER_GENERATION_MAX_TOKENS,
    ANSWER_GENERATION_MODEL,
    ANSWER_GENERATION_PROVIDER,
    ANSWER_GENERATION_RESPONSE_FORMAT,
    ANSWER_GENERATION_STREAM,
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
    if is_option_selection_question(question) and not normalized_options:
        return GeneratedAnswer(
            answer=(
                "这个问题写法属于选择题，但没有提供 A-D 选项，因此无法判断“哪一组选项”正确。"
                "请补充选项，或改成普通问法，例如“请概述《账簿划分和名词解释》的主要内容”"
                "或“交易账簿和银行账簿如何划分？”。"
            ),
            answer_type="clarification",
            generation_status="skipped",
            refused=True,
            refusal_reason="missing_options_for_choice_question",
            grounding_validation={"passed": True, "reason": "missing_options_for_choice_question"},
        )

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
    else:
        deterministic = _deterministic_table_answer(context_chunks, normalized_options, normalized_labels)
        if deterministic is not None:
            _report_verified_claims(deterministic, verified_claim_reporter)
            return deterministic

    formula_answer = _deterministic_formula_answer(question, context_chunks)
    if formula_answer is not None:
        _report_verified_claims(formula_answer, verified_claim_reporter)
        return formula_answer

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
                deterministic_mixed = _deterministic_mixed_table_answer(context_chunks)
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
            )
            repaired.grounding_validation = {**repaired_validation, "repair_attempted": True}
            if repaired_validation["passed"]:
                return repaired
            if not normalized_options:
                fallback = _trusted_extractive_fallback(
                    context_chunks,
                    reason="grounding_validation_failed",
                )
                fallback.grounding_validation = {
                    **fallback.grounding_validation,
                    "repair_attempted": True,
                    "llm_validation": repaired_validation,
                    "degraded_reason": "grounding_validation_failed",
                }
                return fallback
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
        deterministic_mixed = _deterministic_mixed_table_answer(context_chunks)
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

    fallback = _trusted_extractive_fallback(context_chunks)
    _report_verified_claims(fallback, verified_claim_reporter)
    return fallback


def is_option_selection_question(question: str) -> bool:
    normalized = re.sub(r"\s+", "", str(question or ""))
    if not normalized:
        return False
    option_markers = ("选项", "哪一项", "哪项", "哪一组", "哪组", "下列", "以下")
    selection_markers = ("正确", "不正确", "属于", "均属于", "表述", "说法")
    return any(marker in normalized for marker in option_markers) and any(
        marker in normalized for marker in selection_markers
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
    value_text = _format_value(raw_value)
    if option_text:
        answer = f"答案为 {_format_selected_option(option_text, option_label)}。依据表格证据，核验值为 {value_text}。[1]"
    elif metadata.get("calculation_formula"):
        display_formula = str(metadata.get("calculation_display_formula") or metadata["calculation_formula"])
        comparison_target = metadata.get("comparison_target") if isinstance(metadata.get("comparison_target"), dict) else {}
        if metadata.get("comparison_equal") is True and _asks_equality(metadata):
            target_label = comparison_target.get("label") or "目标值"
            answer = f"相等，计算结果为 {value_text}。计算式：{display_formula}，与{target_label}一致。[1]"
        elif metadata.get("comparison_equal") is False and _asks_equality(metadata):
            target_label = comparison_target.get("label") or "目标值"
            target_value = _format_value(comparison_target.get("value") or comparison_target.get("normalized_value"))
            answer = f"不相等，计算结果为 {value_text}。计算式：{display_formula}，与{target_label} {target_value} 不一致。[1]"
        else:
            answer = f"计算结果为 {value_text}。计算式：{display_formula}。[1]"
    else:
        answer = f"查询结果为 {value_text}。[1]"
    claim = AnswerClaim(
        text=answer.rsplit("[1]", 1)[0].strip(),
        citation_ids=["[1]"],
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


def _deterministic_formula_answer(
    question: str,
    context_chunks: list[RetrievalResult],
) -> GeneratedAnswer | None:
    calculation = calculate_formula_answer(question, context_chunks)
    if calculation is None:
        return None
    formula_chunk = next((chunk for chunk in context_chunks if chunk.metadata.get("contains_formula")), None)
    if calculation.answer is not None and formula_chunk is not None:
        claim = AnswerClaim(
            text=calculation.answer.rsplit("[1]", 1)[0].strip(),
            citation_ids=["[1]"],
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
    citation_suffix = " [1]" if formula_chunk is not None else ""
    answer = f"无法计算。{calculation.refusal_reason or '当前公式证据不足。'}{citation_suffix}"
    claim_ids = ["[1]"] if formula_chunk is not None else []
    claims = [
        AnswerClaim(
            text=answer.replace(" [1]", "").strip(),
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
    try:
        with measure("answer_generation.external_api"):
            with open_chat_completion(
                ChatCompletionConfig(
                    provider=ANSWER_GENERATION_PROVIDER,
                    api_key=ANSWER_GENERATION_API_KEY,
                    base_url=ANSWER_GENERATION_BASE_URL,
                    model=ANSWER_GENERATION_MODEL,
                    timeout_seconds=ANSWER_GENERATION_TIMEOUT_SECONDS,
                    include_thinking=ANSWER_GENERATION_INCLUDE_THINKING,
                    response_format=ANSWER_GENERATION_RESPONSE_FORMAT,
                ),
                messages,
                temperature=0,
                max_tokens=ANSWER_GENERATION_MAX_TOKENS,
                response_format=ANSWER_GENERATION_RESPONSE_FORMAT,
                stream=ANSWER_GENERATION_STREAM,
                opener=urllib.request.urlopen,
            ) as response:
                content = _read_streaming_llm_content(
                    response,
                    question=question,
                    context_chunks=context_chunks,
                    options=options,
                    option_labels=option_labels,
                    verified_claim_reporter=verified_claim_reporter,
                    cancellation_checker=cancellation_checker,
                )
    except (urllib.error.URLError, ChatCompletionError) as exc:
        raise OSError("answer generation unavailable") from exc
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


def _read_streaming_llm_content(
    response: Any,
    *,
    question: str,
    context_chunks: list[RetrievalResult],
    options: list[str],
    option_labels: list[str | None],
    verified_claim_reporter: VerifiedClaimReporter | None,
    cancellation_checker: CancellationChecker | None,
) -> str:
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

    for raw_line in iterator:
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
        if verified_claim_reporter is None or _partial_refused(content) is not False:
            continue
        for role, claim in _extract_partial_claims(content):
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
) -> dict[str, Any]:
    validation = validate_grounded_answer(generated.answer or "", generated.claims, context_chunks)
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
    return {
        **validation,
        **option_validation,
        "passed": bool(
            validation["passed"]
            and option_validation["selected_option_supported"]
            and option_validation["selected_option_format_valid"]
        ),
    }


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
    elif operation in {"ratio", "difference"} and len(values) == 2:
        operator = "/" if operation == "ratio" else "-"
        if operator not in formula_compact:
            return False
        positions = [formula_compact.find(reference) for reference in references]
        if positions[0] == positions[1]:
            return False
        left, right = (0, 1) if positions[0] < positions[1] else (1, 0)
        if operation == "ratio":
            if values[right] == 0:
                return False
            recomputed = values[left] / values[right]
        else:
            recomputed = values[left] - values[right]
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
    if best["minimum_fact_coverage"] < 0.30 or (minimum_gap < 0.12 and average_gap < 0.18):
        return None

    if any(
        not fact.get("citation_id") or fact["coverage"] < 0.30 for fact in best["facts"]
    ):
        return None
    citation_ids = list(dict.fromkeys(fact["citation_id"] for fact in best["facts"]))
    option_id = best["option_id"]
    option_label = best["label"]
    option_text = best["text"]
    citation_suffix = "".join(citation_ids)
    selected_text = _strip_sentence_end(_format_selected_option(option_text, option_label))
    answer = f"答案为 {selected_text}。{citation_suffix}"
    claim = AnswerClaim(text=f"答案为 {selected_text}", citation_ids=citation_ids)
    grounding = validate_grounded_answer(answer, [claim], context_chunks)
    if not grounding["passed"]:
        return None
    return GeneratedAnswer(
        answer=answer,
        answer_type="choice_evidence_deterministic",
        generation_status="completed",
        claims=[claim],
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


def _option_evidence_matrix(
    options: list[str],
    option_labels: list[str | None],
    context_chunks: list[RetrievalResult],
) -> list[dict[str, Any]]:
    matrix: list[dict[str, Any]] = []
    for option_index, option in enumerate(options):
        facts = [part.strip() for part in re.split(r"[；;]", str(option)) if part.strip()]
        fact_scores: list[dict[str, Any]] = []
        for fact in facts or [str(option)]:
            scored_chunks = [
                (_character_bigram_recall(fact, chunk.text), chunk.citation_label)
                for chunk in context_chunks
            ]
            entity_grounded_chunks = [
                item
                for item, chunk in zip(scored_chunks, context_chunks, strict=False)
                if _fact_entities_are_grounded(fact, chunk)
            ]
            # Character overlap alone can bind a percentage or date to a
            # nearby paragraph that uses the same regulatory wording but does
            # not contain the claimed value.  Prefer candidates that pass the
            # same entity-to-citation validation used for final answers.
            if entity_grounded_chunks:
                scored_chunks = entity_grounded_chunks
            coverage, citation_id = max(scored_chunks, default=(0.0, None), key=lambda item: item[0])
            fact_scores.append(
                {
                    "fact": fact,
                    "coverage": round(coverage, 4),
                    "citation_id": citation_id if coverage > 0 else None,
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
    citation_id = chunk.citation_label
    validation = validate_grounded_answer(
        f"{fact} {citation_id}",
        [AnswerClaim(text=fact, citation_ids=[citation_id])],
        [chunk],
    )
    return bool(validation["passed"])


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
    return [str(label).strip() if label else None for label in labels[:option_count]]


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
    reason: str = "generation_unavailable",
) -> GeneratedAnswer:
    selected_chunks = context_chunks[:3]
    claims: list[AnswerClaim] = []
    lines: list[str] = []
    for chunk in selected_chunks:
        excerpt = chunk.text.strip()
        if len(excerpt) > 260:
            excerpt = excerpt[:260].rsplit("。", 1)[0] + "。"
        claims.append(AnswerClaim(text=excerpt, citation_ids=[chunk.citation_label]))
        lines.append(f"- {excerpt} {chunk.citation_label}")
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


def _trusted_extractive_fallback(
    context_chunks: list[RetrievalResult],
    *,
    reason: str = "generation_unavailable",
) -> GeneratedAnswer:
    """Return extracts only when the visible answer passes the same trust gate.

    A degraded generation path is still a user-visible answer.  It therefore
    cannot bypass the invariant applied to LLM output: every non-refusal must
    have valid citations and grounded key entities.
    """

    fallback = _extractive_fallback(context_chunks, reason=reason)
    if fallback.grounding_validation.get("passed") is True:
        return fallback
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


def _deterministic_mixed_table_answer(context_chunks: list[RetrievalResult]) -> GeneratedAnswer | None:
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
    regulation_chunk = next(
        (
            item
            for item in context_chunks
            if item is not table_chunk
            and not item.metadata.get("dynamic_table_evidence")
            and item.metadata.get("evidence_role") != "table_context"
        ),
        None,
    )
    if regulation_chunk is None:
        return None
    regulation_excerpt = regulation_chunk.text.strip()
    if len(regulation_excerpt) > 220:
        regulation_excerpt = regulation_excerpt[:220].rsplit("。", 1)[0] + "。"
    table_answer = (table_result.answer or "").strip()
    answer = f"{table_answer}\n\n制度依据：{regulation_excerpt} {regulation_chunk.citation_label}"
    claims = [
        *table_result.claims,
        AnswerClaim(
            text=f"制度依据：{regulation_excerpt}",
            citation_ids=[regulation_chunk.citation_label],
            role="regulatory_basis",
            aspect_ids=[str(regulation_chunk.metadata.get("aspect_id"))]
            if regulation_chunk.metadata.get("aspect_id")
            else [],
        ),
    ]
    validation = validate_grounded_answer(answer, claims, context_chunks)
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


def _format_value(value: Any) -> str:
    numeric = _float_or_none(value)
    if numeric is None:
        return str(value or "")
    return f"{numeric:.10f}".rstrip("0").rstrip(".")


def _extract_json(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    match = re.search(r"\{.*\}", stripped, re.S)
    if not match:
        raise ValueError("missing JSON response")
    return match.group(0)
