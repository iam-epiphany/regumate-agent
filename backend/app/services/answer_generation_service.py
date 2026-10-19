from __future__ import annotations
from backend.app.core.rag_features import feature_gate

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
    ANSWER_GENERATION_DISABLE_THINKING,
    ANSWER_GENERATION_MAX_TOKENS,
    ANSWER_GENERATION_MODEL,
    ANSWER_GENERATION_PROVIDER,
    ANSWER_GENERATION_RESPONSE_FORMAT,
    QA_REQUEST_TOTAL_BUDGET_SECONDS,
)
from backend.app.schemas.qa import AnswerClaim, EvidenceRequirement, RetrievalResult
from backend.app.services.grounding_validation_service import validate_grounded_answer
from backend.app.services.formula_parser_service import (
    calculate_formula_answer,
    formula_refusal_grounding,
)
from backend.app.services.json_utils import extract_json as _extract_json
from backend.app.services.prompt_builder import RAGPromptBuilder, format_prompt_option
from backend.app.services.llm_client import (
    ChatCompletionConfig,
    ChatCompletionError,
)
from backend.app.services.performance_metrics import qa_execution, RequestBudgetExceeded, timed


VerifiedClaimReporter = Callable[[AnswerClaim], None]
CancellationChecker = Callable[[], None]


class GenerationParseError(ValueError):
    """A structured-generation failure with a stable diagnostic category."""

    def __init__(self, message: str, *, kind: str, raw_length: int | None = None,
                 finish_reason: str | None = None):
        super().__init__(message)
        self.kind = kind
        self.raw_length = raw_length
        self.finish_reason = finish_reason


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


@dataclass
class GenerationContext:
    """Everything the generation loop needs, prepared once per answer."""

    question: str
    context_chunks: list[RetrievalResult]
    options: list[str]
    option_labels: list[str | None]
    llm_prompt: str | None
    answer_mode: str
    required: list[str]
    requirements: list[EvidenceRequirement] | None
    table_findings: list[TableFinding]
    deterministic: GeneratedAnswer | None
    cancellation_checker: Callable | None


@qa_execution
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
    requirements: list[EvidenceRequirement] | None = None,
) -> GeneratedAnswer:
    normalized_options = [option for option in options or [] if str(option or "").strip()]
    normalized_labels = _normalize_option_labels(option_labels, len(normalized_options))
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
        requirements=requirements,
    )


def refusal_answer(reason, *, validation=None, unavailable=False) -> GeneratedAnswer:
    """The single refusal shape used by pre-checks and the generation loop."""

    return GeneratedAnswer(
        answer=('问答服务暂不可用，请稍后重试。' if unavailable else
                '当前证据不足以完整回答问题。' + ('缺少：' + '、'.join((validation or {}).get('missing_requirements', []))
                if (validation or {}).get('missing_requirements') else '请补充相关材料或明确查询范围。')),
        answer_type='service_error' if unavailable else 'refusal',
        generation_status='error' if unavailable else 'skipped', refused=True,
        refusal_reason=reason, degraded=unavailable,
        grounding_validation={**(validation or {}), 'passed': False,
            'status': 'error' if unavailable else (validation or {}).get('status', 'not_run'),
            'task_complete': False,
            'failure_stage': (validation or {}).get('failure_stage') or ('generation' if unavailable else 'evidence')})


def _default_table_resolver(kind: str, **kwargs: Any) -> GeneratedAnswer | None:
    """Resolve deterministic candidates through the module-level seams."""

    if kind == "table":
        return _deterministic_table_answer(
            kwargs["context_chunks"], kwargs["options"], kwargs["option_labels"])
    if kind == "formula":
        return _deterministic_formula_answer(kwargs["question"], kwargs["context_chunks"])
    return None


def prepare_generation(
    question: str,
    context_chunks: list[RetrievalResult],
    normalized_options: list[str],
    normalized_labels: list[str | None],
    *,
    llm_prompt: str | None,
    has_sufficient_context: bool,
    cancellation_checker: CancellationChecker | None,
    answer_mode: str,
    required_aspect_ids: list[str] | None,
    requirements: list[EvidenceRequirement] | None,
    table_resolver: Callable[..., GeneratedAnswer | None] | None = None,
) -> GenerationContext | GeneratedAnswer:
    """Run every pre-generation gate and resolve the deterministic candidate."""

    if requirements is not None:
        requirements = [r for r in requirements if bool(getattr(r, 'mandatory', True))]
        required = [r.requirement_id for r in requirements]
    else:
        required = required_aspect_ids or []
    if not context_chunks or not has_sufficient_context:
        return refusal_answer('insufficient_context', validation={'missing_requirements': required})
    missing = _missing_aspects_in_context(context_chunks, required)
    if missing:
        return refusal_answer('missing_required_aspect', validation={'missing_requirements': missing})
    table_findings = _table_findings(context_chunks)
    if answer_mode in {'mixed', 'scenario'}:
        missing_modalities = []
        if not table_findings:
            missing_modalities.append('表格事实')
        if not any(not c.metadata.get('dynamic_table_evidence') and c.metadata.get('evidence_role') != 'table_context'
                   for c in context_chunks):
            missing_modalities.append('制度依据')
        if missing_modalities:
            return refusal_answer('missing_required_aspect', validation={'missing_requirements': missing_modalities})
    resolver = table_resolver or _default_table_resolver
    deterministic = None
    if answer_mode not in {'mixed', 'scenario'}:
        deterministic = resolver(
            "table",
            context_chunks=context_chunks,
            options=normalized_options,
            option_labels=normalized_labels,
        )
    if deterministic is None and _is_formula_request(question):
        deterministic = resolver(
            "formula",
            question=question,
            context_chunks=context_chunks,
        )
    if deterministic is not None and deterministic.refused:
        deterministic.grounding_validation.update(status='not_run', passed=False, task_complete=False)
        return deterministic
    return GenerationContext(
        question=question,
        context_chunks=context_chunks,
        options=normalized_options,
        option_labels=normalized_labels,
        llm_prompt=llm_prompt,
        answer_mode=answer_mode,
        required=required,
        requirements=requirements,
        table_findings=table_findings,
        deterministic=deterministic,
        cancellation_checker=cancellation_checker,
    )


def draft_answer(ctx: GenerationContext, *, attempt: int, correction: str | None) -> GeneratedAnswer:
    """Return the deterministic first draft or ask the model for one."""

    if attempt == 0 and ctx.deterministic is not None:
        return ctx.deterministic
    if not ANSWER_GENERATION_ENABLED or not ANSWER_GENERATION_API_KEY:
        return refusal_answer('generation_unavailable', unavailable=True)
    return _call_llm(
        ctx.question, ctx.context_chunks, ctx.options, ctx.option_labels,
        llm_prompt=ctx.llm_prompt, correction=correction, verified_claim_reporter=None,
        cancellation_checker=ctx.cancellation_checker, answer_mode=ctx.answer_mode,
        table_findings=ctx.table_findings, required_aspect_ids=ctx.required)


def verify_draft(ctx: GenerationContext, generated: GeneratedAnswer) -> dict[str, Any]:
    """Validate one draft through the shared answer validation service."""

    from backend.app.services import answer_validation_service as validation_service

    return validation_service.validate_answer(
        ctx.question, generated.answer or '', generated.claims, ctx.context_chunks,
        requirements=ctx.required,
        deterministic=generated.answer_type in {'table_deterministic', 'formula_deterministic'},
        options=ctx.options, option_labels=ctx.option_labels,
        computation=generated.answer_type == 'formula_deterministic',
        requirement_details=ctx.requirements)


def _repair_correction(ctx: GenerationContext, generated: GeneratedAnswer,
                       validation: dict[str, Any]) -> str:
    cited = {ref for claim in generated.claims for ref in claim.citation_ids}
    evidence = []
    for chunk in ctx.context_chunks:
        if cited and chunk.citation_label not in cited:
            continue
        evidence.append({
            'citation_id': chunk.citation_label,
            'source': chunk.source_doc,
            'text': (chunk.text or '')[:1200],
        })
        if len(evidence) >= 6:
            break
    payload = {
        'validation': validation,
        'draft': {
            'answer': (generated.answer or '')[:4000],
            'claims': [claim.model_dump() for claim in generated.claims],
        },
        'required_requirements': [
            {
                'requirement_id': requirement.requirement_id,
                'question': requirement.question,
                'source_titles': requirement.source_titles,
            }
            for requirement in ctx.requirements or []
        ],
        'evidence': evidence,
    }
    return json.dumps(payload, ensure_ascii=False)


def judge_draft(
    ctx: GenerationContext,
    generated: GeneratedAnswer,
    validation: dict[str, Any],
    attempt: int,
) -> tuple[str, Any]:
    """Map one validation report to accept, refuse or a single repair."""

    if validation['passed']:
        return "accept", None
    if generated.answer_type in {'table_deterministic', 'formula_deterministic'}:
        return "refuse", refusal_answer('grounding_validation_failed', validation=validation)
    if validation['status'] == 'not_run':
        return "refuse", refusal_answer('validation_not_run', validation=validation)
    if validation['status'] == 'error':
        reason = str(validation.get('validation_error_code') or 'validation_service_unavailable')
        return "refuse", refusal_answer(reason, validation=validation, unavailable=True)
    return "repair", _repair_correction(ctx, generated, validation)


def _exact_support_repair_prompt(ctx: GenerationContext) -> str | None:
    """A single re-draft hint for a refusal contradicted by exact support.

    Only literal option statements already found by retrieval are quoted; no
    answer, option label or gold information is injected.  The model remains
    free to refuse again if the support is not substantive.
    """
    if not ctx.options or not ctx.option_labels:
        return None
    statements: list[str] = []
    for chunk in ctx.context_chunks:
        metadata = chunk.metadata or {}
        if metadata.get("evidence_role") != "mcq_exact_support":
            continue
        statement = str(metadata.get("exact_support_statement") or "").strip()
        if statement and statement not in statements:
            statements.append(statement)
    if not statements:
        return None
    quoted = "；".join(f"“{statement}”" for statement in statements[:3])
    return (
        "检索到材料中与候选项直接匹配的原文："
        + quoted
        + "。这是单源选择题：若某选项的核心命题与该原文实质一致，请选择该选项；"
        "其他选项在该材料中不存在可作为不属于该材料的依据，不要仅因其他选项未出现而拒答。"
        "若仍无直接支持，可保持拒答。"
    )


def _normalize_support_text(value: Any) -> str:
    return re.sub(r"[\s\"'“”‘’=：:；;，,。()（）]+", "", str(value or "")).lower()


def _support_recall(expected: str, actual: str) -> float:
    expected_text = _normalize_support_text(expected)
    actual_text = _normalize_support_text(actual)
    if not expected_text:
        return 0.0
    if expected_text in actual_text:
        return 1.0
    if len(expected_text) == 1:
        return 1.0 if expected_text in actual_text else 0.0
    bigrams = [expected_text[index:index + 2] for index in range(len(expected_text) - 1)]
    return sum(bigram in actual_text for bigram in bigrams) / len(bigrams)


def _support_chunks(ctx: GenerationContext) -> list[RetrievalResult]:
    return [
        chunk for chunk in ctx.context_chunks
        if (chunk.metadata or {}).get("evidence_role") == "mcq_exact_support"
    ]


def _unique_supported_option(ctx: GenerationContext) -> int | None:
    """Return the only option with unambiguous exact support, else None.

    This is a last-resort, retrieval-grounded decision for a persistent model
    refusal.  It never uses gold data and never overrides an answered draft.
    """
    if not ctx.options or not ctx.option_labels:
        return None
    support = _support_chunks(ctx)
    if not support:
        return None
    sources = {
        str((chunk.metadata or {}).get("document_id") or chunk.source_doc or "")
        for chunk in ctx.context_chunks
    }
    if len(sources) != 1:
        return None
    statement_matched: list[int] = []
    recalls: list[float] = []
    for option in ctx.options:
        option_norm = _normalize_support_text(option)
        matched = False
        best_recall = 0.0
        for chunk in support:
            metadata = chunk.metadata or {}
            statement = _normalize_support_text(metadata.get("exact_support_statement") or "")
            if len(statement) >= 4 and (statement in option_norm or option_norm in statement):
                matched = True
            best_recall = max(best_recall, _support_recall(option, chunk.text or ""))
        if matched:
            statement_matched.append(len(recalls))
        recalls.append(best_recall)
    if len(statement_matched) == 1:
        index = statement_matched[0]
        return index if recalls[index] >= 0.25 else None
    if len(statement_matched) > 1:
        best = max(statement_matched, key=lambda item: recalls[item])
        others = [recalls[item] for item in statement_matched if item != best]
        second = max(others, default=0.0)
        if recalls[best] >= 0.35 and second <= recalls[best] * 0.8:
            return best
    return None


def _best_support_sentence(text: str, option: str) -> str:
    parts = [part.strip() for part in re.split(r"[。\n；;]+", str(text or "")) if part.strip()]
    if not parts:
        return ""
    best = max(parts, key=lambda part: _support_recall(option, part))
    return best if _support_recall(option, best) >= 0.3 else ""


def _deterministic_mcq_fallback(ctx: GenerationContext) -> GeneratedAnswer | None:
    """Answer a persistent refusal when exactly one option has exact support."""
    index = _unique_supported_option(ctx)
    if index is None:
        return None
    label = ctx.option_labels[index] if index < len(ctx.option_labels) else None
    if not label:
        return None
    option = str(ctx.options[index])
    support = _support_chunks(ctx)
    chunk = max(support, key=lambda item: _support_recall(option, item.text or ""))
    citation = chunk.citation_label
    sentence = _best_support_sentence(chunk.text or "", option)
    evidence = f"依据{citation}原文：“{sentence[:220]}”。" if sentence else f"依据{citation}。"
    answer = f"答案为 {label}。{option}{'' if option.endswith(('。', '；', ';')) else '。'}{evidence}"
    metadata = chunk.metadata or {}
    claim = AnswerClaim(
        text=answer,
        citation_ids=[citation],
        role="conclusion",
        aspect_ids=[str(metadata.get("aspect_id"))] if metadata.get("aspect_id") else [],
    )
    return GeneratedAnswer(
        answer=answer,
        answer_type="llm_grounded",
        generation_status="completed",
        claims=[claim],
        refused=False,
        grounding_validation={
            "passed": True,
            "status": "passed",
            "task_complete": True,
            "facts_supported": True,
            "method": "deterministic_mcq_support",
            "selected_option": index,
            "selected_option_supported": True,
            "selected_option_format_valid": True,
        },
    )


def run_generation_loop(
    ctx: GenerationContext,
    *,
    draft: Callable[..., GeneratedAnswer],
    verify: Callable[[GenerationContext, GeneratedAnswer], dict[str, Any]],
    judge: Callable[[GenerationContext, GeneratedAnswer, dict[str, Any], int], tuple[str, Any]] = judge_draft,
) -> GeneratedAnswer:
    """The unchanged two-attempt draft/verify/repair control flow."""

    correction = None
    format_repair_attempted = False
    semantic_repair_attempted = False
    for attempt in range(2):
        if ctx.cancellation_checker:
            ctx.cancellation_checker()
        try:
            if not (attempt == 0 and ctx.deterministic is not None) and (
                    not ANSWER_GENERATION_ENABLED or not ANSWER_GENERATION_API_KEY):
                # Same position as the original loop: generation availability is
                # checked before any model draft, so every refused draft below
                # takes the single model-refusal branch regardless of its flags.
                return refusal_answer('generation_unavailable', unavailable=True)
            generated = draft(ctx, attempt=attempt, correction=correction)
            if generated.refused:
                support_correction = _exact_support_repair_prompt(ctx) if attempt == 0 else None
                if support_correction:
                    # A refusal is not a final answer when retrieval already
                    # contains a literal option statement.  Spend the existing
                    # one-repair budget asking the model to reconsider, instead
                    # of treating absence of other options as an evidence gap.
                    correction = support_correction
                    continue
                fallback = _deterministic_mcq_fallback(ctx)
                if fallback is not None:
                    # The model still refused after the repair, but retrieval
                    # has one unambiguous exact-support option.  Answer from
                    # that evidence rather than returning an unsupported refusal.
                    return fallback
                generated.answer = generated.answer or generated.refusal_reason or '当前证据不足，无法完整回答。'
                generated.refusal_reason = generated.refusal_reason or 'model_refusal'
                generated.grounding_validation = {'passed': False, 'status': 'not_run', 'task_complete': False,
                                                 'reason': 'model_refusal', 'failure_stage': 'generation'}
                return generated
            validation = verify(ctx, generated)
            generated.grounding_validation = {
                **generated.grounding_validation, **validation,
                'repair_attempted': bool(attempt),
                'format_repair_attempted': format_repair_attempted,
                'semantic_repair_attempted': semantic_repair_attempted,
            }
            decision, payload = judge(ctx, generated, validation, attempt)
            if decision == "accept":
                return generated
            if decision == "refuse":
                return payload
            semantic_repair_attempted = True
            correction = payload
        except RequestBudgetExceeded:
            raise
        except GenerationParseError as exc:
            diagnostic = {
                'status': 'failed', 'failure_stage': 'generation',
                'error_type': exc.kind, 'raw_length': exc.raw_length,
                'finish_reason': exc.finish_reason,
                'format_repair_attempted': True,
            }
            if attempt == 0:
                format_repair_attempted = True
                correction = (
                    '上次输出不是有效的结构化JSON（诊断：'
                    + exc.kind
                    + '）。请严格输出包含 refused、refusal_reason、claims 的JSON，'
                      '每条claim填写text、citation_ids、aspect_ids。'
                )
                continue
            return refusal_answer('generation_format_failure', unavailable=True, validation=diagnostic)
        except ChatCompletionError as exc:
            diagnostic = {
                'status': 'error', 'failure_stage': 'generation',
                'error_type': f'chat_{exc.kind}',
                'finish_reason': exc.finish_reason,
            }
            if exc.kind in {'transport', 'timeout'} and attempt == 0:
                correction = '上次模型调用发生临时传输故障，请重新输出结构化JSON。'
                continue
            if exc.kind == 'truncated' and attempt == 0:
                format_repair_attempted = True
                correction = (
                    '上次输出在JSON结束前被截断。请压缩claims，只保留结论和必要依据，'
                    '不要重复原文，确保输出是完整、可解析的JSON。'
                )
                continue
            reason = (
                'generation_truncated' if exc.kind == 'truncated'
                else 'generation_format_failure' if exc.kind in {'response_json', 'response_schema'}
                else 'generation_unavailable'
            )
            return refusal_answer(reason, unavailable=True, validation=diagnostic)
        except OSError as exc:
            return refusal_answer('generation_unavailable', unavailable=True, validation={
                'status': 'error', 'failure_stage': 'generation',
                'error_type': type(exc).__name__,
            })
        except (ValueError, KeyError, TypeError) as exc:
            diagnostic = {
                'status': 'failed', 'failure_stage': 'generation',
                'error_type': type(exc).__name__,
                'format_repair_attempted': True,
            }
            if attempt == 0:
                format_repair_attempted = True
                correction = (
                    '输出不是有效的结构化答案。请按要求重新输出JSON，并填写引用与claims。'
                )
                continue
            return refusal_answer('generation_format_failure', unavailable=True, validation=diagnostic)
    return refusal_answer('grounding_validation_failed', validation=generated.grounding_validation)


def _generate_answer_impl(question, context_chunks, normalized_options, normalized_labels, *,
                         llm_prompt, has_sufficient_context, verified_claim_reporter,
                         cancellation_checker, answer_mode, required_aspect_ids, requirements):
    prepared = prepare_generation(
        question,
        context_chunks,
        normalized_options,
        normalized_labels,
        llm_prompt=llm_prompt,
        has_sufficient_context=has_sufficient_context,
        cancellation_checker=cancellation_checker,
        answer_mode=answer_mode,
        required_aspect_ids=required_aspect_ids,
        requirements=requirements,
    )
    if isinstance(prepared, GeneratedAnswer):
        return prepared
    return run_generation_loop(prepared, draft=draft_answer, verify=verify_draft)


def _covered_aspects_by_context(context_chunks: list[RetrievalResult]) -> set[str]:
    """Aspects covered by evidence-side metadata on the selected chunks.

    Uses the same chunk-side signals as ``_validate_required_aspect_coverage``
    (``prompt_matched_aspects`` plus ``aspect_id``) so the pre-generation gate
    and the post-generation check share one coverage rule.
    """

    covered: set[str] = set()
    for chunk in context_chunks:
        metadata = chunk.metadata or {}
        covered.update(str(value) for value in metadata.get('supported_requirement_ids') or [])
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

    Every requirement is checked, including choice questions.
    """

    if not required_aspect_ids:
        return []
    eligible = [
        aspect_id
        for aspect_id in required_aspect_ids
        if aspect_id
    ]
    if not eligible:
        return []
    covered = _covered_aspects_by_context(context_chunks)
    return [aspect_id for aspect_id in eligible if aspect_id not in covered]


def _calculation_citation_text(metadata: dict[str, Any], context_chunks: list[RetrievalResult]) -> str:
    """Citation labels of every chunk involved in a calculation.

    Cross-table calculations put each operand's chunk in the context; citing
    them all keeps the answer traceable to every source table.  Falls back to
    an empty string so the caller keeps the anchor label.
    """
    cell_chunk_ids = {
        str(item.get("chunk_id") or "")
        for item in metadata.get("calculation_cells") or []
        if isinstance(item, dict) and item.get("chunk_id")
    }
    if not cell_chunk_ids:
        return ""
    labels = sorted(
        {
            item.citation_label
            for item in context_chunks
            if item.chunk_id in cell_chunk_ids and item.citation_label
        }
    )
    return "".join(labels)


def _deterministic_table_answer(
    context_chunks: list[RetrievalResult],
    options: list[str],
    option_labels: list[str | None],
) -> GeneratedAnswer | None:
    if any(chunk.metadata.get("calculation_operand_only") for chunk in context_chunks):
        return None
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
            grounding_validation={"passed": False, "reason": "table_evidence_not_valid"},
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
    # Cross-table calculations return one cell-level evidence per operand
    # table; cite every involved chunk instead of only the anchor table.
    citation_label = _calculation_citation_text(metadata, context_chunks) or chunk.citation_label
    if option_text:
        # The official option extractor accepts a letter followed by ：/:/。/space
        # or end-of-line, not the comma used between a label and its text.  Keep
        # the label and text, but use a full stop so the structured choice is
        # machine-readable without depending on the evaluator's text fallback.
        option_clause = f"{option_label}。{option_text}" if option_label else str(option_text)
        answer = f"答案为 {option_clause}。依据表格证据，核验值为 {value_with_unit}。{citation_label}"
    elif (options and metadata.get("calculation_result") is not None
          and metadata.get("selection_assumptions")):
        # A default period/row assumption produced a number that matches none of
        # the supplied choices.  Refuse instead of presenting it as a choice;
        # the operands are ambiguous (for example, a period placeholder that
        # resolves to one row while the gold uses another section).
        return refusal_answer("calculation_result_not_in_options")
    elif metadata.get("comparison_operation") in {"max", "min"}:
        if options:
            # The comparison winner is not one of the supplied options; a
            # choice answer would be unselectable and possibly misleading.
            return refusal_answer("comparison_winner_not_in_options")
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
    assumptions = metadata.get("selection_assumptions") or []
    if assumptions:
        answer = "口径说明：" + "；".join(str(item) for item in assumptions) + "\n" + answer
    claim = AnswerClaim(
        text=answer.rsplit(citation_label, 1)[0].strip(),
        # A cross-table answer cites several chunks (“[1][2]”); keep them as
        # separate citation ids so each label validates against the context.
        citation_ids=re.findall(r"\[\d+\]", citation_label) or [citation_label],
        role="calculation" if metadata.get("calculation_result") is not None else "table_fact",
        aspect_ids=[str(chunk.metadata.get("aspect_id"))] if chunk.metadata.get("aspect_id") else [],
    )
    validation = validate_grounded_answer(answer, [claim], context_chunks)
    # Deterministic calculations may create a result not present verbatim in a
    # source chunk; it is grounded by the recorded operands and formula.
    if metadata.get("calculation_result") is not None:
        trace_valid = _calculation_trace_valid(metadata)
        # Entity checks would fail as false negatives for computed values, so
        # they are waived explicitly and recorded instead of being silently
        # dropped; the citation checks are never waived.
        validation["waived_checks"] = {
            "reason": "deterministic_calculation_result",
            "unsupported_entities": validation.get("unsupported_entities") or [],
            "unsupported_claim_entities": validation.get("unsupported_claim_entities") or [],
        }
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


@feature_gate("calculation", empty="none")
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
            # The computed result is not verbatim in a source chunk, so the
            # entity/semantic checks are explicit waivers, not silent drops.
            "waived_checks": {
                "reason": "deterministic_formula_result",
                "unsupported_entities": validation.get("unsupported_entities") or [],
                "unsupported_claim_entities": validation.get("unsupported_claim_entities") or [],
                "semantic_conflict_count": validation.get("semantic_conflict_count") or 0,
                "semantic_insufficient_count": validation.get("semantic_insufficient_count") or 0,
            },
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


@timed('answer.generation')
def _call_llm(question, context_chunks, options, option_labels, *, llm_prompt=None, correction=None,
              verified_claim_reporter=None, cancellation_checker=None, answer_mode='text',
              table_findings=None, required_aspect_ids=None):
    messages = RAGPromptBuilder().build_generation_messages(question, context_chunks, options=options,
        option_labels=option_labels, correction=correction, llm_prompt=llm_prompt, answer_mode=answer_mode,
        table_findings='\n'.join(x.to_prompt_line() for x in table_findings or []),
        required_aspect_ids=required_aspect_ids)
    config = ChatCompletionConfig(provider=ANSWER_GENERATION_PROVIDER, api_key=ANSWER_GENERATION_API_KEY,
        base_url=ANSWER_GENERATION_BASE_URL, model=ANSWER_GENERATION_MODEL,
        timeout_seconds=QA_REQUEST_TOTAL_BUDGET_SECONDS, disable_thinking=ANSWER_GENERATION_DISABLE_THINKING,
        response_format=ANSWER_GENERATION_RESPONSE_FORMAT)
    content = _request_llm_content(config, messages, stream=False, max_tokens=ANSWER_GENERATION_MAX_TOKENS,
        question=question, context_chunks=context_chunks, options=options, option_labels=option_labels,
        verified_claim_reporter=None, cancellation_checker=cancellation_checker,
)
    return _parse_generated_answer_content(content, question=question, context_chunks=context_chunks,
        options=options, option_labels=option_labels, answer_mode=answer_mode)


def _request_llm_content(config, messages, **kwargs):
    from backend.app.services.llm_client import chat_completion_content
    checker = kwargs.get("cancellation_checker")
    if checker:
        checker()
    return chat_completion_content(config, messages, max_tokens=kwargs.get("max_tokens"),
                                   opener=urllib.request.urlopen)


def _parse_generated_answer_content(
    content: str,
    *,
    question: str,
    context_chunks: list[RetrievalResult],
    options: list[str],
    option_labels: list[str | None],
    answer_mode: str,
) -> GeneratedAnswer:
    if not str(content or '').strip():
        raise GenerationParseError('empty generation content', kind='empty', raw_length=0)
    try:
        extracted = _extract_json(content)
    except (ValueError, TypeError) as exc:
        raise GenerationParseError('generation content contains no JSON object',
                                   kind='no_json', raw_length=len(str(content or ''))) from exc
    try:
        parsed = json.loads(extracted)
    except json.JSONDecodeError as exc:
        raise GenerationParseError('generation JSON is invalid', kind='invalid_json',
                                   raw_length=len(str(content or ''))) from exc
    if not isinstance(parsed, dict):
        raise GenerationParseError('generation JSON is not an object', kind='schema_error',
                                   raw_length=len(str(content or '')))
    if type(parsed.get('refused')) is not bool or not isinstance(parsed.get('claims', []), list):
        raise GenerationParseError('generation JSON does not match the answer schema',
                                   kind='schema_error', raw_length=len(str(content or '')))
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
    # Production answer_mode is text/scenario/mixed; MCQ is identified by the
    # presence of options and labels, not by answer_mode.
    is_choice = bool(options and option_labels)
    if is_choice and not refused:
        # Normalize the conclusion sentence to the project's machine-readable
        # form.  The model is asked for it, but older prompts/models may still
        # emit "正确选项为：X。" or the comma form, which the official option
        # extractor cannot recognize.
        for claim in claims:
            claim.text = _normalize_choice_conclusion(claim.text)
    composed_answer = "\n\n".join(claim.text + " " + "".join(ref for ref in claim.citation_ids if ref not in claim.text)
                                       for claim in claims if claim.text).strip()
    fallback_answer = str(parsed.get("answer") or "").strip()
    if is_choice and not refused and fallback_answer:
        fallback_answer = _normalize_choice_conclusion(fallback_answer)
    return GeneratedAnswer(
        answer=composed_answer or fallback_answer or None,
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


# A refusal key or the claims-array header can straddle an SSE chunk boundary,
# so incremental discovery scans the previous chunk's tail plus the new slice.


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
        # Recompute from the same displayed values the calculator used
        # (``_operand_decimal`` prefers the value string), so a result such as
        # 169.578795 is reproduced exactly instead of drifting by float noise.
        def _cell_number(item: dict[str, Any]) -> float:
            raw = item.get("value")
            if isinstance(raw, str) and re.fullmatch(r"-?\d+(?:\.\d+)?", raw.strip()):
                return float(raw.strip())
            return float(item["normalized_value"])

        values = [_cell_number(item) for item in cells]
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
            # This branch is only reachable with operation == "ratio", so the
            # first operand is always the left one.
            left, right = (0, 1)
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


def _normalize_option_labels(option_labels: list[str | None] | None, option_count: int) -> list[str | None]:
    labels = list(option_labels or [])
    if len(labels) < option_count:
        labels.extend([None] * (option_count - len(labels)))
    return [
        re.sub(r"[.、:：]\s*$", "", str(label).strip()) if label else None
        for label in labels[:option_count]
    ]


# Shared with the prompt builder so answer formatting never diverges from
# prompt formatting.
_format_selected_option = format_prompt_option

_CHOICE_CONCLUSION_RE = re.compile(
    r"^\s*(?:正确选项为|正确答案为|答案为)\s*[:：]?\s*([A-D])[\s、,，。：:]*",
    re.IGNORECASE,
)


def _normalize_choice_conclusion(text: str) -> str:
    """Put a choice conclusion in the extractor's accepted form.

    The official option extractor only accepts a letter followed by a colon,
    a full stop, whitespace or end-of-line.  In particular ``答案为 D、`` does
    not match.  Keep the model's selected letter; never invent one.
    """
    match = _CHOICE_CONCLUSION_RE.match(str(text or ""))
    if not match:
        return text
    rest = str(text)[match.end():].lstrip()
    return f"答案为 {match.group(1).upper()}。" + rest


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
