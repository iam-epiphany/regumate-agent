"""One final check of support and task completion, shared by every answer path."""
from __future__ import annotations

import json
from typing import Any

from backend.app.core.config import (SEMANTIC_GROUNDING_API_KEY, SEMANTIC_GROUNDING_BASE_URL,
    ANSWER_GENERATION_MAX_TOKENS,
    SEMANTIC_GROUNDING_DISABLE_THINKING, SEMANTIC_GROUNDING_MODEL, SEMANTIC_GROUNDING_PROVIDER,
    SEMANTIC_GROUNDING_RESPONSE_FORMAT, QA_REQUEST_TOTAL_BUDGET_SECONDS)
from backend.app.schemas.qa import AnswerClaim, EvidenceRequirement, RetrievalResult, ValidationReport
from backend.app.services.evidence_contract_service import source_matches
from backend.app.services.grounding_validation_service import validate_grounded_answer
from backend.app.services.json_utils import extract_json
from backend.app.services.llm_client import ChatCompletionConfig, ChatCompletionError, chat_completion_content
from backend.app.services.performance_metrics import RequestBudgetExceeded, timed
from backend.app.services.prompt_builder import SEMANTIC_SUPPORT_RULE
from backend.app.core.rag_features import features


def validation_passed(report: dict[str, Any] | None) -> bool:
    """The single success predicate shared by the backend and UI contract."""

    value = dict(report or {})
    return (
        value.get('status') == 'passed'
        and value.get('passed') is True
        and value.get('task_complete') is True
        and value.get('citation_integrity') is True
        and value.get('facts_supported') is True
    )


@timed('answer.validation')
def validate_answer(question: str, answer: str, claims: list[AnswerClaim], evidence: list[RetrievalResult],
                    *, requirements: list[str], deterministic: bool = False,
                    options: list[str] | None = None, option_labels: list[str | None] | None = None,
                    computation: bool = False,
                    requirement_details: list[EvidenceRequirement] | None = None) -> dict[str, Any]:
    base = validate_grounded_answer(answer, claims, evidence)
    if computation:
        from backend.app.services.formula_parser_service import calculate_formula_answer
        calculation = calculate_formula_answer(question, evidence)
        if calculation is not None and calculation.answer == answer:
            # Recompute from the cited formula and explicit user inputs; derived
            # numbers need not occur verbatim in the regulatory source.
            base['calculation_verified'] = True
            base['passed'] = not any(base.get(k) for k in [
                'invalid_citation_ids', 'invalid_answer_citation_ids', 'missing_inline_citation_ids',
                'missing_claims', 'missing_claim_citations'])
        else:
            base['passed'] = False
    citation_errors = ['invalid_citation_ids', 'invalid_answer_citation_ids', 'missing_inline_citation_ids',
                       'missing_claims', 'missing_claim_citations']
    report = ValidationReport(citation_integrity=not any(base.get(k) for k in citation_errors))
    cited = {ref for claim in claims for ref in claim.citation_ids}
    covered = set()
    for chunk in evidence:
        if chunk.citation_label not in cited:
            continue
        covered.update(chunk.metadata.get('prompt_matched_aspects') or [])
        covered.update(chunk.metadata.get('aspect_ids') or [])
        covered.update(chunk.metadata.get('supported_requirement_ids') or [])
        if chunk.metadata.get('aspect_id'):
            covered.add(chunk.metadata['aspect_id'])
    required_ids = set(requirements)
    details_all = list(requirement_details or [])
    details = {
        r.requirement_id: r for r in details_all
        if r.mandatory and (not required_ids or r.requirement_id in required_ids)
    }
    semantic_requirements = [
        r.model_dump() for r in details_all
        if r.mandatory and (not required_ids or r.requirement_id in required_ids)
    ]
    report.missing_requirements = [r for r in requirements if r not in covered or
        (r in details and not any(c.citation_label in cited and source_matches(details[r].source_titles, c)
                                 for c in evidence))]
    if not base.get('passed') or report.missing_requirements:
        report.status = 'failed'
        report.reasons = ['invalid_citations_or_entities'] if not report.citation_integrity else ['missing_required_evidence']
    elif deterministic:
        report.status = 'passed'
        report.task_complete = report.facts_supported = True
        report.method = 'deterministic'
    elif not features().semantic:
        report.status = 'not_run'
        report.reasons = ['validation_disabled']
    else:
        report.method = 'model'
        validation_attempts = 0
        validation_error_code = None
        validation_error_type = None
        validation_finish_reason = None
        while True:
            validation_attempts += 1
            try:
                content = chat_completion_content(ChatCompletionConfig(
                    provider=SEMANTIC_GROUNDING_PROVIDER, api_key=SEMANTIC_GROUNDING_API_KEY,
                    base_url=SEMANTIC_GROUNDING_BASE_URL, model=SEMANTIC_GROUNDING_MODEL,
                    timeout_seconds=QA_REQUEST_TOTAL_BUDGET_SECONDS,
                    disable_thinking=SEMANTIC_GROUNDING_DISABLE_THINKING,
                    response_format=SEMANTIC_GROUNDING_RESPONSE_FORMAT), [
                    {'role': 'system', 'content': (
                        '你是答案验收器。只能使用给定原文，禁止外部知识。核对答案是否回答用户的全部问题，'
                        '每项结论是否由其引用的原文支持，特别核对主体、版本、条件、例外、否定、数值和单位。'
                        '用户要求分别依据多个文件时，每份都必须有直接证据。相关摘录不等于回答。'
                        '信息不足必须判为未完成。不要要求问题没有询问的额外事实。'
                        '选择题编号是展示信息，无须在原文出现；但所选选项的每项事实均须被证据支持。'
                        + SEMANTIC_SUPPORT_RULE +
                        '答案可用原文补充选项的条件。已明示的合理默认口径不属于虚构事实。'
                        'reasons仅列关键问题，最多三项，每项不超过三十字。返回JSON：{"complete":true或false,"supported_claims":[1,2],'
                        '"missing_requirements":["缺少的事实或来源"],"reasons":["简短原因"]}。'
                        'supported_claims为从1开始的、已被各自引用直接支持的claim序号。')},
                    {'role': 'user', 'content': json.dumps({'question': question, 'answer': answer,
                        'claims': [c.model_dump() for c in claims], 'required_aspects': requirements,
                        'requirements': semantic_requirements,
                        'options': [{'label': (option_labels or [])[i] if i < len(option_labels or []) else None, 'text': text}
                                    for i, text in enumerate(options or [])],
                        'evidence': [{'citation_id': c.citation_label, 'source': c.source_doc,
                                     'text': c.text, 'aspects': c.metadata.get('prompt_matched_aspects', [])}
                                     for c in evidence]}, ensure_ascii=False)}], max_tokens=ANSWER_GENERATION_MAX_TOKENS)
                data = json.loads(extract_json(content))
                if (type(data.get('complete')) is not bool or not isinstance(data.get('supported_claims'), list)
                    or not isinstance(data.get('missing_requirements', []), list)
                    or not isinstance(data.get('reasons', []), list)):
                    raise ValueError('invalid validation schema')
                supported = {n for n in data['supported_claims'] if type(n) is int and 1 <= n <= len(claims)}
                report.facts_supported = supported == set(range(1, len(claims)+1))
                report.task_complete = data['complete']
                report.missing_requirements = [str(x) for x in data.get('missing_requirements', [])]
                report.reasons = [str(x) for x in data.get('reasons', [])]
                report.status = 'passed' if report.task_complete and report.facts_supported and not report.missing_requirements else 'failed'
                break
            except RequestBudgetExceeded:
                raise
            except ChatCompletionError as exc:
                validation_error_type = f'chat_{exc.kind}'
                validation_finish_reason = exc.finish_reason
                if exc.kind in {'transport', 'timeout'} and validation_attempts < 2:
                    continue
                validation_error_code = (
                    'validation_transport_failure' if exc.kind in {'transport', 'timeout'}
                    else 'validation_truncated' if exc.kind == 'truncated'
                    else 'validation_schema_failure'
                )
                report.status = 'error'
                report.reasons = [validation_error_code]
                break
            except (OSError, json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
                validation_error_type = type(exc).__name__
                validation_error_code = 'validation_schema_failure'
                report.status = 'error'
                report.reasons = [validation_error_code]
                break
    return {**base, **report.model_dump(), 'passed': report.passed,
            'semantic_passed': report.facts_supported if report.status in {'passed', 'failed'} else None,
            'verifier_status': report.status,
            'validation_attempts': locals().get('validation_attempts', 0),
            'validation_error_code': locals().get('validation_error_code'),
            'validation_error_type': locals().get('validation_error_type'),
            'validation_finish_reason': locals().get('validation_finish_reason'),
            'failure_stage': None if report.passed else 'validation',
            'reason': report.reasons[0] if report.reasons else report.status}
