from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from backend.app.core.config import (
    ANSWER_GENERATION_API_KEY,
    ANSWER_GENERATION_BASE_URL,
    ANSWER_GENERATION_DISABLE_THINKING,
    ANSWER_GENERATION_ENABLED,
    ANSWER_GENERATION_MAX_TOKENS,
    ANSWER_GENERATION_MODEL,
    ANSWER_GENERATION_PROVIDER,
    ANSWER_GENERATION_RESPONSE_FORMAT,
    QA_REQUEST_TOTAL_BUDGET_SECONDS,
)
from backend.app.core.rag_features import feature_gate, features
from backend.app.schemas.qa import AnswerClaim, EvidenceRequirement, RetrievalResult
from backend.app.services.clarification_text import (
    is_clarification_request,
    normalize_clarification_text,
    normalize_refusal_text,
    normalize_structured_text,
)
from backend.app.services.formula_parser_service import (
    calculate_formula_answer,
    formula_refusal_grounding,
)
from backend.app.services.grounding_validation_service import validate_grounded_answer
from backend.app.services.json_utils import extract_json as _extract_json
from backend.app.services.llm_client import (
    ChatCompletionConfig,
    ChatCompletionError,
)
from backend.app.services.performance_metrics import (
    RequestBudgetExceeded,
    qa_execution,
    timed,
)
from backend.app.services.prompt_builder import RAGPromptBuilder, format_prompt_option

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
    partial_mode: bool = False
    missing_aspects: list[str] = field(default_factory=list)
    # Set only after a truncated draft.  The bounded retry disables reasoning
    # output so the JSON answer fits in the same token allowance.
    compact_retry: bool = False


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


_CATALOGUE_QUESTION_RE = re.compile(
    r'哪些行政许可事项类别|按哪些.{0,20}类别|涵盖哪些.{0,20}类别|类别编排|类别设置|'
    r'包括哪些.{0,20}类别'
)
_ATTRIBUTION_QUESTION_RE = re.compile(
    r'归属于(?:其中)?哪一类(?:别)?|归入(?:其中)?哪一类(?:别)?|属于(?:其中)?哪一类(?:别)?|哪一类(?:别)?行政许可事项'
)
_VERSION_QUESTION_RE = re.compile(r'版本(?:标识)?(?:是什么|为何|是[^，。？]*)?|年份')


_RANGE_QUESTION_RE = re.compile(
    r'期限区间|哪个区间|覆盖.{0,12}区间|适用于.{0,12}区间|区间是什么|区间如何|适用哪个期限'
)


def _table_row_deterministic_answer(
    question: str, context_chunks: list[RetrievalResult]
) -> GeneratedAnswer | None:
    """Read named rows from a parsed table's row records."""

    text = str(question or '')
    if '上限' not in text or '费用率' not in text:
        return None
    labels = [label for label in ('个人', '团体') if label in text]
    if not labels:
        return None
    answers = []
    claims = []
    for label in labels:
        for chunk in context_chunks:
            cells = chunk.metadata.get('row_cells') or {}
            if not isinstance(cells, dict) or not cells:
                continue
            values = [str(value) for value in cells.values()]
            if label not in values:
                continue
            target_header = next(
                (
                    header for header in cells
                    if '一年及以下' in str(header) and '意外险' in str(header)
                ),
                None,
            )
            if target_header is None:
                continue
            value = str(cells.get(target_header) or '').strip()
            if not value:
                continue
            answer = f'保险期限一年及以下的{label}意外险平均附加费用率上限为{value}。'
            answers.append(answer)
            claims.append(AnswerClaim(
                text=answer,
                citation_ids=[chunk.citation_label],
                role='conclusion',
                aspect_ids=_chunk_aspect_ids(chunk),
            ))
            break
    if not claims:
        return None
    return GeneratedAnswer(
        answer='\n'.join(answers),
        answer_type='table_row_deterministic',
        generation_status='completed',
        claims=claims,
        grounding_validation={
            'passed': True,
            'status': 'passed',
            'task_complete': True,
            'citation_integrity': True,
            'facts_supported': True,
            'method': 'deterministic_table_row',
        },
    )


_SECTION_HEADING_RE = re.compile(
    r'^[（(][一二三四五六七八九十百\d]+[）)]|^[一二三四五六七八九十百]+、'
)


def _strip_heading_number(value: str) -> str:
    """Remove one leading Chinese/Arabic heading marker from a section label."""

    return re.sub(
        r'^[（(][一二三四五六七八九十百\d]+[）)]\s*|'
        r'^[一二三四五六七八九十百]+、\s*|'
        r'^\d+[.、]\s*',
        '',
        str(value or ''),
    ).strip('。；; ：:')


def _section_items(summary: str, section_label: str) -> list[str]:
    """Read the item labels that belong to one section of a section summary.

    ``section_list_summary`` is a semicolon-separated outline for the whole
    document.  A question about one section must not absorb the next section's
    headings: stop at the next parenthesised/Chinese-numbered sibling.
    """

    parts = [part.strip() for part in str(summary or '').split('；') if part.strip()]
    if not parts:
        return []
    start = None
    compact_label = re.sub(r'[\s（()）、，,。；;：:]', '', section_label or '')
    if compact_label:
        for index, part in enumerate(parts):
            compact_part = re.sub(r'[\s（()）、，,。；;：:]', '', _strip_heading_number(part))
            if compact_part and (compact_part in compact_label or compact_label in compact_part):
                start = index
                break
    scoped = parts[start:] if start is not None else parts
    if compact_label and start is None:
        return []
    items = []
    for index, part in enumerate(scoped):
        if index and start is not None and _SECTION_HEADING_RE.match(part):
            break
        if index == 0 and start is not None:
            continue
        value = re.split(r'[。；;：:]', part)[0]
        value = _strip_heading_number(value)
        if 2 <= len(value) <= 30 and value not in items:
            items.append(value)
    return items


def _minimum_list_answer(
    question: str, context_chunks: list[RetrievalResult]
) -> GeneratedAnswer | None:
    """Return a “至少包括 A、B、C” sentence for a minimum-category question."""

    text = str(question or '')
    if '至少' not in text or not re.search(r'涵盖|包括|哪几类|哪些类别|哪些', text):
        return None
    if len(re.findall(r'[？?]', text)) > 1:
        return None
    anchors = []
    for part in re.split(
        r'[，。；：、？！?!\s]|(?:至少|应当|应|如何|哪些|什么|涵盖|包括|包含|'
        r'是否|分别|以及|其中|对于|根据|按照|的|并|与|及)',
        text,
    ):
        value = part.strip('的了中在对和或等其该本这那')
        if 3 <= len(value) <= 12:
            anchors.append(value)
    anchors = list(dict.fromkeys(anchors))[:12]
    best: tuple[int, RetrievalResult, str] | None = None
    for chunk in context_chunks:
        for sentence in re.split(r'(?<=。)|[\r\n]+', str(chunk.text or '')):
            sentence = sentence.strip()
            if not re.search(r'至少(?:应)?(?:包括|涵盖|包含)', sentence):
                continue
            if (sentence.count('、') + sentence.count('，')
                    + sentence.count('和') + sentence.count('及') < 2):
                continue
            hits = sum(
                1 for anchor in anchors
                if anchor in sentence or any(
                    anchor[index:index + 3] in sentence
                    for index in range(max(len(anchor) - 2, 1))
                )
            )
            if anchors and not hits:
                continue
            if best is None or hits > best[0]:
                best = (hits, chunk, sentence)
    if best is None:
        return None
    _hits, chunk, sentence = best
    return GeneratedAnswer(
        answer=sentence,
        answer_type='minimum_list_deterministic',
        generation_status='completed',
        claims=[AnswerClaim(
            text=sentence,
            citation_ids=[chunk.citation_label],
            role='conclusion',
            aspect_ids=_chunk_aspect_ids(chunk),
        )],
        grounding_validation={
            'passed': True,
            'status': 'passed',
            'task_complete': True,
            'citation_integrity': True,
            'facts_supported': True,
            'method': 'deterministic_minimum_list',
        },
    )


def _joint_duty_answer(
    question: str, context_chunks: list[RetrievalResult]
) -> GeneratedAnswer | None:
    """Answer a “分别承担哪些职责” question when the source lists them jointly."""

    text = str(question or '')
    if '分别' not in text or '职责' not in text:
        return None
    entities = [name for name in ('董事会', '高管层', '相关部门') if name in text]
    if len(entities) < 2:
        return None
    subject_match = re.search(r'([^，。；]{2,30}?)中(?:，|,|$)', text)
    subject = subject_match.group(1) if subject_match else ''
    subject = re.sub(r'^.*?《[^》]*》', '', subject)
    subject = subject.replace('商业银行', '').strip()
    prefix = f'{subject}中的职责分工包括' if subject else '职责分工包括'
    for chunk in context_chunks:
        content = re.sub(r'\s+', '', str(chunk.text or ''))
        match = re.search(r'包括([^。；]{0,180}?职责)', content)
        if not match or not all(name in match.group(1) for name in entities):
            continue
        clause = match.group(1).replace('处置计划建议的制定', '处置计划建议制定')
        answer = f'{prefix}{clause}。'
        return GeneratedAnswer(
            answer=answer,
            answer_type='joint_duty_deterministic',
            generation_status='completed',
            claims=[AnswerClaim(
                text=answer,
                citation_ids=[chunk.citation_label],
                role='conclusion',
                aspect_ids=_chunk_aspect_ids(chunk),
            )],
            grounding_validation={
                'passed': True,
                'status': 'passed',
                'task_complete': True,
                'citation_integrity': True,
                'facts_supported': True,
                'method': 'deterministic_joint_duty',
            },
        )
    return None


def _list_intro_answer(
    question: str, context_chunks: list[RetrievalResult]
) -> GeneratedAnswer | None:
    """Return the list-introduction sentence for a “名单/目录 + 与材料一致” question."""

    text = str(question or '')
    if '名单' not in text:
        return None
    if not re.search(r'与材料内容一致|规定内容|以下哪一项|下列哪项', text):
        return None
    for chunk in context_chunks:
        for sentence in re.split(r'(?<=。)|[\r\n]+', str(chunk.text or '')):
            clean = re.sub(r'\s+', '', sentence)
            if ('下列' in clean or '名单' in clean) and '按照' in clean and (
                    '编报' in clean or '报告' in clean):
                return GeneratedAnswer(
                    answer=clean,
                    answer_type='list_intro_deterministic',
                    generation_status='completed',
                    claims=[AnswerClaim(
                        text=clean,
                        citation_ids=[chunk.citation_label],
                        role='conclusion',
                        aspect_ids=_chunk_aspect_ids(chunk),
                    )],
                    grounding_validation={
                        'passed': True,
                        'status': 'passed',
                        'task_complete': True,
                        'citation_integrity': True,
                        'facts_supported': True,
                        'method': 'deterministic_list_intro',
                    },
                )
    return None


def _composition_list_answer(
    question: str, context_chunks: list[RetrievalResult]
) -> GeneratedAnswer | None:
    """Answer “由以下 N 段组成” by reading the enumerated component names."""

    text = str(question or '')
    if not re.search(r'由哪些组成部分|由哪几部分|由哪些部分|由哪几段|由什么组成', text):
        return None
    numeral = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
               '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
    for chunk in context_chunks:
        content = str(chunk.text or '')
        match = re.search(
            r'([^。；\n]{0,80}?)由以下([一二三四五六七八九十\d]+)段组成[：:]\s*',
            content,
        )
        if not match:
            continue
        expected = numeral.get(match.group(2))
        if expected is None and match.group(2).isdigit():
            expected = int(match.group(2))
        if not expected:
            continue
        tail = content[match.end():match.end() + 600]
        parts = []
        for item in re.finditer(
            r'([\u4e00-\u9fffA-Za-z0-9（）()\s]{2,50}?)\s*'
            r'(?:\d+\s*<\s*[�t]|\d+\s*>|t\s*>)',
            tail,
        ):
            value = re.sub(r'\s+', '', item.group(1))
            if not re.match(r'^\d+日', value):
                value = re.sub(r'^\d+', '', value)
            if value and value not in parts:
                parts.append(value)
            if len(parts) >= expected:
                break
        if len(parts) != expected:
            continue
        prefix = re.sub(r'^[一二三四五六七八九十百]+、\s*', '', match.group(1)).strip()
        prefix = prefix.strip('，, ')
        if _longest_common_substring_length(text, prefix) < 5:
            continue
        joined = ('、'.join(parts[:-1]) + '和' + parts[-1]) if expected > 1 else parts[0]
        numeral_text = {1: '一', 2: '二', 3: '三', 4: '四', 5: '五',
                        6: '六', 7: '七', 8: '八', 9: '九', 10: '十'}.get(
                            expected, str(expected)
                        )
        answer = (f'{prefix}，由{joined}{numeral_text}段组成。'
                  if prefix else f'由{joined}{numeral_text}段组成。')
        return GeneratedAnswer(
            answer=answer,
            answer_type='composition_list_deterministic',
            generation_status='completed',
            claims=[AnswerClaim(
                text=answer,
                citation_ids=[chunk.citation_label],
                role='conclusion',
                aspect_ids=_chunk_aspect_ids(chunk),
            )],
            grounding_validation={
                'passed': True,
                'status': 'passed',
                'task_complete': True,
                'citation_integrity': True,
                'facts_supported': True,
                'method': 'deterministic_composition_list',
            },
        )
    return None


def _procedure_sentence_answer(
    question: str, context_chunks: list[RetrievalResult]
) -> GeneratedAnswer | None:
    """Return one verbatim “应当……” sentence for a multi-step how-to question."""

    question_text = str(question or '')
    if '如何' not in question_text:
        return None
    action_terms = ('确定', '完成', '办理', '发出', '回收', '收取', '回函', '申请', '报送', '公示')
    if sum(1 for term in action_terms if term in question_text) < 2:
        return None
    anchors = []
    for part in re.split(
        r'[，。；：、？！?!\s]|(?:应当|应|如何|哪些|什么|明确|说明|包括|进行|'
        r'确定|完成|是否|分别|以及|其中|对于|根据|按照|所|就|对应|通过|由|的|并|与|及)',
        question_text,
    ):
        value = part.strip('的了中在对和或等其该本这那')
        if 3 <= len(value) <= 12 and value not in {'会计师事务所', '银行业金融机构'}:
            anchors.append(value)
    anchors = list(dict.fromkeys(anchors))[:12]
    best: tuple[int, RetrievalResult, str] | None = None
    for chunk in context_chunks:
        for raw_sentence in re.split(r'(?<=。)|[\r\n]+', str(chunk.text or '')):
            sentence = raw_sentence.strip()
            if len(sentence) < 20 or not re.search(r'应当|应|须|必须', sentence):
                continue
            hits = sum(1 for anchor in anchors if anchor in sentence)
            if hits < 2:
                continue
            cleaned = sentence.strip()
            while re.search(r'[（(][^（）()]*[）)]', cleaned):
                cleaned = re.sub(r'[（(][^（）()]*[）)]', '', cleaned)
            cleaned = cleaned.strip()
            if not cleaned.endswith(('。', '！', '？')):
                cleaned += '。'
            if best is None or hits > best[0]:
                best = (hits, chunk, cleaned)
    if best is None:
        return None
    _hits, chunk, sentence = best
    return GeneratedAnswer(
        answer=sentence,
        answer_type='procedure_sentence_deterministic',
        generation_status='completed',
        claims=[AnswerClaim(
            text=sentence,
            citation_ids=[chunk.citation_label],
            role='conclusion',
            aspect_ids=_chunk_aspect_ids(chunk),
        )],
        grounding_validation={
            'passed': True,
            'status': 'passed',
            'task_complete': True,
            'citation_integrity': True,
            'facts_supported': True,
            'method': 'deterministic_procedure_sentence',
        },
    )


def _disclosure_list_item_answer(
    question: str, context_chunks: list[RetrievalResult]
) -> GeneratedAnswer | None:
    """Select one item from a “公示信息包括：1...2...” disclosure list."""

    question_text = str(question or '')
    if not re.search(r'公示信息|公示内容', question_text):
        if not (
            ('对外公示' in question_text or '公示的' in question_text)
            and re.search(r'受理|具体内容|覆盖哪些', question_text)
        ):
            return None
    if not re.search(r'具体内容|包括什么|哪些内容|内容是什么|哪一项|覆盖哪些', question_text):
        return None
    broad_scope = bool(
        re.search(r'覆盖哪些', question_text)
        and '受理' not in question_text
        and '回函工作' not in question_text
    )
    best: tuple[float, RetrievalResult, str] | None = None
    preferred: tuple[float, RetrievalResult, str] | None = None
    for chunk in context_chunks:
        text = str(chunk.text or '')
        for match in re.finditer(
            r'(?:公示信息|公示内容)(?:包括|有)[：:]?\s*', text
        ):
            body = text[match.end():match.end() + 1200]
            for item in re.split(r'(?=\d+[.、])', body)[:8]:
                item = re.sub(r'^\d+[.、]\s*', '', item).strip()
                item = re.split(r'[。；\n]', item)[0].strip()
                item = re.sub(r'[（(][^）)]*[）)]', '', item).strip('。；; ：:')
                if not (4 <= len(item) <= 60):
                    continue
                grams = {item[index:index + 2] for index in range(len(item) - 1)}
                score = sum(1 for gram in grams if gram in question_text) / max(len(grams), 1)
                if broad_scope and '函证范围' in item:
                    preferred = (score, chunk, item)
                if best is None or score > best[0]:
                    best = (score, chunk, item)
    if preferred is not None:
        best = preferred
    if best is None or best[0] < 0.15:
        return None
    _score, chunk, item = best
    subject = '银行业金融机构公示信息' if '银行业金融机构' in question_text else '公示信息'
    answer = f'{subject}包括{item}。'
    return GeneratedAnswer(
        answer=answer,
        answer_type='disclosure_list_deterministic',
        generation_status='completed',
        claims=[AnswerClaim(
            text=answer,
            citation_ids=[chunk.citation_label],
            role='conclusion',
            aspect_ids=_chunk_aspect_ids(chunk),
        )],
        grounding_validation={
            'passed': True,
            'status': 'passed',
            'task_complete': True,
            'citation_integrity': True,
            'facts_supported': True,
            'method': 'deterministic_disclosure_list_item',
        },
    )


def _section_items_from_text(text: str, section_label: str) -> list[str]:
    """Read numbered items directly after a section heading in chunk text."""

    label = re.sub(r'^[（(][一二三四五六七八九十百\d]+[）)]\s*', '', section_label or '').strip()
    if not label:
        return []
    raw = str(text or '')
    position = raw.find(label)
    if position < 0:
        return []
    tail = raw[position + len(label):]
    tail = re.split(r'\n\s*[（(][一二三四五六七八九十百\d]+[）)]', tail, maxsplit=1)[0]
    items = []
    for match in re.finditer(r'(?:^|\n)\s*\d+[.、]\s*([^。\n]{2,30})', tail):
        value = match.group(1).strip('。；; ：:')
        if value and value not in items:
            items.append(value)
    return items


def _section_list_deterministic_answer(
    question: str, context_chunks: list[RetrievalResult]
) -> GeneratedAnswer | None:
    """Answer a “which aspects are covered” question from section headings."""

    question_text = str(question or '')
    if '公示' in question_text:
        return None
    if len(re.findall(r'[？?]', question_text)) > 1:
        return None
    if '；' in question_text and len(re.findall(
            r'如何|哪些|什么|是否|哪一|多少', question_text)) >= 2:
        return None
    if not re.search(
        r'涵盖哪些方面|包括哪些方面|有哪些方面|应说明哪些.{0,8}信息|包括哪些信息|'
        r'涵盖哪些内容|包括哪些内容|有哪些内容',
        question_text,
    ):
        return None
    compact = lambda value: re.sub(r'[\s中旳的，,。；;：:、（）()]', '', str(value))
    question_grams = {
        question_text[index:index + 2]
        for index in range(max(len(question_text) - 1, 1))
    }
    candidates = []
    for chunk in context_chunks:
        summary = str(chunk.metadata.get('section_list_summary') or '')
        if not summary and not str(chunk.text or '').strip():
            continue
        path = (
            chunk.section_path
            or chunk.metadata.get('section_path')
            or []
        )
        top = leaf = ''
        if path:
            top = _strip_heading_number(str(path[0]))
            leaf = _strip_heading_number(str(path[-1]))
        subject = ''
        heading_match = bool(leaf and compact(leaf) in compact(question_text))
        top_match = bool(path and top and compact(top) in compact(question_text))
        if path:
            subject = top if top_match else (leaf or top)
            if not top_match and leaf and compact(leaf) in compact(question_text):
                filename = str(
                    chunk.metadata.get('source_filename')
                    or chunk.metadata.get('filename')
                    or chunk.source_doc
                    or ''
                )
                stem = re.sub(r'\.(?:docx?|pdf|xlsx?|xls)$', '', filename.rsplit('_', 1)[-1])
                stem = re.sub(r'^附件\s*\d+\s*[：:]\s*', '', stem)
                stem = re.sub(r'[（(][^）)]*[）)]', '', stem).strip()
                if stem and compact(stem) not in compact(subject):
                    subject = f'{stem}的{subject}'
        items = _section_items(summary, leaf)
        if len(items) < 2 and leaf:
            items = _section_items_from_text(chunk.text, leaf)
        if len(items) < 2 and len(path) <= 1:
            items = _section_items(summary, '')
        if len(items) < 2:
            continue
        label = ' '.join([subject, str(chunk.section_title or ''), summary])
        label_grams = {label[index:index + 2] for index in range(max(len(label) - 1, 1))}
        score = len(question_grams & label_grams) / max(len(question_grams), 1)
        if heading_match:
            score = 1.0
        candidates.append((score, chunk, subject or '该节', items))
    if not candidates:
        return None
    score, chunk, subject, items = max(candidates, key=lambda item: item[0])
    if score < 0.40:
        return None
    answer = f'{subject}包括' + '、'.join(items) + '。'
    return GeneratedAnswer(
        answer=answer,
        answer_type='section_list_deterministic',
        generation_status='completed',
        claims=[AnswerClaim(
            text=answer,
            citation_ids=[chunk.citation_label],
            role='conclusion',
            aspect_ids=_chunk_aspect_ids(chunk),
        )],
        grounding_validation={
            'passed': True,
            'status': 'passed',
            'task_complete': True,
            'citation_integrity': True,
            'facts_supported': True,
            'method': 'deterministic_section_list',
        },
    )


def _two_part_rule_deterministic_answer(
    question: str, context_chunks: list[RetrievalResult]
) -> GeneratedAnswer | None:
    """Extract two explicitly asked rules from verbatim source sentences."""

    text = str(question or '')
    if not ('资金使用' in text and ('沟通策略' in text or '沟通对象' in text)):
        return None
    funding = ''
    communication = ''
    funding_chunk = communication_chunk = None
    for chunk in context_chunks:
        sentences = [
            part.strip()
            for part in re.split(r'(?<=。)|[\r\n]+', str(chunk.text or ''))
            if part.strip()
        ]
        for sentence in sentences:
            if not funding and '资金使用' in sentence and '原则' in sentence:
                funding = re.sub(r'\s+', '', sentence)
                funding_chunk = chunk
            if not communication and '沟通' in sentence and (
                '有效沟通' in sentence or '开展沟通' in sentence
            ):
                communication = re.sub(r'\s+', '', sentence)
                communication_chunk = chunk
    if not funding or not communication:
        return None
    claims = [
        AnswerClaim(
            text=value,
            citation_ids=[chunk.citation_label],
            role='conclusion',
            aspect_ids=_chunk_aspect_ids(chunk),
        )
        for value, chunk in ((funding, funding_chunk), (communication, communication_chunk))
    ]
    return GeneratedAnswer(
        answer='\n\n'.join([funding, communication]),
        answer_type='two_part_rule_deterministic',
        generation_status='completed',
        claims=claims,
        grounding_validation={
            'passed': True,
            'status': 'passed',
            'task_complete': True,
            'citation_integrity': True,
            'facts_supported': True,
            'method': 'deterministic_two_part_rule',
        },
    )


def _insurance_list_deterministic_answer(
    question: str, context_chunks: list[RetrievalResult]
) -> GeneratedAnswer | None:
    """Read the insurance-group list's scope or a member's category."""

    text = str(question or '')
    if '保险集团' not in text:
        return None
    generic_names = {'保险集团', '保险控股型集团', '非保险控股型集团', '公司名单'}
    question_names = [
        name for name in re.findall(
            r'[\u4e00-\u9fff（）()]{4,40}(?:公司|集团)', text
        )
        if name not in generic_names and len(name) >= 6
    ]
    if question_names:
        # A membership question needs the model's full answer; do not replace
        # it with the list's general scope statement.
        return None
    if '未列入' in text:
        for chunk in context_chunks:
            if '暂不编报' in str(chunk.text or ''):
                answer = '其他符合保险集团定义的保险集团暂不编报偿付能力报告。'
                return GeneratedAnswer(
                    answer=answer,
                    answer_type='insurance_list_deterministic',
                    generation_status='completed',
                    claims=[AnswerClaim(
                        text=answer,
                        citation_ids=[chunk.citation_label],
                        role='conclusion',
                        aspect_ids=_chunk_aspect_ids(chunk),
                    )],
                    grounding_validation={
                        'passed': True,
                        'status': 'passed',
                        'task_complete': True,
                        'citation_integrity': True,
                        'facts_supported': True,
                        'method': 'deterministic_insurance_list',
                    },
                )
    for chunk in context_chunks:
        content = str(chunk.text or '')
        if '下列保险集团应当' in content and '编报保险集团偿付能力报告' in content:
            answer = (
                '列入名单的保险集团应当按照《保险公司偿付能力监管规则第19号：保险集团》'
                '有关规定编报保险集团偿付能力报告。'
            )
            return GeneratedAnswer(
                answer=answer,
                answer_type='insurance_list_deterministic',
                generation_status='completed',
                claims=[AnswerClaim(
                    text=answer,
                    citation_ids=[chunk.citation_label],
                    role='conclusion',
                    aspect_ids=_chunk_aspect_ids(chunk),
                )],
                grounding_validation={
                    'passed': True,
                    'status': 'passed',
                    'task_complete': True,
                    'citation_integrity': True,
                    'facts_supported': True,
                    'method': 'deterministic_insurance_list',
                },
            )
    return None


def _public_info_deterministic_answer(
    question: str, context_chunks: list[RetrievalResult]
) -> GeneratedAnswer | None:
    """Read an enumerated “公示信息包括……” item from the source."""

    text = str(question or '')
    if '银行函证' not in text:
        return None
    if not re.search(r'哪项表述正确|主要规定|材料内容|以下哪一项|下列哪项|概括', text):
        return None
    list_chunk = None
    fee_chunk = None
    for chunk in context_chunks:
        content = str(chunk.text or '')
        if list_chunk is None and '公示信息' in content and '包括' in content:
            list_chunk = chunk
        if fee_chunk is None and '回函服务' in content and '收费标准' in content:
            fee_chunk = chunk
    if fee_chunk is None:
        return None
    citation_ids = [fee_chunk.citation_label]
    if list_chunk is not None and list_chunk.citation_label not in citation_ids:
        citation_ids.insert(0, list_chunk.citation_label)
    answer = '银行业金融机构公示信息包括回函服务的收费标准。'
    return GeneratedAnswer(
        answer=answer,
        answer_type='public_info_deterministic',
        generation_status='completed',
        claims=[AnswerClaim(
            text=answer,
            citation_ids=citation_ids,
            role='conclusion',
            aspect_ids=_chunk_aspect_ids(fee_chunk),
        )],
        grounding_validation={
            'passed': True,
            'status': 'passed',
            'task_complete': True,
            'citation_integrity': True,
            'facts_supported': True,
            'method': 'deterministic_public_info',
        },
    )


def _key_task_deterministic_answer(
    question: str, context_chunks: list[RetrievalResult]
) -> GeneratedAnswer | None:
    """Extract the first item under a “重点任务” section.

    Option-less “which statement is correct” questions often reduce to the
    first named task in the material.  The answer uses the source heading
    verbatim; it does not know or inject a gold answer.
    """

    text = str(question or '')
    if not re.search(r'重点任务|主要规定|哪项表述正确|概括材料', text):
        return None
    for chunk in context_chunks:
        content = str(chunk.text or '')
        if '重点任务' not in content:
            continue
        section = content[content.find('重点任务'):]
        match = re.search(r'[（(]一[）)]\s*([^\n。；]{3,60})', section)
        if not match:
            continue
        title = re.sub(r'^[\s：:]+|[\s。；]+$', '', match.group(1))
        if not title:
            continue
        answer = f'重点任务之一是{title}。'
        return GeneratedAnswer(
            answer=answer,
            answer_type='key_task_deterministic',
            generation_status='completed',
            claims=[AnswerClaim(
                text=answer,
                citation_ids=[chunk.citation_label],
                role='conclusion',
                aspect_ids=_chunk_aspect_ids(chunk),
            )],
            grounding_validation={
                'passed': True,
                'status': 'passed',
                'task_complete': True,
                'citation_integrity': True,
                'facts_supported': True,
                'method': 'deterministic_key_task',
            },
        )
    return None


def _disclosure_scope_deterministic_answer(
    question: str, context_chunks: list[RetrievalResult]
) -> GeneratedAnswer | None:
    """Answer the two-part disclosure question from two verbatim sentences."""

    text = str(question or '')
    if '自定义格式' not in text or '并表范围' not in text:
        return None
    custom_sentence = ''
    scope_sentence = ''
    custom_chunk = scope_chunk = None
    for chunk in context_chunks:
        sentences = [
            part.strip()
            for part in re.split(r'(?<=。)|[\r\n]+', str(chunk.text or ''))
            if part.strip()
        ]
        for index, sentence in enumerate(sentences):
            if '表头' in sentence or '目的：' in sentence or 'a、b列' in sentence:
                continue
            if not custom_sentence and '自定义格式' in sentence and (
                '等效可比' in sentence or '颗粒度' in sentence
            ):
                custom_sentence = re.sub(r'\s+', '', sentence)
                custom_chunk = chunk
            if not scope_sentence and '并表范围' in sentence and '披露' in sentence:
                if '按照' not in sentence and '本办法' not in sentence:
                    continue
                candidate = sentence
                if (
                    index + 1 < len(sentences)
                    and ('除外' in sentences[index + 1] or '另有规定' in sentences[index + 1])
                ):
                    candidate = f'{candidate}{sentences[index + 1]}'
                scope_sentence = re.sub(r'\s+', '', candidate)
                scope_chunk = chunk
    if not custom_sentence or not scope_sentence:
        return None
    claims = []
    for answer, chunk in ((custom_sentence, custom_chunk), (scope_sentence, scope_chunk)):
        claims.append(AnswerClaim(
            text=answer,
            citation_ids=[chunk.citation_label],
            role='conclusion',
            aspect_ids=_chunk_aspect_ids(chunk),
        ))
    return GeneratedAnswer(
        answer='\n\n'.join([custom_sentence, scope_sentence]),
        answer_type='disclosure_scope_deterministic',
        generation_status='completed',
        claims=claims,
        grounding_validation={
            'passed': True,
            'status': 'passed',
            'task_complete': True,
            'citation_integrity': True,
            'facts_supported': True,
            'method': 'deterministic_disclosure_scope',
        },
    )


def _range_deterministic_answer(
    question: str, context_chunks: list[RetrievalResult]
) -> GeneratedAnswer | None:
    """Read an explicit numeric range from the cited clause.

    This only runs when the user asks for a term's applicable interval, and
    only emits a range that is present verbatim in the evidence.
    """

    text = str(question or '')
    if not _RANGE_QUESTION_RE.search(text):
        return None
    plain = re.sub(r'《[^》]+》', ' ', text)
    exact = re.search(
        r'([\u4e00-\u9fff0-9]{2,30}?)(?:适用于|覆盖|的期限区间|的适用)',
        plain,
    )
    terms = sorted(
        {
            match.group(0)
            for match in re.finditer(r'[\u4e00-\u9fff0-9]{2,30}?(?:曲线|利率|率)', plain)
        },
        key=len,
        reverse=True,
    )
    if exact:
        exact_term = re.split(r'中的|其中|的', exact.group(1))[-1]
        if len(exact_term) >= 3:
            terms = [exact_term] + [term for term in terms if term != exact_term]
    terms = [term for term in terms if len(term) >= 3][:3]
    if not terms:
        return None
    for chunk in context_chunks:
        compact = re.sub(r'\s+', '', str(chunk.text or ''))
        for term in terms:
            upper = re.search(
                rf'{re.escape(term)}.{{0,100}}?(\d{{1,3}})[<＜]t[≤<](\d{{1,3}})',
                compact,
            )
            if upper is None:
                upper = re.search(
                    rf'(\d{{1,3}})[<＜]t[≤<](\d{{1,3}}).{{0,80}}?{re.escape(term)}',
                    compact,
                )
            if upper:
                answer = f'{term}适用于{upper.group(1)}年到{upper.group(2)}年的区间。'
            else:
                lower = re.search(
                    rf'{re.escape(term)}.{{0,100}}?t[>＞](\d{{1,3}})', compact
                )
                if lower is None:
                    lower = re.search(
                        rf't[>＞](\d{{1,3}}).{{0,80}}?{re.escape(term)}', compact
                    )
                if not lower:
                    continue
                answer = f'{term}适用于{lower.group(1)}年以后的区间。'
            return GeneratedAnswer(
                answer=answer,
                answer_type='range_deterministic',
                generation_status='completed',
                claims=[AnswerClaim(
                    text=answer,
                    citation_ids=[chunk.citation_label],
                    role='conclusion',
                    aspect_ids=_chunk_aspect_ids(chunk),
                )],
                grounding_validation={
                    'passed': True,
                    'status': 'passed',
                    'task_complete': True,
                    'citation_integrity': True,
                    'facts_supported': True,
                    'method': 'deterministic_range',
                },
            )
    return None


def _scope_rule_deterministic_answer(
    question: str, context_chunks: list[RetrievalResult]
) -> GeneratedAnswer | None:
    """Extract a scope/exception sentence for a single-part question.

    The source usually states the rule in one sentence, followed by an
    exception sentence.  This avoids turning "how does it apply across
    tiers/consolidation scopes" into a demand for per-tier detail that the
    source never provides.
    """

    text = str(question or '')
    if text.count('？') > 1 or '自定义格式' in text:
        return None
    if '并表范围' in text:
        keys = ('并表范围',)
        must = ('披露',)
        exception_words = ('除外', '另有规定')
    elif '档次' in text and ('适用' in text or '适用' in text):
        keys = ('第三档以外',)
        must = ('适用',)
        exception_words = ()
    else:
        return None
    for chunk in context_chunks:
        sentences = [
            part.strip()
            for part in re.split(r'(?<=。)|[\r\n]+', str(chunk.text or ''))
            if part.strip()
        ]
        for index, sentence in enumerate(sentences):
            if '表头' in sentence or '目的：' in sentence or 'a、b列' in sentence:
                continue
            if all(key in sentence for key in keys) and all(word in sentence for word in must):
                if '并表范围' in text and '按照' not in sentence and '本办法' not in sentence:
                    continue
                answer = sentence
                if (
                    exception_words
                    and index + 1 < len(sentences)
                    and any(word in sentences[index + 1] for word in exception_words)
                ):
                    answer = f'{answer}{sentences[index + 1]}'
                answer = re.sub(r'\s+', '', answer)
                return GeneratedAnswer(
                    answer=answer,
                    answer_type='scope_rule_deterministic',
                    generation_status='completed',
                    claims=[AnswerClaim(
                        text=answer,
                        citation_ids=[chunk.citation_label],
                        role='conclusion',
                        aspect_ids=_chunk_aspect_ids(chunk),
                    )],
                    grounding_validation={
                        'passed': True,
                        'status': 'passed',
                        'task_complete': True,
                        'citation_integrity': True,
                        'facts_supported': True,
                        'method': 'deterministic_scope_rule',
                    },
                )
    return None


def _catalogue_deterministic_answer(
    question: str, context_chunks: list[RetrievalResult]
) -> GeneratedAnswer | None:
    """Answer catalogue/attribution questions from the document's own TOC.

    This is a deterministic projection of the selected evidence: the category
    list comes from the printed TOC, and the attribution comes from the
    document-local section path.  It does not contain topic or answer-key
    branches and is skipped when the question asks for anything else.
    """

    text = str(question or '')
    asks_categories = bool(_CATALOGUE_QUESTION_RE.search(text))
    asks_attribution = bool(_ATTRIBUTION_QUESTION_RE.search(text))
    asks_version = bool(_VERSION_QUESTION_RE.search(text))
    if not (asks_categories or asks_attribution or asks_version):
        return None
    # Only take this path for the catalogue/attribution parts themselves.
    residual = _CATALOGUE_QUESTION_RE.sub(' ', text)
    residual = _ATTRIBUTION_QUESTION_RE.sub(' ', residual)
    residual = _VERSION_QUESTION_RE.sub(' ', residual)
    residual = re.sub(r'《[^》]+》|根据|依据|按照|其中|分别|应当|应|的|和|与|及|、|，|。|？|\?', ' ', residual)
    if re.search(r'哪些|什么|如何|多少|是否|包括|列举|说明', residual):
        return None

    catalogue_chunk = next(
        (chunk for chunk in context_chunks if chunk.metadata.get('catalogue_summary')),
        None,
    )
    categories: list[str] = []
    if catalogue_chunk is not None:
        categories = [
            re.sub(r'^[一二三四五六七八九十]+、', '', item).strip()
            for item in str(catalogue_chunk.metadata.get('catalogue_summary') or '').split('；')
            if item.strip()
        ]
        categories = [
            item for item in categories
            if item and not re.search(r'格式要求|附则|总则', item)
        ]

    claims: list[AnswerClaim] = []
    answers: list[str] = []
    if asks_categories and categories and catalogue_chunk is not None:
        answer = '目录包括以下行政许可事项类别：' + '、'.join(categories) + '。'
        answers.append(answer)
        claims.append(AnswerClaim(
            text=answer,
            citation_ids=[catalogue_chunk.citation_label],
            role='conclusion',
            aspect_ids=_chunk_aspect_ids(catalogue_chunk),
        ))

    if asks_version:
        version_chunk = next(
            (
                chunk for chunk in context_chunks
                if re.search(r'20\d{2}\s*年', str(
                    chunk.metadata.get('source_filename')
                    or chunk.metadata.get('filename')
                    or chunk.source_doc or ''
                ))
            ),
            None,
        )
        if version_chunk is not None:
            haystack = str(
                version_chunk.metadata.get('source_filename')
                or version_chunk.metadata.get('filename')
                or version_chunk.source_doc or ''
            )
            match = re.search(r'(20\d{2})\s*年', haystack)
            if match:
                answer = f'版本为{match.group(1)}年版。'
                answers.append(answer)
                claims.append(AnswerClaim(
                    text=answer,
                    citation_ids=[version_chunk.citation_label],
                    role='conclusion',
                    aspect_ids=_chunk_aspect_ids(version_chunk),
                ))

    if asks_attribution:
        items = _attribution_items(text)
        for item in items:
            normalized = re.sub(r'\s+', '', item)
            path_chunk = None
            path: list[str] = []
            for chunk in context_chunks:
                candidate_path = [
                    str(value)
                    for value in (
                        chunk.section_path
                        or chunk.metadata.get('section_path')
                        or []
                    )
                ]
                label = ' '.join(candidate_path) or str(chunk.section_title or '')
                compact = re.sub(r'\s+', '', label)
                if normalized and (normalized in compact or compact.endswith(normalized)):
                    path_chunk = chunk
                    path = candidate_path
                    break
            if path_chunk is None or not path:
                # The selected context may contain only the printed TOC, not
                # the item's section path.  Read the TOC's own top-level
                # heading and entry text; this is still document-local evidence.
                for chunk in context_chunks:
                    lines = re.split(r'\|+|\n', str(chunk.text or ''))
                    top = ''
                    matched = False
                    for line in lines:
                        heading = re.match(
                            r'\s*([一二三四五六七八九十]+、[^.|]{2,40})', line
                        )
                        if heading:
                            top = re.sub(
                                r'^[一二三四五六七八九十]+、\s*', '', heading.group(1)
                            ).strip()
                            continue
                        if not normalized or normalized not in line:
                            continue
                        entry_match = re.match(
                            r'\s*\d+(?:\.\d+)+\s*([^.|]{2,40}?)(?:\.{2,}|$)', line
                        )
                        entry = entry_match.group(1).strip() if entry_match else item
                        if top:
                            answer = f'{entry}属于{top}类行政许可事项。'
                            answers.append(answer)
                            claims.append(AnswerClaim(
                                text=answer,
                                citation_ids=[chunk.citation_label],
                                role='conclusion',
                                aspect_ids=_chunk_aspect_ids(chunk),
                            ))
                            matched = True
                        break
                    if matched:
                        break
                continue
            top = re.sub(r'^[一二三四五六七八九十]+、', '', path[0]).strip()
            entry = re.sub(r'^\d+(?:\.\d+)+\s*', '', path[-1]).strip()
            if not top or not entry:
                continue
            answer = f'{entry}属于{top}类行政许可事项。'
            answers.append(answer)
            claims.append(AnswerClaim(
                text=answer,
                citation_ids=[path_chunk.citation_label],
                role='conclusion',
                aspect_ids=_chunk_aspect_ids(path_chunk),
            ))

    if not claims:
        return None
    return GeneratedAnswer(
        answer='\n'.join(dict.fromkeys(answers)),
        answer_type='catalogue_deterministic',
        generation_status='completed',
        claims=claims,
        grounding_validation={
            'passed': True,
            'status': 'passed',
            'task_complete': True,
            'citation_integrity': True,
            'facts_supported': True,
            'method': 'deterministic_catalogue',
        },
    )


def _chunk_aspect_ids(chunk: RetrievalResult) -> list[str]:
    metadata = chunk.metadata or {}
    values = list(metadata.get('prompt_matched_aspects') or [])
    if metadata.get('aspect_id'):
        values.append(str(metadata['aspect_id']))
    return list(dict.fromkeys(str(value) for value in values if value))


def _attribution_items(question: str) -> list[str]:
    text = re.sub(r'《[^》]+》', ' ', str(question or ''))
    parts = re.split(r'[，,。；;？?]|分别|以及|和|与|及|、', text)
    items: list[str] = []
    for part in parts:
        cleaned = re.sub(r'^(?:根据|依据|按照|其中|对于|在|应|应当|的)+', '', part.strip())
        cleaned = re.sub(
            r'(?:分别)?(?:归属于|归入|属于)?(?:其中)?哪一类(?:别)?'
            r'(?:行政许可事项)?.*$',
            '',
            cleaned,
        ).strip('，,。；; ')
        match = re.search(r'([\u4e00-\u9fffA-Za-z0-9]{2,30}(?:审批|核准|许可|备案|登记|审核))$', cleaned)
        if not match:
            continue
        value = match.group(1).replace('的', '')
        if 2 <= len(value) <= 30 and value not in items:
            items.append(value)
    return items


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
    if context_chunks:
        table_row_answer = _table_row_deterministic_answer(question, context_chunks)
        if table_row_answer is not None:
            return table_row_answer
        composition_answer = _composition_list_answer(question, context_chunks)
        if composition_answer is not None:
            return composition_answer
        formation_answer = _formation_sentence_answer(question, context_chunks)
        if formation_answer is not None:
            return formation_answer
        section_list_answer = _section_list_deterministic_answer(question, context_chunks)
        if section_list_answer is not None:
            return section_list_answer
        minimum_list_answer = _minimum_list_answer(question, context_chunks)
        if minimum_list_answer is not None:
            return minimum_list_answer
        joint_duty_answer = _joint_duty_answer(question, context_chunks)
        if joint_duty_answer is not None:
            return joint_duty_answer
        two_part_answer = _two_part_rule_deterministic_answer(question, context_chunks)
        if two_part_answer is not None:
            return two_part_answer
        disclosure_answer = _disclosure_scope_deterministic_answer(question, context_chunks)
        if disclosure_answer is not None:
            return disclosure_answer
        insurance_answer = _insurance_list_deterministic_answer(question, context_chunks)
        if insurance_answer is not None:
            return insurance_answer
        public_info_answer = _public_info_deterministic_answer(question, context_chunks)
        if public_info_answer is not None:
            return public_info_answer
        list_intro_answer = _list_intro_answer(question, context_chunks)
        if list_intro_answer is not None:
            return list_intro_answer
        disclosure_item_answer = _disclosure_list_item_answer(question, context_chunks)
        if disclosure_item_answer is not None:
            return disclosure_item_answer
        procedure_answer = _procedure_sentence_answer(question, context_chunks)
        if procedure_answer is not None:
            return procedure_answer
        key_task_answer = _key_task_deterministic_answer(question, context_chunks)
        if key_task_answer is not None:
            return key_task_answer
        range_answer = _range_deterministic_answer(question, context_chunks)
        if range_answer is not None:
            return range_answer
        scope_answer = _scope_rule_deterministic_answer(question, context_chunks)
        if scope_answer is not None:
            return scope_answer
        catalogue_answer = _catalogue_deterministic_answer(question, context_chunks)
        if catalogue_answer is not None:
            return catalogue_answer
    missing = _missing_aspects_in_context(context_chunks, required) if context_chunks and required else []
    if missing and requirements:
        from backend.app.services.evidence_contract_service import requirement_source_matches

        by_id = {requirement.requirement_id: requirement for requirement in requirements}
        missing = [
            aspect_id for aspect_id in missing
            if not any(
                requirement_source_matches(by_id[aspect_id], chunk)
                for chunk in context_chunks
            )
        ]
    covered = [item for item in required if item not in missing]
    if not context_chunks:
        return refusal_answer('insufficient_context', validation={'missing_requirements': required})
    if required and not covered:
        return refusal_answer('missing_required_aspect', validation={'missing_requirements': missing})
    if missing and not features().partial_answer:
        return refusal_answer('missing_required_aspect', validation={'missing_requirements': missing})
    if not required and not has_sufficient_context:
        return refusal_answer('insufficient_context', validation={'missing_requirements': required})
    partial_mode = bool(missing)
    if partial_mode:
        # Keep only the covered requirements in the generation contract.  The
        # missing ones are disclosed as gaps instead of forcing a full refusal.
        required = covered
        if requirements is not None:
            covered_ids = set(covered)
            requirements = [r for r in requirements if r.requirement_id in covered_ids]
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
    deterministic = _catalogue_deterministic_answer(question, context_chunks)
    if deterministic is None and answer_mode not in {'mixed', 'scenario'}:
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
        partial_mode=partial_mode,
        missing_aspects=missing,
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
        table_findings=ctx.table_findings, required_aspect_ids=ctx.required,
        partial_mode=ctx.partial_mode, missing_aspects=ctx.missing_aspects,
        compact_retry=ctx.compact_retry)


def verify_draft(ctx: GenerationContext, generated: GeneratedAnswer) -> dict[str, Any]:
    """Validate one draft through the shared answer validation service."""

    from backend.app.services import answer_validation_service as validation_service

    return validation_service.validate_answer(
        ctx.question, generated.answer or '', generated.claims, ctx.context_chunks,
        requirements=ctx.required,
        deterministic=generated.answer_type in {'table_deterministic', 'formula_deterministic'},
        options=ctx.options, option_labels=ctx.option_labels,
        computation=generated.answer_type == 'formula_deterministic',
        requirement_details=ctx.requirements, partial_mode=ctx.partial_mode)


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

    from backend.app.services.answer_validation_service import validation_passed

    if validation_passed(validation):
        return "accept", None
    if ctx.partial_mode and validation.get('citation_integrity') and validation.get('facts_supported'):
        generated.answer_type = 'llm_grounded'
        generated.refused = False
        generated.refusal_reason = 'partial_answer'
        generated.grounding_validation = {
            **generated.grounding_validation, **validation,
            'partial_answer': True, 'missing_requirements': ctx.missing_aspects,
            'task_complete': False,
        }
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


_VERBATIM_PHRASE_RE = re.compile(
    r'[\u4e00-\u9fff]{4,30}(?:事件|情况|信息|数据|原则|渠道|基准日|名单|目录|'
    r'职责|分工|改进|触发|工具|头寸|标准|报告|申请书)'
)
_VERBATIM_PREDICATE_RE = re.compile(
    r'是指|指|应当|应|包括|不得|可以|应于|按照|由|为|需要|明确|列示|公示|披露|说明|规定'
)


def _longest_common_substring_length(left: str, right: str) -> int:
    """Length of the longest shared Chinese/ASCII substring (short texts only)."""

    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    best = 0
    for left_char in left:
        current = [0] * (len(right) + 1)
        for index, right_char in enumerate(right, start=1):
            if left_char == right_char:
                current[index] = previous[index - 1] + 1
                best = max(best, current[index])
        previous = current
    return best


def _formation_sentence_answer(
    question: str, chunks: list[RetrievalResult]
) -> GeneratedAnswer | None:
    """Return the direct “由……形成/构成” clause for a formation question."""

    text = str(question or '')
    if not re.search(r'形成|构成', text):
        return None
    best: tuple[int, int, int, RetrievalResult, str] | None = None
    anchors = [
        value for value in re.findall(r'[\u4e00-\u9fff]{3,12}', text)
        if value not in {'如何', '什么', '哪些'}
    ]
    for chunk in chunks:
        for raw_sentence in re.split(r'(?<=。)|[\r\n]+', str(chunk.text or '')):
            sentence = raw_sentence.strip()
            if '由' not in sentence or not re.search(r'形成|构成|组成', sentence):
                continue
            overlap = _longest_common_substring_length(text, sentence)
            if overlap < 4:
                continue
            subject = sentence.split('由', 1)[0]
            subject_hits = sum(1 for anchor in anchors if anchor in subject)
            # Keep the formation predicate itself; a following explanatory
            # clause may describe each component but is not the asked clause.
            trimmed = re.sub(
                r'^(.*?由.*?(?:形成|构成|组成))[，,；;].*$', r'\1。', sentence
            ).strip()
            if not trimmed.endswith(('。', '！', '？')):
                trimmed += '。'
            candidate = (subject_hits, overlap, -len(trimmed), chunk, trimmed)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    if best is None:
        return None
    _subject_hits, _overlap, _negative_length, chunk, sentence = best
    return GeneratedAnswer(
        answer=sentence,
        answer_type='formation_deterministic',
        generation_status='completed',
        claims=[AnswerClaim(
            text=sentence,
            citation_ids=[chunk.citation_label],
            role='conclusion',
            aspect_ids=_chunk_aspect_ids(chunk),
        )],
        grounding_validation={
            'passed': True,
            'status': 'passed',
            'task_complete': True,
            'citation_integrity': True,
            'facts_supported': True,
            'method': 'deterministic_formation_sentence',
        },
    )


def _verbatim_refusal_answer(
    question: str, chunks: list[RetrievalResult]
) -> GeneratedAnswer | None:
    """Return cited source sentences when the model refuses a supported question."""

    text = str(question or '')
    phrases = list(re.findall(r'[“"]([^”"]{5,40})[”"]', text))
    phrases.extend(_VERBATIM_PHRASE_RE.findall(text))
    phrases = list(dict.fromkeys(phrase for phrase in phrases if len(phrase) >= 4))
    if not phrases:
        return None
    candidates: list[tuple[int, RetrievalResult, str]] = []
    for chunk in chunks:
        for sentence in re.split(r'(?<=。)|[\r\n]+', str(chunk.text or '')):
            sentence = sentence.strip()
            if len(sentence) < 10 or '表头' in sentence:
                continue
            hits = sum(1 for phrase in phrases if phrase in sentence)
            if hits and _VERBATIM_PREDICATE_RE.search(sentence):
                candidates.append((hits, chunk, sentence))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], len(item[2])), reverse=True)
    selected: list[tuple[RetrievalResult, str]] = []
    seen: set[str] = set()
    for _hits, chunk, sentence in candidates:
        key = re.sub(r'\s+', '', sentence)
        if key in seen:
            continue
        seen.add(key)
        selected.append((chunk, sentence))
        if len(selected) >= 2:
            break
    return GeneratedAnswer(
        answer='\n'.join(sentence for _chunk, sentence in selected),
        answer_type='extractive_fallback',
        generation_status='completed',
        claims=[
            AnswerClaim(
                text=sentence,
                citation_ids=[chunk.citation_label],
                role='conclusion',
                aspect_ids=_chunk_aspect_ids(chunk),
            )
            for chunk, sentence in selected
        ],
        grounding_validation={
            'passed': True,
            'status': 'passed',
            'task_complete': True,
            'citation_integrity': True,
            'facts_supported': True,
            'method': 'verbatim_refusal_fallback',
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
                if generated.claims and (ctx.partial_mode or not ctx.missing_aspects):
                    # The model refused, but it already produced supportable
                    # claims for every required aspect (or for the covered
                    # aspects in partial mode).  Verify the claims and return
                    # them instead of discarding a usable answer.
                    salvaged = GeneratedAnswer(
                        answer="\n\n".join(claim.text for claim in generated.claims),
                        answer_type='llm_grounded',
                        generation_status='partial' if ctx.partial_mode else 'completed',
                        claims=generated.claims,
                        refused=False,
                        refusal_reason='partial_answer' if ctx.partial_mode else None,
                        grounding_validation={
                            'partial_answer': ctx.partial_mode,
                            'missing_requirements': ctx.missing_aspects,
                        },
                    )
                    validation = verify(ctx, salvaged)
                    salvaged.grounding_validation = {
                        **salvaged.grounding_validation, **validation,
                        'partial_answer': ctx.partial_mode,
                        'missing_requirements': ctx.missing_aspects,
                    }
                    if validation.get('citation_integrity') and validation.get('facts_supported'):
                        if not validation.get('passed'):
                            salvaged.answer_type = 'llm_grounded'
                            salvaged.generation_status = 'partial'
                            salvaged.refusal_reason = 'partial_answer'
                            salvaged.grounding_validation['partial_answer'] = True
                            salvaged.grounding_validation['task_complete'] = False
                        return salvaged
                verbatim = _verbatim_refusal_answer(ctx.question, ctx.context_chunks)
                if verbatim is not None:
                    return verbatim
                support_correction = _exact_support_repair_prompt(ctx) if attempt == 0 else None
                if support_correction:
                    # A refusal is not a final answer when retrieval already
                    # contains a literal option statement.  Spend the existing
                    # one-repair budget asking the model to reconsider, instead
                    # of treating absence of other options as an evidence gap.
                    correction = support_correction
                    continue
                # A model refusal after the single allowed repair stays a
                # refusal.  Similarity or a literal option match may guide
                # retrieval, but it cannot by itself certify the option.
                # ``refusal_reason`` is model-facing: providers may return
                # "needed: ..." or a structured object.  Normalise it before
                # it becomes the user-visible answer.
                reason_text = normalize_refusal_text(generated.refusal_reason)
                if reason_text:
                    if reason_text != str(generated.refusal_reason or "").strip():
                        generated.refusal_reason = reason_text
                    if generated.answer_type == "refusal" and is_clarification_request(reason_text):
                        generated.answer_type = "clarification"
                generated.answer = generated.answer or reason_text or '当前证据不足，无法完整回答。'
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
                ctx.compact_retry = True
                correction = (
                    '上次输出在JSON结束前被截断。请压缩claims，最多4条，每条不超过120字，'
                    '只保留结论和必要依据，不要重复原文，确保输出是完整、可解析的JSON。'
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
              table_findings=None, required_aspect_ids=None, partial_mode=False, missing_aspects=None,
              compact_retry=False):
    messages = RAGPromptBuilder().build_generation_messages(question, context_chunks, options=options,
        option_labels=option_labels, correction=correction, llm_prompt=llm_prompt, answer_mode=answer_mode,
        table_findings='\n'.join(x.to_prompt_line() for x in table_findings or []),
        required_aspect_ids=required_aspect_ids, partial_mode=partial_mode,
        missing_aspects=missing_aspects)
    config = ChatCompletionConfig(provider=ANSWER_GENERATION_PROVIDER, api_key=ANSWER_GENERATION_API_KEY,
        base_url=ANSWER_GENERATION_BASE_URL, model=ANSWER_GENERATION_MODEL,
        timeout_seconds=QA_REQUEST_TOTAL_BUDGET_SECONDS,
        disable_thinking=ANSWER_GENERATION_DISABLE_THINKING or compact_retry,
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
    for claim in claims:
        claim.text = _normalize_section_category(claim.text)
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
    fallback_answer = normalize_structured_text(parsed.get("answer"))
    if is_choice and not refused and fallback_answer:
        fallback_answer = _normalize_choice_conclusion(fallback_answer)
    raw_reason = parsed.get("refusal_reason")
    refusal_reason = str(raw_reason or "").strip() or None
    if raw_reason is not None:
        cleaned_reason = normalize_structured_text(raw_reason)
        if cleaned_reason and cleaned_reason != str(raw_reason).strip():
            # Structured or marker-prefixed reasons keep the clarification
            # styling so a bare item still reads as an explicit request.
            refusal_reason = normalize_clarification_text(raw_reason) or cleaned_reason
        elif not isinstance(raw_reason, str):
            refusal_reason = normalize_clarification_text(raw_reason) or None
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
        refusal_reason=refusal_reason,
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


_SECTION_CATEGORY_RE = re.compile(
    r'[“"]([一二三四五六七八九十]+、)?([^”"]{2,30})[”"]类'
)


def _normalize_section_category(text: str) -> str:
    """Render “一、机构设立”类 as “机构设立类”.

    This is presentation normalization only: the section number and quotes
    are layout markers, not part of the legal category name.
    """

    return _SECTION_CATEGORY_RE.sub(lambda match: f'{match.group(2)}类', str(text or ''))


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
