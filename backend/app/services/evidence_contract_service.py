"""Question-derived requirements shared by retrieval, selection and validation."""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from backend.app.schemas.qa import EvidenceBundle, EvidenceDecision, EvidenceRequirement, RetrievalResult


def target_terms(question: str) -> list[str]:
    """Keep quoted entities and overlapping phrases; never expand fixed topics."""
    quoted = re.findall(r'[“\"‘\']([^”\"’\']{2,40})[”\"’\']', question)
    text = re.sub(r'《[^》]*》', ' ', question)
    text = re.sub(r'是如何定义的|如何定义|怎样定义|定义|是什么|是指什么|如何|怎样|怎么|哪些|什么|是否|应该|应当|计算|统计|请|根据|依据|说明|解释|分别|以及|并且|的|与|和|及|在|对|中', ' ', text)
    terms = list(quoted)
    for phrase in re.findall(r'[\u4e00-\u9fffA-Za-z0-9_]{2,}', text):
        if len(phrase) <= 18:
            terms.append(phrase)
        else:
            terms.extend(phrase[i:i+4] for i in range(len(phrase)-3))
    return list(dict.fromkeys(t for t in terms if len(t) >= 2))[:32]


def title_key(value: str) -> str:
    from backend.app.services.retrieval_metadata_filter_service import _normalize_title
    text = unicodedata.normalize('NFKC', value).casefold()
    text = re.sub(r'\((?:pdf|word|excel|docx?|xlsx?)\)', '', text)
    text = re.sub(r'^\d{1,5}[_\-\s]+', '', text)
    text = re.sub(r'\.(?:pdf|docx?|xlsx?|csv)$', '', text)
    return _normalize_title(re.sub(r'[^\w\u4e00-\u9fff]', '', text).replace('_', ''))


_LEADING_VERSION_QUALIFIER_RE = re.compile(r'^\s*([^\s，,。；;：:（）()]{2,30}?版)(?=.+)')
_PURE_VERSION_QUALIFIER_RE = re.compile(r'^\s*([^\s，,。；;：:（）()]{2,30}?版)\s*$')
_CORE_NAME_MARKER_RE = re.compile(r'示例|办法|规则|通知|附件|制度|规定|指南|指引|规范|意见|条例|细则')


def _title_core_and_qualifiers(value: str) -> tuple[str, list[str]]:
    """Split a source title into its core name and version/scope qualifiers.

    Qualifiers may be written after the core in parentheses
    (``恢复计划示例（商业银行版）``) or before it
    (``商业银行版恢复计划示例``).  Keeping both forms as qualifiers lets the
    same comparator validate either wording without treating the qualifier as
    part of the document name.  A title consisting only of a version qualifier
    (``商业银行版``) yields an empty core and one qualifier.
    """

    text = str(value or '')
    qualifiers = re.findall(r'[（(]([^）)]+)[）)]', text)
    core = re.sub(r'[（(][^）)]*[）)]', '', text)
    leading = _LEADING_VERSION_QUALIFIER_RE.match(core)
    if leading:
        qualifiers.insert(0, leading.group(1))
        core = core[leading.end():]
    else:
        pure = _PURE_VERSION_QUALIFIER_RE.match(core)
        if pure and not _CORE_NAME_MARKER_RE.search(pure.group(1)):
            qualifiers.insert(0, pure.group(1))
            core = ''
    return core, list(dict.fromkeys(item.strip() for item in qualifiers if item.strip()))


def _title_core_key(value: str) -> str:
    """Title identity without version/scope qualifiers.

    Qualifiers are checked separately so a period/version phrase cannot be
    silently dropped, while a table-located document whose title omits the
    question's ``（季度）`` qualifier can still be recognised when the cell
    metadata proves the period.
    """

    core, _ = _title_core_and_qualifiers(value)
    return title_key(core)


def _title_actual_text(chunk: RetrievalResult) -> str:
    metadata = chunk.metadata or {}
    values: list[str] = [chunk.source_doc]
    for key in ('filename', 'source_filename', 'source_title', 'sheet_name', 'table_title',
                'column_label', 'row_label', 'unit', 'period', 'inferred_year', 'inferred_month',
                'inferred_quarter', 'version_status', 'publication_date', 'table_headers'):
        value = metadata.get(key)
        if value not in (None, '', [], {}):
            values.append(json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value))
    return ' '.join(values)


def _qualifier_matches(qualifier: str, actual_text: str, metadata: dict[str, Any], chunk: RetrievalResult) -> bool:
    raw = unicodedata.normalize('NFKC', str(qualifier or '')).strip()
    compact = re.sub(r'[\s\-_/]', '', raw).casefold()
    actual_compact = re.sub(r'[\s\-_/]', '', actual_text).casefold()
    if not compact:
        return True
    if compact in actual_compact:
        return True
    # File-format qualifiers are strict (existing contract).
    if compact in {'pdf', 'word', 'doc', 'docx', 'excel', 'xls', 'xlsx'}:
        expected = {'word': 'docx', 'excel': 'xlsx'}.get(compact, compact)
        formats = {str(metadata.get(key) or '').lower().lstrip('.') for key in ('source_format', 'file_type')}
        formats.update(re.findall(r'\.(pdf|docx?|xlsx?)$', chunk.source_doc.lower()))
        return expected in formats
    # Period qualifiers: the document title may omit them while its table
    # metadata carries the year/month/quarter.  This is the same identity the
    # spreadsheet locator already validated.
    period_metadata = metadata.get('period')
    period_values = period_metadata if isinstance(period_metadata, dict) else {}
    year_match = re.fullmatch(r'(20\d{2})\s*年?', raw)
    if year_match:
        year = int(year_match.group(1))
        if year in {metadata.get('inferred_year'), period_values.get('year')}:
            return True
        return bool(re.search(rf'{year}\s*年', actual_text))
    month_match = re.fullmatch(r'(\d{1,2})\s*月', raw)
    if month_match:
        month = int(month_match.group(1))
        if month in {metadata.get('inferred_month'), period_values.get('month')}:
            return True
        return bool(re.search(rf'{month}\s*月', actual_text))
    if '季度' in compact:
        if any(str(metadata.get(key) or '') for key in ('quarter', 'inferred_quarter')):
            return True
        if period_values.get('quarter') is not None:
            return True
        return bool(re.search(r'[一二三四1-4]\s*季度|季度', actual_text))
    if '月度' in compact or compact in {'月', '月份'}:
        if metadata.get('inferred_month') is not None or period_values.get('month') is not None:
            return True
        return bool(re.search(r'\d{1,2}\s*月', actual_text))
    # Version/edition qualifiers must be visible in the source identity or
    # version metadata; they are never silently dropped.
    if any(token in compact for token in ('版', '修订', '试行', '暂行', '征求意见')):
        version = str(metadata.get('version_status') or '')
        publication = str(metadata.get('publication_date') or '')
        haystack = re.sub(r'[\s\-_/]', '', version + publication).casefold()
        return compact in haystack or compact in actual_compact
    return False


def _title_matches(title: str, actual_text: str, metadata: dict[str, Any], chunk: RetrievalResult,
                   *, full_text: str | None = None) -> bool:
    core, qualifiers = _title_core_and_qualifiers(title)
    core_key = _title_core_key(core)
    qualifier_text = full_text if full_text is not None else actual_text
    if not core_key:
        # A pure version qualifier (“商业银行版”) has no core name; it can
        # only be validated against the qualifier evidence itself.
        return bool(qualifiers) and all(
            _qualifier_matches(qualifier, qualifier_text, metadata, chunk)
            for qualifier in qualifiers
        )
    actual_core = _title_core_key(actual_text)
    if core_key not in actual_core:
        return False
    for qualifier in qualifiers:
        if not _qualifier_matches(qualifier, qualifier_text, metadata, chunk):
            return False
    return True


def source_matches(titles: list[str], chunk: RetrievalResult) -> bool:
    if not titles:
        return True
    actual_text = _title_actual_text(chunk)
    actual_values = [chunk.source_doc, chunk.metadata.get('filename'),
                     chunk.metadata.get('source_filename'), chunk.metadata.get('source_title')]
    actual_values = [str(value) for value in actual_values if value]
    for title in titles:
        if not any(_title_matches(title, value, chunk.metadata or {}, chunk, full_text=actual_text)
                   for value in actual_values):
            return False
    return True


_COMPARISON_MARKER_RE = re.compile(r'是否一致|是否相同|有何异同|有何差异|是否不同|异同|差异|比较|对比|分别')
_ATTRIBUTE_END_RE = re.compile(r'是否一致|是否相同|有何异同|有何差异|是否不同|异同|差异')
_GENERIC_ATTRIBUTE_TERMS = {'规定', '要求', '表述', '内容', '情况', '方面', '界定'}
_GENERIC_TARGET_TERMS = {'规定', '一致', '差异', '包括', '内容', '情况', '方面', '要求'}
_DOCUMENT_STYLE_RE = re.compile(r'示例|办法|规则|通知|附件|制度|规定|指南|指引|规范|意见|条例|细则|版')
_QUOTED_PHRASE_RE = re.compile(r'[“"‘\']([^”"’\']{2,80})[”"’\']')
_COMPARISON_SOURCE_SEPARATOR_RE = re.compile(r'[和与及、]')


def is_cross_document_comparison(question: str) -> bool:
    """Whether the wording asks to compare at least two named sources.

    This is deliberately wording-level: no document names, topics or IDs are
    encoded.  A comparison verb alone (for example a numeric "difference"
    question) is not enough; the question must also coordinate two operands.
    """

    text = str(question or '')
    return bool(_COMPARISON_MARKER_RE.search(text) and _COMPARISON_SOURCE_SEPARATOR_RE.search(text))


def _clean_attribute_term(value: str) -> str:
    text = unicodedata.normalize('NFKC', str(value or '')).strip()
    text = text.strip('“”"\'‘’《》（）()[]【】,，。；;：: ')
    text = re.sub(r'^(?:关于|对于|针对|就|对|在|其中|其|该|两者|双方|二者|各)+', '', text)
    text = text.strip('“”"\'‘’《》（）()[]【】,，。；;：: ')
    text = re.sub(r'(?:有哪些|有何|有什么|存在哪些|是否存在|有)$', '', text)
    text = text.strip('的“”"\'‘’《》（）()[]【】,，。；;：: ')
    text = re.sub(r'(?:的)?(?:规定|要求|表述|内容|情况|方面|界定)$', '', text)
    text = text.strip('的“”"\'‘’《》（）()[]【】,，。；;：: ')
    if not text or text in _GENERIC_ATTRIBUTE_TERMS or len(text) < 2 or len(text) > 40:
        return ''
    if '《' in text or '》' in text:
        return ''
    return text


def _is_quoted_entity(value: str) -> bool:
    text = str(value or '').strip()
    return bool(re.fullmatch(r'[“"‘\'][^”"’\']{1,80}[”"’\']', text))


def _attribute_from_prefix(prefix: str) -> str:
    """Extract the compared attribute from text before a comparison predicate."""

    patterns = (
        r'中(.+?)的(?:规定|要求|表述|内容|情况|方面|界定)?$',
        r'对(.+?)的(.+?)$',
    )
    for index, pattern in enumerate(patterns):
        match = re.search(pattern, prefix)
        if not match:
            continue
        candidate = match.group(1) if index == 0 else match.group(2)
        # “中，“关键功能”的定义” captures the quoted entity, not the
        # compared property.  Leave it to the trailing “的” phrase below.
        if _is_quoted_entity(candidate):
            continue
        cleaned = _clean_attribute_term(candidate)
        if cleaned:
            return cleaned
    if '的' in prefix:
        cleaned = _clean_attribute_term(prefix.rsplit('的', 1)[-1])
        if cleaned:
            return cleaned
    for quoted in reversed(_QUOTED_PHRASE_RE.findall(prefix)):
        cleaned = _clean_attribute_term(quoted)
        if cleaned:
            return cleaned
    candidate = prefix.rsplit('的', 1)[-1] if '的' in prefix else prefix
    return _clean_attribute_term(candidate)


def comparison_attribute_terms(question: str) -> list[str]:
    """Extract the property that must be compared across the named sources.

    The extractor stays grammatical rather than topic-specific: it looks at
    the phrase immediately preceding a comparison predicate, removes generic
    wording such as "规定", and falls back to the last quoted phrase when the
    predicate itself is generic (for example "关于“职责分工”的规定是否一致").
    """

    text = unicodedata.normalize('NFKC', str(question or ''))
    text = re.sub(r'《[^》]*》', ' ', text)
    terms: list[str] = []
    for match in _ATTRIBUTE_END_RE.finditer(text):
        prefix = text[:match.start()]
        prefix = re.split(r'[，,。；;：:？?！!\n]', prefix)[-1]
        cleaned = _attribute_from_prefix(prefix)
        if cleaned:
            terms.append(cleaned)
    if not terms:
        # A planner may rewrite "X 和 Y 的 Z 是否一致" into "X 中 Z 是什么"
        # or "X 中关于 Z 的规定".  Keep the same attribute in those forms.
        clauses = [part for part in re.split(r'[，,。；;：:？?！!\n]', text) if part.strip()]
        clause = clauses[-1] if clauses else text
        clause = re.split(r'是否一致|是否相同|有何异同|有何差异|是否不同|异同|差异', clause)[0]
        cleaned = _attribute_from_prefix(clause)
        if cleaned:
            terms.append(cleaned)
    if not terms:
        asked = re.search(r'的([^，。；;：:？?]{2,40}?)(?:是什么|如何定义|怎样定义|怎么定义|包括哪些|有哪些|指什么)', text)
        if asked:
            cleaned = _clean_attribute_term(asked.group(1))
            if cleaned:
                terms.append(cleaned)
        elif '定义' in text:
            terms.append('定义')
    return list(dict.fromkeys(term for term in terms if term))[:5]


def document_style_phrases(values: Any) -> list[str]:
    """Return user-stated source-looking phrases from a subject scope.

    Only wording shape is used (document/version suffixes and a bounded
    length), so the helper is independent of a particular corpus file name.
    """

    phrases: list[str] = []
    for value in values or ():
        text = str(value or '')
        for part in re.split(r'[，,。；;：:、\n]|[和与及]', text):
            candidate = part.strip(' “”"\'‘’《》,，。；;：: ')
            if 4 <= len(candidate) <= 80 and _DOCUMENT_STYLE_RE.search(candidate):
                phrases.append(candidate)
    return list(dict.fromkeys(phrases))


def _attribute_haystack(chunk: RetrievalResult) -> tuple[str, str]:
    metadata = chunk.metadata or {}
    parts = [str(chunk.text or ''), str(chunk.section_title or ''),
             ' '.join(str(item) for item in (chunk.section_path or [])),
             str(chunk.source_doc or '')]
    for key in ('source_title', 'filename', 'source_filename'):
        value = metadata.get(key)
        if value:
            parts.append(str(value))
    normalized = unicodedata.normalize('NFKC', ' '.join(parts))
    return normalized, re.sub(r'\s+', '', normalized).casefold()


def requirement_attribute_matches(requirement: EvidenceRequirement, chunk: RetrievalResult) -> bool:
    """Check that a chunk evidences the compared attribute, not just its source."""

    terms = list(requirement.attribute_terms or [])
    if not terms:
        return True
    normalized, compact = _attribute_haystack(chunk)
    term_values = list(dict.fromkeys(
        term for term in terms
        if term and term not in _GENERIC_ATTRIBUTE_TERMS
    ))
    if not term_values:
        return False
    if requirement.intent == 'define':
        targets = [
            term for term in (requirement.target_terms or term_values)
            if 2 <= len(term) <= 40 and term not in _GENERIC_TARGET_TERMS
        ]
        for target in targets:
            pattern = (
                rf'(?:所称)?{re.escape(target)}\s*(?:[（(][^）)]{{0,20}}[）)])?'
                r'\s*(?:，|,)?\s*(?:是指|指|定义为|系指|是)'
            )
            if re.search(pattern, normalized):
                return True
        return False
    for term in term_values:
        if re.sub(r'\s+', '', unicodedata.normalize('NFKC', term)).casefold() in compact:
            return True
    if requirement.intent in {'lookup', 'list'}:
        for term in requirement.target_terms or ():
            cleaned = re.sub(r'\s+', '', unicodedata.normalize('NFKC', str(term or ''))).casefold()
            if len(cleaned) >= 4 and cleaned not in _GENERIC_TARGET_TERMS and cleaned in compact:
                return True
    return False


def requirements_for_plan(plan: Any) -> list[EvidenceRequirement]:
    requirements = []
    # Attribute terms are hints for locating and ranking a compared clause.
    # They are deliberately not used as a hard evidence filter: the final
    # semantic validator decides whether a clause actually supports a claim.
    plan_attribute_terms = (
        comparison_attribute_terms(str(getattr(plan, 'original_question', '') or ''))
        if str(getattr(plan, 'question_mode', 'open') or 'open') == 'open' else []
    )
    for aspect in plan.aspects:
        source = str(aspect.table_filters.get('source_title') or '')
        sources = re.findall(r'《([^》]+)》', aspect.question)
        if source:
            sources = [source]
        mandatory = bool(getattr(aspect, 'mandatory', True))
        attribute_terms: list[str] = []
        fallback_sources: list[str] = []
        if plan_attribute_terms and aspect.modality == 'text':
            scope_phrases = document_style_phrases(getattr(aspect, 'subject_scope', ()))
            # A single user-stated, source-looking scope phrase can identify
            # the source of a source-isolated aspect even when the planner did
            # not use book-title marks.  Multiple phrases stay a comparison
            # aspect and keep the source-only contract.
            if len(scope_phrases) == 1:
                attribute_terms = list(plan_attribute_terms)
                if not source:
                    fallback_sources = [scope_phrases[0]]
        resolved_sources = list(dict.fromkeys(sources or fallback_sources))
        requirements.append(EvidenceRequirement(
            requirement_id=aspect.aspect_id, question=aspect.question,
            intent=aspect.intent, mandatory=mandatory,
            target_terms=list(aspect.targets), attribute_terms=attribute_terms,
            source_titles=resolved_sources
            if aspect.modality == 'text' or len(aspect.selectors) <= 1 else [],
            subject_scope=list(aspect.subject_scope),
            constraints=dict(aspect.table_filters), selectors=list(aspect.selectors)))
        if aspect.modality in {'table', 'mixed'} and len(aspect.selectors) >= 2:
            for index, selector in enumerate(aspect.selectors):
                title = selector.get('source_title') or aspect.table_filters.get('source_title')
                requirements.append(EvidenceRequirement(
                    requirement_id=f'{aspect.aspect_id}_input{index+1}', parent_id=aspect.aspect_id,
                    selector_index=index, question=aspect.question, intent='operand', mandatory=mandatory,
                    source_titles=[str(title)] if title else [],
                    constraints={**aspect.table_filters, **selector}, selectors=[dict(selector)]))
    return requirements


def requirement_source_matches(requirement: EvidenceRequirement, chunk: RetrievalResult) -> bool:
    binding = (chunk.metadata.get('requirement_bindings') or {}).get(requirement.requirement_id)
    if binding is not None:
        actual = str(binding.get('filename') or binding.get('source_title') or '')
        full_text = actual + ' ' + _title_actual_text(chunk)
        return all(_title_matches(title, actual, chunk.metadata or {}, chunk, full_text=full_text)
                   for title in requirement.source_titles)
    return source_matches(requirement.source_titles, chunk)


def build_evidence_bundle(requirements: list[EvidenceRequirement], chunks: list[RetrievalResult],
                          candidates: list[RetrievalResult] | None = None) -> EvidenceBundle:
    coverage: dict[str, list[str]] = {}
    decisions = []
    selected_ids = {c.chunk_id for c in chunks}
    # Cell matching has already run once in the deterministic table service.
    # Bind each requested operand to its actual ordered source-cell record.
    for chunk in candidates if candidates is not None else chunks:
        parent = chunk.metadata.get('aspect_id')
        cells = chunk.metadata.get('calculation_cells') or chunk.metadata.get('comparison_cells') or []
        if not cells or not chunk.metadata.get('dynamic_table_evidence'):
            continue
        for requirement in requirements:
            index = requirement.selector_index
            if requirement.parent_id == parent and index is not None and index < len(cells):
                chunk.metadata.setdefault('requirement_bindings', {})[requirement.requirement_id] = cells[index]
                supported = chunk.metadata.setdefault('supported_requirement_ids', [])
                if requirement.requirement_id not in supported:
                    supported.append(requirement.requirement_id)
    for requirement in requirements:
        ids = []
        seen_requirement = False
        for chunk in candidates if candidates is not None else chunks:
            tags = set(chunk.metadata.get('prompt_matched_aspects') or [])
            tags.update(chunk.metadata.get('aspect_ids') or [])
            tags.update(chunk.metadata.get('supported_requirement_ids') or [])
            if chunk.metadata.get('aspect_id'):
                tags.add(chunk.metadata['aspect_id'])
            if requirement.requirement_id not in tags and len(requirements) != 1:
                continue
            seen_requirement = True
            source_ok = requirement_source_matches(requirement, chunk)
            stage = 'filtered' if not source_ok else 'selected' if chunk.chunk_id in selected_ids else 'not_selected'
            reason = ('explicit_source_conflict' if not source_ok else
                      'context_budget' if stage == 'not_selected' else
                      'source_and_requirement_match')
            decisions.append(EvidenceDecision(chunk_id=chunk.chunk_id, requirement_id=requirement.requirement_id,
                stage=stage, reason=reason))
            if source_ok and chunk.chunk_id in selected_ids:
                ids.append(chunk.chunk_id)
        coverage[requirement.requirement_id] = list(dict.fromkeys(ids))
        if not seen_requirement:
            decisions.append(EvidenceDecision(chunk_id='', requirement_id=requirement.requirement_id,
                                               stage='missing', reason='not_recalled'))
    missing_all = [r.requirement_id for r in requirements if not coverage[r.requirement_id] or
                   (r.source_titles and not any(source_matches(r.source_titles, c) for c in chunks)) or
                   any(c.metadata.get('unit_incomplete') and c.chunk_id in coverage[r.requirement_id]
                       for c in chunks)]
    by_id = {r.requirement_id: r for r in requirements}
    missing = [rid for rid in missing_all if by_id[rid].mandatory]
    missing_optional = [rid for rid in missing_all if not by_id[rid].mandatory]
    return EvidenceBundle(requirements=requirements, evidence=chunks, coverage=coverage,
                          missing_requirements=missing, missing_optional_requirements=missing_optional,
                          decisions=decisions)
