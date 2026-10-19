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


def _title_core_key(value: str) -> str:
    """Title identity without parenthetical qualifiers.

    Qualifiers are checked separately so a period/version phrase cannot be
    silently dropped, while a table-located document whose title omits the
    question's ``（季度）`` qualifier can still be recognised when the cell
    metadata proves the period.
    """

    return title_key(re.sub(r'[（(][^）)]*[）)]', '', str(value or '')))


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
    core = _title_core_key(title)
    if not core:
        return False
    actual_core = _title_core_key(actual_text)
    if core not in actual_core:
        return False
    qualifier_text = full_text if full_text is not None else actual_text
    for qualifier in re.findall(r'[（(]([^）)]+)[）)]', str(title or '')):
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


def requirements_for_plan(plan: Any) -> list[EvidenceRequirement]:
    requirements = []
    for aspect in plan.aspects:
        source = str(aspect.table_filters.get('source_title') or '')
        sources = re.findall(r'《([^》]+)》', aspect.question)
        if source:
            sources = [source]
        mandatory = bool(getattr(aspect, 'mandatory', True))
        requirements.append(EvidenceRequirement(
            requirement_id=aspect.aspect_id, question=aspect.question,
            intent=aspect.intent, mandatory=mandatory,
            target_terms=list(aspect.targets), source_titles=list(dict.fromkeys(sources))
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
            matches = requirement_source_matches(requirement, chunk)
            stage = 'filtered' if not matches else 'selected' if chunk.chunk_id in selected_ids else 'not_selected'
            decisions.append(EvidenceDecision(chunk_id=chunk.chunk_id, requirement_id=requirement.requirement_id,
                stage=stage, reason='explicit_source_conflict' if not matches else 'context_budget' if stage == 'not_selected' else 'source_and_requirement_match'))
            if matches and chunk.chunk_id in selected_ids:
                ids.append(chunk.chunk_id)
        coverage[requirement.requirement_id] = list(dict.fromkeys(ids))
        if not seen_requirement:
            decisions.append(EvidenceDecision(chunk_id='', requirement_id=requirement.requirement_id,
                                               stage='missing', reason='not_recalled'))
    missing_all = [r.requirement_id for r in requirements if not coverage[r.requirement_id] or
                   (r.source_titles and not any(source_matches(r.source_titles, c) for c in chunks)) or
                   any(c.metadata.get('unit_incomplete') and c.chunk_id in coverage[r.requirement_id] for c in chunks)]
    by_id = {r.requirement_id: r for r in requirements}
    missing = [rid for rid in missing_all if by_id[rid].mandatory]
    missing_optional = [rid for rid in missing_all if not by_id[rid].mandatory]
    return EvidenceBundle(requirements=requirements, evidence=chunks, coverage=coverage,
                          missing_requirements=missing, missing_optional_requirements=missing_optional,
                          decisions=decisions)
