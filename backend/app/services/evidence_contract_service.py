"""Question-derived requirements shared by retrieval, selection and validation."""
from __future__ import annotations

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


def source_matches(titles: list[str], chunk: RetrievalResult) -> bool:
    if not titles:
        return True
    actual = [title_key(str(v)) for v in [chunk.source_doc, chunk.metadata.get('filename'),
              chunk.metadata.get('source_filename'), chunk.metadata.get('source_title')] if v]
    for title in titles:
        if not any(title_key(title) in a for a in actual):
            return False
        requested_format = re.search(r'[（(]\s*(pdf|word|docx?|excel|xlsx?)\s*[）)]\s*$', title, re.I)
        if requested_format:
            expected = {'word': 'docx', 'excel': 'xlsx'}.get(requested_format[1].lower(), requested_format[1].lower())
            source_formats = {str(chunk.metadata.get(key) or '').lower().lstrip('.') for key in ['source_format', 'file_type']}
            source_formats.update(re.findall(r'\.(pdf|docx?|xlsx?)$', chunk.source_doc.lower()))
            if expected not in source_formats:
                return False
    return True


def requirements_for_plan(plan: Any) -> list[EvidenceRequirement]:
    requirements = []
    for aspect in plan.aspects:
        source = str(aspect.table_filters.get('source_title') or '')
        sources = re.findall(r'《([^》]+)》', aspect.question)
        if source:
            sources = [source]
        requirements.append(EvidenceRequirement(
            requirement_id=aspect.aspect_id, question=aspect.question,
            intent=aspect.intent,
            target_terms=list(aspect.targets), source_titles=list(dict.fromkeys(sources))
            if aspect.modality == 'text' or len(aspect.selectors) <= 1 else [],
            subject_scope=list(aspect.subject_scope),
            constraints=dict(aspect.table_filters), selectors=list(aspect.selectors)))
        if aspect.modality in {'table', 'mixed'} and len(aspect.selectors) >= 2:
            for index, selector in enumerate(aspect.selectors):
                title = selector.get('source_title') or aspect.table_filters.get('source_title')
                requirements.append(EvidenceRequirement(
                    requirement_id=f'{aspect.aspect_id}_input{index+1}', parent_id=aspect.aspect_id,
                    selector_index=index, question=aspect.question, intent='operand',
                    source_titles=[str(title)] if title else [],
                    constraints={**aspect.table_filters, **selector}, selectors=[dict(selector)]))
    return requirements


def requirement_source_matches(requirement: EvidenceRequirement, chunk: RetrievalResult) -> bool:
    binding = (chunk.metadata.get('requirement_bindings') or {}).get(requirement.requirement_id)
    if binding is not None:
        actual = title_key(str(binding.get('filename') or binding.get('source_title') or ''))
        return all(title_key(title) in actual for title in requirement.source_titles)
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
    missing = [r.requirement_id for r in requirements if not coverage[r.requirement_id] or
               (r.source_titles and not any(source_matches(r.source_titles, c) for c in chunks)) or
               any(c.metadata.get('unit_incomplete') and c.chunk_id in coverage[r.requirement_id] for c in chunks)]
    return EvidenceBundle(requirements=requirements, evidence=chunks, coverage=coverage,
                          missing_requirements=missing, decisions=decisions)
