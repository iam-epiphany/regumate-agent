from dataclasses import dataclass
import re

from backend.app.core.config import (
    DIRECT_EVIDENCE_COVERAGE,
    FINAL_CITATION_LIMIT,
    MIN_EVIDENCE_COVERAGE,
    MIN_RERANK_SCORE,
    RERANK_TOP_K,
    RETRIEVAL_TOP_K,
)
from backend.app.schemas.qa import Citation
from backend.app.services.embedding_service import EmbeddingServiceError, embed_query
from backend.app.services.rerank_service import RerankServiceError, RerankedChunk, rerank_candidates
from backend.app.services.vector_store_service import VectorSearchResult, VectorStoreError, hybrid_search


class RetrievalServiceUnavailable(RuntimeError):
    pass


@dataclass
class RetrievalMatch:
    citation: Citation
    score: float
    rerank_score: float
    coverage_score: float = 0.0
    evidence_role: str = "related_context"


def retrieve_citations(question: str) -> list[RetrievalMatch]:
    try:
        query_embedding = embed_query(question)
        candidates = hybrid_search(query_embedding, limit=RETRIEVAL_TOP_K)
        reranked = rerank_candidates(question=question, candidates=candidates, limit=RERANK_TOP_K)
    except (EmbeddingServiceError, VectorStoreError, RerankServiceError) as exc:
        raise RetrievalServiceUnavailable(str(exc)) from exc

    terms = question_terms(question)
    reliable = [_annotate(item, terms, question) for item in reranked if _is_reliable(item, terms, question)]
    reliable.sort(key=_rank_key, reverse=True)
    selected = _limit_document_repetition(reliable, limit=FINAL_CITATION_LIMIT)
    return [
        RetrievalMatch(
            citation=_to_citation(item),
            score=item.candidate.score,
            rerank_score=item.rerank_score,
            coverage_score=item.coverage_score,
            evidence_role=item.evidence_role,
        )
        for item in selected
    ]


@dataclass
class _AnnotatedChunk:
    candidate: VectorSearchResult
    rerank_score: float
    coverage_score: float
    evidence_role: str


def _is_reliable(item: RerankedChunk, terms: list[str], question: str) -> bool:
    if not item.candidate.chunk_id or not item.candidate.text.strip() or item.rerank_score < MIN_RERANK_SCORE:
        return False
    coverage = evidence_coverage(terms, item.candidate.text)
    if coverage < MIN_EVIDENCE_COVERAGE:
        return False
    if _is_table_context(item.candidate, question, coverage):
        return coverage >= DIRECT_EVIDENCE_COVERAGE
    return True


def _annotate(item: RerankedChunk, terms: list[str], question: str) -> _AnnotatedChunk:
    coverage = evidence_coverage(terms, item.candidate.text)
    return _AnnotatedChunk(
        candidate=item.candidate,
        rerank_score=item.rerank_score,
        coverage_score=coverage,
        evidence_role=_evidence_role(item.candidate, question, coverage),
    )


def _rank_key(item: _AnnotatedChunk) -> tuple[float, float, float]:
    role_weight = {
        "direct_evidence": 3.0,
        "table_evidence": 2.0,
        "related_context": 1.0,
        "table_context": 0.0,
    }.get(item.evidence_role, 0.0)
    return (role_weight, item.coverage_score, item.rerank_score)


def _limit_document_repetition(items: list[_AnnotatedChunk], limit: int) -> list[_AnnotatedChunk]:
    selected: list[_AnnotatedChunk] = []
    per_document: dict[str, int] = {}
    for item in items:
        document_id = item.candidate.document_id
        if per_document.get(document_id, 0) >= 2 and len(items) > limit:
            continue
        selected.append(item)
        per_document[document_id] = per_document.get(document_id, 0) + 1
        if len(selected) >= limit:
            break
    return selected


def _to_citation(item: _AnnotatedChunk) -> Citation:
    candidate = item.candidate
    return Citation(
        document_id=candidate.document_id,
        chunk_id=candidate.chunk_id,
        filename=candidate.filename,
        section_title=candidate.section_title,
        page_number=candidate.page_number,
        excerpt=_safe_excerpt(candidate.text),
        score=candidate.score,
        rerank_score=item.rerank_score,
        chunk_type=candidate.chunk_type,
        evidence_role=item.evidence_role,
    )


def question_terms(question: str) -> list[str]:
    normalized = _normalize_for_match(question)
    domain_phrases = [
        "普惠小微",
        "同一借款人",
        "借款人",
        "多笔贷款",
        "贷款余额",
        "余额",
        "统计",
        "用途",
        "金额口径",
        "客户范围",
        "处理口径",
        "情形",
        "不纳入",
        "合并判断",
        "资产合计",
        "逾期贷款",
        "不良贷款",
        "风险分类",
        "绿色信贷",
        "差错更正",
    ]
    terms = [phrase for phrase in domain_phrases if phrase in normalized]
    terms.extend(token for token in re.findall(r"[A-Za-z0-9_]{2,}", normalized))
    if not terms:
        terms.extend(_fallback_chinese_terms(normalized))
    return list(dict.fromkeys(terms))


def evidence_coverage(terms: list[str], text: str) -> float:
    if not terms:
        return 0.0
    normalized_text = _normalize_for_match(text)
    matched = sum(1 for term in terms if term in normalized_text)
    return matched / len(terms)


def _evidence_role(candidate: VectorSearchResult, question: str, coverage: float) -> str:
    if candidate.chunk_type == "table":
        return "table_evidence" if _question_asks_table(question) or coverage >= DIRECT_EVIDENCE_COVERAGE else "table_context"
    return "direct_evidence" if coverage >= DIRECT_EVIDENCE_COVERAGE else "related_context"


def _is_table_context(candidate: VectorSearchResult, question: str, coverage: float) -> bool:
    return candidate.chunk_type == "table" and not _question_asks_table(question) and coverage < DIRECT_EVIDENCE_COVERAGE


def _question_asks_table(question: str) -> bool:
    normalized = _normalize_for_match(question)
    return any(term in normalized for term in ["表格", "情形", "处理口径", "哪些情况", "哪些情形", "不纳入"])


def _normalize_for_match(text: str) -> str:
    normalized = re.sub(r"\s+", "", text)
    replacements = {
        "同一个": "同一",
        "怎样": "怎么",
        "如何": "怎么",
        "应该": "",
        "应当": "",
        "怎么": "",
        "如何": "",
        "什么": "",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    return normalized


def _fallback_chinese_terms(text: str) -> list[str]:
    stop_words = ["应该", "应当", "怎么", "如何", "什么", "是否", "可以", "进行", "统计"]
    terms: list[str] = []
    for sequence in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        cleaned = sequence
        for word in stop_words:
            cleaned = cleaned.replace(word, "")
        if 2 <= len(cleaned) <= 8:
            terms.append(cleaned)
        elif len(cleaned) > 8:
            terms.extend(cleaned[index : index + 4] for index in range(0, len(cleaned) - 3, 4))
    return terms


def _safe_excerpt(text: str, limit: int = 320) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    sentences = re.findall(r".+?(?:[。！？!?；;]|$)", cleaned, flags=re.S)
    excerpt = ""
    for sentence in sentences:
        if len(excerpt) + len(sentence) > limit:
            break
        excerpt += sentence
    return excerpt.strip() or cleaned[:limit].strip()
