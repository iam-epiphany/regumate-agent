import json

from sqlalchemy.orm import Session

from backend.app.models.document import QALog
from backend.app.schemas.qa import Citation, QAResponse
from backend.app.services.audit_service import log_action
from backend.app.services.retrieval_service import (
    RetrievalMatch,
    RetrievalServiceUnavailable,
    evidence_coverage,
    question_terms,
    retrieve_citations,
)


REFUSAL_ANSWER = "知识库中未找到足够依据，无法给出确定回答。"


def answer_question(db: Session, question: str) -> QAResponse:
    cleaned_question = question.strip()
    if not cleaned_question:
        return QAResponse(answer="请输入问题。", citations=[], confidence=0.0, refused=True)

    matches = retrieve_citations(cleaned_question)
    if not matches:
        response = QAResponse(answer=REFUSAL_ANSWER, citations=[], confidence=0.0, refused=True)
        _save_qa_log(db, cleaned_question, response)
        _log_qa_audit(db, "qa_refused", cleaned_question, response)
        return response

    direct_matches = _direct_matches(matches)
    if not direct_matches:
        citations = [match.citation for match in matches]
        response = QAResponse(
            answer="知识库仅检索到相关背景，未找到能直接回答该问题的制度口径。",
            citations=citations,
            confidence=_confidence_from_matches(matches),
            refused=True,
        )
        _save_qa_log(db, cleaned_question, response)
        _log_qa_audit(db, "qa_refused", cleaned_question, response)
        return response

    citations = [match.citation for match in direct_matches]
    answer = _build_template_answer(cleaned_question, citations)
    response = QAResponse(
        answer=answer,
        citations=citations,
        confidence=_confidence_from_matches(direct_matches),
        refused=False,
    )
    _save_qa_log(db, cleaned_question, response)
    _log_qa_audit(db, "qa_answered", cleaned_question, response)
    return response


def _build_template_answer(question: str, citations: list[Citation]) -> str:
    sentences = _best_answer_sentences(question, citations)
    if not sentences:
        return "知识库中未找到足够依据，无法给出确定回答。"

    return f"根据知识库引用，结论如下：{''.join(sentences)}"


def _confidence_from_matches(matches: list[RetrievalMatch]) -> float:
    best = matches[0]
    if best.evidence_role in {"direct_evidence", "table_evidence"} and best.coverage_score >= 0.75 and best.rerank_score >= 0.75:
        return 0.85
    if best.evidence_role in {"direct_evidence", "table_evidence"} and best.coverage_score >= 0.45:
        return 0.65
    return 0.35


def _direct_matches(matches: list[RetrievalMatch]) -> list[RetrievalMatch]:
    return [match for match in matches if match.evidence_role in {"direct_evidence", "table_evidence"}]


def _best_answer_sentences(question: str, citations: list[Citation]) -> list[str]:
    terms = question_terms(question)
    candidates: list[tuple[float, str]] = []
    for citation in citations:
        if citation.chunk_type == "table":
            table_answer = _table_answer_excerpt(citation.excerpt)
            if table_answer:
                candidates.append((evidence_coverage(terms, table_answer), table_answer))
            continue
        for sentence in _split_answer_sentences(citation.excerpt):
            coverage = evidence_coverage(terms, sentence)
            if coverage > 0:
                candidates.append((coverage, sentence))

    candidates.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    selected: list[str] = []
    for _, sentence in candidates:
        if sentence in selected:
            continue
        selected.append(sentence)
        if len(selected) >= 2:
            break
    return selected


def _split_answer_sentences(text: str) -> list[str]:
    import re

    sentences = re.findall(r".+?(?:[。！？!?；;]|$)", text.strip(), flags=re.S)
    return [sentence.strip() for sentence in sentences if sentence.strip() and not sentence.lstrip().startswith("|")]


def _table_answer_excerpt(text: str) -> str:
    if "表格行证据：" in text:
        return text.replace("表格行证据：", "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    table_lines = [line for line in lines if "|" in line and not set(line.strip("|").replace(" ", "")) <= {"-", ":"}]
    return "\n".join(table_lines[:4])


def _save_qa_log(db: Session, question: str, response: QAResponse) -> None:
    log = QALog(
        question=question,
        answer=response.answer,
        refused=response.refused,
        confidence=response.confidence,
        citation_count=len(response.citations),
    )
    db.add(log)
    db.commit()


def _log_qa_audit(db: Session, action: str, question: str, response: QAResponse) -> None:
    detail = json.dumps(
        {
            "question": question,
            "answer": response.answer,
            "refused": response.refused,
            "confidence": response.confidence,
        },
        ensure_ascii=False,
    )
    log_action(db, action, "question", None, detail[:4000])
