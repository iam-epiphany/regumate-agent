import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.document import Document, DocumentChunk, QALog
from backend.app.schemas.qa import Citation, QAResponse
from backend.app.services.audit_service import log_action


REFUSAL_ANSWER = "知识库中未找到足够依据，无法给出确定回答。"


def answer_question(db: Session, question: str) -> QAResponse:
    cleaned_question = question.strip()
    if not cleaned_question:
        return QAResponse(answer="请输入问题。", citations=[], confidence=0.0, refused=True)

    matches = search_knowledge_base(db, cleaned_question, limit=3)
    if not matches:
        response = QAResponse(answer=REFUSAL_ANSWER, citations=[], confidence=0.0, refused=True)
        _save_qa_log(db, cleaned_question, response)
        log_action(db, "qa_refused", "question", None, cleaned_question[:200])
        return response

    citations = [_to_citation(chunk, document) for _, chunk, document in matches]
    confidence = min(1.0, matches[0][0] / 10)
    answer = _build_template_answer(cleaned_question, citations)
    response = QAResponse(answer=answer, citations=citations, confidence=round(confidence, 2), refused=False)
    _save_qa_log(db, cleaned_question, response)
    log_action(db, "qa_answered", "question", None, cleaned_question[:200])
    return response


def search_knowledge_base(db: Session, question: str, limit: int = 3) -> list[tuple[int, DocumentChunk, Document]]:
    tokens = _tokens(question)
    if not tokens:
        return []

    statement = select(DocumentChunk, Document).join(Document, Document.document_id == DocumentChunk.document_id)
    scored: list[tuple[int, DocumentChunk, Document]] = []
    for chunk, document in db.execute(statement).all():
        searchable = f"{chunk.text} {chunk.section_title or ''} {document.filename}".lower()
        score = sum(1 for token in tokens if token in searchable)
        if question.strip().lower() in searchable:
            score += 5
        if score > 0:
            scored.append((score, chunk, document))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:limit]


def _tokens(text: str) -> set[str]:
    normalized = re.sub(r"[，。！？；：、,.!?;:\s]+", " ", text.lower()).strip()
    tokens = set(re.findall(r"[a-z0-9_]+", normalized))
    cjk_sequences = re.findall(r"[\u4e00-\u9fff]{2,}", normalized)
    for sequence in cjk_sequences:
        if len(sequence) <= 8:
            tokens.add(sequence)
        for index in range(0, max(len(sequence) - 1, 0)):
            tokens.add(sequence[index : index + 2])
    return {token for token in tokens if len(token) >= 2}


def _to_citation(chunk: DocumentChunk, document: Document) -> Citation:
    return Citation(
        document_id=document.document_id,
        chunk_id=chunk.chunk_id,
        filename=document.filename,
        section_title=chunk.section_title,
        page_number=chunk.page_number,
        excerpt=chunk.text[:260],
    )


def _build_template_answer(question: str, citations: list[Citation]) -> str:
    excerpts = "；".join(citation.excerpt for citation in citations[:2])
    return f"根据知识库中检索到的监管制度片段，关于“{question}”可以参考以下内容：{excerpts}"


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
