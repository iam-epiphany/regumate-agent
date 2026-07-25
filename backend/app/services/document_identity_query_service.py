from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.models.document import Document
from backend.app.schemas.qa import QAResponse


_IDENTITY_TERMS = (
    "身份信息",
    "身份卡",
    "版本状态",
    "现行有效",
    "已失效",
    "已废止",
    "已被替代",
    "替代关系",
)


def answer_document_identity_question(db: Session, question: str) -> QAResponse | None:
    """Answer explicit identity/version questions from the identity ledger, not RAG prose.

    The handler intentionally does not infer legal effect from dates, filenames, or body
    text. Unknown is a valid outcome and remains distinct from a confirmed identity card.
    """

    if not _is_identity_intent(question):
        return None
    title = _quoted_title(question)
    if not title:
        return None
    documents = list(db.scalars(select(Document).where(Document.status == "indexed")))
    ranked = sorted(
        (
            (_title_score(title, document), document)
            for document in documents
            if _title_score(title, document) > 0
        ),
        key=lambda item: (-item[0], item[1].document_id),
    )
    if not ranked:
        return QAResponse(
            answer=f"当前知识库中未找到与《{title}》匹配的文档身份卡，无法判断其版本或法律效力。",
            citations=[],
            confidence=0.0,
            refused=True,
            answer_type="identity_refusal",
            generation_status="skipped",
            grounding_validation={"passed": True, "reason": "document_identity_not_found"},
            refusal_reason="document_identity_not_found",
            refusal_code="document_identity_not_found",
        )
    top_score = ranked[0][0]
    top_documents = [document for score, document in ranked if score == top_score]
    if len(top_documents) != 1:
        return QAResponse(
            answer=f"《{title}》匹配到多份文档身份卡，当前无法唯一确定目标文档；请补充完整文件名或文号。",
            citations=[],
            confidence=0.0,
            refused=True,
            answer_type="identity_refusal",
            generation_status="skipped",
            grounding_validation={"passed": True, "reason": "document_identity_ambiguous"},
            refusal_reason="document_identity_ambiguous",
            refusal_code="document_identity_ambiguous",
        )
    document = top_documents[0]
    status = (document.version_status or "unknown").strip().lower()
    review_status = (document.identity_review_status or "unreviewed").strip().lower()
    if status == "unknown":
        return QAResponse(
            answer=(
                f"《{title}》的文档身份卡将版本状态记录为“未知”，因此无法确认其现行有效、已失效或已被替代。"
                "身份信息仅用于检索和版本风险提示，不代表系统已核验法规法律效力。"
            ),
            citations=[],
            confidence=1.0,
            refused=True,
            answer_type="identity_refusal",
            generation_status="deterministic",
            grounding_validation={
                "passed": True,
                "reason": "version_status_unknown",
                "document_id": document.document_id,
                "file_sha256": document.file_sha256,
                "identity_review_status": review_status,
            },
            refusal_reason="version_status_unknown",
            refusal_code="version_status_unknown",
        )
    if review_status != "confirmed":
        return QAResponse(
            answer=(
                f"《{title}》的身份卡版本字段为“{status}”，但当前仍处于待核对状态，"
                "系统不会据此作出法律效力结论。"
            ),
            citations=[],
            confidence=0.5,
            refused=True,
            answer_type="identity_refusal",
            generation_status="deterministic",
            grounding_validation={"passed": True, "reason": "identity_not_confirmed", "document_id": document.document_id},
            refusal_reason="identity_not_confirmed",
            refusal_code="identity_not_confirmed",
        )
    return QAResponse(
        answer=(
            f"《{title}》已人工核对的文档身份卡将版本状态记录为“{status}”。"
            "该记录用于检索和版本风险提示，不等同于系统对法规法律效力作出鉴定。"
        ),
        citations=[],
        confidence=1.0,
        refused=False,
        answer_type="identity_metadata",
        generation_status="deterministic",
        grounding_validation={"passed": True, "reason": "confirmed_identity_snapshot", "document_id": document.document_id},
    )


def _quoted_title(question: str) -> str | None:
    match = re.search(r"《([^》]+)》", question)
    return match.group(1).strip() if match else None


def _is_identity_intent(question: str) -> bool:
    if any(term in question for term in _IDENTITY_TERMS):
        return True
    if "法律效力" not in question:
        return False
    return bool(
        re.search(r"(?:该|此|这份|文件|制度|法规).{0,10}法律效力", question)
        or re.search(r"法律效力.{0,10}(?:状态|如何|是否|能否|判断|确认)", question)
    )


def _title_score(title: str, document: Document) -> int:
    needle = _normalize(title)
    if not needle:
        return 0
    candidates = [document.filename, document.title or ""]
    best = 0
    for candidate in candidates:
        normalized = _normalize(candidate)
        if normalized == needle:
            best = max(best, 100)
        elif needle in normalized:
            best = max(best, 80 + min(19, len(needle)))
        elif normalized and normalized in needle:
            best = max(best, 60 + min(19, len(normalized)))
    return best


def _normalize(value: str) -> str:
    value = re.sub(r"^\d+_", "", value)
    value = re.sub(r"\.(?:docx?|pdf|xlsx?|csv|html?)$", "", value, flags=re.IGNORECASE)
    return re.sub(r"[\s_（）()《》【】\-—:：]+", "", value).casefold()
