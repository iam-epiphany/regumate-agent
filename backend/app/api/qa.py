from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.schemas.qa import QARequest, QAResponse
from backend.app.services.rag_service import answer_question
from backend.app.services.retrieval_service import RetrievalServiceUnavailable


router = APIRouter(prefix="/qa", tags=["qa"])


@router.post("/ask", response_model=QAResponse)
def ask_question(payload: QARequest, db: Session = Depends(get_db)) -> QAResponse:
    """可信问答：先检索知识库，再基于引用片段回答。"""

    try:
        return answer_question(db, payload.question)
    except RetrievalServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
