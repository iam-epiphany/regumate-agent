from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.schemas.qa import LLMContextPackage, QARequest, QAResponse
from backend.app.services.rag_service import answer_question, retrieve_context_package
from backend.app.services.retrieval_service import RetrievalServiceUnavailable


router = APIRouter(prefix="/qa", tags=["qa"])


@router.post("/ask", response_model=QAResponse)
def ask_question(payload: QARequest, db: Session = Depends(get_db)) -> QAResponse:
    """可信问答：当前 RAG-only 阶段仅返回 LLM 上下文包，不生成最终答案。"""

    try:
        return answer_question(db, payload.question)
    except RetrievalServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/retrieve", response_model=LLMContextPackage)
def retrieve_context(payload: QARequest, db: Session = Depends(get_db)) -> LLMContextPackage:
    """检索知识库并组装后续 LLM 可直接使用的上下文包。"""

    try:
        return retrieve_context_package(db, payload.question)
    except RetrievalServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
