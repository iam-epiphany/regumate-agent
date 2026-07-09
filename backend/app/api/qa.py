import json
from queue import Queue
from threading import Thread
from typing import Any, Iterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.app.core.database import SessionLocal, get_db
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


@router.post("/ask/stream")
def ask_question_stream(payload: QARequest) -> StreamingResponse:
    """可信问答流式观测：通过 SSE 推送 RAG 阶段进度，最终返回 QAResponse。"""

    return StreamingResponse(
        _stream_qa_events(payload.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/retrieve", response_model=LLMContextPackage)
def retrieve_context(payload: QARequest, db: Session = Depends(get_db)) -> LLMContextPackage:
    """检索知识库并组装后续 LLM 可直接使用的上下文包。"""

    try:
        return retrieve_context_package(db, payload.question)
    except RetrievalServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _stream_qa_events(question: str) -> Iterator[str]:
    events: Queue[tuple[str, dict[str, Any]] | None] = Queue()

    def progress_reporter(event: dict[str, Any]) -> None:
        events.put(("progress", event))

    def run_qa() -> None:
        db = SessionLocal()
        try:
            response = answer_question(db, question, progress_reporter=progress_reporter)
            events.put(("final", response.model_dump(mode="json")))
        except RetrievalServiceUnavailable as exc:
            events.put(
                (
                    "progress",
                    {
                        "stage": "retrieval",
                        "status": "failed",
                        "title": "检索系统暂不可用",
                        "detail": str(exc),
                    },
                )
            )
            events.put(("error", {"detail": str(exc)}))
        except Exception as exc:
            events.put(
                (
                    "progress",
                    {
                        "stage": "prompt_build",
                        "status": "failed",
                        "title": "上下文构造失败",
                        "detail": str(exc),
                    },
                )
            )
            events.put(("error", {"detail": "问答接口暂未返回数据。"}))
        finally:
            db.close()
            events.put(None)

    Thread(target=run_qa, daemon=True).start()

    while True:
        item = events.get()
        if item is None:
            break
        event_name, payload = item
        yield _sse_frame(event_name, payload)


def _sse_frame(event_name: str, payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event_name}\ndata: {data}\n\n"
