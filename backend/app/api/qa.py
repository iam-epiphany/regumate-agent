import json
from queue import Queue
from threading import Thread
from typing import Any, Iterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from backend.app.core.database import SessionLocal, get_db
from backend.app.schemas.qa import (
    LLMContextPackage,
    QARequest,
    QAResponse,
    QATaskCreateResponse,
    QATaskStatusResponse,
)
from backend.app.services.qa_task_service import create_qa_task, get_qa_task_status, list_recent_qa_task_statuses
from backend.app.services.rag_service import answer_question, retrieve_context_package
from backend.app.services.retrieval_service import RetrievalServiceUnavailable


router = APIRouter(prefix="/qa", tags=["qa"])


@router.post("/ask", response_model=QAResponse)
def ask_question(payload: QARequest, db: Session = Depends(get_db)) -> QAResponse:
    """Return a deterministic table answer or a generated, grounded text answer."""

    try:
        return answer_question(
            db,
            payload.question,
            options=payload.options,
            include_debug=payload.include_debug,
        )
    except RetrievalServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/tasks", response_model=QATaskCreateResponse)
def create_question_task(payload: QARequest) -> QATaskCreateResponse:
    """Create a durable QA task that can be polled after page switches or reloads."""

    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    return create_qa_task(payload)


@router.get("/tasks", response_model=list[QATaskStatusResponse])
def list_question_tasks(limit: int = 5, db: Session = Depends(get_db)) -> list[QATaskStatusResponse]:
    return list_recent_qa_task_statuses(db, limit=limit)


@router.get("/tasks/{task_id}", response_model=QATaskStatusResponse)
def get_question_task(task_id: str, db: Session = Depends(get_db)) -> QATaskStatusResponse:
    task = get_qa_task_status(db, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="问答任务不存在或已过期")
    return task


@router.post("/ask/stream")
def ask_question_stream(payload: QARequest) -> StreamingResponse:
    """可信问答流式观测：通过 SSE 推送 RAG 阶段进度，最终返回 QAResponse。"""

    return StreamingResponse(
        _stream_qa_events(payload.question, payload.options, payload.include_debug),
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
        return retrieve_context_package(db, payload.question, options=payload.options)
    except RetrievalServiceUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _stream_qa_events(
    question: str,
    options: list[str] | None = None,
    include_debug: bool = False,
) -> Iterator[str]:
    events: Queue[tuple[str, dict[str, Any]] | None] = Queue()

    def progress_reporter(event: dict[str, Any]) -> None:
        events.put(("progress", event))

    def run_qa() -> None:
        db = SessionLocal()
        try:
            response = answer_question(
                db,
                question,
                options=options,
                include_debug=include_debug,
                progress_reporter=progress_reporter,
            )
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
