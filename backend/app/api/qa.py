import json
from queue import Queue
from threading import Thread
from time import monotonic, sleep
from typing import Any, Iterator

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from backend.app.core.database import SessionLocal, get_db
from backend.app.schemas.qa import (
    LLMContextPackage,
    QARequest,
    QAResponse,
    QATaskRequest,
    QATaskCreateResponse,
    QATaskStatusResponse,
)
from backend.app.services.qa_task_service import (
    cancel_qa_task,
    create_qa_task,
    get_qa_task_status,
    list_recent_qa_task_statuses,
)
from backend.app.services.rag_service import answer_question, retrieve_context_package
from backend.app.services.retrieval_service import RetrievalServiceUnavailable
from backend.app.core.config import QA_REQUEST_TOTAL_BUDGET_SECONDS
from backend.app.services.performance_metrics import RequestBudgetExceeded, record_trace_event, request_budget_context, request_trace_context
from uuid import uuid4


router = APIRouter(prefix="/qa", tags=["qa"])


@router.post("/ask", response_model=QAResponse)
def ask_question(payload: QARequest, response: Response, db: Session = Depends(get_db), x_request_id: str | None = Header(default=None)) -> QAResponse:
    """Return a deterministic table answer or a generated, grounded text answer."""

    request_id = x_request_id if x_request_id and len(x_request_id) <= 128 else uuid4().hex
    response.headers["X-Request-ID"] = request_id
    def progress(event: dict[str, Any]) -> None:
        record_trace_event(
            str(event.get("stage") or "unknown"), str(event.get("status") or "unknown"),
            event.get("summary") if isinstance(event.get("summary"), dict) else None,
        )
    try:
        with request_trace_context(request_id), request_budget_context(QA_REQUEST_TOTAL_BUDGET_SECONDS):
            record_trace_event("request_received", "started", {"question_length": len(payload.question), "option_count": len(payload.options or [])})
            result = answer_question(
                db, payload.question, options=payload.options,
                include_debug=payload.include_debug, progress_reporter=progress,
            )
            record_trace_event("response_returned", "completed", {"citation_count": len(result.citations)})
            return result
    except OperationalError as exc:
        # The SQLite database on a Docker volume mount can transiently fail to
        # open under concurrent load ("unable to open database file").  One
        # bounded retry with a fresh session absorbs the glitch; a second
        # failure is a real storage problem and surfaces as 500.
        record_trace_event("request_retry", "db_transient", {"error": str(exc)[:120]})
        sleep(1.0)
        try:
            with SessionLocal() as retry_db:
                return answer_question(
                    retry_db, payload.question, options=payload.options,
                    include_debug=payload.include_debug, progress_reporter=progress,
                )
        except (OperationalError, RetrievalServiceUnavailable, RequestBudgetExceeded) as retry_exc:
            if isinstance(retry_exc, OperationalError):
                raise HTTPException(status_code=500, detail="数据库临时不可用，请稍后重试。") from retry_exc
            if isinstance(retry_exc, RetrievalServiceUnavailable):
                raise HTTPException(status_code=503, detail=str(retry_exc)) from retry_exc
            raise HTTPException(status_code=504, detail=str(retry_exc)) from retry_exc
    except RetrievalServiceUnavailable as exc:
        # Model-inference lock contention or a transient Qdrant hiccup under
        # concurrent load; one bounded retry with a fresh session absorbs it.
        record_trace_event("request_retry", "retrieval_transient", {"error": str(exc)[:120]})
        sleep(1.0)
        try:
            with SessionLocal() as retry_db:
                return answer_question(
                    retry_db, payload.question, options=payload.options,
                    include_debug=payload.include_debug, progress_reporter=progress,
                )
        except (OperationalError, RetrievalServiceUnavailable, RequestBudgetExceeded) as retry_exc:
            if isinstance(retry_exc, OperationalError):
                raise HTTPException(status_code=500, detail="数据库临时不可用，请稍后重试。") from retry_exc
            if isinstance(retry_exc, RetrievalServiceUnavailable):
                raise HTTPException(status_code=503, detail=str(retry_exc)) from retry_exc
            raise HTTPException(status_code=504, detail=str(retry_exc)) from retry_exc
    except RequestBudgetExceeded as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc


@router.post("/tasks", response_model=QATaskCreateResponse)
def create_question_task(payload: QATaskRequest) -> QATaskCreateResponse:
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


@router.get("/tasks/{task_id}/stream")
def stream_question_task(task_id: str, db: Session = Depends(get_db)) -> StreamingResponse:
    """Stream durable task snapshots, including verified answer previews."""

    if get_qa_task_status(db, task_id) is None:
        raise HTTPException(status_code=404, detail="问答任务不存在或已过期")
    return StreamingResponse(
        _stream_qa_task_snapshots(task_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/tasks/{task_id}/cancel", response_model=QATaskStatusResponse)
def cancel_question_task(task_id: str, db: Session = Depends(get_db)) -> QATaskStatusResponse:
    """Cancel a queued or running QA task without losing its audit trail."""

    task = cancel_qa_task(db, task_id)
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


def _stream_qa_task_snapshots(task_id: str) -> Iterator[str]:
    last_payload = ""
    last_heartbeat = monotonic()
    terminal_statuses = {"completed", "refused", "failed", "cancelled"}
    while True:
        with SessionLocal() as db:
            task = get_qa_task_status(db, task_id)
        if task is None:
            yield _sse_frame("error", {"detail": "问答任务不存在或已过期"})
            return

        serialized = task.model_dump_json()
        if serialized != last_payload:
            last_payload = serialized
            last_heartbeat = monotonic()
            yield _sse_frame("task", task.model_dump(mode="json"))
        if task.status in terminal_statuses:
            return
        if monotonic() - last_heartbeat >= 15:
            last_heartbeat = monotonic()
            yield ": keep-alive\n\n"
        sleep(0.25)
