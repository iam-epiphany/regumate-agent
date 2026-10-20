from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from queue import Empty, Full, Queue
from threading import Lock, Thread
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.core.config import (
    QA_COMPLEX_REQUEST_TOTAL_BUDGET_SECONDS,
    QA_QUEUE_CAPACITY,
    QA_REQUEST_TOTAL_BUDGET_SECONDS,
)
from backend.app.core.database import SessionLocal
from backend.app.models.document import QATask
from backend.app.schemas.qa import (
    ApiError,
    QAAnswerPreview,
    QAResponse,
    QATaskCreateResponse,
    QATaskRequest,
    QATaskStatusResponse,
    RagProgressEvent,
)
from backend.app.services.audit_service import record_event
from backend.app.services.rag_service import answer_question
from backend.app.services.retrieval_service import RetrievalServiceUnavailable
from backend.app.services.performance_metrics import RequestBudgetExceeded, request_budget_context, model_call_budget


logger = logging.getLogger(__name__)

_QUEUE: Queue[str] = Queue(maxsize=max(1, QA_QUEUE_CAPACITY))
_PENDING: set[str] = set()
_LOCK = Lock()
_STARTED = False


class QATaskCancelled(Exception):
    """Stop the durable QA pipeline after a user cancellation request."""


def start_qa_task_worker() -> None:
    """Start one bounded worker and resume durable queued QA tasks."""

    global _STARTED
    newly_started = False
    with _LOCK:
        if not _STARTED:
            Thread(target=_worker_loop, name="regumate-qa-worker", daemon=True).start()
            _STARTED = True
            newly_started = True
    if newly_started:
        _recover_interrupted_tasks()
    _fill_queue_from_db()


def create_qa_task(payload: QATaskRequest) -> QATaskCreateResponse:
    """Create or recover the task identified by the browser request id."""

    task: QATask | None = None
    with SessionLocal() as db:
        task = db.scalar(
            select(QATask).where(QATask.client_request_id == payload.client_request_id)
        )
        if task is None:
            task = QATask(
                task_id=uuid4().hex,
                client_request_id=payload.client_request_id,
                question=payload.question.strip(),
                options_json=json.dumps(payload.options, ensure_ascii=False),
                include_debug=payload.include_debug,
                status="queued",
                progress_json="[]",
            )
            db.add(task)
            try:
                db.commit()
            except IntegrityError:
                db.rollback()
                task = db.scalar(
                    select(QATask).where(
                        QATask.client_request_id == payload.client_request_id
                    )
                )
                if task is None:
                    raise
        response = QATaskCreateResponse(
            task_id=task.task_id,
            client_request_id=payload.client_request_id,
            status=task.status,
        )
    start_qa_task_worker()
    if response.status in {"queued", "running"}:
        _schedule_in_memory(response.task_id)
    return response


def get_qa_task_status(db: Session, task_id: str) -> QATaskStatusResponse | None:
    task = db.get(QATask, task_id)
    if task is None:
        return None
    return _task_to_response(task)


def cancel_qa_task(db: Session, task_id: str) -> QATaskStatusResponse | None:
    """Cancel a queued/running task and make the operation idempotent."""

    task = db.get(QATask, task_id)
    if task is None:
        return None
    if task.status in {"queued", "running"}:
        task.status = "cancelled"
        task.answer_preview_json = None
        task.error = None
        task.error_code = None
        task.error_stage = None
        task.error_retryable = False
        task.completed_at = _now()
        task.updated_at = task.completed_at
        db.commit()
        db.refresh(task)
        try:
            record_event(
                db,
                "qa_cancelled",
                "question",
                task.task_id,
                detail=f"用户停止生成。问题：{task.question}",
                severity="info",
                event_key=f"qa_cancelled:{task.task_id}",
                summary="问答生成已停止",
                user_message="用户主动停止了本次回答生成。",
                details={"question": task.question, "status": "cancelled"},
            )
        except Exception:
            db.rollback()
            logger.exception("Could not persist cancellation audit for %s", task.task_id)
        db.refresh(task)
    return _task_to_response(task)


def list_recent_qa_task_statuses(db: Session, limit: int = 5) -> list[QATaskStatusResponse]:
    safe_limit = min(max(limit, 1), 20)
    tasks = db.scalars(
        select(QATask).order_by(QATask.created_at.desc()).limit(safe_limit)
    ).all()
    return [_task_to_response(task) for task in tasks]


def qa_task_status_counts() -> dict[str, int]:
    with SessionLocal() as db:
        rows = db.execute(select(QATask.status, QATask.task_id)).all()
    counts: dict[str, int] = {}
    for status, _ in rows:
        counts[str(status)] = counts.get(str(status), 0) + 1
    counts["queue_capacity"] = _QUEUE.maxsize
    counts["queue_depth"] = _QUEUE.qsize()
    return counts


def _worker_loop() -> None:
    while True:
        try:
            task_id = _QUEUE.get(timeout=5)
        except Empty:
            try:
                _fill_queue_from_db()
            except Exception:
                logger.exception("Failed to refill QA queue")
            continue
        try:
            _run_task(task_id)
        except Exception as exc:
            logger.exception("Unexpected QA worker failure for %s", task_id)
            _record_unexpected_failure(task_id, exc)
        finally:
            with _LOCK:
                _PENDING.discard(task_id)
            _QUEUE.task_done()
        try:
            _fill_queue_from_db()
        except Exception:
            logger.exception("Failed to refill QA queue")


def _run_task(task_id: str) -> bool:
    with SessionLocal() as db:
        task = db.get(QATask, task_id)
        if task is None or task.status not in {"queued", "running"}:
            return False
        previous_attempts = int(task.attempt_count or 0)
        budgets = [e['summary']['_runtime_budget'] for e in _json_events(task.progress_json) if _is_budget_event(e)]
        task.status = "running"
        task.answer_preview_json = None
        task.attempt_count = int(task.attempt_count or 0) + 1
        task.error = None
        task.error_code = None
        task.error_stage = None
        task.error_retryable = False
        task.updated_at = _now()
        db.commit()

        options = _json_list(task.options_json)

        def progress_reporter(event: dict[str, Any]) -> None:
            _raise_if_task_cancelled(db, task_id)
            _append_progress_event(db, task_id, event)

        def answer_preview_reporter(preview: QAAnswerPreview) -> None:
            _raise_if_task_cancelled(db, task_id)
            _save_answer_preview(db, task_id, preview)

        def cancellation_checker() -> None:
            _raise_if_task_cancelled(db, task_id)

        try:
            if not budgets and previous_attempts:
                raise RequestBudgetExceeded('旧任务未记录已用运行预算，请重新提问。',
                                            stage='task.recovery', code='budget_not_recorded')
            budget = budgets[-1] if budgets else {'started_at': _now().isoformat(), 'model_calls': 0}
            category = str(budget.get('category') or 'standard')
            total_seconds = float(
                budget.get('total_seconds')
                or (QA_COMPLEX_REQUEST_TOTAL_BUDGET_SECONDS if category == 'complex'
                    else QA_REQUEST_TOTAL_BUDGET_SECONDS)
            )
            started_at_iso = str(budget['started_at'])
            elapsed = max(0.0, (_now()-datetime.fromisoformat(started_at_iso)).total_seconds())
            remaining = total_seconds-elapsed
            if remaining <= 0:
                raise RequestBudgetExceeded('恢复的任务已超过运行预算，请重新提问。', stage='task.recovery')
            def save_budget(calls: int, *, state=None) -> None:
                from backend.app.services.performance_metrics import request_budget_snapshot
                snapshot = request_budget_snapshot(model_calls=calls)
                runtime_category = getattr(
                    state, 'category',
                    snapshot['category'] if snapshot.get('category') != 'unknown' else category,
                )
                runtime_total = getattr(
                    state, 'total_seconds',
                    snapshot['total_seconds'] if snapshot.get('total_seconds') is not None else total_seconds,
                )
                runtime = {
                    'started_at': started_at_iso,
                    'model_calls': calls,
                    'category': runtime_category,
                    'total_seconds': runtime_total,
                    'upgraded': bool(getattr(state, 'upgraded', snapshot.get('upgraded', False))),
                }
                _append_progress_event(db, task_id, {'stage': 'planning', 'status': 'completed', 'title': '任务状态已保存', 'detail': '',
                    'summary': {'_runtime_budget': runtime}})
            def save_budget_state(state) -> None:
                from backend.app.services.performance_metrics import model_call_count
                save_budget(model_call_count(), state=state)
            save_budget(int(budget['model_calls']))
            with request_budget_context(
                total_seconds,
                elapsed_seconds=elapsed,
                category=category,
                on_change=save_budget_state,
            ), model_call_budget(calls=int(budget['model_calls']), on_reserve=save_budget):
                response = answer_question(
                    db,
                    task.question,
                    options=options,
                    include_debug=task.include_debug,
                    progress_reporter=progress_reporter,
                    answer_preview_reporter=answer_preview_reporter,
                    cancellation_checker=cancellation_checker,
                )
            if response.answer_type == "service_error":
                _fail_task(db, task_id, response.answer or "问答服务暂不可用。",
                           code=response.refusal_reason or "service_unavailable", stage="llm_generation",
                           title="问答服务暂不可用", retryable=False)
                return False
            db.expire_all()
            task = db.get(QATask, task_id)
            if task is None:
                return False
            if task.status == "cancelled":
                return False
            task.answer_json = response.model_dump_json()
            task.answer_preview_json = None
            task.status = "refused" if response.refused else "completed"
            task.error = None
            task.error_code = None
            task.error_stage = None
            task.error_retryable = False
            task.completed_at = _now()
            task.updated_at = task.completed_at
            db.commit()
            return False
        except QATaskCancelled:
            db.rollback()
            return False
        except RequestBudgetExceeded as exc:
            db.expire_all()
            task = db.get(QATask, task_id)
            if task is None or task.status == 'cancelled':
                return False
            _fail_task(db, task_id, str(exc), code=exc.code, stage=exc.stage,
                       title='问答请求已达到运行预算', retryable=False)
            return False
        except RetrievalServiceUnavailable as exc:
            db.expire_all()
            task = db.get(QATask, task_id)
            if task is None or task.status == "cancelled":
                return False
            # A retry of the whole task would silently obtain a new call/deadline budget.
            retryable = False
            _fail_task(
                db,
                task_id,
                str(exc),
                code="retrieval_unavailable",
                stage="retrieval",
                title="检索系统暂不可用",
                retryable=retryable,
            )
            return retryable
        except Exception:
            db.expire_all()
            task = db.get(QATask, task_id)
            if task is None or task.status == "cancelled":
                return False
            logger.exception("QA task %s failed", task_id)
            _fail_task(
                db,
                task_id,
                "问答任务执行失败。",
                code="qa_task_failed",
                stage="llm_generation",
                title="问答任务执行失败",
                retryable=False,
            )
            return False


def _append_progress_event(db: Session, task_id: str, event: dict[str, Any]) -> None:
    task = db.get(QATask, task_id)
    if task is None:
        return
    events = _json_events(task.progress_json)
    events.append(event)
    # Bound task snapshots so a long diagnostic run cannot grow without limit.
    budget_events = [item for item in events if _is_budget_event(item)]
    public_events = [item for item in events if not _is_budget_event(item)]
    task.progress_json = json.dumps(budget_events[-1:] + public_events[-99:], ensure_ascii=False)
    task.updated_at = _now()
    db.commit()


def _save_answer_preview(db: Session, task_id: str, preview: QAAnswerPreview) -> None:
    task = db.get(QATask, task_id)
    if task is None or task.status == "cancelled":
        return
    task.answer_preview_json = preview.model_dump_json()
    task.updated_at = _now()
    db.commit()


def _raise_if_task_cancelled(db: Session, task_id: str) -> None:
    db.expire_all()
    task = db.get(QATask, task_id)
    if task is not None and task.status == "cancelled":
        raise QATaskCancelled()


def _fail_task(
    db: Session,
    task_id: str,
    detail: str,
    *,
    code: str,
    stage: str,
    title: str,
    retryable: bool,
) -> None:
    public_stage = stage if stage in {'planning', 'retrieval', 'rerank', 'context_selection',
                                      'prompt_build', 'llm_generation', 'grounding_validation'} else (
        'llm_generation' if stage == 'model_call' else 'planning' if stage == 'task.recovery' else 'prompt_build')
    _append_progress_event(
        db,
        task_id,
        {
            "stage": public_stage,
            "status": "failed",
            "title": title,
            "detail": detail,
        },
    )
    task = db.get(QATask, task_id)
    if task is None:
        return
    task.status = "queued" if retryable else "failed"
    task.answer_preview_json = None
    task.error = detail
    task.error_code = code
    task.error_stage = stage
    task.error_retryable = retryable
    task.completed_at = None if retryable else _now()
    task.updated_at = _now()
    db.commit()


def _recover_interrupted_tasks() -> None:
    with SessionLocal() as db:
        tasks = db.scalars(
            select(QATask).where(QATask.status.in_(("queued", "running")))
        ).all()
        for task in tasks:
            task.status = "queued"
            task.answer_preview_json = None
            task.error = "应用重启后正在恢复未完成的问答任务。"
            task.error_code = None
            task.error_stage = None
            task.error_retryable = False
            task.completed_at = None
            task.updated_at = _now()
        db.commit()


def _fill_queue_from_db() -> None:
    available = _QUEUE.maxsize - _QUEUE.qsize()
    if available <= 0:
        return
    with SessionLocal() as db:
        task_ids = db.scalars(
            select(QATask.task_id)
            .where(QATask.status == "queued")
            .order_by(QATask.updated_at.asc(), QATask.created_at.asc())
            .limit(available)
        ).all()
    for task_id in task_ids:
        _schedule_in_memory(task_id)


def _schedule_in_memory(task_id: str) -> bool:
    with _LOCK:
        if task_id in _PENDING:
            return True
        try:
            _QUEUE.put_nowait(task_id)
        except Full:
            return False
        _PENDING.add(task_id)
        return True


def _record_unexpected_failure(task_id: str, exc: Exception) -> None:
    try:
        with SessionLocal() as db:
            task = db.get(QATask, task_id)
            if task is None:
                return
            if task.status == "cancelled":
                return
            task.status = "failed"
            task.answer_preview_json = None
            task.error = "问答任务执行失败。"
            task.error_code = "qa_worker_failed"
            task.error_stage = "llm_generation"
            task.error_retryable = False
            task.completed_at = _now()
            task.updated_at = task.completed_at
            db.commit()
    except Exception:
        logger.exception("Could not persist QA worker failure: %s", exc)


def _task_to_response(task: QATask) -> QATaskStatusResponse:
    answer = None
    if task.answer_json:
        try:
            answer = QAResponse.model_validate_json(task.answer_json)
        except ValueError:
            answer = None
    progress_events = []
    for event in _json_events(task.progress_json):
        if not isinstance(event, dict) or _is_budget_event(event):
            continue
        try:
            progress_events.append(RagProgressEvent.model_validate(event))
        except ValueError:
            progress_events.append(RagProgressEvent(stage='prompt_build', status='skipped',
                title='历史阶段状态未记录', detail=str(event.get('detail') or ''),
                summary={'recorded_stage': event.get('stage')}))
    answer_preview = None
    if task.answer_preview_json:
        try:
            answer_preview = QAAnswerPreview.model_validate_json(task.answer_preview_json)
        except ValueError:
            answer_preview = None
    error = None
    if task.error_code:
        error = ApiError(
            code=task.error_code,
            message=task.error or "问答任务执行失败。",
            stage=task.error_stage,
            retryable=bool(task.error_retryable),
            request_id=task.client_request_id or task.task_id,
        )
    return QATaskStatusResponse(
        task_id=task.task_id,
        client_request_id=task.client_request_id,
        question=task.question,
        options=_json_list(task.options_json),
        include_debug=task.include_debug,
        status=task.status,
        progress_events=progress_events,
        answer_preview=answer_preview,
        answer=answer,
        error=error,
        created_at=_iso(task.created_at),
        updated_at=_iso(task.updated_at),
        completed_at=_iso(task.completed_at) if task.completed_at else None,
    )


def _json_events(raw: str | None) -> list[dict[str, Any]]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def _json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [str(item) for item in value] if isinstance(value, list) else []


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _is_budget_event(event: dict[str, Any]) -> bool:
    summary = event.get('summary')
    return isinstance(summary, dict) and isinstance(summary.get('_runtime_budget'), dict)
