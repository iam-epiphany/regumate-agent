from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import os
import sys
from threading import RLock
from threading import Event, Thread
from time import perf_counter, perf_counter_ns, process_time
from functools import wraps
from typing import Any, Callable, Generic, Iterator, ParamSpec, TypeVar
from uuid import uuid4


T = TypeVar("T")
P = ParamSpec("P")


@dataclass
class _Metric:
    count: int = 0
    total_ms: float = 0.0
    max_ms: float = 0.0
    last_ms: float = 0.0


_METRICS: dict[str, _Metric] = {}
_METRICS_LOCK = RLock()
_CURRENT_TRACE: ContextVar[dict[str, Any] | None] = ContextVar("regumate_performance_trace", default=None)
_CURRENT_REQUEST_ID: ContextVar[str | None] = ContextVar("regumate_request_id", default=None)
_CURRENT_REQUEST_DEADLINE: ContextVar[float | None] = ContextVar("regumate_request_deadline", default=None)
_CURRENT_REQUEST_BUDGET: ContextVar["RequestBudgetState | None"] = ContextVar("regumate_request_budget", default=None)
_CURRENT_MODEL_BUDGET: ContextVar[dict[str, Any] | None] = ContextVar("regumate_model_budget", default=None)
_CURRENT_CANCELLATION: ContextVar[Any] = ContextVar("regumate_cancellation", default=None)
_TRACE_HISTORY: list[dict[str, Any]] = []
_TRACE_HISTORY_LIMIT = 50
_INFLIGHT_REQUESTS: dict[str, dict[str, Any]] = {}
_RESOURCE_LOCK = RLock()
_RESOURCE_STOP = Event()
_RESOURCE_THREAD: Thread | None = None
_RESOURCE_STATUS: dict[str, Any] = {
    "sampling": False,
    "interval_seconds": 1.0,
    "samples": 0,
    "current": {},
    "peaks": {},
    "history": [],
}


def observe_timing(name: str, elapsed_ms: float) -> None:
    with _METRICS_LOCK:
        metric = _METRICS.setdefault(name, _Metric())
        metric.count += 1
        metric.total_ms += elapsed_ms
        metric.max_ms = max(metric.max_ms, elapsed_ms)
        metric.last_ms = elapsed_ms
    trace = _CURRENT_TRACE.get()
    if trace is not None:
        stage = trace["stages"].setdefault(name, {"count": 0, "total_ms": 0.0, "max_ms": 0.0})
        stage["count"] += 1
        stage["total_ms"] += elapsed_ms
        stage["max_ms"] = max(stage["max_ms"], elapsed_ms)


@contextmanager
def measure(name: str) -> Iterator[None]:
    started = perf_counter_ns()
    try:
        yield
    finally:
        observe_timing(name, (perf_counter_ns() - started) / 1_000_000)


def timed(name: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorator(function: Callable[P, T]) -> Callable[P, T]:
        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with measure(name):
                return function(*args, **kwargs)

        return wrapper

    return decorator


def trace_operation(kind: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Capture all nested ``measure`` calls for one QA or indexing operation."""

    def decorator(function: Callable[P, T]) -> Callable[P, T]:
        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            trace: dict[str, Any] = {
                "trace_id": uuid4().hex,
                "request_id": _CURRENT_REQUEST_ID.get(),
                "kind": kind,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "stages": {},
            }
            started = perf_counter_ns()
            token = _CURRENT_TRACE.set(trace)
            try:
                result = function(*args, **kwargs)
            except Exception:
                trace["status"] = "failed"
                raise
            else:
                trace["status"] = "completed"
                return result
            finally:
                trace["total_ms"] = round((perf_counter_ns() - started) / 1_000_000, 3)
                trace["stages"] = {
                    name: {
                        "count": value["count"],
                        "total_ms": round(value["total_ms"], 3),
                        "max_ms": round(value["max_ms"], 3),
                    }
                    for name, value in sorted(trace["stages"].items())
                }
                _CURRENT_TRACE.reset(token)
                with _METRICS_LOCK:
                    _TRACE_HISTORY.append(trace)
                    del _TRACE_HISTORY[:-_TRACE_HISTORY_LIMIT]

        return wrapper

    return decorator


@contextmanager
def request_trace_context(request_id: str) -> Iterator[None]:
    """Attach an externally visible request ID to the current synchronous trace."""

    token = _CURRENT_REQUEST_ID.set(request_id)
    started_at = datetime.now(timezone.utc).isoformat()
    with _METRICS_LOCK:
        _INFLIGHT_REQUESTS[request_id] = {
            "request_id": request_id,
            "started_at": started_at,
            "last_stage": "request_received",
            "last_status": "started",
        }
    try:
        yield
    finally:
        with _METRICS_LOCK:
            _INFLIGHT_REQUESTS.pop(request_id, None)
        _CURRENT_REQUEST_ID.reset(token)


class RequestBudgetExceeded(TimeoutError):
    """Raised at cooperative boundaries before a request can become unbounded."""

    def __init__(self, message: str, *, stage: str = 'request', code: str = 'request_budget_exhausted'):
        super().__init__(message)
        self.stage = stage
        self.code = code


@dataclass
class RequestBudgetState:
    """One request's deadline, category and one-time complex upgrade.

    The state object is shared by every nested execution wrapper.  Nested
    calls can never introduce a fresh deadline; only the explicit
    ``upgrade_request_budget`` call may extend the original deadline once.
    """

    started_at: float
    deadline: float
    total_seconds: float
    category: str = "standard"
    upgraded: bool = False
    on_change: Callable[["RequestBudgetState"], None] | None = None
    started_at_iso: str = ""

    def snapshot(self, *, model_calls: int = 0, model_call_limit: int = 5) -> dict[str, Any]:
        elapsed_ms = round(max(0.0, perf_counter() - self.started_at) * 1000, 2)
        return {
            "category": "complex" if self.category == "complex" else "standard",
            "total_seconds": round(self.total_seconds, 3),
            "elapsed_ms": elapsed_ms,
            "remaining_seconds": round(max(0.0, self.deadline - perf_counter()), 3),
            "model_calls": model_calls,
            "model_call_limit": model_call_limit,
            "upgraded": self.upgraded,
        }


@contextmanager
def request_budget_context(
    total_seconds: float,
    *,
    elapsed_seconds: float = 0.0,
    category: str = "standard",
    on_change: Callable[[RequestBudgetState], None] | None = None,
) -> Iterator[None]:
    """Enter one request budget; nested contexts inherit the outer deadline.

    ``elapsed_seconds`` lets a recovered background task reconstruct the
    original start time.  It is used to compute the deadline and is never a
    new budget start.
    """

    parent_state = _CURRENT_REQUEST_BUDGET.get()
    parent_deadline = _CURRENT_REQUEST_DEADLINE.get()
    if parent_state is not None or parent_deadline is not None:
        # Re-entering the same pipeline is not a new request.  Keep the
        # existing deadline even when the nested caller asks for more time.
        deadline = parent_state.deadline if parent_state is not None else parent_deadline
        token = _CURRENT_REQUEST_DEADLINE.set(deadline)
        try:
            yield
        finally:
            _CURRENT_REQUEST_DEADLINE.reset(token)
        return

    now = perf_counter()
    started_at = now - max(0.0, float(elapsed_seconds or 0.0))
    deadline = started_at + max(0.001, float(total_seconds))
    state = RequestBudgetState(
        started_at=started_at,
        deadline=deadline,
        total_seconds=max(0.001, float(total_seconds)),
        category="complex" if category == "complex" else "standard",
        on_change=on_change,
        started_at_iso=datetime.now(timezone.utc).isoformat(),
    )
    budget_token = _CURRENT_REQUEST_BUDGET.set(state)
    deadline_token = _CURRENT_REQUEST_DEADLINE.set(deadline)
    try:
        yield
    finally:
        _CURRENT_REQUEST_DEADLINE.reset(deadline_token)
        _CURRENT_REQUEST_BUDGET.reset(budget_token)


def upgrade_request_budget(total_seconds: float, *, category: str = "complex") -> bool:
    """Apply the one permitted complex-task deadline extension.

    The new deadline is anchored at the original start time, so time already
    spent on planning, retrieval or repair is deducted rather than reset.
    """

    state = _CURRENT_REQUEST_BUDGET.get()
    if state is None or state.upgraded:
        return False
    requested = max(0.001, float(total_seconds))
    if requested <= state.total_seconds:
        return False
    state.total_seconds = requested
    state.deadline = state.started_at + requested
    state.category = "complex" if category == "complex" else state.category
    state.upgraded = True
    _CURRENT_REQUEST_DEADLINE.set(state.deadline)
    if state.on_change is not None:
        try:
            state.on_change(state)
        except Exception:  # noqa: BLE001 - observability persistence must not break answering
            pass
    record_trace_event("budget", "upgraded", {
        "category": state.category,
        "total_seconds": round(state.total_seconds, 3),
    })
    return True


def request_budget_snapshot(*, model_calls: int | None = None,
                            model_call_limit: int | None = None) -> dict[str, Any]:
    state = _CURRENT_REQUEST_BUDGET.get()
    if state is None:
        return {
            "category": "unknown",
            "total_seconds": None,
            "elapsed_ms": 0.0,
            "remaining_seconds": None,
            "model_calls": model_call_count() if model_calls is None else model_calls,
            "model_call_limit": model_call_limit or 5,
            "upgraded": False,
        }
    from backend.app.core.config import QA_MODEL_CALL_LIMIT
    return state.snapshot(
        model_calls=model_call_count() if model_calls is None else model_calls,
        model_call_limit=model_call_limit or QA_MODEL_CALL_LIMIT,
    )


def model_call_count() -> int:
    return int((_CURRENT_MODEL_BUDGET.get() or {}).get("calls", 0))


def reserve_model_call(timeout: float) -> float:
    checker = _CURRENT_CANCELLATION.get()
    if checker:
        checker()
    state = _CURRENT_MODEL_BUDGET.get()
    if state is not None:
        from backend.app.core.config import QA_MODEL_CALL_LIMIT
        if state["calls"] >= QA_MODEL_CALL_LIMIT:
            raise RequestBudgetExceeded("model call limit exhausted", stage='model_call', code='model_call_limit_exceeded')
        state["calls"] += 1
        if state.get('on_reserve'):
            state['on_reserve'](state['calls'])
        record_trace_event("model_call", "started", {"call_count": state["calls"]})
    remaining = remaining_request_budget_seconds()
    if remaining is not None and remaining <= 0:
        raise RequestBudgetExceeded("request deadline exhausted")
    return min(timeout, remaining) if remaining is not None else timeout


@contextmanager
def model_call_budget(*, calls: int = 0, on_reserve: Callable[[int], None] | None = None):
    """Restore a durable task's spent calls before entering the shared pipeline."""
    token = _CURRENT_MODEL_BUDGET.set({'calls': calls, 'on_reserve': on_reserve})
    try:
        yield
    finally:
        _CURRENT_MODEL_BUDGET.reset(token)


def qa_execution(function):
    """One budget for direct, HTTP, streaming and queued execution."""
    @wraps(function)
    def execute(*args, **kwargs):
        from backend.app.core.config import QA_REQUEST_TOTAL_BUDGET_SECONDS
        existing = _CURRENT_MODEL_BUDGET.get()
        token = _CURRENT_MODEL_BUDGET.set(existing if existing is not None else {"calls": 0})
        cancellation = _CURRENT_CANCELLATION.set(kwargs.get("cancellation_checker") or _CURRENT_CANCELLATION.get())
        try:
            with request_budget_context(QA_REQUEST_TOTAL_BUDGET_SECONDS):
                result = function(*args, **kwargs)
                if hasattr(result, "grounding_validation"):
                    result.grounding_validation["model_call_count"] = model_call_count()
                return result
        finally:
            _CURRENT_CANCELLATION.reset(cancellation)
            _CURRENT_MODEL_BUDGET.reset(token)
    return execute


def remaining_request_budget_seconds() -> float | None:
    state = _CURRENT_REQUEST_BUDGET.get()
    deadline = state.deadline if state is not None else _CURRENT_REQUEST_DEADLINE.get()
    return None if deadline is None else max(0.0, deadline - perf_counter())


def check_request_budget(stage: str) -> None:
    checker = _CURRENT_CANCELLATION.get()
    if checker:
        checker()
    remaining = remaining_request_budget_seconds()
    if remaining is not None and remaining <= 0:
        record_trace_event(stage, "timeout", {"reason": "request_budget_exhausted"})
        raise RequestBudgetExceeded(f"QA request budget exhausted during {stage}", stage=stage)


def record_trace_event(stage: str, status: str, summary: dict[str, Any] | None = None) -> None:
    """Record safe, size-only progress metadata without retaining prompts or secrets."""

    trace = _CURRENT_TRACE.get()
    if trace is None:
        return
    events = trace.setdefault("events", [])
    safe_summary = {
        str(key): value
        for key, value in (summary or {}).items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }
    now = datetime.now(timezone.utc).isoformat()
    event = {
        "stage": stage,
        "status": status,
        "at": now,
        "summary": safe_summary,
    }
    spans = trace.setdefault("stage_spans", {})
    if status in {"started", "running"}:
        previous = spans.get(stage)
        if not isinstance(previous, dict) or previous.get("end_at"):
            spans[stage] = {"start_at": now, "end_at": None, "elapsed_ms": None, "status": status}
    elif status in {"completed", "failed", "timeout", "cancelled", "skipped"}:
        span = spans.setdefault(stage, {"start_at": now, "end_at": None, "elapsed_ms": None})
        span["end_at"] = now
        span["status"] = status
        try:
            start = datetime.fromisoformat(str(span["start_at"]))
            span["elapsed_ms"] = round((datetime.fromisoformat(now) - start).total_seconds() * 1000, 3)
        except (TypeError, ValueError):
            span["elapsed_ms"] = None
    events.append(event)
    request_id = _CURRENT_REQUEST_ID.get()
    if request_id:
        with _METRICS_LOCK:
            inflight = _INFLIGHT_REQUESTS.get(request_id)
            if inflight is not None:
                inflight.update({"last_stage": stage, "last_status": status, "last_event_at": now})


def inflight_request_snapshot() -> list[dict[str, Any]]:
    """Return safe live request state for diagnosing a stalled service."""

    with _METRICS_LOCK:
        return [dict(value) for value in _INFLIGHT_REQUESTS.values()]


def timing_metrics_snapshot() -> dict[str, dict[str, float | int]]:
    with _METRICS_LOCK:
        return {
            name: {
                "count": metric.count,
                "last_ms": round(metric.last_ms, 3),
                "avg_ms": round(metric.total_ms / metric.count, 3) if metric.count else 0.0,
                "max_ms": round(metric.max_ms, 3),
            }
            for name, metric in sorted(_METRICS.items())
        }


def trace_history_snapshot(limit: int = 10) -> list[dict[str, Any]]:
    with _METRICS_LOCK:
        return [dict(item) for item in _TRACE_HISTORY[-max(0, limit):]]


def start_resource_sampling(interval_seconds: float = 1.0) -> None:
    """Sample lightweight process and PyTorch memory counters in the background."""

    global _RESOURCE_THREAD
    with _RESOURCE_LOCK:
        if _RESOURCE_THREAD is not None and _RESOURCE_THREAD.is_alive():
            return
        _RESOURCE_STOP.clear()
        _RESOURCE_STATUS.update({"sampling": True, "interval_seconds": interval_seconds})
        _RESOURCE_THREAD = Thread(
            target=_resource_sampling_loop,
            args=(max(0.1, interval_seconds),),
            name="regumate-resource-sampler",
            daemon=True,
        )
        _RESOURCE_THREAD.start()


def stop_resource_sampling() -> None:
    global _RESOURCE_THREAD
    _RESOURCE_STOP.set()
    thread = _RESOURCE_THREAD
    if thread is not None:
        thread.join(timeout=2)
    with _RESOURCE_LOCK:
        _RESOURCE_STATUS["sampling"] = False
        _RESOURCE_THREAD = None


def resource_metrics_snapshot() -> dict[str, Any]:
    with _RESOURCE_LOCK:
        return {
            **_RESOURCE_STATUS,
            "current": dict(_RESOURCE_STATUS.get("current") or {}),
            "peaks": dict(_RESOURCE_STATUS.get("peaks") or {}),
            "history": list(_RESOURCE_STATUS.get("history") or []),
        }


def _resource_sampling_loop(interval_seconds: float) -> None:
    previous_wall = perf_counter()
    previous_cpu = process_time()
    while not _RESOURCE_STOP.wait(interval_seconds):
        current_wall = perf_counter()
        current_cpu = process_time()
        wall_delta = max(current_wall - previous_wall, 1e-9)
        cpu_percent = max(0.0, (current_cpu - previous_cpu) / wall_delta * 100.0)
        previous_wall, previous_cpu = current_wall, current_cpu
        current = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "process_cpu_percent": round(cpu_percent, 2),
            "rss_bytes": _rss_bytes(),
        }
        current.update(_cuda_memory_metrics())
        with _RESOURCE_LOCK:
            _RESOURCE_STATUS["samples"] = int(_RESOURCE_STATUS.get("samples") or 0) + 1
            _RESOURCE_STATUS["current"] = current
            history = _RESOURCE_STATUS.setdefault("history", [])
            history.append(current)
            # Keep up to two hours at the default one-second cadence so a full
            # 300-case evaluation retains its complete resource envelope.
            del history[:-7200]
            peaks = _RESOURCE_STATUS.setdefault("peaks", {})
            for key, value in current.items():
                if isinstance(value, (int, float)):
                    peaks[key] = max(float(peaks.get(key) or 0), float(value))


def _rss_bytes() -> int | None:
    try:
        pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
        return pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def _cuda_memory_metrics() -> dict[str, int]:
    torch = sys.modules.get("torch")
    if torch is None:
        return {}
    try:
        if not torch.cuda.is_available():
            return {}
        return {
            "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
            "cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    except (AttributeError, RuntimeError):
        return {}


class ByteLRUCache(Generic[T]):
    """A bounded, process-local LRU with observable hit/miss/eviction counters."""

    def __init__(
        self,
        *,
        max_bytes: int,
        max_items: int,
        size_of: Callable[[T], int],
    ) -> None:
        self.max_bytes = max(0, max_bytes)
        self.max_items = max(0, max_items)
        self._size_of = size_of
        self._items: OrderedDict[str, tuple[T, int]] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = RLock()

    def get(self, key: str) -> T | None:
        with self._lock:
            item = self._items.pop(key, None)
            if item is None:
                self._misses += 1
                return None
            self._items[key] = item
            self._hits += 1
            return item[0]

    def put(self, key: str, value: T) -> None:
        if self.max_bytes <= 0 or self.max_items <= 0:
            return
        size = max(1, int(self._size_of(value)))
        if size > self.max_bytes:
            return
        with self._lock:
            previous = self._items.pop(key, None)
            if previous is not None:
                self._bytes -= previous[1]
            self._items[key] = (value, size)
            self._bytes += size
            while self._items and (
                len(self._items) > self.max_items or self._bytes > self.max_bytes
            ):
                _, (_, evicted_size) = self._items.popitem(last=False)
                self._bytes -= evicted_size
                self._evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0

    def snapshot(self) -> dict[str, int | bool]:
        with self._lock:
            return {
                "enabled": self.max_bytes > 0 and self.max_items > 0,
                "items": len(self._items),
                "bytes": self._bytes,
                "max_items": self.max_items,
                "max_bytes": self.max_bytes,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
            }
