from threading import RLock
from time import perf_counter_ns

from backend.app.services.performance_metrics import observe_timing
from backend.app.core.config import MODEL_INFERENCE_LOCK_WAIT_SECONDS
from backend.app.services.performance_metrics import check_request_budget, remaining_request_budget_seconds


# BGE-M3 and the reranker can share one GPU. Serializing their forward passes
# prevents concurrent upload indexing and QA requests from exhausting VRAM.
class InstrumentedRLock:
    def __init__(self) -> None:
        self._lock = RLock()
        self._acquired_at = 0

    def __enter__(self) -> "InstrumentedRLock":
        started = perf_counter_ns()
        # Waiting for the lock is queueing, not inference: it must not consume
        # the request budget, or a slow concurrent neighbour would fail this
        # request while it is merely waiting its turn.  The budget is checked
        # after the lock is acquired, immediately before the model pass.
        remaining = remaining_request_budget_seconds()
        timeout = MODEL_INFERENCE_LOCK_WAIT_SECONDS if remaining is None else min(MODEL_INFERENCE_LOCK_WAIT_SECONDS, remaining)
        if not self._lock.acquire(timeout=max(0.001, timeout)):
            raise TimeoutError("Timed out waiting for the shared model inference lock")
        check_request_budget("model_lock.wait")
        observe_timing("model_lock.wait", (perf_counter_ns() - started) / 1_000_000)
        self._acquired_at = perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        observe_timing("model_lock.hold", (perf_counter_ns() - self._acquired_at) / 1_000_000)
        self._lock.release()


MODEL_INFERENCE_LOCK = InstrumentedRLock()
