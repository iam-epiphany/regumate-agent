"""Server-owned RAG capabilities. No dataset identifiers or scoring inputs.

Unset configuration is exactly the production pipeline. Experimental processes
must use a separate database and a collection explicitly reserved for experiments.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import asdict, dataclass
from functools import lru_cache, wraps
import inspect
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class RagFeatures:
    hybrid: bool = True
    rerank: bool = True
    structured_index: bool = True
    supplements: bool = True
    neighbors: bool = True
    references: bool = True
    coverage_gate: bool = True
    semantic: bool = True
    calculation: bool = True
    chunking: str = "structured"
    contextual_embedding: bool = True


@lru_cache(maxsize=1)
def features() -> RagFeatures:
    raw = os.getenv("REGUMATE_RAG_FEATURES")
    if not raw:
        return RagFeatures()
    values = json.loads(raw)
    if not isinstance(values, dict) or set(values) - set(RagFeatures.__dataclass_fields__):
        raise ValueError("Unknown RAG feature configuration")
    for key, value in values.items():
        if key == "chunking":
            if value not in {"structured", "recursive"}:
                raise ValueError("Invalid chunking mode")
        elif type(value) is not bool:
            raise ValueError(f"RAG feature {key} must be boolean")
    db_dir = Path(os.getenv("REGUMATE_DB_DIR", "")).resolve()
    root = Path(os.environ["REGUMATE_EXPERIMENT_ROOT"]).resolve()
    if not db_dir.is_relative_to(root) or db_dir == root:
        raise ValueError("Experiments require a database below the experiment root")
    if not os.getenv("QDRANT_COLLECTION", "").startswith("regumate_ablation_"):
        raise ValueError("Experiments require an isolated Qdrant collection")
    return RagFeatures(**values)


_TRACE: ContextVar[dict | None] = ContextVar("rag_experiment_trace", default=None)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(type(value).__name__)


def event(kind: str, **values: Any) -> None:
    trace = _TRACE.get()
    if trace is not None:
        # Snapshot now: downstream repair often mutates the same answer object.
        trace["events"].append(json.loads(json.dumps(
            {"kind": kind, **values}, ensure_ascii=False, default=_jsonable,
        )))


def feature_gate(name: str, *, empty: str = "list", passthrough: str | None = None):
    """Gate at the implementation entry, including callers through re-exports."""
    def decorate(function):
        signature = inspect.signature(function)
        @wraps(function)
        def wrapped(*args, **kwargs):
            enabled = getattr(features(), name)
            if not enabled:
                event("module", module=name, function=function.__name__, enabled=False)
                if passthrough:
                    return signature.bind(*args, **kwargs).arguments[passthrough]
                return [] if empty == "list" else None
            result = function(*args, **kwargs)
            result_ids = [getattr(getattr(item, "citation", item), "chunk_id", None)
                          for item in result] if isinstance(result, (list, tuple)) else []
            event("module", module=name, function=function.__name__, enabled=True,
                  result_count=len(result) if isinstance(result, (list, tuple)) else int(result is not None),
                  chunk_ids=[value for value in result_ids if value])
            return result
        return wrapped
    return decorate


def capture_validation(function):
    signature = inspect.signature(function)
    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs).arguments
        result = function(*args, **kwargs)
        event("validation", answer=bound["answer"], claims=bound["claims"],
              evidence=bound["context_chunks"], verdict=result)
        return result
    return wrapped


def capture_question(function):
    signature = inspect.signature(function)
    @wraps(function)
    def wrapped(*args, **kwargs):
        path = os.getenv("REGUMATE_EXPERIMENT_TRACE")
        if not path:
            return function(*args, **kwargs)
        selected = features()  # validates isolation before any query
        bound = signature.bind(*args, **kwargs).arguments
        trace = {"trace_id": uuid4().hex, "features": asdict(selected), "events": [],
                 "question": bound.get("question")}
        token = _TRACE.set(trace)
        start = perf_counter()
        try:
            result = function(*args, **kwargs)
            trace["response"] = result
            return result
        except Exception as exc:
            trace["error_type"] = type(exc).__name__
            raise
        finally:
            trace["elapsed_ms"] = (perf_counter() - start) * 1000
            _TRACE.reset(token)
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(trace, ensure_ascii=False, default=_jsonable) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
    return wrapped
