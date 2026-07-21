from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark BGE-M3 and BGE reranker without retrieval or external LLM latency."
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--performance-mode", choices=["gpu", "cpu_balanced", "cpu_low_resource"])
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--torch-interop-threads", type=int, default=1)
    parser.add_argument("--embedding-batches", default="1,4,8,16")
    parser.add_argument("--rerank-batches", default="2,4,8,24")
    parser.add_argument("--candidate-counts", default="1,4,8,16,24")
    parser.add_argument("--input-lengths", default="128,384,768")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    os.environ["MODEL_DEVICE"] = args.device
    os.environ["REGUMATE_PERFORMANCE_MODE"] = args.performance_mode or (
        "gpu" if args.device == "cuda" else "cpu_balanced"
    )
    os.environ["MODEL_BACKEND"] = "pytorch"
    os.environ["QUERY_EMBEDDING_CACHE_BYTES"] = "0"
    os.environ["RERANK_SCORE_CACHE_BYTES"] = "0"
    if args.torch_threads > 0:
        value = str(args.torch_threads)
        os.environ["TORCH_NUM_THREADS"] = value
        os.environ["OMP_NUM_THREADS"] = value
        os.environ["MKL_NUM_THREADS"] = value
    os.environ["TORCH_NUM_INTEROP_THREADS"] = str(args.torch_interop_threads)

    from backend.app.core import config
    from backend.app.core.performance_profile import resolve_performance_profile
    from backend.app.services.embedding_service import embed_texts
    from backend.app.services.model_device_service import get_model_device_info
    from backend.app.services.rerank_service import rerank_candidates
    from backend.app.services.vector_store_service import VectorSearchResult

    device = get_model_device_info()
    profile = resolve_performance_profile(device.selected_device)
    base_query = "商业银行资本充足率监管要求如何计算并引用正确制度依据？"
    input_lengths = _int_list(args.input_lengths)
    texts = [_text_of_length(base_query, length) for length in input_lengths]

    embed_texts([base_query], batch_size=1)
    warmup_candidate = _candidate("warmup", texts[-1])
    rerank_candidates(question=base_query, candidates=[warmup_candidate], limit=1)

    embedding_results: list[dict[str, Any]] = []
    for batch_size in _int_list(args.embedding_batches):
        batch = [_text_of_length(f"{base_query}{index}", input_lengths[index % len(input_lengths)]) for index in range(batch_size)]
        embedding_results.append(
            _run_case(
                name=f"embedding_batch_{batch_size}",
                function=lambda batch=batch, batch_size=batch_size: embed_texts(batch, batch_size=batch_size),
                warmup=args.warmup,
                repeats=args.repeats,
                dimensions={"batch_size": batch_size},
            )
        )

    rerank_results: list[dict[str, Any]] = []
    for candidate_count in _int_list(args.candidate_counts):
        for input_length in input_lengths:
            candidates = [
                _candidate(f"chunk-{candidate_count}-{input_length}-{index}", _text_of_length(base_query, input_length))
                for index in range(candidate_count)
            ]
            for batch_size in _int_list(args.rerank_batches):
                # Override the resolved profile for this single-process microbenchmark.
                config.RERANK_BATCH_SIZE = batch_size
                rerank_results.append(
                    _run_case(
                        name=f"rerank_c{candidate_count}_l{input_length}_b{batch_size}",
                        function=lambda candidates=candidates: rerank_candidates(
                            question=base_query,
                            candidates=candidates,
                            limit=len(candidates),
                        ),
                        warmup=args.warmup,
                        repeats=args.repeats,
                        dimensions={
                            "candidate_count": candidate_count,
                            "input_length_chars": input_length,
                            "batch_size": batch_size,
                        },
                    )
                )

    artifact = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": device.to_debug_dict(),
        "profile": profile.to_dict(),
        "configuration": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "torch_threads": args.torch_threads,
            "torch_interop_threads": args.torch_interop_threads,
        },
        "embedding": embedding_results,
        "reranker": rerank_results,
        "script_sha256": _sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "profile": artifact["profile"]}, ensure_ascii=False))
    return 0


def _candidate(chunk_id: str, text: str):
    from backend.app.services.vector_store_service import VectorSearchResult

    return VectorSearchResult(
        chunk_id=chunk_id,
        document_id="benchmark-document",
        filename="benchmark.md",
        section_title="性能测试",
        page_number=1,
        text=text,
        embedding_text=text,
        token_count=max(1, len(text) // 2),
        score=0.5,
    )


def _run_case(
    *,
    name: str,
    function: Callable[[], Any],
    warmup: int,
    repeats: int,
    dimensions: dict[str, Any],
) -> dict[str, Any]:
    for _ in range(warmup):
        function()
    values: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        function()
        values.append((time.perf_counter_ns() - started) / 1_000_000)
    return {"name": name, **dimensions, "elapsed_ms": _summary(values)}


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "avg": round(statistics.mean(values), 3),
        "p50": round(statistics.median(values), 3),
        "p95": round(ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)], 3),
        "max": round(max(values), 3),
    }


def _text_of_length(seed: str, length: int) -> str:
    repeated = (seed + "；") * max(1, math.ceil(length / (len(seed) + 1)))
    return repeated[:length]


def _int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("benchmark matrix values must be positive integers")
    return values


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
