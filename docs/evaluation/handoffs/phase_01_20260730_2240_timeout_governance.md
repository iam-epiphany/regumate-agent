# Phase 1 timeout-governance handoff (2026-07-30 22:40 +08:00)

Status: `blocked`; Phase 2 and new-question generation remain prohibited.

The retained checkpoints are immutable diagnostic evidence:

- `data/evaluation/phase_01/diagnostic_full_20260730_2000/` (original interrupted run);
- `data/evaluation/phase_01/diagnostic_timeout_governed_20260730_2220/` (runner timeout only);
- `data/evaluation/phase_01/diagnostic_timeout_governed_v2_20260730_2237/` (new code identity, stopped at 101).

The latest run used the real `/api/qa/ask`, SQLite, Qdrant and configured DeepSeek chain. It recorded an atomic checkpoint after every case. It reached 100 completed cases (79 correct, no runtime errors), then Q101 was persisted as a runtime timeout at 180024 ms. It must not be scored as a complete diagnostic.

Root-cause evidence: the real runtime has `MODEL_DEVICE=cuda` configured but CUDA unavailable, so it executes BGE embedding/reranking on CPU. Historic Q101--Q104 traces contain `model_lock.wait`, `model_lock.hold`, `embedding.inference` and `rerank.inference`, not checked-out SQLite connections. The generic repair added per-request IDs, trace spans, a 150-second cooperative request budget, a 30-second model-lock wait bound, and rerank batches capped at four candidates. The remaining failure is one synchronous native model forward pass that does not yield to the cooperative budget; the runner timing out cannot cancel that in-flight server thread.

Before another complete official diagnostic, introduce a genuinely killable/process-isolated or otherwise bounded model-forward execution path, with resource cleanup tests. Start a fresh 300-case diagnostic after that code/config identity changes; do not append to any checkpoint above.

At handoff, runner PID 45688 was stopped and the `regumate-app` container was stopped after logs, process data, Docker status/stats, and the checkpoint were saved in `interruption_snapshot_20260730_2240/`. Qdrant remains healthy; no uncontrolled runner or API request remains.
