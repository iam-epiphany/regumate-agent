# Phase 1 handoff — GPU diagnostic complete

## Run identity and status

- Result: `data/evaluation/phase_01/diagnostic_gpu_full_20260730_2340/official_300_results.json`
- Report: `data/evaluation/phase_01/diagnostic_gpu_full_20260730_2340/official_300_report.md`
- Mode/status: `diagnostic` / `diagnostic_complete`; this is not an acceptance result.
- Runtime: CUDA selected on the configured GPU. The run identity records code commit/diff, corpus snapshot, evaluator and runner hashes, QA hash, runtime GPU details, Qdrant state and endpoint.
- Executed: 300/300, with no checkpoint reuse or result splicing.
- Correct: 271/300. Runtime errors: 8. Phase 1 remains `blocked`; Phase 2 was not entered.

## Evidence-backed failure clustering

| Class | Count | Scope | Evidence | Priority / general direction |
| --- | ---: | --- | --- | --- |
| `spreadsheet_retrieval` / `table_header_or_cell_location` | 21 | Excel, table-calculation | Each failure was an evidence-preserving refusal with `table_evidence_not_found`; table retrieval and comparison cases otherwise completed. | P0: inspect structured table candidate selection, header/cell coordinate resolution and calculation operand assembly; add source-agnostic regression fixtures. |
| `answer_generation` / `environment_error` | 8 | Word, single/multi-fact | App logs show `NameError: perf_counter is not defined` in the LLM total-budget code path. | P0 fixed: import `perf_counter`; retain bounded generation budget. Re-run on a new code identity. |
| `request_timeout`, `cancellation_cleanup`, `database_or_connection_pool`, `qdrant_error` | 0 | all | The completed GPU run had zero runner timeout and no residual in-flight request; the former long-tail sample passed after the explicit-material scope fix. | Monitor in subsequent full run. |

No evidence supports fixed-question, fixed-file, fixed-answer, or fixed-regulatory-fact logic. The historical fixed-fact implementation remains absent from production.

## Repairs and verification performed after the diagnostic

- Constrained MCQ exact matching to explicitly resolved materials before considering a whole-corpus seed scan. This generic scope-first behavior removed the prior long-tail; the former sample completed on GPU in about 1.4 seconds.
- Corrected the answer-generation budget path's missing `perf_counter` import and preserved backward-compatible streaming-helper use for tests.
- Corrected the runner's native Markdown report rendering so it does not consume the legacy evaluator artifact shape.
- Verification: `python -m pytest backend/app/tests/test_rag_api.py backend/app/tests/test_official_qa_runner.py -q` => 90 passed; `python -m pytest backend/app/tests/test_answer_generation_service.py -q` => 93 passed.

## Required next action

Remain in Phase 1. Rebuild the GPU overlay after the post-diagnostic code change; resolve the 21 general spreadsheet-evidence failures with regression tests; then start a fresh complete GPU acceptance run from Q001. Acceptance may be recorded only at 300/300 with zero runtime errors plus a fresh hardcoding audit and all quality gates.

## Process state at handoff

- Official runner: completed and exited.
- API service: running under the GPU Compose overlay, healthy at last check.
- Qdrant: running and healthy at last check.
- No background diagnostic runner or unbounded request has been left running.
