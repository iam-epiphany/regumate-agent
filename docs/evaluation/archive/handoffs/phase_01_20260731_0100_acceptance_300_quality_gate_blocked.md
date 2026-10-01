# Phase 1 handoff — official acceptance achieved, quality gate blocked

## Current state

- Phase: `phase_01_baseline_and_no_hardcode`
- Status: `blocked`
- Next phase: prohibited. No new 100-question set was generated.
- Valid completion commit: none.

## Completed evidence

- Fresh GPU acceptance was run from Q001 with one locked run identity, real `/api/qa/ask`, SQLite, Qdrant and the configured DeepSeek chain.
- Result: `300/300`, `runtime_error=0`, completion status `acceptance_passed`.
- Artifact: `data/evaluation/phase_01/acceptance_gpu_full_20260731_0200/official_300_results.json`.
- The preceding general regression set was `29/29`, zero errors: `data/evaluation/phase_01/regression_29_gpu_v4_20260731_0155/contest_qa_all_results.json`.
- CUDA preflight recorded `NVIDIA GeForce RTX 5070 Laptop GPU`; CPU was not used for either formal run.

## General fixes verified

- Spreadsheet calculations retain explicitly selected generic columns after selector resolution.
- Choice answers use evidence-supported deterministic recovery without question-specific facts or answer maps.
- Request/checkpoint handling persists every case and tolerates transient Windows `os.replace` access denial by retrying the atomic replacement while preserving the prior checkpoint.
- Query embedding and reranking use bounded inference batches so a broad request cannot monopolize the model channel.

## Validation performed

- `python -m pytest backend/app/tests -q` — 438 passed (one third-party deprecation warning).
- Full quality gate report: `outputs/evaluation/quality_gate/20260731_005844/report.json`.
- Passed: baseline identity, no-hardcoding audit, secret scan, evaluation isolation, backend tests, frontend lint, frontend test, frontend build.

## Blocking condition

`data/evaluation/hard_challenge_50/round_1/lock.json` declares build-audit SHA-256 `95367c97f59830748e63ea13869b55578ab7a336e691c1e43b49cc2fc46ea217`, but no file with that hash exists in the frozen directory and Git history contains no recoverable version. The full gate therefore fails `frozen_artifacts`.

Do not manufacture a file, alter the lock, or weaken the verifier. Obtain the original artifact through authorized source control or its artifact owner, retain its bytes unchanged, then rerun the full gate. Because the run identity captures the worktree, perform a fresh Q001 GPU acceptance after the gate-relevant working tree is final before marking Phase 1 complete.

Recovery audit performed on 2026-07-31: the reachable local history, remote `origin/main` and `origin/dev`, and visible `D:\Agent-Project` build-audit files contain no file with the declared SHA-256. A broad local Git dangling-object scan was also attempted but could not complete within its bounded read-only window because of repository object volume. No evidence was changed during these checks.

## Process state at handoff

- Official runner: completed; no runner process remains.
- API app/Qdrant: still running as the controlled GPU Docker Compose service; no unbounded evaluation task is active.
