# Independent 100 — first blind run and generic repair

## Evidence retained

- Frozen public batch and lock: `data/evaluation/independent_100_20260731/batch/`.
- Immutable first blind output: `first_blind_outputs.jsonl`.
- First safe diagnostic: `first_blind_safe_diagnostic.json`, scored 35/100.
- Post-fix outputs are separate regression artifacts; they do not replace the first run.  The clean post-fix regression also scored 35/100.

## Root-cause summary from safe diagnostics

- 33 failures are retrieval/citation failures concentrated in duplicate row/column labels and multi-table/period ambiguity.
- 12 failures are answer-generation/calculation failures, concentrated in cross-document and formula/table calculation cases.
- 20 refusal cases failed the scorer’s expected code in the batch runs.  Direct API verification shows the generic early-boundary refusal is active; runner interruption and existing request handling remain a separate runtime reliability concern.

## Generic repairs made

- Added early, deterministic refusal handling for explicit unsupported request boundaries before retrieval or model generation.
- Added a public-only runner with identity-checked durable checkpoints and immutable final output creation.  This prevents loss of completed cases when an external execution window expires.
- The repair preserved official 300/300 GPU acceptance and passed the full quality gate at `outputs/evaluation/quality_gate/20260731_115706/report.json`.

## Status

This batch does **not** meet the 90% target and must not be presented as a successful generalization validation.  Do not rewrite or delete it.  A future continuation must confirm which table questions are genuinely invalid because their public wording is non-unique, retain those records, and create independently audited replacement cases; valid cases must remain regression evidence.

## Focused failure gate

After the generic repair, a public-only root test confirmed that an explicit prediction request returns `unsupported_prediction` without citations.  A public table-ambiguity probe still returned an evidence refusal.  The 65 cases that failed the immutable first blind run were then rerun in `failed_65_post_fix_outputs.jsonl` and isolated scoring returned **0/65**, below the required **59/65** gate.  Per the gate, no new full-100 rerun was started.
