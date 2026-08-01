# Phase 2 completed — evaluation pipeline

- Phase 2 implementation commit: `17e24e3455cc6b4d7ac658ca15f47fa757189c5c`.
- Scope: evaluation-only tooling; production QA, model configuration, SQLite, Qdrant, CUDA, official QA and frozen Phase 1 artifacts were not changed.
- Public root: `data/evaluation/generalization_100/`; private root: sibling `../ReguMate-Eval-Private/generalization_100/`.
- Isolation: process isolation only. The private root is physically outside the repository but remains readable by this local account; this is explicitly not represented as strong permission isolation.

## Delivered controls

- Proposal filtering performs normalized-string, lexical/character semantic, evidence/conclusion and reasoning-structure duplicate checks.
- Generator projection writes only public question metadata. Auditor independently verifies schema, corpus references, refusal completeness and Decimal calculations.
- Freezing records public/private/scorer/corpus hashes. Verification rejects changed artifacts. Runner reads only the public question file and protects first-run output paths from overwrite.
- Offline scorer reads the private key and writes a safe diagnostic package without answers, answer numbers, evidence text, source names or locations.

## Validation

- A two-case synthetic-only validation was generated outside all formal rounds: one answerable and one refusal case. It passed audit, freezing, integrity verification and scoring (2/2). It is not a Round 1 result.
- `python -m pytest backend/app/tests/test_generalization_pipeline.py -q` passed.
- The full quality gate is run before this phase's valid commit; it re-verifies the existing Phase 1 GPU acceptance rather than rerunning DeepSeek because production was unchanged.

## Boundary for Round 1

No Round 1 question, gold key or first-run result exists. The authoring/audit process must generate and independently audit a fresh 100-case set before it is frozen. After freezing, optimisation may consume only the safe diagnostic package.
