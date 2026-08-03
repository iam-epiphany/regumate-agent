# Independent 100-case evaluation

This evaluation is a new, standalone batch.  It does not read, amend, deduplicate against, or otherwise use any earlier frozen Round 1 questions, gold files, first-run output, or score report.

## Batch contract

- Exactly 100 cases: 80 answerable and 20 refusal/clarification cases.
- Answerable capability quotas: fact/definition 20, rule/scope 12, single-document synthesis 10, cross-document evidence 8, table lookup 12, table calculation 10, formula calculation 4, and policy/reporting joint judgement 4.
- Difficulty quotas: easy 20, medium 55, hard 25.  Difficulty comes from evidence composition, table structure, calculation, unit/period, or version scope—not wording traps.
- At least 40 distinct official-corpus documents support answerable cases.  Each case is relevant to a bank reporting, compliance, risk, finance, or supervisory workflow.

## Isolation and lifecycle

1. The authoring process reads only the official 500-document corpus and writes full candidate/gold records below the sibling private root.
2. An independent auditor verifies source existence, table cell coordinates, Decimal calculations, units, periods, formula inputs, answerability and refusal boundary.
3. `generalization_pipeline.py validate-batch`, `audit`, and `freeze` must all pass before public questions are frozen.
4. Public `questions.jsonl` may contain only case ID, question text, answerable flag, question type, difficulty, and business relevance.  Answers, evidence, paths, cells, formulas, and refusal keys stay private.
5. The runner reads only public questions.  The scorer runs separately with private gold and emits only safe diagnostic fields.  A first-run output path is immutable.

## Acceptance order

The app must be CUDA-ready before any formal blind run.  The current rigorous
100-case batch is repaired and summarized first.  A second, independently
authored and audited 100-case batch is then frozen and blind-tested.  The
official 300-case acceptance is intentionally last and must not run until the
second batch reaches its acceptance gate.  CPU execution is diagnostic-only.
Repairs are based solely on safe failure categories and must be general
capabilities rather than case, document, answer, or evaluation-specific
branches.

## Rigorous v3 regression checkpoint

The immutable rigorous-v3 first blind run scored 80/100.  A separate regression
of its 20 failed cases recovered 12 cases.  Merging those regression outputs
with the 80 unchanged first-run passes produced a 92/100 checkpoint with a
93.68% required-source evidence hit rate.  The regression does not replace the
first blind evidence.

Capability results at this checkpoint are: fact/definition 13/15, rule/scope
12/12, single-document synthesis 8/8, cross-document evidence 5/10, table
lookup 15/15, table calculation 9/10, formula calculation 5/5,
policy/reporting joint judgement 5/5, and refusal/clarification 20/20.

The remaining cross-document failures must not be converted into source- or
answer-specific production rules.  In particular, one frozen case uses a
truncated clause that appears twice in the same source document with different
continuations.  That ambiguity is retained as an audit finding and is not used
to force the retriever toward either occurrence.
