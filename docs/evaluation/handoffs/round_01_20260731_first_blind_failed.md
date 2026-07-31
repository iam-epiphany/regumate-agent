# Round 1 first blind run — failed and preserved

## Outcome

Round 1's first blind run completed against the frozen public 100-case file and is retained at `data/evaluation/generalization_100/round_01/first_blind_outputs.jsonl`. The independent scorer wrote only the safe package at `data/evaluation/generalization_100/round_01/first_blind_safe_diagnostic.json`; it reported `0/100`.

This is a failed first blind run, not a successful Round 1 result and not a regression score. No private gold answer, source, evidence text, cell or expected value was used during error analysis.

## Evidence retained

- Public questions and current integrity lock: `data/evaluation/generalization_100/round_01/`.
- Preliminary pre-run lock: `lock.pre_first_run_invalidated.json`. It is retained because the runner was corrected before any API call so that it records API refusal fields; the subsequent lock is the one used by the first run.
- Private-side independent audit and integrity reports remain under the sibling private evaluation root.
- The initial full quality gate passed before Round 1: `outputs/evaluation/quality_gate/20260731_090857/report.json`.

## Safe root-cause attribution

The safe diagnostic shows that many answerable public questions contain the literal `nan` as a row or column label and that their expected evidence was not retrieved. This is a general spreadsheet-authoring bug: empty pandas cells were converted with `str(value)` before candidate selection. It is not a production RAG issue and is not tied to any case, title, answer or document identifier.

The authoring tool now normalizes empty/NaN/NaT values to an empty string before label selection. The first run remains unchanged. Refusal-code compatibility also requires a generic contract test before replacement cases are produced; the safe package is intentionally insufficient to infer expected private codes.

## Required continuation

1. Add generic authoring tests for blank labels and a runner/scorer refusal-contract test using synthetic-only fixtures.
2. Generate replacement cases using the repaired generator; independently audit, freeze and first-run them as replacement evidence without modifying this case set, lock, outputs or diagnostic.
3. Apply only safe diagnostics to any production fix, then run Round 1 regression separately.
4. Run the full quality gate and record the new state/handoff before a valid Round 1 completion commit.
