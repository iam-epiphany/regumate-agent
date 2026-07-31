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

The authoring tool now normalizes empty/NaN/NaT values to an empty string before label selection. The first run remains unchanged. Refusal-code compatibility also requires a generic contract test; the safe package is intentionally insufficient to infer expected private codes.

The safe evidence supports a narrower classification than the initial continuation note: 70 answerable questions contain `nan` and are candidate invalid cases; only those may eventually be replaced after case-level confirmation. The other 10 answerable questions contain usable labels but were all refused without citations, so they are real Round 1 system failures and must remain in the regression set. The 20 refusal cases also remain; their failures cannot yet be attributed to the system or to the evaluator contract without private-key-independent contract testing.

## Required continuation

1. Add generic authoring tests for blank labels and a runner/scorer refusal-contract test using synthetic-only fixtures.
2. Confirm invalidity only for the 70 malformed cases. If confirmed, produce audited substitutes solely for those cases; do not replace valid system-failure cases.
3. Diagnose and repair the generic table-retrieval/refusal behavior using only safe diagnostics, then run the retained valid cases as a Round 1 regression separately.
4. Run the full quality gate and record the new state/handoff before a valid Round 1 completion commit.
