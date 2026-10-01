# Round 1 continuation — generic fix incomplete

## Preserved evidence and isolation

The failed first blind run, its public question set, locks, raw API outputs and safe diagnostic remain unchanged. This continuation read only the public questions, safe diagnostic, production source, public corpus and runtime index metadata. It did not read the sibling private evaluation directory, `gold.jsonl`, private audit output, expected answers, source evidence locations or expected values.

## Work completed

- Ran the Full quality gate successfully before modifying production code: `outputs/evaluation/quality_gate/20260731_094843/report.json`.
- Confirmed the safe package has 80 answerable failures and 20 refusal-boundary failures; 70 answerable public questions visibly contain the literal `nan`. The remaining 10 answerable questions and all 20 refusal cases remain valid regression evidence until separately resolved.
- Added a general query-planner invariant: table title and period explicitly stated by a user override incompatible LLM-generated filters, and an explicit quarter removes an unsupported inferred month (and conversely).
- Added a synthetic regression test for a valid quarterly workbook lookup where an LLM plan wrongly supplies a monthly constraint.
- Verified the focused suite: `python -m pytest backend\app\tests\test_query_planner_service.py backend\app\tests\test_spreadsheet_retrieval_service.py backend\app\tests\test_generalization_pipeline.py -q` → `70 passed`.

## Unresolved blocker

A live API probe for a valid preserved table-style question still returned `table_evidence_not_found` after rebuilding the app image. This proves that the observed false-refusal path is not closed by filter merging alone. The next continuation must instrument or safely expose the final query aspect/filter set in a synthetic-only test and identify the remaining generic planner/runtime divergence. It must also define and test the refusal-code contract without consulting private gold.

Do not replace the 70 candidate-invalid questions until case-level invalidity is independently confirmed. Do not run a new Round 1 first blind test or overwrite any retained result.

## Commit and acceptance boundary

There is no valid completion commit. Production code changed after the Full gate, so the historic official 300/300 acceptance artifact no longer has the required matching code identity. A fresh full official acceptance from Q001, followed by the Full quality gate, is required before any valid commit. The requested Round 1 generation, audited replacement, final regression and commit are therefore not complete.

## Continuation update: table locator and refusal-code contract

The remaining table rejection was reproduced using only the preserved public question and the runtime index. The generic cause was twofold: ordinal prefixes from a corpus catalogue (for example `148_`) were not treated as a title alias after table-text normalization, and the parser did not reliably distinguish an explicitly named worksheet from the requested row label. The repair now normalizes short ordinal title prefixes, recognizes `工作表“…”`, and treats `请给出“行”在“列”列` as a deterministic row/column request. The valid preserved public probe now returns a cited, deterministic table-cell answer.

The API also now supplies a generic machine-readable refusal code when a safe refusal occurs before the model returns one. The classifier only categorizes explicitly requested boundaries (real-time/out-of-scope, forecast, subjective advice, missing calculation inputs, or missing context); it contains no evaluation-case content or regulatory answer.

Verification performed without reading the private evaluation directory: `python -m pytest backend\app\tests\test_query_planner_service.py backend\app\tests\test_spreadsheet_retrieval_service.py backend\app\tests\test_generalization_pipeline.py -q` passed 72 tests; `python -m pytest backend\app\tests -q` passed 448 tests. The app was rebuilt and the public table probe confirmed a table citation.

The required official acceptance remains blocked by environment identity: `/api/health/ready` reported `selected_device=cpu` and `cuda_available=false`. Protocol section 4 disallows using a CPU run as the official 300-case acceptance. Do not commit or start Round 1 regression until CUDA is restored and a fresh acceptance from Q001 succeeds.
