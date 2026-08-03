# Independent 100 evaluation handoff — GPU acceptance

## Completed

- Recreated `regumate-app` with `docker-compose.gpu.yml`; `/api/health/ready` reported CUDA selected and available on the NVIDIA GPU.
- Ran a fresh, resumable official acceptance under the current code identity.  The first foreground process was interrupted by the tool time limit after its checkpoint was written; the same immutable checkpoint was resumed by a separate hidden process.
- Completed result: 300/300 correct, zero runtime errors, CUDA selected, GPU performance mode.  Result directory: `data/evaluation/independent_100_20260731/official_acceptance_20260731_104202/`.
- Full quality gate passed: `outputs/evaluation/quality_gate/20260731_105603/report.json`.
- Backend suite passed: 450 tests.  Frontend lint, tests (34), and build passed.
- Strengthened the isolated evaluation pipeline with fixed capability/difficulty quotas, source-coverage validation, public projection leakage checks, freeze-time batch validation, and safe diagnostic category aggregation.

## Still required

No independent questions have been authored, frozen, or blind-run yet.  The authoring process must create the new private gold below `../ReguMate-Eval-Private/independent_100_20260731/batch` from the official corpus only, then project public questions to `data/evaluation/independent_100_20260731/batch/public/questions.jsonl`.  It must not read any historical Round 1 files or private artifacts.

Before any blind run, run `generalization_pipeline.py audit`, `validate-batch`, and `freeze`; retain the first runner output and score only via the isolated scorer.  Any repair must be based solely on its safe diagnostic package.
