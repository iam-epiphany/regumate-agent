param(
    [Parameter(Mandatory = $true)]
    [string]$BuildId,
    [ValidateSet("gpu", "cpu")]
    [string]$Profile = "gpu",
    [string]$ChallengeResults = "data\evaluation\trust_challenge_results.jsonl",
    [switch]$SkipIngest
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:REGUMATE_BUILD_ID = $BuildId
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$RunDir = Join-Path $ProjectRoot "data\evaluation\release_$BuildId"
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
$CheckpointPath = Join-Path $RunDir "checkpoint.json"
$State = [ordered]@{ build_id = $BuildId; profile = $Profile; stages = [ordered]@{} }
$ContestResultPath = $null
$OodResultPath = $null

function Save-Checkpoint {
    $State | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $CheckpointPath -Encoding utf8
}

function Invoke-Stage([string]$Name, [scriptblock]$Action) {
    $State.stages[$Name] = [ordered]@{ status = "running"; started_at = (Get-Date).ToUniversalTime().ToString("o") }
    Save-Checkpoint
    try {
        & $Action
        if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
        $State.stages[$Name].status = "completed"
        $State.stages[$Name].completed_at = (Get-Date).ToUniversalTime().ToString("o")
        Save-Checkpoint
    } catch {
        $State.stages[$Name].status = "failed"
        $State.stages[$Name].error = $_.Exception.Message
        $State.stages[$Name].completed_at = (Get-Date).ToUniversalTime().ToString("o")
        Save-Checkpoint
        throw
    }
}

if ($BuildId -eq "dev") { throw "Formal release BuildId cannot be dev." }

Invoke-Stage "preflight" { powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\preflight.ps1 }
Invoke-Stage "backend_tests" { python -m pytest backend\app\tests -q }
Invoke-Stage "frontend_lint_test_build" {
    Push-Location frontend
    try {
        npm run lint
        npm test -- --run
        npm run build
    } finally { Pop-Location }
}
Invoke-Stage "cpu_inference_benchmark" {
    python scripts\benchmark_model_inference.py `
        --device cpu `
        --performance-mode cpu_balanced `
        --warmup 1 `
        --repeats 3 `
        --output (Join-Path $RunDir "cpu_inference_benchmark.json")
}
Invoke-Stage "compose_build" {
    if ($Profile -eq "gpu") { docker compose -f docker-compose.yml -f docker-compose.gpu.yml build app }
    else { docker compose build app }
}
Invoke-Stage "source_metadata_audit" {
    python scripts\audit_official_metadata.py `
        --require-count 500 `
        --report (Join-Path $RunDir "source_metadata_audit.json")
}
Invoke-Stage "evaluation_isolation" {
    python scripts\audit_evaluation_isolation.py `
        --report (Join-Path $RunDir "evaluation_isolation.json")
}
Invoke-Stage "contest_300_and_ood" {
    $arguments = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ".\scripts\run_contest_dataset_test.ps1", "-Mode", "Full", "-IncludeOod")
    if ($SkipIngest) { $arguments += "-SkipUpload" }
    if ($Profile -eq "cpu") { $arguments += "-AllowCpu" }
    powershell @arguments
}
Invoke-Stage "contest_metric_gate" {
    $ContestResult = Get-ChildItem -LiteralPath "data\evaluation\contest_dataset_test" -Recurse -File -Filter "contest_qa_all_results.json" |
        Sort-Object LastWriteTimeUtc | Select-Object -Last 1
    if (-not $ContestResult) { throw "No contest QA result was produced." }
    $script:ContestResultPath = $ContestResult.FullName
    $script:OodResultPath = Join-Path $ContestResult.Directory.FullName "ood\contest_qa_ood_results.json"
    if (-not (Test-Path -LiteralPath $script:OodResultPath)) { throw "No OOD result was produced: $script:OodResultPath" }
    $BaselineResult = "data\evaluation\contest_qa_test\20260718_025139\contest_qa_all_results.json"
    if (-not (Test-Path -LiteralPath $BaselineResult)) { throw "Frozen GPU baseline is missing: $BaselineResult" }
    python scripts\verify_release_metrics.py `
        --result $script:ContestResultPath `
        --baseline $BaselineResult `
        --ood-result $script:OodResultPath `
        --report (Join-Path $RunDir "contest_metric_gate.json")
}
Invoke-Stage "contest_report" {
    python scripts\build_contest_report.py `
        --all-results $script:ContestResultPath `
        --ood-results $script:OodResultPath `
        --output (Join-Path $RunDir "final_contest_report.json")
}
Invoke-Stage "trust_challenge" {
    $ChallengeLock = Join-Path $ProjectRoot 'data\evaluation\trust_challenge_100\lock.json'
    $FirstRunSeal = Join-Path $ProjectRoot 'outputs\evaluation\trust_challenge_100\holdout_first_run_seal.json'
    $FirstReport = Join-Path $ProjectRoot 'outputs\evaluation\trust_challenge_100\holdout_first_report.json'
    foreach ($artifact in @($ChallengeLock, $FirstRunSeal, $FirstReport)) {
        if (-not (Test-Path -LiteralPath $artifact)) { throw "Frozen challenge evidence is missing: $artifact" }
    }
    $ChallengeReport = Get-Content -LiteralPath $FirstReport -Raw | ConvertFrom-Json
    Copy-Item -LiteralPath $FirstReport -Destination (Join-Path $RunDir 'trust_challenge_report.json') -Force
    if (-not $ChallengeReport.summary.gate_passed) {
        Write-Warning 'The first holdout did not pass. Preserving the sealed failure report; full 100-case publication was not run.'
    }
}
Invoke-Stage "secret_scan" { python scripts\scan_secrets.py }
Invoke-Stage "release_identity" {
    python scripts\build_release_manifest.py `
        --output (Join-Path $RunDir "release_manifest.json") `
        --image "regumate/app:contest-v3" `
        --artifact data\evaluation\trust_challenge_100\questions.jsonl `
        --artifact data\evaluation\trust_challenge_100\lock.json `
        --artifact outputs\evaluation\trust_challenge_100\holdout_first_run_seal.json `
        --artifact outputs\evaluation\trust_challenge_100\holdout_failure_report.md `
        --artifact data\evaluation\final\parse_manifest.json `
        --artifact data\evaluation\final\ingest_manifest.json `
        --artifact (Join-Path $RunDir "source_metadata_audit.json") `
        --artifact (Join-Path $RunDir "evaluation_isolation.json") `
        --artifact (Join-Path $RunDir "contest_metric_gate.json") `
        --artifact (Join-Path $RunDir "final_contest_report.json") `
        --artifact (Join-Path $RunDir "trust_challenge_report.json")
}

Write-Host "Release validation passed. Checkpoint: $CheckpointPath" -ForegroundColor Green
