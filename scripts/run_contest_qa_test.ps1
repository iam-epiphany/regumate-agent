param(
    [ValidateSet("Full", "Quick")]
    [string]$Mode = "Full",
    [ValidateRange(1, 300)]
    [int]$Limit = 20,
    [ValidateRange(5, 600)]
    [int]$TimeoutSeconds = 180,
    [ValidateRange(1, 1440)]
    [int]$KnowledgeBaseTimeoutMinutes = 30,
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [ValidateSet("diagnostic", "acceptance", "fail-fast")]
    [string]$RunMode = "diagnostic",
    [string]$Output,
    [switch]$Resume,
    [switch]$SkipWarmup,
    [switch]$IncludeOod
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$BaseUrlWasSpecified = $PSBoundParameters.ContainsKey("BaseUrl")
$QdrantUrl = "http://127.0.0.1:6333"
$ContainerBaseUrl = "http://127.0.0.1:8000"

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "== $Message ==" -ForegroundColor Cyan
}

function Get-ReguMateRuntimeState {
    $statePath = Join-Path $ProjectRoot ".run-state\runtime.json"
    if (-not (Test-Path -LiteralPath $statePath)) { return $null }
    try {
        return Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    } catch {
        return $null
    }
}

function Sync-ReguMateRuntimeUrls {
    $state = Get-ReguMateRuntimeState
    if (-not $state) { return }
    if (-not $script:BaseUrlWasSpecified -and $state.app_url) {
        $script:BaseUrl = [string]$state.app_url
    }
    if ($state.qdrant_url) {
        $script:QdrantUrl = [string]$state.qdrant_url
    }
}

function Get-QdrantPoints([string]$Collection) {
    try {
        $uri = $QdrantUrl + '/collections/' + $Collection
        $payload = Invoke-RestMethod -Uri $uri -TimeoutSec 10
        return [int]$payload.result.points_count
    } catch {
        return $null
    }
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker was not found. Install and start Docker Desktop first."
}

Write-Step "Check and start ReguMate"
$AppImage = if ($env:REGUMATE_APP_IMAGE) { $env:REGUMATE_APP_IMAGE } else { "regumate/app:contest-v3" }
docker image inspect $AppImage *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "$AppImage is missing. Building from source..." -ForegroundColor Yellow
    docker compose build app
    if ($LASTEXITCODE -ne 0) { throw "ReguMate app image build failed." }
}

powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_demo.ps1 -RequireGpu
if ($LASTEXITCODE -ne 0) { throw "ReguMate startup failed." }
Sync-ReguMateRuntimeUrls
Write-Host "Using ReguMate API: $BaseUrl"

# /api/health/rag 返回与 /api/health/ready 相同的字段但不会在知识库
# 尚未导入资料时返回 503，评测脚本据此给出明确的“先上传”提示。
$ready = Invoke-RestMethod -Uri "$BaseUrl/api/health/rag" -TimeoutSec 30
$device = $ready.model_device
Write-Host "Model device: selected_device=$($device.selected_device), cuda_available=$($device.cuda_available), device=$($device.cuda_device_name)"
if ($device.selected_device -ne "cuda" -or -not $device.cuda_available) {
    throw "Official evaluation requires CUDA. selected_device=$($device.selected_device) fallback_reason=$($device.fallback_reason)"
}
$points = Get-QdrantPoints $ready.qdrant_collection
if ($null -ne $points) {
    Write-Host "Current Qdrant collection: $($ready.qdrant_collection), points=$points"
    if ($points -lt 10000) {
        throw "Current Qdrant collection has too few vector points, points=$points expected_at_least=10000. Run scripts\upload_contest_knowledge_base.ps1 first."
    }
} else {
    Write-Host "Current Qdrant collection: $($ready.qdrant_collection), points=unknown" -ForegroundColor Yellow
}

$contestRoot = Join-Path $ProjectRoot "data\contest_dataset"
$attachments = Join-Path $contestRoot "dataset\nfra_page_attachments_500"
$qa = Join-Path $contestRoot "QA$([char]0x6570)$([char]0x636E).xlsx"
if (-not (Test-Path -LiteralPath $attachments)) {
    throw "Contest attachment directory was not found: $attachments"
}
if (-not (Test-Path -LiteralPath $qa)) {
    throw "Contest QA workbook was not found: $qa"
}

$caseLimit = if ($Mode -eq "Quick") { $Limit } else { 0 }
$arguments = @(
    "compose", "exec", "-T", "app", "python", "scripts/run_contest_qa_test.py",
    "--base-url", $ContainerBaseUrl,
    "--contest-root", "/app/data/contest_dataset",
    "--split", "all",
    "--timeout", "$TimeoutSeconds",
    "--retries", "1",
    "--request-delay", "0.05",
    "--poll-interval", "5",
    "--ingest-timeout-minutes", "$KnowledgeBaseTimeoutMinutes",
    "--skip-attachment-check"
)

if ($Mode -eq "Full") {
    $relativeOutput = if ($Output) { $Output } else { "evaluation/系统测试结果/官方300题/contest_qa_test/$(Get-Date -Format 'yyyyMMdd_HHmmss')" }
    if ([System.IO.Path]::IsPathRooted($relativeOutput)) {
        throw "Output must be relative to the project root so the container can persist it: $relativeOutput"
    }
    $effectiveTimeout = [Math]::Min([Math]::Max($TimeoutSeconds, 5), 180)
    $gitCommit = (git rev-parse HEAD).Trim()
    $gitDiffHash = & python -c "import hashlib, subprocess; print(hashlib.sha256(subprocess.check_output(['git', 'diff', '--binary', 'HEAD'])).hexdigest())"
    if ($LASTEXITCODE -ne 0 -or -not $gitDiffHash) { throw "Unable to capture the host worktree diff fingerprint." }
    $baselineIdentity = Get-Content -LiteralPath .\docs\evaluation\baselines\phase_01_identity.json -Raw | ConvertFrom-Json
    $corpusSnapshot = [string]$baselineIdentity.corpus.snapshot_sha256
    if ($corpusSnapshot -notmatch '^[0-9a-f]{64}$') { throw "Baseline corpus snapshot is missing or invalid." }
    $arguments = @(
        "compose", "exec", "-T", "app", "python", "scripts/evaluation/run_official_qa.py",
        "--base-url", $ContainerBaseUrl,
        "--mode", $RunMode,
        "--output", ("/app/" + $relativeOutput.Replace('\', '/')),
        "--request-timeout-seconds", "$effectiveTimeout",
        "--retries", "0",
        "--request-delay", "0.1",
        "--git-commit", $gitCommit,
        "--worktree-diff-sha256", $gitDiffHash.Trim(),
        "--corpus-snapshot-sha256", $corpusSnapshot
    )
    if ($Resume) { $arguments += "--resume" }
} elseif ($caseLimit -gt 0) {
    $arguments += @("--limit", "$caseLimit")
}
if ($SkipWarmup) {
    $arguments += "--skip-warmup"
}
if ($IncludeOod) {
    $arguments += "--include-ood"
}
Write-Step "Run contest_dataset QA evaluation"
Write-Host "Mode: $Mode; run mode: $RunMode"
if ($Mode -eq "Quick") { Write-Host "Quick limit: $Limit" }
if ($Mode -eq "Full") { Write-Host "Full diagnostic/acceptance runs all 300 cases and does not stop when the score can no longer reach 300/300." }
Write-Host "This script expects the 500 contest files to be already uploaded and indexed."
Write-Host "Results will be written under data\evaluation\contest_qa_test\<timestamp>\"

docker @arguments
if ($LASTEXITCODE -ne 0) {
    throw "contest_dataset QA test failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "contest_dataset QA test completed." -ForegroundColor Green
