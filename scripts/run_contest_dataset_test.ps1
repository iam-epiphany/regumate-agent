param(
    [ValidateSet("Full", "Quick")]
    [string]$Mode = "Full",
    [ValidateRange(1, 300)]
    [int]$Limit = 20,
    [ValidateRange(5, 600)]
    [int]$TimeoutSeconds = 60,
    [ValidateRange(10, 1440)]
    [int]$IngestTimeoutMinutes = 360,
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [switch]$SkipWarmup,
    [switch]$SkipUpload,
    [switch]$IncludeOod,
    [switch]$AllowCpu
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$BaseUrlWasSpecified = $PSBoundParameters.ContainsKey("BaseUrl")

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
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker was not found. Install and start Docker Desktop first."
}

if ($AllowCpu) {
    $env:MODEL_DEVICE = "cpu"
    $env:REGUMATE_TORCH_FLAVOR = "cpu"
    if (-not $env:REGUMATE_APP_IMAGE) {
        $env:REGUMATE_APP_IMAGE = "regumate/app:contest-cpu"
    }
}

Write-Step "Check and start ReguMate"
$AppImage = if ($env:REGUMATE_APP_IMAGE) { $env:REGUMATE_APP_IMAGE } else { "regumate/app:contest-v3" }
docker image inspect $AppImage *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "$AppImage is missing. Building from source..." -ForegroundColor Yellow
    docker compose build app
    if ($LASTEXITCODE -ne 0) { throw "ReguMate app image build failed." }
}

powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_demo.ps1
if ($LASTEXITCODE -ne 0) { throw "ReguMate startup failed." }
Sync-ReguMateRuntimeUrls
Write-Host "Using ReguMate API: $BaseUrl"

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
    "compose", "exec", "-T", "app", "python", "scripts/run_contest_dataset_test.py",
    "--base-url", $BaseUrl,
    "--contest-root", "/app/data/contest_dataset",
    "--split", "all",
    "--timeout", "$TimeoutSeconds",
    "--retries", "1",
    "--request-delay", "0.05",
    "--poll-interval", "5",
    "--ingest-timeout-minutes", "$IngestTimeoutMinutes"
)

if ($caseLimit -gt 0) {
    $arguments += @("--limit", "$caseLimit")
}
if ($SkipWarmup) {
    $arguments += "--skip-warmup"
}
if ($SkipUpload) {
    $arguments += "--skip-upload"
}
if ($IncludeOod) {
    $arguments += "--include-ood"
}
if ($AllowCpu) {
    $arguments += "--allow-cpu"
}

Write-Step "Legacy combined flow: upload 500 contest files and run QA evaluation"
Write-Host "Mode: $Mode"
if ($Mode -eq "Quick") { Write-Host "Quick limit: $Limit" }
Write-Host "Results will be written under data\evaluation\contest_dataset_test\<timestamp>\"

docker @arguments
if ($LASTEXITCODE -ne 0) {
    throw "contest_dataset test failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "contest_dataset test completed." -ForegroundColor Green
