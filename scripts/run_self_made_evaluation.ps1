param(
    [string]$BaseUrl = "",
    [double]$Timeout = 60,
    [double]$IngestTimeoutMinutes = 10
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$argsList = @(
    "scripts\generate_self_made_artifacts.py",
    "--timeout",
    [string]$Timeout,
    "--ingest-timeout-minutes",
    [string]$IngestTimeoutMinutes
)
if ($BaseUrl) {
    $argsList += @("--base-url", $BaseUrl.TrimEnd("/"))
} elseif (Test-Path -LiteralPath ".run-state\runtime.json") {
    $runtime = Get-Content -LiteralPath ".run-state\runtime.json" -Raw | ConvertFrom-Json
    if ($runtime.app_url) {
        try {
            Invoke-RestMethod -Uri "$($runtime.app_url)/api/health/ready" -TimeoutSec 5 | Out-Null
            $argsList += @("--base-url", [string]$runtime.app_url)
        } catch {
            Write-Host "Runtime state exists but backend is not reachable; generating offline self-made reports only."
        }
    }
}

python @argsList
