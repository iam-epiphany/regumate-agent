param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [ValidateRange(30, 7200)]
    [int]$TimeoutSeconds = 900,
    [ValidateRange(1, 30)]
    [int]$ProgressIntervalSeconds = 1
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$BaseUrlWasSpecified = $PSBoundParameters.ContainsKey("BaseUrl")

function Get-ReguMateRuntimeState {
    $statePath = Join-Path $ProjectRoot ".run-state\runtime.json"
    if (-not (Test-Path -LiteralPath $statePath)) { return $null }
    try {
        return Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    } catch {
        return $null
    }
}

function Resolve-ReguMateBaseUrl {
    $state = Get-ReguMateRuntimeState
    if (-not $BaseUrlWasSpecified -and $state -and $state.app_url) {
        return [string]$state.app_url
    }
    return $BaseUrl
}

function Format-Elapsed([TimeSpan]$Elapsed) {
    if ($Elapsed.TotalMinutes -ge 1) {
        return ("{0:00}:{1:00}" -f [int]$Elapsed.TotalMinutes, $Elapsed.Seconds)
    }
    return ("{0}s" -f [int]$Elapsed.TotalSeconds)
}

$ResolvedBaseUrl = (Resolve-ReguMateBaseUrl).TrimEnd("/")
$WarmupUri = "$ResolvedBaseUrl/api/health/warmup"
$startedAt = Get-Date
$deadline = $startedAt.AddSeconds($TimeoutSeconds)

Write-Host "Warm up ReguMate models: $WarmupUri" -ForegroundColor Cyan
Write-Host "First run may download BGE-M3 and BGE reranker into data/model_cache. Keep this window open."

$job = Start-Job -ScriptBlock {
    param([string]$Uri, [int]$Timeout)
    $ErrorActionPreference = "Stop"
    Invoke-RestMethod -Method Post -Uri $Uri -TimeoutSec $Timeout
} -ArgumentList $WarmupUri, $TimeoutSeconds

try {
    $lastPrintedSecond = -1
    while ($job.State -in @("NotStarted", "Running")) {
        $now = Get-Date
        if ($now -ge $deadline) {
            Stop-Job -Job $job -ErrorAction SilentlyContinue
            throw "Model warmup timed out after $TimeoutSeconds seconds. Check Docker network access to HuggingFace and run: docker compose logs --tail 200 app"
        }

        $elapsed = $now - $startedAt
        $elapsedSeconds = [int]$elapsed.TotalSeconds
        $percent = [Math]::Min(95, [Math]::Max(3, [int](($elapsedSeconds / [Math]::Max($TimeoutSeconds, 1)) * 100)))
        $status = "Elapsed $(Format-Elapsed $elapsed). Downloading/loading models may take several minutes on the first run."
        Write-Progress -Activity "ReguMate model warmup" -Status $status -PercentComplete $percent

        if ($elapsedSeconds -gt 0 -and $elapsedSeconds % 30 -eq 0 -and $elapsedSeconds -ne $lastPrintedSecond) {
            Write-Host ("Still warming up... elapsed {0}" -f (Format-Elapsed $elapsed))
            $lastPrintedSecond = $elapsedSeconds
        }

        Start-Sleep -Seconds $ProgressIntervalSeconds
    }

    Write-Progress -Activity "ReguMate model warmup" -Completed

    if ($job.State -ne "Completed") {
        $errorText = (Receive-Job -Job $job -ErrorAction SilentlyContinue -ErrorVariable warmupErrors | Out-String).Trim()
        if ($warmupErrors) {
            $errorText = (($warmupErrors | ForEach-Object { $_.ToString() }) -join [Environment]::NewLine)
        }
        if ([string]::IsNullOrWhiteSpace($errorText)) {
            $errorText = "Model warmup failed with job state: $($job.State)"
        }
        throw $errorText
    }

    $result = Receive-Job -Job $job -ErrorAction Stop
    $elapsedTotal = (Get-Date) - $startedAt
    Write-Host ("Model warmup completed in {0}." -f (Format-Elapsed $elapsedTotal)) -ForegroundColor Green
    $result | ConvertTo-Json -Depth 8
} finally {
    Remove-Job -Job $job -Force -ErrorAction SilentlyContinue
}
