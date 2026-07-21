$ErrorActionPreference = 'Continue'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$statePath = Join-Path $ProjectRoot '.run-state\local-dev.json'
if (-not (Test-Path -LiteralPath $statePath)) {
    exit 0
}

function Stop-ReguMateProcessTree([int]$ProcessId) {
    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (-not $process) { return }

    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId=$ProcessId" -ErrorAction SilentlyContinue
    foreach ($child in $children) {
        Stop-ReguMateProcessTree ([int]$child.ProcessId)
    }

    Write-Host "Stopping local hot-reload process $ProcessId ($($process.ProcessName))..."
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
}

$state = Get-Content -LiteralPath $statePath -Encoding UTF8 | ConvertFrom-Json
foreach ($processId in @($state.backend_pid, $state.frontend_pid)) {
    if (-not $processId) { continue }
    Stop-ReguMateProcessTree ([int]$processId)
}

Remove-Item -LiteralPath $statePath -Force -ErrorAction SilentlyContinue
