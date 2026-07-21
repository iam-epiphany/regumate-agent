$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

function Import-ReguMateEnvFile {
    $envPath = Join-Path $ProjectRoot '.env'
    if (-not (Test-Path -LiteralPath $envPath)) { return }
    foreach ($line in Get-Content -LiteralPath $envPath -Encoding UTF8) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith('#')) { continue }
        $parts = $trimmed -split '=', 2
        if ($parts.Count -ne 2) { continue }
        $name = $parts[0].Trim()
        $value = $parts[1].Trim().Trim('"').Trim("'")
        if ($name -and -not [Environment]::GetEnvironmentVariable($name, 'Process')) {
            [Environment]::SetEnvironmentVariable($name, $value, 'Process')
        }
    }
}

function Get-ReguMateEnv([string]$Name, [string]$Default = '') {
    $value = [Environment]::GetEnvironmentVariable($Name, 'Process')
    if ($value) { return $value }
    return $Default
}

function Get-ReguMatePort([string]$Name, [int]$Default) {
    $value = Get-ReguMateEnv $Name "$Default"
    $port = 0
    if (-not [int]::TryParse($value, [ref]$port) -or $port -lt 1 -or $port -gt 65535) {
        throw "$Name must be a TCP port between 1 and 65535. Current value: '$value'."
    }
    return $port
}

function Get-FreeTcpPort([int[]]$ExcludedPorts = @()) {
    do {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
        try {
            $listener.Start()
            $candidate = ([System.Net.IPEndPoint]$listener.LocalEndpoint).Port
        } finally {
            $listener.Stop()
        }
    } while ($ExcludedPorts -contains $candidate)
    return $candidate
}

function Resolve-LocalPort([string]$Label, [int]$PreferredPort, [int[]]$ReservedPorts = @()) {
    $listener = Get-NetTCPConnection -LocalPort $PreferredPort -State Listen -ErrorAction SilentlyContinue
    if (($ReservedPorts -notcontains $PreferredPort) -and -not $listener) {
        return $PreferredPort
    }
    $selected = Get-FreeTcpPort $ReservedPorts
    Write-Host "$Label port $PreferredPort is already in use; using $selected for local hot reload." -ForegroundColor Yellow
    return $selected
}

function Test-QdrantReady([int]$Port) {
    try {
        $probe = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/healthz" -TimeoutSec 2
        return $probe -match 'healthz check passed'
    } catch {
        return $false
    }
}

function Wait-HttpOk([string]$Url, [int]$TimeoutSeconds, [string]$Name) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        try {
            $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) { return }
        } catch { }
        Start-Sleep -Seconds 1
    } while ((Get-Date) -lt $deadline)
    throw "$Name did not become reachable at $Url within $TimeoutSeconds seconds."
}

function Wait-ReguMateBackend([string]$Url, [int]$TimeoutSeconds) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        try {
            $health = Invoke-RestMethod -Uri "$Url/api/health" -TimeoutSec 5
            if ($health.status -eq 'ok') { return }
        } catch { }
        Start-Sleep -Seconds 1
    } while ((Get-Date) -lt $deadline)
    throw "Backend did not become healthy at $Url within $TimeoutSeconds seconds."
}

function ConvertTo-ReguMateProcessArgument([string]$Argument) {
    if ($null -eq $Argument) { return '""' }
    if ($Argument -notmatch '[\s"]') { return $Argument }
    return '"' + ($Argument -replace '"', '\"') + '"'
}

function Start-LoggedProcess([string]$Name, [string]$FilePath, [string[]]$ArgumentList, [string]$WorkingDirectory, [string]$OutLog, [string]$ErrLog) {
    Write-Host "Starting $Name..."
    $commandLine = ($ArgumentList | ForEach-Object { ConvertTo-ReguMateProcessArgument $_ }) -join ' '
    return Start-Process -FilePath $FilePath `
        -ArgumentList $commandLine `
        -WorkingDirectory $WorkingDirectory `
        -RedirectStandardOutput $OutLog `
        -RedirectStandardError $ErrLog `
        -WindowStyle Hidden `
        -PassThru
}

Import-ReguMateEnvFile

$machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
$env:Path = @($env:Path, $machinePath, $userPath) -join ';'

$stateDir = Join-Path $ProjectRoot '.run-state'
New-Item -ItemType Directory -Path $stateDir -Force | Out-Null

$appPort = Resolve-LocalPort 'Backend' (Get-ReguMatePort 'REGUMATE_APP_PORT' 8000)
$frontendPort = Resolve-LocalPort 'Frontend' (Get-ReguMatePort 'REGUMATE_FRONTEND_PORT' 5173) @($appPort)
$qdrantHttpPort = Get-ReguMatePort 'REGUMATE_QDRANT_HTTP_PORT' 6333

if (-not (Test-QdrantReady $qdrantHttpPort)) {
    where.exe docker.exe *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Qdrant is not reachable at 127.0.0.1:$qdrantHttpPort and Docker is not available to start it."
    }
    Write-Host "Starting Qdrant container for local hot reload..."
    docker compose up -d qdrant
    if ($LASTEXITCODE -ne 0) { throw 'Failed to start Qdrant via Docker Compose.' }
    $deadline = (Get-Date).AddSeconds(60)
    do {
        if (Test-QdrantReady $qdrantHttpPort) { break }
        Start-Sleep -Seconds 1
    } while ((Get-Date) -lt $deadline)
    if (-not (Test-QdrantReady $qdrantHttpPort)) {
        throw "Qdrant did not become ready at 127.0.0.1:$qdrantHttpPort."
    }
}

$python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    $python = 'python'
}

& $python -c "import uvicorn" *> $null
if ($LASTEXITCODE -ne 0) {
    throw 'Local Python environment is missing uvicorn. This legacy local script is no longer the standard path; use docker-run.bat instead.'
}

where.exe npm.cmd *> $null
if ($LASTEXITCODE -ne 0) { throw 'npm.cmd was not found. Install Node.js or use the bundled Node environment before running local frontend hot reload.' }

$frontendDir = Join-Path $ProjectRoot 'frontend'
if (-not (Test-Path -LiteralPath (Join-Path $frontendDir 'node_modules'))) {
    Write-Host 'frontend/node_modules not found; running npm install...'
    Push-Location $frontendDir
    try {
        npm install
        if ($LASTEXITCODE -ne 0) { throw 'npm install failed.' }
    } finally {
        Pop-Location
    }
}

[Environment]::SetEnvironmentVariable('PYTHONIOENCODING', 'utf-8', 'Process')
[Environment]::SetEnvironmentVariable('QDRANT_URL', "http://127.0.0.1:$qdrantHttpPort", 'Process')
[Environment]::SetEnvironmentVariable('CORS_ORIGINS', "http://localhost:$appPort,http://127.0.0.1:$appPort,http://localhost:$frontendPort,http://127.0.0.1:$frontendPort", 'Process')
[Environment]::SetEnvironmentVariable('VITE_API_PROXY_TARGET', "http://127.0.0.1:$appPort", 'Process')
[Environment]::SetEnvironmentVariable('VITE_HMR_CLIENT_PORT', "$frontendPort", 'Process')
[Environment]::SetEnvironmentVariable('REGUMATE_FRONTEND_DEV_SERVER', "http://127.0.0.1:$frontendPort", 'Process')
[Environment]::SetEnvironmentVariable('REGUMATE_APP_PORT', "$appPort", 'Process')
[Environment]::SetEnvironmentVariable('REGUMATE_FRONTEND_PORT', "$frontendPort", 'Process')
[Environment]::SetEnvironmentVariable('REGUMATE_QDRANT_HTTP_PORT', "$qdrantHttpPort", 'Process')

$backendOut = Join-Path $stateDir 'local-backend.out.log'
$backendErr = Join-Path $stateDir 'local-backend.err.log'
$frontendOut = Join-Path $stateDir 'local-frontend.out.log'
$frontendErr = Join-Path $stateDir 'local-frontend.err.log'

$backendArgs = @('-m', 'uvicorn', 'backend.main:app', '--reload', '--reload-dir', (Join-Path $ProjectRoot 'backend'), '--host', '127.0.0.1', '--port', "$appPort")
$frontendArgs = @('run', 'dev', '--', '--host', '127.0.0.1', '--port', "$frontendPort")

$backend = Start-LoggedProcess 'local backend' $python $backendArgs $ProjectRoot $backendOut $backendErr
$frontend = Start-LoggedProcess 'local frontend' 'npm.cmd' $frontendArgs $frontendDir $frontendOut $frontendErr

$apiUrl = "http://127.0.0.1:$appPort"
$frontendUrl = "http://127.0.0.1:$frontendPort"

try {
    Wait-ReguMateBackend $apiUrl 90
    Wait-HttpOk $frontendUrl 60 'Frontend'
} catch {
    Write-Host "[FAIL] Startup probe failed. Backend log: $backendErr Frontend log: $frontendErr" -ForegroundColor Red
    throw
}

$state = @{
    mode = 'local-hot-reload'
    backend_pid = $backend.Id
    frontend_pid = $frontend.Id
    app_port = $appPort
    frontend_port = $frontendPort
    api_port = $appPort
    qdrant_http_port = $qdrantHttpPort
    app_url = $apiUrl
    frontend_url = $frontendUrl
    api_url = $apiUrl
    qdrant_url = "http://127.0.0.1:$qdrantHttpPort"
    updated_at = (Get-Date).ToString('o')
}
$state | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $stateDir 'local-dev.json') -Encoding UTF8
$state | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $stateDir 'runtime.json') -Encoding UTF8

Write-Host ''
Write-Host "ReguMate local hot reload is running at $apiUrl" -ForegroundColor Green
Write-Host "Vite dev server: $frontendUrl" -ForegroundColor Green
Write-Host "Qdrant dashboard: http://127.0.0.1:$qdrantHttpPort/dashboard" -ForegroundColor Green
Write-Host ''
Write-Host 'Logs:'
Write-Host "  Get-Content -Wait -LiteralPath `"$backendOut`""
Write-Host "  Get-Content -Wait -LiteralPath `"$frontendOut`""
