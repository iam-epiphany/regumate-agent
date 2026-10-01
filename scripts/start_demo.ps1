param(
    [switch]$RequireGpu
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

function Test-DockerGpuAvailable {
    if ($env:REGUMATE_DOCKER_GPU -eq '0') { return $false }
    if ($env:REGUMATE_DOCKER_GPU -eq '1') { return $true }

    $appImage = Get-ReguMateEnv 'REGUMATE_APP_IMAGE' 'regumate/app:contest-v3'
    docker image inspect $appImage --format '{{.Id}}' *> $null
    if ($LASTEXITCODE -ne 0) { return $false }

    docker run --rm --gpus all --entrypoint python $appImage -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" *> $null
    return $LASTEXITCODE -eq 0
}

function Get-ReguMateEnv([string]$Name, [string]$Default = '') {
    $processValue = [Environment]::GetEnvironmentVariable($Name)
    if ($processValue) { return $processValue }
    $envFileValue = Get-ReguMateEnvFileValue $Name
    if ($envFileValue) { return $envFileValue }
    return $Default
}

function Get-ReguMateEnvFileValue([string]$Name) {
    if (Test-Path -LiteralPath '.env') {
        $match = Select-String -LiteralPath '.env' -Pattern "^$([regex]::Escape($Name))=(.*)$" | Select-Object -First 1
        if ($match) { return $match.Matches[0].Groups[1].Value.Trim().Trim('"').Trim("'") }
    }
    return ''
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

function Test-ReguMateBackendPort([int]$Port) {
    try {
        $probe = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -TimeoutSec 2
        return $probe.status -eq 'ok' -and $probe.message -match 'ReguMate'
    } catch {
        return $false
    }
}

function Test-QdrantHttpPort([int]$Port) {
    try {
        $probe = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/healthz" -TimeoutSec 2
        return $probe -match 'healthz check passed'
    } catch {
        return $false
    }
}

function Resolve-ReguMateHostPort([string]$Name, [int]$PreferredPort, [scriptblock]$KnownServiceProbe, [int[]]$ReservedPorts = @()) {
    $listener = Get-NetTCPConnection -LocalPort $PreferredPort -State Listen -ErrorAction SilentlyContinue
    if (($ReservedPorts -notcontains $PreferredPort) -and (-not $listener -or (& $KnownServiceProbe $PreferredPort))) {
        return $PreferredPort
    }

    $selectedPort = Get-FreeTcpPort $ReservedPorts
    Write-Host "$Name port $PreferredPort is already in use; using available port $selectedPort for this run." -ForegroundColor Yellow
    return $selectedPort
}

function Resolve-ReguMateHostPorts {
    $reservedPorts = @()
    $appPort = Resolve-ReguMateHostPort 'ReguMate app' (Get-ReguMatePort 'REGUMATE_APP_PORT' 8000) ${function:Test-ReguMateBackendPort} $reservedPorts
    $reservedPorts += $appPort
    $qdrantHttpPort = Resolve-ReguMateHostPort 'Qdrant HTTP' (Get-ReguMatePort 'REGUMATE_QDRANT_HTTP_PORT' 6333) ${function:Test-QdrantHttpPort} $reservedPorts
    $reservedPorts += $qdrantHttpPort
    $qdrantGrpcPort = Resolve-ReguMateHostPort 'Qdrant gRPC' (Get-ReguMatePort 'REGUMATE_QDRANT_GRPC_PORT' 6334) { param([int]$Port) return (Test-QdrantHttpPort $qdrantHttpPort) } $reservedPorts

    [Environment]::SetEnvironmentVariable('REGUMATE_APP_PORT', "$appPort", 'Process')
    [Environment]::SetEnvironmentVariable('REGUMATE_QDRANT_HTTP_PORT', "$qdrantHttpPort", 'Process')
    [Environment]::SetEnvironmentVariable('REGUMATE_QDRANT_GRPC_PORT', "$qdrantGrpcPort", 'Process')

    return @{
        app_port = $appPort
        qdrant_http_port = $qdrantHttpPort
        qdrant_grpc_port = $qdrantGrpcPort
        app_url = "http://127.0.0.1:$appPort"
        qdrant_url = "http://127.0.0.1:$qdrantHttpPort"
    }
}

function Save-ReguMateRuntimeState([hashtable]$Ports) {
    $stateDir = Join-Path $ProjectRoot '.run-state'
    New-Item -ItemType Directory -Path $stateDir -Force | Out-Null
    $statePath = Join-Path $stateDir 'runtime.json'
    $Ports + @{ updated_at = (Get-Date).ToString('o') } |
        ConvertTo-Json -Depth 4 |
        Set-Content -LiteralPath $statePath -Encoding UTF8
}

function Sync-QdrantCollectionEnv {
    $envFileCollection = Get-ReguMateEnvFileValue 'QDRANT_COLLECTION'
    if ([string]::IsNullOrWhiteSpace($envFileCollection)) {
        $envFileCollection = 'regumate_chunks'
    }
    $processCollection = [Environment]::GetEnvironmentVariable('QDRANT_COLLECTION')
    if ($processCollection -and $processCollection -ne $envFileCollection) {
        Write-Host "Ignoring inherited QDRANT_COLLECTION=$processCollection; using .env/default QDRANT_COLLECTION=$envFileCollection." -ForegroundColor Yellow
    }
    [Environment]::SetEnvironmentVariable('QDRANT_COLLECTION', $envFileCollection, 'Process')
}

function Invoke-ReguMateCompose([string[]]$ComposeCommand) {
    $args = @('compose')
    if (Test-DockerGpuAvailable) {
        Write-Host 'Docker GPU support detected; enabling docker-compose.gpu.yml.' -ForegroundColor Green
        $args += @('-f', 'docker-compose.yml', '-f', 'docker-compose.gpu.yml')
    } else {
        if ($RequireGpu) {
            throw 'GPU evaluation requires Docker GPU access, but CUDA is unavailable in the app image/container runtime.'
        }
        Write-Host 'Docker GPU support was not detected; starting without GPU and letting ReguMate fall back to CPU.' -ForegroundColor Yellow
    }
    $args += $ComposeCommand
    & docker @args
}

Sync-QdrantCollectionEnv
$ports = Resolve-ReguMateHostPorts
Save-ReguMateRuntimeState $ports
& "$PSScriptRoot\preflight.ps1"
Invoke-ReguMateCompose @('up', '-d', '--no-build')
if ($LASTEXITCODE -ne 0) { throw 'Docker Compose startup failed.' }

$deadline = (Get-Date).AddMinutes(3)
do {
    try {
        $health = Invoke-RestMethod -Uri "$($ports.qdrant_url)/healthz" -TimeoutSec 3
        if ($health -match 'healthz check passed') { break }
    } catch { }
    Start-Sleep -Seconds 2
} while ((Get-Date) -lt $deadline)

if ((Get-Date) -ge $deadline) { throw 'Qdrant was not ready in 3 minutes. Run docker compose logs qdrant.' }

$backend = $null
$backendDeadline = (Get-Date).AddMinutes(3)
do {
    try {
        $candidate = Invoke-RestMethod -Uri "$($ports.app_url)/api/health" -TimeoutSec 5
        if ($candidate.status -eq 'ok') {
            $backend = $candidate
            break
        }
    } catch { }
    Start-Sleep -Seconds 2
} while ((Get-Date) -lt $backendDeadline)

if (-not $backend) { throw 'ReguMate backend was not ready in 3 minutes. Run docker compose logs app.' }

$expectedCollection = Get-ReguMateEnv 'QDRANT_COLLECTION' 'regumate_chunks'
$rag = $null
$ragDeadline = (Get-Date).AddMinutes(3)
do {
    try {
        $rag = Invoke-RestMethod -Uri "$($ports.app_url)/api/health/rag" -TimeoutSec 30
        break
    } catch { }
    Start-Sleep -Seconds 2
} while ((Get-Date) -lt $ragDeadline)

if (-not $rag) { throw 'ReguMate RAG status was not ready in 3 minutes. Run docker compose logs app.' }

if ($rag.qdrant_collection -ne $expectedCollection) {
    throw "ReguMate backend is not using the contest collection. Expected '$expectedCollection', got '$($rag.qdrant_collection)'. Stop other dev services and run: docker compose up -d --no-build"
}

Save-ReguMateRuntimeState $ports
Write-Host "ReguMate is running at $($ports.app_url)" -ForegroundColor Green
Write-Host "Qdrant dashboard: $($ports.qdrant_url)/dashboard" -ForegroundColor Green
Write-Host "Qdrant collection: $($rag.qdrant_collection)" -ForegroundColor Green
$selectedDevice = $rag.model_device.selected_device
if ($selectedDevice -eq 'cuda') {
    Write-Host "Model device: cuda ($($rag.model_device.cuda_device_name))" -ForegroundColor Green
} else {
    Write-Host "Model device: $selectedDevice. CUDA is unavailable or unsuitable; CPU fallback is active." -ForegroundColor Yellow
    if ($rag.model_device.fallback_reason) {
        Write-Host "Fallback reason: $($rag.model_device.fallback_reason)" -ForegroundColor Yellow
    }
}
if ($RequireGpu -and ($selectedDevice -ne 'cuda' -or -not $rag.model_device.cuda_available)) {
    throw 'GPU evaluation preflight failed: the running app did not select CUDA. Do not run the official 300 on CPU.'
}
$allowUnready = (Get-ReguMateEnv 'REGUMATE_ALLOW_UNREADY' '') -eq '1'

if (-not $rag.ready -and -not $allowUnready) {
    throw 'RAG is not ready. The demo may refuse valid contest questions. Check /api/health/rag, model files, Qdrant collection, and final_runtime data. Set REGUMATE_ALLOW_UNREADY=1 only for first-time ingestion/debugging.'
}
