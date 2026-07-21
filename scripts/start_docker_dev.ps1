param(
    [switch]$Build
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

function Get-ReguMateEnvFileValue([string]$Name) {
    $envPath = Join-Path $ProjectRoot '.env'
    if (-not (Test-Path -LiteralPath $envPath)) { return '' }
    $match = Select-String -LiteralPath $envPath -Pattern "^$([regex]::Escape($Name))=(.*)$" | Select-Object -First 1
    if ($match) { return $match.Matches[0].Groups[1].Value.Trim().Trim('"').Trim("'") }
    return ''
}

function Get-ReguMateEnv([string]$Name, [string]$Default = '') {
    $processValue = [Environment]::GetEnvironmentVariable($Name)
    if ($processValue) { return $processValue }
    $envFileValue = Get-ReguMateEnvFileValue $Name
    if ($envFileValue) { return $envFileValue }
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

function Remove-ReguMateContainer([string]$Name) {
    docker rm -f $Name *> $null
}

function Ensure-ReguMateNetwork([string]$Name) {
    docker network inspect $Name *> $null
    if ($LASTEXITCODE -ne 0) {
        docker network create $Name | Out-Null
    }
}

function Test-DockerGpuAvailable([string]$AppImage) {
    if ($env:REGUMATE_DOCKER_GPU -eq '0') { return $false }
    if ($env:REGUMATE_DOCKER_GPU -eq '1') { return $true }
    docker run --rm --gpus all --entrypoint python $AppImage -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" *> $null
    return $LASTEXITCODE -eq 0
}

function Wait-ReguMateHttp([string]$Url, [int]$TimeoutSeconds, [string]$Name) {
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

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Docker Desktop is required for ReguMate Docker development mode.'
}
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    throw 'Docker Desktop is not running.'
}

$appPort = Get-ReguMatePort 'REGUMATE_APP_PORT' 8000
$frontendPort = Get-ReguMatePort 'REGUMATE_DOCKER_FRONTEND_PORT' 5174
$qdrantHttpPort = Get-ReguMatePort 'REGUMATE_QDRANT_HTTP_PORT' 6333
$qdrantGrpcPort = Get-ReguMatePort 'REGUMATE_QDRANT_GRPC_PORT' 6334
$torchFlavor = Get-ReguMateEnv 'REGUMATE_TORCH_FLAVOR' 'cuda'
$appImage = Get-ReguMateEnv 'REGUMATE_APP_IMAGE' "regumate/app:dev-$torchFlavor"
$networkName = Get-ReguMateEnv 'REGUMATE_DOCKER_DEV_NETWORK' 'regumate-dev'

if ($torchFlavor -notin @('cuda', 'cpu')) {
    throw "REGUMATE_TORCH_FLAVOR must be 'cuda' or 'cpu'. Current value: '$torchFlavor'."
}

docker image inspect $appImage *> $null
if ($Build -or $LASTEXITCODE -ne 0) {
    Write-Host "Building $appImage with REGUMATE_TORCH_FLAVOR=$torchFlavor..."
    docker build `
        --build-arg "REGUMATE_TORCH_FLAVOR=$torchFlavor" `
        --build-arg "REGUMATE_BUILD_ID=docker-dev" `
        -t $appImage `
        .
    if ($LASTEXITCODE -ne 0) { throw 'Docker app image build failed.' }
}

$dataDir = Join-Path $ProjectRoot 'data'
$modelsDir = Join-Path $dataDir 'models'
$qdrantDir = Join-Path $dataDir 'qdrant'
New-Item -ItemType Directory -Force -Path $dataDir, $modelsDir, $qdrantDir | Out-Null

Ensure-ReguMateNetwork $networkName

Remove-ReguMateContainer 'regumate-dev-frontend'
Remove-ReguMateContainer 'regumate-dev-app'
Remove-ReguMateContainer 'regumate-dev-qdrant'

Write-Host 'Starting Qdrant dev container...'
docker run -d `
    --name regumate-dev-qdrant `
    --network $networkName `
    -p "127.0.0.1:${qdrantHttpPort}:6333" `
    -p "127.0.0.1:${qdrantGrpcPort}:6334" `
    -v "${qdrantDir}:/qdrant/storage" `
    qdrant/qdrant:v1.18.0 | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Failed to start Qdrant dev container.' }

Wait-ReguMateHttp "http://127.0.0.1:$qdrantHttpPort/healthz" 60 'Qdrant'

$gpuArgs = @()
if (Test-DockerGpuAvailable $appImage) {
    Write-Host 'Docker GPU support detected; app dev container will use --gpus all.' -ForegroundColor Green
    $gpuArgs = @('--gpus', 'all')
} else {
    Write-Host 'Docker GPU support was not detected; app dev container will start in CPU fallback mode.' -ForegroundColor Yellow
}

$envFileArgs = @()
if (Test-Path -LiteralPath (Join-Path $ProjectRoot '.env')) {
    $envFileArgs = @('--env-file', (Join-Path $ProjectRoot '.env'))
}

$corsOrigins = "http://localhost:$frontendPort,http://127.0.0.1:$frontendPort,http://localhost:$appPort,http://127.0.0.1:$appPort"

Write-Host 'Starting ReguMate backend dev container...'
docker run -d `
    --name regumate-dev-app `
    --network $networkName `
    @gpuArgs `
    @envFileArgs `
    -p "127.0.0.1:${appPort}:8000" `
    -e 'QDRANT_URL=http://regumate-dev-qdrant:6333' `
    -e "CORS_ORIGINS=$corsOrigins" `
    -e 'REGUMATE_FRONTEND_DEV_SERVER=http://regumate-dev-frontend:5174' `
    -e 'MODEL_DEVICE=cuda' `
    -e 'REGUMATE_BUILD_ID=docker-dev' `
    -v "${ProjectRoot}\backend:/app/backend" `
    -v "${ProjectRoot}\scripts:/app/scripts" `
    -v "${dataDir}:/app/data" `
    -v "${modelsDir}:/app/data/models:ro" `
    $appImage `
    python -m uvicorn backend.main:app --reload --reload-dir /app/backend --host 0.0.0.0 --port 8000 | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Failed to start ReguMate backend dev container.' }

Write-Host 'Starting ReguMate frontend dev container...'
docker run -d `
    --name regumate-dev-frontend `
    --network $networkName `
    -p "127.0.0.1:${frontendPort}:5174" `
    -e 'VITE_API_PROXY_TARGET=http://regumate-dev-app:8000' `
    -e "VITE_HMR_CLIENT_PORT=$frontendPort" `
    -v "${ProjectRoot}\frontend:/app/frontend" `
    -v 'regumate-frontend-node-modules:/app/frontend/node_modules' `
    -w /app/frontend `
    node:22-bookworm-slim `
    sh -c 'test -d node_modules/.vite || npm install && npm run dev -- --host 0.0.0.0 --port 5174' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Failed to start ReguMate frontend dev container.' }

Wait-ReguMateHttp "http://127.0.0.1:$appPort/api/health" 90 'ReguMate backend'
Wait-ReguMateHttp "http://127.0.0.1:$frontendPort" 90 'Vite frontend'

$stateDir = Join-Path $ProjectRoot '.run-state'
New-Item -ItemType Directory -Force -Path $stateDir | Out-Null
$state = @{
    mode = 'docker-run-dev'
    app_port = $appPort
    frontend_port = $frontendPort
    qdrant_http_port = $qdrantHttpPort
    qdrant_grpc_port = $qdrantGrpcPort
    app_url = "http://127.0.0.1:$appPort"
    frontend_url = "http://127.0.0.1:$frontendPort"
    qdrant_url = "http://127.0.0.1:$qdrantHttpPort"
    app_image = $appImage
    torch_flavor = $torchFlavor
    updated_at = (Get-Date).ToString('o')
}
$state | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $stateDir 'docker-dev.json') -Encoding UTF8
$state | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $stateDir 'runtime.json') -Encoding UTF8

Write-Host ''
Write-Host "ReguMate Docker development frontend: http://127.0.0.1:$frontendPort" -ForegroundColor Green
Write-Host "ReguMate backend API: http://127.0.0.1:$appPort" -ForegroundColor Green
Write-Host "Qdrant dashboard: http://127.0.0.1:$qdrantHttpPort/dashboard" -ForegroundColor Green
Write-Host ''
Write-Host 'Logs:'
Write-Host '  docker logs -f regumate-dev-app'
Write-Host '  docker logs -f regumate-dev-frontend'
