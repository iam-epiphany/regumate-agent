$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

function Pass([string]$Message) { Write-Host "[PASS] $Message" -ForegroundColor Green }
function Warn([string]$Message) { Write-Host "[WARN] $Message" -ForegroundColor Yellow }
function Fail([string]$Message) { throw "[FAIL] $Message" }
function Get-ReguMateEnv([string]$Name, [string]$Default = '') {
    $processValue = [Environment]::GetEnvironmentVariable($Name)
    if ($processValue) { return $processValue }
    if (Test-Path -LiteralPath '.env') {
        $match = Select-String -LiteralPath '.env' -Pattern "^$([regex]::Escape($Name))=(.*)$" | Select-Object -First 1
        if ($match) { return $match.Matches[0].Groups[1].Value.Trim().Trim('"').Trim("'") }
    }
    return $Default
}
function Get-ReguMatePort([string]$Name, [int]$Default) {
    $value = Get-ReguMateEnv $Name "$Default"
    $port = 0
    if (-not [int]::TryParse($value, [ref]$port) -or $port -lt 1 -or $port -gt 65535) {
        Fail "$Name must be a TCP port between 1 and 65535. Current value: '$value'."
    }
    return $port
}
function Test-DockerGpuAvailable {
    if ($env:REGUMATE_DOCKER_GPU -eq '0') { return $false }
    if ($env:REGUMATE_DOCKER_GPU -eq '1') { return $true }

    $appImage = Get-ReguMateEnv 'REGUMATE_APP_IMAGE' 'regumate/app:contest-v3'
    docker image inspect $appImage --format '{{.Id}}' *> $null
    if ($LASTEXITCODE -ne 0) { return $false }

    docker run --rm --gpus all --entrypoint python $appImage -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" *> $null
    return $LASTEXITCODE -eq 0
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { Fail 'Docker Desktop is not installed.' }
docker info *> $null
if ($LASTEXITCODE -ne 0) { Fail 'Docker Desktop is not running.' }
Pass 'Docker Desktop is running'

docker compose version *> $null
if ($LASTEXITCODE -ne 0) { Fail 'Docker Compose is unavailable.' }
docker compose config --quiet
if ($LASTEXITCODE -ne 0) { Fail 'docker-compose.yml is invalid.' }
Pass 'Docker Compose configuration is valid'
docker compose -f docker-compose.yml -f docker-compose.gpu.yml config --quiet
if ($LASTEXITCODE -ne 0) { Fail 'docker-compose.gpu.yml is invalid.' }
Pass 'Docker Compose GPU override configuration is valid'

$wslVersion = wsl.exe --status 2>$null
if ($LASTEXITCODE -eq 0 -and ($wslVersion -join "`n") -match '2') { Pass 'WSL2 is available for Docker Desktop' }
else { Warn 'WSL2 status could not be confirmed. Docker Desktop GPU support on Windows requires WSL2 backend.' }

$offlineMode = (Get-ReguMateEnv 'REGUMATE_OFFLINE_MODE' 'false').ToLowerInvariant() -in @('1', 'true', 'yes', 'on')
if ($offlineMode) {
    $modelDirs = @(
        'data\models\bge-m3',
        'data\models\bge-reranker-v2-m3'
    )
    foreach ($modelDir in $modelDirs) {
        $configPath = Join-Path $modelDir 'config.json'
        if (-not (Test-Path -LiteralPath $configPath)) { Fail "Missing offline model config: $configPath" }
        $weights = Get-ChildItem -LiteralPath $modelDir -File -ErrorAction SilentlyContinue | Where-Object {
            $_.Extension -in '.bin', '.safetensors'
        }
        if (-not $weights) { Fail "Missing offline model weights (*.bin or *.safetensors): $modelDir" }
    }
    Pass 'BGE-M3 and reranker model files are present'
} else {
    Pass 'Online model mode is enabled; local BGE model directories are not required before first start'
}

$contestRootItem = Get-ChildItem -LiteralPath (Join-Path $ProjectRoot 'data') -Recurse -Directory -Filter 'nfra_page_attachments_500' -ErrorAction SilentlyContinue | Select-Object -First 1
$contestQaItem = Get-ChildItem -LiteralPath (Join-Path $ProjectRoot 'data') -Recurse -File -Filter 'QA*.xlsx' -ErrorAction SilentlyContinue | Select-Object -First 1
if ($contestRootItem -and $contestQaItem) {
    $contestRoot = $contestRootItem.FullName
    $contestFiles = @(
        Get-ChildItem -LiteralPath $contestRoot -File | Where-Object {
            $_.Extension.ToLowerInvariant() -in '.txt','.md','.doc','.docx','.pdf','.xls','.xlsx'
        }
    )
    if ($contestFiles.Count -ne 500) { Fail "Official attachment count is $($contestFiles.Count), expected 500." }
    Pass 'Official QA workbook and all 500 attachments are present'
} elseif ($offlineMode) {
    Fail 'Missing official QA workbook or 500-attachment directory.'
} else {
    Warn 'Official contest dataset is not included; upload documents in the UI or add contest data before bulk ingestion.'
}

$secretFiles = @(
    Get-ChildItem -LiteralPath 'backend','frontend\src','scripts','docs' -Recurse -File -ErrorAction SilentlyContinue
) + @(
    Get-Item -LiteralPath 'README.md','docker-compose.yml','Dockerfile' -ErrorAction SilentlyContinue
)
$secretFinding = $secretFiles | Select-String -Pattern '\bsk-[A-Za-z0-9_-]{20,}\b' | Select-Object -First 1
if ($secretFinding) { Fail "Possible API key in source file: $($secretFinding.Path):$($secretFinding.LineNumber)" }
Pass 'No API key-shaped token was found in delivery source files'

$drive = Get-PSDrive -Name ([System.IO.Path]::GetPathRoot($ProjectRoot).TrimEnd('\').TrimEnd(':'))
if ($drive.Free -lt 15GB) { Warn 'Less than 15GB free disk space; Docker build, model download, or indexing may fail.' }
else { Pass 'Free disk space meets the recommendation' }

$appPort = Get-ReguMatePort 'REGUMATE_APP_PORT' 8000
$qdrantHttpPort = Get-ReguMatePort 'REGUMATE_QDRANT_HTTP_PORT' 6333
$qdrantGrpcPort = Get-ReguMatePort 'REGUMATE_QDRANT_GRPC_PORT' 6334

$backendIsReguMate = $false
try {
    $backendProbe = Invoke-RestMethod -Uri "http://127.0.0.1:$appPort/api/health" -TimeoutSec 2
    $backendIsReguMate = $backendProbe.status -eq 'ok' -and $backendProbe.message -match 'ReguMate'
} catch { }
$qdrantIsHealthy = $false
try {
    $qdrantProbe = Invoke-RestMethod -Uri "http://127.0.0.1:$qdrantHttpPort/healthz" -TimeoutSec 2
    $qdrantIsHealthy = $qdrantProbe -match 'healthz check passed'
} catch { }

$portRoles = @(
    @{ port = $appPort; role = 'ReguMate app'; verified = $backendIsReguMate },
    @{ port = $qdrantHttpPort; role = 'Qdrant HTTP'; verified = $qdrantIsHealthy },
    @{ port = $qdrantGrpcPort; role = 'Qdrant gRPC'; verified = $qdrantIsHealthy }
)
foreach ($portRole in $portRoles) {
    $port = [int]$portRole.port
    $listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if (-not $listener) { continue }
    if ($portRole.verified) {
        Pass "Port $port ($($portRole.role)) is already owned by the running ReguMate stack"
    } else {
        Warn "Port $port ($($portRole.role)) is already in use by an unverified process."
    }
}

$keyFromProcess = [Environment]::GetEnvironmentVariable('DEEPSEEK_API_KEY')
$keyFromFile = (Test-Path -LiteralPath '.env') -and (
    Select-String -LiteralPath '.env' -Pattern '^DEEPSEEK_API_KEY=.+$' -Quiet
)
if ($keyFromProcess -or $keyFromFile) { Pass 'DeepSeek key is configured (value hidden)' }
else { Warn 'DEEPSEEK_API_KEY is missing; text QA will use extractive degradation.' }

$configuredAppImage = Get-ReguMateEnv 'REGUMATE_APP_IMAGE' 'regumate/app:contest-v3'
$appImage = docker image inspect $configuredAppImage --format '{{.Id}}' 2>$null
if ($LASTEXITCODE -ne 0) { Warn "$configuredAppImage is missing; run.bat will build it from source." }
else {
    Pass 'ReguMate app image is present'
    if (Test-DockerGpuAvailable) {
        Pass 'Docker can run the ReguMate app image with NVIDIA GPU access'
    } else {
        Warn 'Docker GPU access is unavailable. Startup will continue and ReguMate will report CPU fallback in the web status panel.'
    }
    docker run --rm --entrypoint soffice $configuredAppImage --version *> $null
    if ($LASTEXITCODE -ne 0) { Fail 'LibreOffice is unavailable in the ReguMate app image.' }
    docker run --rm --entrypoint /bin/sh $configuredAppImage -c 'antiword -h >/dev/null 2>&1' *> $null
    if ($LASTEXITCODE -ne 0) { Fail 'antiword is unavailable in the ReguMate app image.' }
    Pass 'LibreOffice and antiword are available in the app image'
}

$qdrantImage = docker image inspect 'qdrant/qdrant:v1.18.0' --format '{{.Id}}' 2>$null
if ($LASTEXITCODE -ne 0) { Warn 'qdrant/qdrant:v1.18.0 is missing; Docker Compose will pull it from Docker Hub.' }
else { Pass 'Qdrant image is present' }

if ($qdrantIsHealthy) {
    Pass 'Running Qdrant service passed /healthz'
} else {
    Warn 'Qdrant is not running yet; start_demo.ps1 will start it.'
}

Write-Host 'Preflight completed. Review and resolve any warnings shown above before the contest demo.'
