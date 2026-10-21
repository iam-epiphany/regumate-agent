$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Candidates = @(
    (Join-Path $ProjectRoot 'dist-delivery\ReguMate-Agent-Contest'),
    (Join-Path $ProjectRoot 'dist-delivery\ReguMate-Agent-Delivery'),
    (Join-Path $ProjectRoot 'dist-delivery\ReguMate-Agent')
)
$DeliveryRoot = $Candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $DeliveryRoot) {
    throw 'No delivery directory found. Run scripts/build_delivery_package.ps1 first.'
}

$required = @(
    '.env.example',
    'README.md',
    'docker-compose.yml',
    'Dockerfile',
    'backend\main.py',
    'frontend\package.json',
    'docs\contest-version\core_cases.json',
    'outputs\official300_open\open_questions_all.jsonl'
)
foreach ($relative in $required) {
    $path = Join-Path $DeliveryRoot $relative
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing required delivery file: $relative"
    }
}

$envFile = Join-Path $DeliveryRoot '.env'
if (Test-Path -LiteralPath $envFile) {
    throw 'Delivery must not contain a real .env; ship .env.example only.'
}
foreach ($name in @('FILE_MANIFEST.sha256', 'SHA256SUMS.txt')) {
    if (Get-ChildItem -LiteralPath $DeliveryRoot -Recurse -File -Force -Filter $name -ErrorAction SilentlyContinue) {
        throw "Delivery must not contain generated hash manifest: $name"
    }
}

$forbidden = Get-ChildItem -LiteralPath $DeliveryRoot -Recurse -File -Force -ErrorAction SilentlyContinue |
    Where-Object {
        $_.FullName -match '\\(node_modules|\.git|\.idea|\.pytest_cache|__pycache__|\.run-state|tmp)(\\|$)' -or
        $_.Name -eq '.env' -or
        $_.Name -match '\.(pyc|pyo|tsbuildinfo)$'
    } | Select-Object -First 1
if ($forbidden) {
    throw "Delivery contains forbidden file: $($forbidden.FullName)"
}

$template = Get-Content -LiteralPath (Join-Path $DeliveryRoot '.env.example')
if ($template | Where-Object { $_ -match '^\s*[A-Z0-9_]*(?:API_KEY|TOKEN|PASSWORD)\s*=\s*\S' }) {
    throw 'Delivery configuration template contains a non-empty credential.'
}
if ([IO.File]::ReadAllText((Join-Path $DeliveryRoot 'README.md')) -ne [IO.File]::ReadAllText((Join-Path $ProjectRoot 'README.md'))) {
    throw 'Delivery README differs from the current source.'
}

$archivePath = $DeliveryRoot + '.zip'
if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
    throw "Submission archive missing: $archivePath"
}
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [IO.Compression.ZipFile]::OpenRead($archivePath)
try {
    $prefix = (Split-Path -Leaf $DeliveryRoot) + '/'
    $archiveNames = @($archive.Entries | ForEach-Object { $_.FullName.Replace('\', '/') })
    foreach ($relative in $required) {
        $entryName = $prefix + $relative.Replace('\', '/')
        if ($entryName -notin $archiveNames) { throw "Required archive entry missing: $entryName" }
    }
    $forbiddenEntry = $archiveNames | Where-Object {
        $_ -match '(^|/)(\.env|node_modules|\.git|__pycache__|\.pytest_cache|tmp)(/|$)' -or
        $_ -match '\.(pyc|pyo|tsbuildinfo)$'
    } | Select-Object -First 1
    if ($forbiddenEntry) { throw "Forbidden archive entry: $forbiddenEntry" }
} finally {
    $archive.Dispose()
}

$files = Get-ChildItem -LiteralPath $DeliveryRoot -Recurse -File -Force
$sizeMb = [math]::Round((($files | Measure-Object Length -Sum).Sum / 1MB), 2)
Write-Host "Delivery verified: $($files.Count) files, $sizeMb MB." -ForegroundColor Green
Write-Host "Only .env.example is present; no real .env and no generated SHA-256 manifest." -ForegroundColor Green
Write-Host "Root: $DeliveryRoot"
Write-Host "Submission archive verified: $archivePath"
