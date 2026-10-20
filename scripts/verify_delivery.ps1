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
    'docs\contest-version\core_cases.json'
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

$files = Get-ChildItem -LiteralPath $DeliveryRoot -Recurse -File -Force
$sizeMb = [math]::Round((($files | Measure-Object Length -Sum).Sum / 1MB), 2)
Write-Host "Delivery verified: $($files.Count) files, $sizeMb MB." -ForegroundColor Green
Write-Host "Only .env.example is present; no real .env and no generated SHA-256 manifest." -ForegroundColor Green
Write-Host "Root: $DeliveryRoot"
