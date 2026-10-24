$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DeliveryFiles = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'delivery_files.json') -Raw | ConvertFrom-Json
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
    '.env',
    'README.md',
    'docker-compose.yml',
    'Dockerfile',
    'backend\main.py',
    'frontend\package.json',
    'outputs\official300_open\open_questions_all.jsonl'
) + @($DeliveryFiles.scripts)
foreach ($relative in $required) {
    $path = Join-Path $DeliveryRoot $relative
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing required delivery file: $relative"
    }
}

$expectedScripts = @($DeliveryFiles.scripts | ForEach-Object { $_.Replace('\', '/') } | Sort-Object)
$actualScripts = @(Get-ChildItem -LiteralPath (Join-Path $DeliveryRoot 'scripts') -Recurse -File -Force | ForEach-Object {
    $_.FullName.Substring($DeliveryRoot.Length + 1).Replace('\', '/')
} | Sort-Object)
if (Compare-Object $expectedScripts $actualScripts) {
    throw 'Delivery scripts differ from scripts/delivery_files.json.'
}
foreach ($test in $DeliveryFiles.development_only_backend_tests) {
    if (Test-Path -LiteralPath (Join-Path $DeliveryRoot "backend\app\tests\$test")) {
        throw "Delivery contains a development-only test: $test"
    }
}

$envFile = Join-Path $DeliveryRoot '.env'
$envValues = @{}
foreach ($line in Get-Content -LiteralPath $envFile) {
    if ($line -match '^\s*([A-Z][A-Z0-9_]*)\s*=\s*(.*)$') {
        $envValues[$Matches[1]] = $Matches[2].Trim().Trim([char[]]@("'", '"'))
    }
}
foreach ($name in @('MODEL_PROVIDER', 'MODEL_API_KEY', 'MODEL_BASE_URL', 'MODEL_NAME', 'EMBEDDING_API_KEY', 'SERVICE_PORT')) {
    if (-not $envValues.ContainsKey($name)) { throw "Missing contest environment variable: $name" }
}
foreach ($name in @('MODEL_PROVIDER', 'MODEL_BASE_URL', 'MODEL_NAME')) {
    if (-not $envValues[$name]) { throw "Contest environment variable is empty: $name" }
}
$servicePort = 0
if (-not [int]::TryParse($envValues['SERVICE_PORT'], [ref]$servicePort) -or $servicePort -lt 1 -or $servicePort -gt 65535) {
    throw 'SERVICE_PORT must be an integer between 1 and 65535.'
}
if ([IO.File]::ReadAllText($envFile) -ne [IO.File]::ReadAllText((Join-Path $ProjectRoot '.env'))) {
    throw 'Delivery .env differs from the authorized source configuration.'
}
foreach ($name in @('FILE_MANIFEST.sha256', 'SHA256SUMS.txt')) {
    if (Get-ChildItem -LiteralPath $DeliveryRoot -Recurse -File -Force -Filter $name -ErrorAction SilentlyContinue) {
        throw "Delivery must not contain generated hash manifest: $name"
    }
}

$forbidden = Get-ChildItem -LiteralPath $DeliveryRoot -Recurse -File -Force -ErrorAction SilentlyContinue |
    Where-Object {
        $_.FullName -match '\\(node_modules|\.git|\.idea|\.pytest_cache|__pycache__|\.run-state|tmp)(\\|$)' -or
        ($_.Name -eq '.env' -and $_.FullName -ne $envFile) -or
        $_.Name -eq '.env.example' -or
        $_.Name -match '\.(pyc|pyo|tsbuildinfo)$'
    } | Select-Object -First 1
if ($forbidden) {
    throw "Delivery contains forbidden file: $($forbidden.FullName)"
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
    $archiveScripts = @($archiveNames | Where-Object {
        $_.StartsWith($prefix + 'scripts/') -and -not $_.EndsWith('/')
    } | ForEach-Object { $_.Substring($prefix.Length) } | Sort-Object)
    if (Compare-Object $expectedScripts $archiveScripts) {
        throw 'Archive scripts differ from scripts/delivery_files.json.'
    }
    foreach ($test in $DeliveryFiles.development_only_backend_tests) {
        if (($prefix + "backend/app/tests/$test") -in $archiveNames) {
            throw "Archive contains a development-only test: $test"
        }
    }
    $envEntry = $archive.GetEntry($prefix + '.env')
    $reader = [IO.StreamReader]::new($envEntry.Open())
    try {
        if ($reader.ReadToEnd() -ne [IO.File]::ReadAllText($envFile)) {
            throw 'Archive .env differs from the authorized delivery configuration.'
        }
    } finally {
        $reader.Dispose()
    }
    $forbiddenEntry = $archiveNames | Where-Object {
        $_ -match '(^|/)(\.env\.example|node_modules|\.git|__pycache__|\.pytest_cache|tmp)(/|$)' -or
        ($_ -match '(^|/)\.env$' -and $_ -ne ($prefix + '.env')) -or
        $_ -match '\.(pyc|pyo|tsbuildinfo)$'
    } | Select-Object -First 1
    if ($forbiddenEntry) { throw "Forbidden archive entry: $forbiddenEntry" }
} finally {
    $archive.Dispose()
}

$files = Get-ChildItem -LiteralPath $DeliveryRoot -Recurse -File -Force
$sizeMb = [math]::Round((($files | Measure-Object Length -Sum).Sum / 1MB), 2)
Write-Host "Delivery verified: $($files.Count) files, $sizeMb MB." -ForegroundColor Green
Write-Host "Root .env included with contest model configuration; no duplicate template or generated SHA-256 manifest." -ForegroundColor Green
Write-Host "Root: $DeliveryRoot"
Write-Host "Submission archive verified: $archivePath"
Write-Host "Script allowlist verified: $($expectedScripts.Count) files; development tools remain source-only."
