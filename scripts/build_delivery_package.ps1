$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$DeliveryName = 'ReguMate-Agent-Delivery'
$DistRoot = Join-Path $ProjectRoot 'dist-delivery'
$DeliveryRoot = Join-Path $DistRoot $DeliveryName
$ProjectRootFull = (Resolve-Path -LiteralPath $ProjectRoot).Path.TrimEnd('\')

function Ensure-InProject([string]$Path) {
    $full = [System.IO.Path]::GetFullPath($Path)
    if (-not $full.StartsWith($ProjectRootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to operate outside project root: $full"
    }
    return $full
}

function Copy-Directory([string]$Source, [string]$Destination, [string[]]$ExcludeDirs = @(), [string[]]$ExcludeFiles = @()) {
    if (-not (Test-Path -LiteralPath $Source)) { throw "Missing source directory: $Source" }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    $args = @(
        $Source,
        $Destination,
        '/E',
        '/NFL',
        '/NDL',
        '/NJH',
        '/NJS',
        '/NP',
        '/R:2',
        '/W:1'
    )
    if ($ExcludeDirs.Count -gt 0) { $args += @('/XD') + $ExcludeDirs }
    if ($ExcludeFiles.Count -gt 0) { $args += @('/XF') + $ExcludeFiles }
    & robocopy @args | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "robocopy failed from $Source to $Destination with code $LASTEXITCODE" }
}

function Copy-FileToDelivery([string]$RelativePath) {
    $source = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path -LiteralPath $source)) { throw "Missing source file: $RelativePath" }
    $target = Join-Path $DeliveryRoot $RelativePath
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
    Copy-Item -LiteralPath $source -Destination $target -Force
}

New-Item -ItemType Directory -Force -Path $DistRoot | Out-Null

$DeliveryRootFull = Ensure-InProject $DeliveryRoot
if (Test-Path -LiteralPath $DeliveryRootFull) {
    Remove-Item -LiteralPath $DeliveryRootFull -Recurse -Force
}
foreach ($archivePath in @(
    (Join-Path $DistRoot "$DeliveryName.zip"),
    (Join-Path $DistRoot "$DeliveryName.zip.sha256.txt")
)) {
    if (Test-Path -LiteralPath $archivePath) {
        $archivePathFull = Ensure-InProject $archivePath
        Remove-Item -LiteralPath $archivePathFull -Force
    }
}
New-Item -ItemType Directory -Force -Path $DeliveryRootFull | Out-Null

$rootFiles = @(
    '.dockerignore',
    '.env',
    '.gitignore',
    'docker-compose.gpu.yml',
    'docker-compose.yml',
    'Dockerfile',
    'pytest.ini',
    'README.md',
    'requirements-cuda.txt',
    'requirements-dev.in',
    'requirements.in',
    'requirements.txt',
    'docker-compose.dev.yml',
    'docker-run.bat',
    'docker-run.sh',
    'rebuild-run.bat',
    'run.bat',
    'run.sh',
    'stop.sh',
    'stop.bat'
)
foreach ($file in $rootFiles) { Copy-FileToDelivery $file }

Copy-Directory (Join-Path $ProjectRoot 'backend') (Join-Path $DeliveryRoot 'backend') `
    -ExcludeDirs @('__pycache__', '.pytest_cache') `
    -ExcludeFiles @('*.pyc', '*.pyo', '*.log')

Copy-Directory (Join-Path $ProjectRoot 'frontend') (Join-Path $DeliveryRoot 'frontend') `
    -ExcludeDirs @('node_modules', 'dist', 'build') `
    -ExcludeFiles @('*.tsbuildinfo', '*.log')

Copy-Directory (Join-Path $ProjectRoot 'scripts') (Join-Path $DeliveryRoot 'scripts') `
    -ExcludeDirs @('__pycache__') `
    -ExcludeFiles @(
        '*.pyc',
        '*.pyo',
        '*.log',
        'audit_hard_challenge_50.py',
        'build_hard_challenge_50.py',
        'build_hard_challenge_50_round2.py'
    )

$IterationLogFileName = -join ([char[]](
    0x7cfb, 0x7edf, 0x4f18, 0x5316, 0x4e0e, 0x8bc4, 0x6d4b,
    0x8fed, 0x4ee3, 0x8bb0, 0x5f55, 0x2e, 0x6d, 0x64
))

Copy-Directory (Join-Path $ProjectRoot 'docs') (Join-Path $DeliveryRoot 'docs') `
    -ExcludeDirs @('contest_excel_100_final.json') `
    -ExcludeFiles @(
        '*.docx',
        'legacy_doc_parsing_report.md',
        'official_excel_cells_report.md',
        'CODEX_*.md',
        $IterationLogFileName
    )
$internalDocs = @('CODEX_OBJECTIVE.md', 'CODEX_STATUS.md', $IterationLogFileName)
foreach ($internalDoc in $internalDocs) {
    $internalPath = Join-Path (Join-Path $DeliveryRoot 'docs') $internalDoc
    if (Test-Path -LiteralPath $internalPath) {
        Remove-Item -LiteralPath $internalPath -Force
    }
}

New-Item -ItemType Directory -Force -Path (Join-Path $DeliveryRoot 'data') | Out-Null
Copy-Directory (Join-Path $ProjectRoot 'data\models') (Join-Path $DeliveryRoot 'data\models')
Copy-Directory (Join-Path $ProjectRoot 'data\qdrant') (Join-Path $DeliveryRoot 'data\qdrant')
Copy-Directory (Join-Path $ProjectRoot 'data\contest_dataset') (Join-Path $DeliveryRoot 'data\contest_dataset')
Copy-Directory (Join-Path $ProjectRoot 'data\regulations') (Join-Path $DeliveryRoot 'data\regulations') `
    -ExcludeFiles @('银行业监管制度测试样例_模拟版.pdf')
Copy-Directory (Join-Path $ProjectRoot 'data\evaluation\final_runtime') (Join-Path $DeliveryRoot 'data\evaluation\final_runtime') `
    -ExcludeDirs @('audit_archives', 'models', 'qdrant') `
    -ExcludeFiles @('*.log')
New-Item -ItemType Directory -Force -Path (Join-Path $DeliveryRoot 'data\evaluation\final') | Out-Null
if (Test-Path -LiteralPath (Join-Path $ProjectRoot 'outputs\evaluation')) {
    Copy-Directory (Join-Path $ProjectRoot 'outputs\evaluation') (Join-Path $DeliveryRoot 'outputs\evaluation') `
        -ExcludeDirs @('__pycache__') `
        -ExcludeFiles @('*.pyc', '*.pyo', '*.log')
}
foreach ($file in @(
    'data\evaluation\final\frozen_parameters.json',
    'data\evaluation\final\ingest_manifest.json',
    'data\evaluation\final\official_excel_cells.json',
    'data\evaluation\final\parse_manifest.json'
)) {
    Copy-FileToDelivery $file
}

New-Item -ItemType Directory -Force -Path (Join-Path $DeliveryRoot 'dist-delivery') | Out-Null
$optionalDeliveryFiles = @(
    'dist-delivery\regumate-images.tar',
    'dist-delivery\SHA256SUMS.txt'
)
$missingOptionalDeliveryFiles = @()
foreach ($file in $optionalDeliveryFiles) {
    if (Test-Path -LiteralPath (Join-Path $ProjectRoot $file)) {
        Copy-FileToDelivery $file
    } else {
        $missingOptionalDeliveryFiles += $file
    }
}

$excludedFinding = Get-ChildItem -LiteralPath $DeliveryRoot -Recurse -Force -ErrorAction SilentlyContinue | Where-Object {
    $relative = $_.FullName.Substring($DeliveryRoot.Length).TrimStart('\')
    $insideQdrant = $relative -like 'data\qdrant\*'
    $_.FullName -match '\\(node_modules|\.git|\.idea|\.pytest_cache|__pycache__|\.run-state|contest_staging|audit_archives|dev_runtime)\\' -or
    ($_.Name -match '\.(pyc|pyo|tsbuildinfo)$') -or
    ($_.Name -match '\.log$' -and -not $insideQdrant)
} | Select-Object -First 1
if ($excludedFinding) { throw "Unexpected generated file in delivery package: $($excludedFinding.FullName)" }

$fileCount = (Get-ChildItem -LiteralPath $DeliveryRoot -Recurse -File -Force | Measure-Object).Count
$size = (Get-ChildItem -LiteralPath $DeliveryRoot -Recurse -File -Force | Measure-Object Length -Sum).Sum
$contentsPath = Join-Path $DeliveryRoot 'DELIVERY_CONTENTS.txt'
$contents = @(
    "ReguMate delivery directory",
    "Generated: $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))",
    "Files: $fileCount",
    ("Uncompressed size: {0:N2} GB" -f ($size / 1GB)),
    "",
    "Top-level contents:"
) + (
    Get-ChildItem -LiteralPath $DeliveryRoot -Force | Sort-Object Name | ForEach-Object {
        if ($_.PSIsContainer) { "[dir]  $($_.Name)" } else { "[file] $($_.Name) $($_.Length) bytes" }
    }
) + @(
    "",
    "Excluded from delivery: .git, IDE files, Python/Node caches, node_modules, frontend dist, development databases, temporary staging directories, historical audit logs, repeated evaluation runs, and duplicate .env.example.",
    "Qdrant internal *.log files are retained because they are part of the persisted vector index."
)
if ($missingOptionalDeliveryFiles.Count -gt 0) {
    $contents += @(
        "",
        "Optional delivery files not present at generation time:"
    ) + ($missingOptionalDeliveryFiles | ForEach-Object { "- $_" })
}
$contents | Set-Content -LiteralPath $contentsPath -Encoding utf8

Write-Host "Delivery directory: $DeliveryRoot" -ForegroundColor Green
Write-Host "Files: $fileCount"
Write-Host ("Uncompressed size: {0:N2} GB" -f ($size / 1GB))
Write-Host "Manifest: $contentsPath" -ForegroundColor Green
