param(
    [switch]$IncludeRuntimeData
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$DeliveryName = 'ReguMate-Agent-Contest'
$DistRoot = Join-Path $ProjectRoot 'dist-delivery'
$DeliveryRoot = Join-Path $DistRoot $DeliveryName
$ProjectRootFull = (Resolve-Path -LiteralPath $ProjectRoot).Path.TrimEnd('\')

function Ensure-InProject([string]$Path) {
    $full = [System.IO.Path]::GetFullPath($Path)
    if (-not $full.StartsWith($ProjectRootFull + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
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
    '.env.example',
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
    'docker-compose.dev.yml'
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
        '*.log',
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
$dataDirectories = @('data\contest_dataset', 'data\regulations')
if ($IncludeRuntimeData) {
    # Optional demo-ready mode.  Windows cannot reliably copy Qdrant's long
    # payload-index paths, so runtime vector data is opt-in.
    $dataDirectories = @('data\models', 'data\qdrant') + $dataDirectories
}
foreach ($directory in $dataDirectories) {
    $source = Join-Path $ProjectRoot $directory
    if (Test-Path -LiteralPath $source) {
        Copy-Directory $source (Join-Path $DeliveryRoot $directory)
    }
}
$runtimeEvaluation = Join-Path $ProjectRoot 'data\evaluation\final_runtime'
if ($IncludeRuntimeData -and (Test-Path -LiteralPath $runtimeEvaluation)) {
    Copy-Directory $runtimeEvaluation (Join-Path $DeliveryRoot 'data\evaluation\final_runtime') `
        -ExcludeDirs @('audit_archives', 'models', 'qdrant') `
        -ExcludeFiles @('*.log')
}
# These are source inputs, not generated results. The default runner needs the
# complete question set on fresh checkouts and inside source-only deliveries.
Copy-FileToDelivery 'outputs\official300_open\open_questions_all.jsonl'
foreach ($inputFile in Get-ChildItem -LiteralPath (Join-Path $ProjectRoot 'outputs\official300_open') -Filter 'open_questions*.jsonl' -File) {
    if ($inputFile.Name -ne 'open_questions_all.jsonl') {
        Copy-FileToDelivery ('outputs\official300_open\' + $inputFile.Name)
    }
}

$excludedFinding = Get-ChildItem -LiteralPath $DeliveryRoot -Recurse -Force -ErrorAction SilentlyContinue | Where-Object {
    $relative = $_.FullName.Substring($DeliveryRoot.Length).TrimStart('\')
    $insideQdrant = $relative -like 'data\qdrant\*'
    $_.FullName -match '\\(node_modules|\.git|\.idea|\.pytest_cache|__pycache__|\.run-state|contest_staging|audit_archives|dev_runtime)\\' -or
    ($_.Name -match '\.(pyc|pyo|tsbuildinfo)$') -or
    ($_.Name -eq '.env') -or
    ($_.Name -eq 'SHA256SUMS.txt') -or
    ($_.Name -eq 'FILE_MANIFEST.sha256') -or
    ($_.Name -match '\.log$' -and -not $insideQdrant)
} | Select-Object -First 1
if ($excludedFinding) { throw "Unexpected generated file in delivery package: $($excludedFinding.FullName)" }

$fileCount = (Get-ChildItem -LiteralPath $DeliveryRoot -Recurse -File -Force | Measure-Object).Count
$size = (Get-ChildItem -LiteralPath $DeliveryRoot -Recurse -File -Force | Measure-Object Length -Sum).Sum
$contentsPath = Join-Path $DeliveryRoot 'DELIVERY_CONTENTS.txt'
$contents = @(
    "ReguMate delivery directory",
    "Generated: $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))",
    "Source commit: $(git rev-parse HEAD)",
    "Runtime data included: $([bool]$IncludeRuntimeData)",
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
    "Excluded from delivery: .git, IDE files, Python/Node caches, node_modules, frontend dist, development databases, temporary staging directories, historical audit logs, repeated evaluation runs, the real .env, and generated SHA-256 manifests. The package ships .env.example only.",
    "Qdrant internal *.log files are retained because they are part of the persisted vector index."
)
$contents | Set-Content -LiteralPath $contentsPath -Encoding utf8

$archivePath = Ensure-InProject (Join-Path $DistRoot "$DeliveryName.zip")
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory(
    $DeliveryRootFull,
    $archivePath,
    [System.IO.Compression.CompressionLevel]::Optimal,
    $true
)

Write-Host "Delivery directory: $DeliveryRoot" -ForegroundColor Green
Write-Host "Files: $fileCount"
Write-Host ("Uncompressed size: {0:N2} GB" -f ($size / 1GB))
Write-Host "Manifest: $contentsPath" -ForegroundColor Green
Write-Host "Submission archive: $archivePath" -ForegroundColor Green
