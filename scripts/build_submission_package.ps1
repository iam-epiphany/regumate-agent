param(
    [string]$DeepSeekApiKey = $env:REGUMATE_SUBMISSION_DEEPSEEK_API_KEY
)

$ErrorActionPreference = 'Stop'

if (-not $DeepSeekApiKey) {
    throw 'DeepSeek API key is required. Set REGUMATE_SUBMISSION_DEEPSEEK_API_KEY before running this script.'
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$PackageName = 'ReguMate-Agent'
$DistRoot = Join-Path $ProjectRoot 'dist-delivery'
$StagingRoot = Join-Path $DistRoot "$PackageName-staging"
$PackageRoot = Join-Path $StagingRoot $PackageName
$VisiblePackageRoot = Join-Path $DistRoot $PackageName
$ArchivePath = Join-Path $DistRoot "$PackageName.zip"
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

function Copy-FileToPackage([string]$RelativePath) {
    $source = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path -LiteralPath $source)) { throw "Missing source file: $RelativePath" }
    $target = Join-Path $PackageRoot $RelativePath
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
    Copy-Item -LiteralPath $source -Destination $target -Force
}

New-Item -ItemType Directory -Force -Path $DistRoot | Out-Null
foreach ($path in @($StagingRoot, $ArchivePath, "$ArchivePath.sha256.txt")) {
    if (Test-Path -LiteralPath $path) {
        $safePath = Ensure-InProject $path
        Remove-Item -LiteralPath $safePath -Recurse -Force
    }
}
New-Item -ItemType Directory -Force -Path $PackageRoot | Out-Null

$rootFiles = @(
    '.dockerignore',
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
                    'scripts/launcher/'
)
foreach ($file in $rootFiles) { Copy-FileToPackage $file }
foreach ($file in @('scripts\launcher\docker-run.bat', 'scripts\launcher\docker-run.sh')) {
    if (Test-Path -LiteralPath (Join-Path $ProjectRoot $file)) {
        Copy-FileToPackage $file
    }
}

Copy-Directory (Join-Path $ProjectRoot 'backend') (Join-Path $PackageRoot 'backend') `
    -ExcludeDirs @('__pycache__', '.pytest_cache') `
    -ExcludeFiles @('*.pyc', '*.pyo', '*.log')
Copy-Directory (Join-Path $ProjectRoot 'frontend') (Join-Path $PackageRoot 'frontend') `
    -ExcludeDirs @('node_modules', 'dist', 'build') `
    -ExcludeFiles @('*.tsbuildinfo', '*.log')
Copy-Directory (Join-Path $ProjectRoot 'scripts') (Join-Path $PackageRoot 'scripts') `
    -ExcludeDirs @('__pycache__') `
    -ExcludeFiles @(
        '*.pyc',
        '*.pyo',
        '*.log',
        'build_delivery_package.ps1',
        'audit_hard_challenge_50.py',
        'build_hard_challenge_50.py',
        'build_hard_challenge_50_round2.py',
        'build_offline_bundle.ps1',
        'build_submission_package.ps1',
        'verify_delivery.ps1'
    )
$IterationLogFileName = -join ([char[]](
    0x7cfb, 0x7edf, 0x4f18, 0x5316, 0x4e0e, 0x8bc4, 0x6d4b,
    0x8fed, 0x4ee3, 0x8bb0, 0x5f55, 0x2e, 0x6d, 0x64
))

Copy-Directory (Join-Path $ProjectRoot 'docs') (Join-Path $PackageRoot 'docs') `
    -ExcludeDirs @('evaluation', 'contest_excel_100_final.json') `
    -ExcludeFiles @(
        '*.docx',
        '*report*.json',
        '*report*.md',
        'CODEX_*.md',
        $IterationLogFileName
    )
$internalDocs = @('CODEX_OBJECTIVE.md', 'CODEX_STATUS.md', $IterationLogFileName)
foreach ($internalDoc in $internalDocs) {
    $internalPath = Join-Path (Join-Path $PackageRoot 'docs') $internalDoc
    if (Test-Path -LiteralPath $internalPath) {
        Remove-Item -LiteralPath $internalPath -Force
    }
}
foreach ($report in @(
    'docs\evaluation\archive\reports\final_contest_report.md',
    'docs\evaluation\archive\reports\final_contest_report.json',
    'docs\evaluation\archive\reports\performance_baseline_20260718.md',
    'docs\evaluation\archive\reports\performance_experiments\phase1-engineering.md',
    'docs\evaluation\archive\reports\performance_experiments\indexing-baseline.md'
)) {
    if (Test-Path -LiteralPath (Join-Path $ProjectRoot $report)) {
        Copy-FileToPackage $report
    }
}
$evaluationMarkdownReports = Get-ChildItem -LiteralPath (Join-Path $ProjectRoot 'docs\evaluation') -File -Filter '*.md' -ErrorAction SilentlyContinue
foreach ($report in $evaluationMarkdownReports) {
    if ($report.Name -in @('legacy_doc_parsing_report.md', 'official_excel_cells_report.md')) {
        continue
    }
    $relative = $report.FullName.Substring($ProjectRootFull.Length).TrimStart('\')
    Copy-FileToPackage $relative
}
# 冻结豁免策略文件：backend 测试（test_phase1_evaluation_gates.py）依赖它，
# 必须随包分发以保证 README 的 pytest 命令在交付包内可执行。
foreach ($frozenPolicy in @('docs\evaluation\legacy_exceptions\frozen_artifact_exceptions.v1.json')) {
    if (Test-Path -LiteralPath (Join-Path $ProjectRoot $frozenPolicy)) {
        Copy-FileToPackage $frozenPolicy
    }
}
Copy-Directory (Join-Path $ProjectRoot 'data\contest_dataset') (Join-Path $PackageRoot 'data\contest_dataset')
# 自命题 200 题评测集：README 一键测评（run_all_evaluations.py）与评分器依赖
# 其中的 questions.jsonl 与 gold.jsonl，必须随包分发。
Copy-Directory (Join-Path $ProjectRoot 'data\自命题200题评测集') (Join-Path $PackageRoot 'data\自命题200题评测集') `
    -ExcludeDirs @('__pycache__') `
    -ExcludeFiles @('*.pyc', '*.pyo', '*.log')
# data/regulations 是可选的额外监管语料（存在则随包分发，缺失不影响构建）。
if (Test-Path -LiteralPath (Join-Path $ProjectRoot 'data\regulations')) {
    Copy-Directory (Join-Path $ProjectRoot 'data\regulations') (Join-Path $PackageRoot 'data\regulations') `
        -ExcludeFiles @('银行业监管制度测试样例_模拟版.pdf')
}
foreach ($artifact in @(
    'data\evaluation\final\parse_manifest.json',
    'data\evaluation\final\ingest_manifest.json',
    'data\evaluation\source_metadata_audit_current.json',
    'data\evaluation\official_metadata_audit_current.json',
    'data\evaluation\evaluation_isolation_current.json'
)) {
    if (Test-Path -LiteralPath (Join-Path $ProjectRoot $artifact)) {
        Copy-FileToPackage $artifact
    }
}
if (Test-Path -LiteralPath (Join-Path $ProjectRoot 'data\evaluation\trust_challenge_100')) {
    Copy-Directory (Join-Path $ProjectRoot 'data\evaluation\trust_challenge_100') `
        (Join-Path $PackageRoot 'data\evaluation\trust_challenge_100') `
        -ExcludeDirs @('__pycache__') `
        -ExcludeFiles @('*.pyc', '*.pyo', '*.log')
}
if (Test-Path -LiteralPath (Join-Path $ProjectRoot 'data\evaluation\hard_challenge_70\round_1')) {
    Copy-Directory (Join-Path $ProjectRoot 'data\evaluation\hard_challenge_70\round_1') `
        (Join-Path $PackageRoot 'data\evaluation\hard_challenge_70\round_1') `
        -ExcludeDirs @('__pycache__') `
        -ExcludeFiles @('*.pyc', '*.pyo', '*.log')
}
foreach ($artifact in @(
    'outputs\evaluation\document_identity_backfill_500.json',
    'outputs\evaluation\vector_index_audit_after_formula_reindex.json',
    'outputs\evaluation\trust_challenge_100\holdout_first_run.json',
    'outputs\evaluation\trust_challenge_100\holdout_first_report.json',
    'outputs\evaluation\trust_challenge_100\holdout_first_run_seal.json',
    'outputs\evaluation\trust_challenge_100\post_holdout_delivery_changes.json',
    'outputs\evaluation\trust_challenge_100\holdout_failure_report.md',
    'outputs\evaluation\hard_challenge_70\round_1\run.json',
    'outputs\evaluation\hard_challenge_70\round_1\report.json',
    'outputs\evaluation\hard_challenge_70\round_1\failure_summary.md'
)) {
    if (Test-Path -LiteralPath (Join-Path $ProjectRoot $artifact)) {
        Copy-FileToPackage $artifact
    }
}
$latestRelease = Get-ChildItem -LiteralPath (Join-Path $ProjectRoot 'data\evaluation') -Directory -Filter 'release_*' -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTimeUtc | Select-Object -Last 1
if ($latestRelease) {
    Copy-Directory $latestRelease.FullName (Join-Path $PackageRoot 'data\evaluation\release')
}

$envLines = @(
    '# ReguMate small-package runtime configuration. This file intentionally contains the configured API key.',
    '# LLM_* uses an OpenAI-compatible Chat Completions API root. Do not include /chat/completions in LLM_BASE_URL.',
    '# DeepSeek: LLM_BASE_URL=https://api.deepseek.com and LLM_MODEL=deepseek-v4-flash.',
    '# OpenAI: LLM_BASE_URL=https://api.openai.com/v1 and set LLM_MODEL to the model you are allowed to use.',
    '# Other OpenAI-compatible providers: set LLM_BASE_URL to the provider-compatible API root.',
    '# Local BGE-M3 embedding and BGE reranker are unaffected by LLM_*.',
    'LLM_PROVIDER=openai_compatible',
    "LLM_API_KEY=$DeepSeekApiKey",
    'LLM_BASE_URL=https://api.deepseek.com',
    'LLM_MODEL=deepseek-v4-flash',
    'LLM_RESPONSE_FORMAT=json_object',
    'LLM_STREAM=true',
    'LLM_DISABLE_THINKING=false',
    "DEEPSEEK_API_KEY=$DeepSeekApiKey",
    'REGUMATE_BUILD_ID=regumate-agent-small',
    'REGUMATE_OFFLINE_MODE=false',
    'HF_HUB_OFFLINE=0',
    'TRANSFORMERS_OFFLINE=0',
    'HF_DATASETS_OFFLINE=0',
    'QDRANT_COLLECTION=regumate_chunks',
    'QDRANT_AUTO_CREATE_COLLECTION=true',
    'QUERY_PLANNER_MODEL=deepseek-v4-flash',
    'ANSWER_GENERATION_MODEL=deepseek-v4-flash',
    'SEMANTIC_GROUNDING_MODE=risk_based',
    'SEMANTIC_GROUNDING_MODEL=deepseek-v4-flash',
    'MODEL_DEVICE=auto',
    'MODEL_GPU_MIN_FREE_MEMORY_GB=1.0',
    'REGUMATE_PERFORMANCE_MODE=auto',
    'MODEL_BACKEND=pytorch',
    'MODEL_WARMUP_POLICY=background',
    'RERANK_BATCH_SIZE=0',
    'RERANK_MAX_LENGTH=1024',
    'TORCH_NUM_THREADS=0',
    'TORCH_NUM_INTEROP_THREADS=0',
    'EMBEDDING_BATCH_SIZE=4',
    'EMBEDDING_MAX_BATCH_SIZE=8',
    'REGUMATE_ALLOW_UNREADY=1'
)
$envLines | Set-Content -LiteralPath (Join-Path $PackageRoot '.env') -Encoding utf8

$bad = Get-ChildItem -LiteralPath $PackageRoot -Recurse -Force | Where-Object {
    $relative = $_.FullName.Substring($PackageRoot.Length).TrimStart('\')
    $relative -match '(^|\\)(node_modules|\.git|\.idea|\.pytest_cache|__pycache__|\.run-state|contest_staging|audit_archives|dev_runtime|dist-delivery)(\\|$)' -or
    $relative -match '^data\\(models|qdrant)(\\|$)' -or
    $relative -match '^data\\evaluation\\final_runtime(\\|$)' -or
    $_.Name -match '\.(pyc|pyo|log|tsbuildinfo)$'
} | Select-Object -First 1
if ($bad) { throw "Unexpected excluded file in submission package: $($bad.FullName)" }

$manifest = @(
    'ReguMate-Agent small package',
    "Generated: $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))",
    '',
    'Included contest artifacts: data/contest_dataset/dataset/nfra_page_attachments_500, the official QA workbook under data/contest_dataset, and the self-authored 200-question evaluation set with gold answers under data/自命题200题评测集.',
    'Included validation assets: backend/frontend tests, parse/ingest manifests, source metadata audit, the locked official-document trust challenge artifacts, and the frozen hard70 round_1 artifacts.',
    'Excluded large artifacts: offline Docker image archive, data/models, data/qdrant, data/evaluation/final_runtime, frontend/node_modules, frontend/dist, historical evaluation outputs.',
    'Runtime expectation: online Docker build, online Qdrant image pull, online HuggingFace model download, contest_dataset ingestion through scripts/upload_contest_knowledge_base.ps1, QA evaluation through scripts/run_contest_qa_test.ps1.'
)
$manifest | Set-Content -LiteralPath (Join-Path $PackageRoot 'SUBMISSION_CONTENTS.txt') -Encoding utf8

$fileManifestPath = Join-Path $PackageRoot 'FILE_MANIFEST.sha256'
$hashLines = foreach ($file in Get-ChildItem -LiteralPath $PackageRoot -Recurse -File -Force | Sort-Object FullName) {
    if ($file.FullName -eq $fileManifestPath) { continue }
    $relative = $file.FullName.Substring($PackageRoot.Length).TrimStart('\').Replace('\', '/')
    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).Hash.ToLower()
    "$hash  $relative"
}
$hashLines | Set-Content -LiteralPath $fileManifestPath -Encoding utf8

$fileCount = (Get-ChildItem -LiteralPath $PackageRoot -Recurse -File -Force | Measure-Object).Count
$size = (Get-ChildItem -LiteralPath $PackageRoot -Recurse -File -Force | Measure-Object Length -Sum).Sum

if ($size -gt 500MB) {
    throw ("Package directory is {0:N2} MB, exceeding 500MB." -f ($size / 1MB))
}

if (Test-Path -LiteralPath $VisiblePackageRoot) {
    Remove-Item -LiteralPath $VisiblePackageRoot -Recurse -Force
}
Copy-Directory $PackageRoot $VisiblePackageRoot
Remove-Item -LiteralPath $StagingRoot -Recurse -Force
Write-Host "Package directory: $VisiblePackageRoot" -ForegroundColor Green
Write-Host ("Directory size: {0:N2} MB" -f ($size / 1MB))
Write-Host "Files: $fileCount"

# 生成提交用 zip 归档与 SHA-256 校验文件（旧包在脚本开头已删除）。
# 注意：此处 $PackageRoot（staging）已在上面删除，从最终目录 $VisiblePackageRoot 打包。
$ArchiveTempPath = Join-Path $DistRoot "$PackageName.tmp.zip"
if (Test-Path -LiteralPath $ArchiveTempPath) { Remove-Item -LiteralPath $ArchiveTempPath -Force }
Compress-Archive -LiteralPath $VisiblePackageRoot -DestinationPath $ArchiveTempPath -CompressionLevel Optimal
Move-Item -LiteralPath $ArchiveTempPath -Destination $ArchivePath -Force
$hash = Get-FileHash -Algorithm SHA256 -LiteralPath $ArchivePath
"$($hash.Hash.ToLower())  $PackageName.zip" | Set-Content -LiteralPath "$ArchivePath.sha256.txt" -Encoding utf8
Write-Host "Archive: $ArchivePath" -ForegroundColor Green
Write-Host "SHA256:  $($hash.Hash.ToLower())"
