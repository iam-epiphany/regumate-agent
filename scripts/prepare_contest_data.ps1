$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ContestRoot = Join-Path $ProjectRoot 'data\contest_dataset'
$Source = Join-Path $ContestRoot 'dataset\nfra_page_attachments_500'
$QaSource = (Get-ChildItem -LiteralPath $ContestRoot -File -Filter 'QA*.xlsx' | Select-Object -First 1).FullName
$Target = Join-Path $ProjectRoot 'data\contest_staging'

if (-not (Test-Path -LiteralPath $Source)) { throw "Contest attachment directory not found: $Source" }
if (-not $QaSource -or -not (Test-Path -LiteralPath $QaSource)) { throw "Contest QA workbook not found under $ContestRoot" }
New-Item -ItemType Directory -Force -Path $Target | Out-Null

$records = @()
$files = Get-ChildItem -LiteralPath $Source -File | Sort-Object Name
if ($files.Count -ne 500) { throw "Expected 500 contest files, found $($files.Count)." }

for ($index = 0; $index -lt $files.Count; $index++) {
    $file = $files[$index]
    $hash = Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName
    $stagedName = ('{0:D3}_{1}{2}' -f ($index + 1), $hash.Hash.Substring(0, 16).ToLower(), $file.Extension.ToLower())
    $destination = Join-Path $Target $stagedName
    if (-not (Test-Path -LiteralPath $destination) -or (Get-Item -LiteralPath $destination).Length -ne $file.Length) {
        Copy-Item -LiteralPath $file.FullName -Destination $destination -Force
    }
    $records += [ordered]@{
        staged_name = $stagedName
        original_name = $file.Name
        sha256 = $hash.Hash.ToLower()
        size = $file.Length
    }
    if ((($index + 1) % 50) -eq 0) { Write-Host "Prepared $($index + 1)/500" }
}

Copy-Item -LiteralPath $QaSource -Destination (Join-Path $Target 'qa.xlsx') -Force
$manifest = [ordered]@{ file_count = $records.Count; files = $records }
$manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Target 'source_manifest.json') -Encoding utf8

$actual = Get-ChildItem -LiteralPath $Target -File | Where-Object { $_.Name -match '^\d{3}_[0-9a-f]{16}\.' }
if ($actual.Count -ne 500) { throw "Staging validation failed: expected 500, found $($actual.Count)." }
Write-Host "Contest staging directory is ready: $Target" -ForegroundColor Green
