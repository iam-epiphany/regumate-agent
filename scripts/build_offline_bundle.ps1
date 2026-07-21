$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$ProjectRootFull = (Resolve-Path -LiteralPath $ProjectRoot).Path.TrimEnd('\')
$Dist = Join-Path $ProjectRoot 'dist-delivery'
New-Item -ItemType Directory -Force -Path $Dist | Out-Null

docker compose build app
if ($LASTEXITCODE -ne 0) { throw 'App image build failed.' }
docker image inspect 'regumate/app:contest-v3' *> $null
docker image inspect 'qdrant/qdrant:v1.18.0' *> $null

$Archive = Join-Path $Dist 'regumate-images.tar'
docker save --output $Archive 'regumate/app:contest-v3' 'qdrant/qdrant:v1.18.0'
if ($LASTEXITCODE -ne 0) { throw 'Docker image export failed.' }

$artifacts = @($Archive) + @(
    Get-ChildItem -LiteralPath (Join-Path $ProjectRoot 'data\models\bge-m3') -Recurse -File
) + @(
    Get-ChildItem -LiteralPath (Join-Path $ProjectRoot 'data\models\bge-reranker-v2-m3') -Recurse -File
)
$lines = foreach ($artifact in $artifacts) {
    if ($artifact -is [System.IO.FileInfo]) { $artifact = $artifact.FullName }
    if (-not (Test-Path -LiteralPath $artifact)) { throw "Missing delivery artifact: $artifact" }
    $hash = Get-FileHash -Algorithm SHA256 -LiteralPath $artifact
    $artifactFull = (Resolve-Path -LiteralPath $hash.Path).Path
    if (-not $artifactFull.StartsWith($ProjectRootFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Delivery artifact is outside the project root: $artifactFull"
    }
    $relative = $artifactFull.Substring($ProjectRootFull.Length).TrimStart('\').Replace('\', '/')
    "$($hash.Hash.ToLower())  $relative"
}
$lines | Set-Content -LiteralPath (Join-Path $Dist 'SHA256SUMS.txt') -Encoding utf8
Write-Host "Offline image archive created: $Archive" -ForegroundColor Green
