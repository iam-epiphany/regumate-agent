param(
    [string]$SourceCacheRoot = "D:\AI-Cache",
    [string]$EmbeddingSource = "",
    [string]$RerankerSource = "",
    [string]$DestinationRoot = "data\models"
)

$ErrorActionPreference = "Stop"

function Test-ModelDirectory {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        return $false
    }

    $configPath = Join-Path $Path "config.json"
    $hasConfig = Test-Path -LiteralPath $configPath -PathType Leaf

    $tokenizerFiles = @("tokenizer.json", "tokenizer_config.json", "sentencepiece.bpe.model", "vocab.txt", "vocab.json")
    $hasTokenizer = $false
    foreach ($file in $tokenizerFiles) {
        $tokenizerPath = Join-Path $Path $file
        if (Test-Path -LiteralPath $tokenizerPath -PathType Leaf) {
            $hasTokenizer = $true
            break
        }
    }

    $weightFiles = Get-ChildItem -LiteralPath $Path -File -ErrorAction SilentlyContinue | Where-Object {
        $_.Name -like "*.safetensors" -or $_.Name -like "*.bin"
    }
    $hasWeights = [bool]$weightFiles

    return ($hasConfig -and $hasTokenizer -and $hasWeights)
}

function Resolve-ModelSource {
    param(
        [string]$ExplicitSource,
        [string]$SourceCacheRoot,
        [string]$RepoCacheName,
        [string]$DisplayName
    )

    if ($ExplicitSource) {
        $resolved = Resolve-Path -LiteralPath $ExplicitSource -ErrorAction Stop
        return $resolved.Path
    }

    $candidates = @(
        (Join-Path $SourceCacheRoot "huggingface\hub\$RepoCacheName\snapshots"),
        (Join-Path $SourceCacheRoot "huggingface\$RepoCacheName\snapshots")
    )

    foreach ($snapshotsDir in $candidates) {
        if (Test-Path -LiteralPath $snapshotsDir -PathType Container) {
            $snapshots = Get-ChildItem -LiteralPath $snapshotsDir -Directory | Sort-Object LastWriteTime -Descending
            foreach ($snapshot in $snapshots) {
                if (Test-ModelDirectory -Path $snapshot.FullName) {
                    return $snapshot.FullName
                }
            }
        }
    }

    throw "Model cache not found for $DisplayName. Pass an explicit source directory or check $SourceCacheRoot."
}

function Copy-ModelDirectory {
    param(
        [string]$Source,
        [string]$Destination,
        [string]$DisplayName
    )

    if (Test-Path -LiteralPath $Destination) {
        Remove-Item -LiteralPath $Destination -Recurse -Force
    }

    New-Item -ItemType Directory -Path $Destination | Out-Null
    Get-ChildItem -LiteralPath $Source -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $Destination -Recurse -Force
    }

    if (-not (Test-ModelDirectory -Path $Destination)) {
        throw "$DisplayName copy completed, but destination is incomplete: $Destination"
    }

    Write-Host "$DisplayName prepared at: $Destination"
}

$destinationRootPath = Join-Path (Get-Location) $DestinationRoot
$embeddingDestination = Join-Path $destinationRootPath "bge-m3"
$rerankerDestination = Join-Path $destinationRootPath "bge-reranker-v2-m3"

$embeddingSourcePath = Resolve-ModelSource `
    -ExplicitSource $EmbeddingSource `
    -SourceCacheRoot $SourceCacheRoot `
    -RepoCacheName "models--BAAI--bge-m3" `
    -DisplayName "BGE-M3"

$rerankerSourcePath = Resolve-ModelSource `
    -ExplicitSource $RerankerSource `
    -SourceCacheRoot $SourceCacheRoot `
    -RepoCacheName "models--BAAI--bge-reranker-v2-m3" `
    -DisplayName "BGE reranker"

New-Item -ItemType Directory -Path $destinationRootPath -Force | Out-Null
Copy-ModelDirectory -Source $embeddingSourcePath -Destination $embeddingDestination -DisplayName "BGE-M3"
Copy-ModelDirectory -Source $rerankerSourcePath -Destination $rerankerDestination -DisplayName "BGE reranker"

Write-Host "Offline model directories are ready. Include data\models\bge-m3 and data\models\bge-reranker-v2-m3 in the delivery package."
