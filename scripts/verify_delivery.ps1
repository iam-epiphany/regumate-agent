$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Manifest = Join-Path $ProjectRoot 'dist-delivery\SHA256SUMS.txt'

$SmallPackageHash = Join-Path $ProjectRoot 'dist-delivery\ReguMate-Agent.zip.sha256.txt'

function Test-HashManifest([string]$Path, [string]$BasePath) {
    $checked = 0
    foreach ($line in Get-Content -LiteralPath $Path -Encoding UTF8) {
        if (-not $line.Trim()) { continue }
        if ($line -notmatch '^([0-9a-fA-F]{64})\s{2}(.+)$') {
            throw "Invalid SHA256 entry: $line"
        }
        $expected = $Matches[1].ToLowerInvariant()
        $relative = $Matches[2].Replace('/', '\')
        $path = Join-Path $BasePath $relative
        if (-not (Test-Path -LiteralPath $path)) { throw "Missing delivery artifact: $relative" }
        $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash.ToLowerInvariant()
        if ($actual -ne $expected) { throw "SHA-256 mismatch: $relative" }
        $checked += 1
    }
    return $checked
}

if (Test-Path -LiteralPath $Manifest) {
    $checked = Test-HashManifest $Manifest $ProjectRoot
    Write-Host "Offline delivery integrity verified: $checked files." -ForegroundColor Green
    exit 0
}

if (Test-Path -LiteralPath $SmallPackageHash) {
    $checked = Test-HashManifest $SmallPackageHash (Join-Path $ProjectRoot 'dist-delivery')
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archivePath = Join-Path $ProjectRoot 'dist-delivery\ReguMate-Agent.zip'
    $archive = [System.IO.Compression.ZipFile]::OpenRead($archivePath)
    try {
        $badEntry = $archive.Entries | Where-Object {
            $_.FullName -match '(^|/)\.env\.example$'
        } | Select-Object -First 1
        if ($badEntry) {
            throw 'Small package contains duplicate .env.example.'
        }
    } finally {
        $archive.Dispose()
    }
    Write-Host "Small submission package integrity verified: $checked archive." -ForegroundColor Green
    exit 0
}

throw 'Missing dist-delivery\SHA256SUMS.txt and dist-delivery\ReguMate-Agent.zip.sha256.txt. Build a delivery or submission package first.'
