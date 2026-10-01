$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Manifest = Join-Path $ProjectRoot 'dist-delivery\SHA256SUMS.txt'
$SubmissionRoot = Join-Path $ProjectRoot 'dist-delivery\ReguMate-Agent'
$SubmissionManifest = Join-Path $SubmissionRoot 'FILE_MANIFEST.sha256'

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

function Test-NoForbiddenFiles([string]$Root) {
    $iterationLogFileName = -join ([char[]](
        0x7cfb, 0x7edf, 0x4f18, 0x5316, 0x4e0e, 0x8bc4, 0x6d4b,
        0x8fed, 0x4ee3, 0x8bb0, 0x5f55, 0x2e, 0x6d, 0x64
    ))
    $forbiddenNames = @(
        'AGENTS.md',
        '.env.example',
        'CODEX_OBJECTIVE.md',
        'CODEX_STATUS.md',
        $iterationLogFileName
    )
    $bad = Get-ChildItem -LiteralPath $Root -Recurse -File -Force | Where-Object {
        $relative = $_.FullName.Substring($Root.Length).TrimStart('\')
        $_.Name -in $forbiddenNames -or
            $relative -match '(^|\\)(\.git|\.idea|\.pytest_cache|__pycache__|node_modules|tmp)(\\|$)' -or
            $relative -match '(^|\\)frontend\\dist\\' -or
            $relative -match '(^|\\)data\\(models|qdrant)\\' -or
            $relative -match '(^|\\)data\\evaluation\\final_runtime\\' -or
            $relative -match 'self[-_]made|contest-data-self-made|regression_round|hard_challenge_50'
    } | Select-Object -First 1
    if ($bad) {
        throw "Delivery contains forbidden file: $($bad.FullName)"
    }
}

if (Test-Path -LiteralPath $Manifest) {
    $checked = Test-HashManifest $Manifest $ProjectRoot
    Test-NoForbiddenFiles (Join-Path $ProjectRoot 'dist-delivery')
    Write-Host "Offline delivery integrity verified: $checked files." -ForegroundColor Green
    exit 0
}

if (Test-Path -LiteralPath $SubmissionManifest) {
    $checked = Test-HashManifest $SubmissionManifest $SubmissionRoot
    $listed = @(
        Get-Content -LiteralPath $SubmissionManifest -Encoding UTF8 |
            Where-Object { $_.Trim() } |
            ForEach-Object {
                if ($_ -notmatch '^[0-9a-fA-F]{64}\s{2}(.+)$') { throw "Invalid SHA256 entry: $_" }
                $Matches[1].Replace('/', '\')
            }
    )
    $actualFiles = @(
        Get-ChildItem -LiteralPath $SubmissionRoot -Recurse -File -Force |
            Where-Object { $_.FullName -ne $SubmissionManifest } |
            ForEach-Object { $_.FullName.Substring($SubmissionRoot.Length).TrimStart('\') }
    )
    if (@(Compare-Object $listed $actualFiles).Count -ne 0) {
        throw 'Submission file set does not match FILE_MANIFEST.sha256.'
    }
    Test-NoForbiddenFiles $SubmissionRoot
    Write-Host "Submission directory integrity verified: $checked files." -ForegroundColor Green
    exit 0
}

throw 'Missing dist-delivery\SHA256SUMS.txt and dist-delivery\ReguMate-Agent\FILE_MANIFEST.sha256. Build a delivery or submission directory first.'
