param(
    [ValidateRange(5, 600)]
    [int]$TimeoutSeconds = 60,
    [ValidateRange(10, 1440)]
    [int]$IngestTimeoutMinutes = 360,
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [switch]$SkipWarmup,
    [switch]$AllowCpu
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
$BaseUrlWasSpecified = $PSBoundParameters.ContainsKey("BaseUrl")
$QdrantUrl = "http://127.0.0.1:6333"

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "== $Message ==" -ForegroundColor Cyan
}

function Get-ReguMateRuntimeState {
    $statePath = Join-Path $ProjectRoot ".run-state\runtime.json"
    if (-not (Test-Path -LiteralPath $statePath)) { return $null }
    try {
        return Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    } catch {
        return $null
    }
}

function Sync-ReguMateRuntimeUrls {
    $state = Get-ReguMateRuntimeState
    if (-not $state) { return }
    if (-not $script:BaseUrlWasSpecified -and $state.app_url) {
        $script:BaseUrl = [string]$state.app_url
    }
    if ($state.qdrant_url) {
        $script:QdrantUrl = [string]$state.qdrant_url
    }
}

function Invoke-JsonRequest([string]$Method, [string]$Uri, [object]$Body = $null, [int]$Timeout = 60) {
    if ($null -eq $Body) {
        return Invoke-RestMethod -Method $Method -Uri $Uri -TimeoutSec $Timeout
    }
    return Invoke-RestMethod -Method $Method -Uri $Uri -Body $Body -ContentType "application/json" -TimeoutSec $Timeout
}

function Assert-GpuReady([string]$Url) {
    $ready = Invoke-RestMethod -Uri "$Url/api/health/ready" -TimeoutSec 30
    $device = $ready.model_device
    Write-Host "Model device: selected_device=$($device.selected_device), cuda_available=$($device.cuda_available), device=$($device.cuda_device_name)"
    if ($device.selected_device -ne "cuda") {
        Write-Host "ReguMate is using CPU fallback. fallback_reason=$($device.fallback_reason)" -ForegroundColor Yellow
    }
}

function Get-ExistingDocuments([string]$Url) {
    $payload = Invoke-RestMethod -Uri "$Url/api/documents" -TimeoutSec 60
    $map = @{}
    foreach ($document in @($payload.documents)) {
        if ($document.filename -and -not $map.ContainsKey($document.filename)) {
            $map[$document.filename] = $document
        }
        $fixedName = Repair-Utf8Mojibake $document.filename
        if ($fixedName -and -not $map.ContainsKey($fixedName)) {
            $map[$fixedName] = $document
        }
    }
    return $map
}

function Repair-Utf8Mojibake([string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value)) { return $Value }
    if ($Value -notmatch '[\u0080-\u00FF]') {
        return $Value
    }
    try {
        $bytes = [System.Text.Encoding]::GetEncoding('ISO-8859-1').GetBytes($Value)
        return [System.Text.Encoding]::UTF8.GetString($bytes)
    } catch {
        return $Value
    }
}

function Get-ContentType([string]$Extension) {
    switch ($Extension.ToLowerInvariant()) {
        ".txt" { return "text/plain" }
        ".md" { return "text/markdown" }
        ".doc" { return "application/msword" }
        ".docx" { return "application/vnd.openxmlformats-officedocument.wordprocessingml.document" }
        ".pdf" { return "application/pdf" }
        ".xls" { return "application/vnd.ms-excel" }
        ".xlsx" { return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" }
        default { return "application/octet-stream" }
    }
}

function Send-ContestFile([string]$Url, [System.IO.FileInfo]$File) {
    Add-Type -AssemblyName System.Net.Http
    $client = [System.Net.Http.HttpClient]::new()
    $client.Timeout = [TimeSpan]::FromSeconds([Math]::Max($TimeoutSeconds, 300))
    $content = [System.Net.Http.MultipartFormDataContent]::new()
    $bytes = [System.IO.File]::ReadAllBytes($File.FullName)
    $fileContent = [System.Net.Http.ByteArrayContent]::new($bytes)
    $fileContent.Headers.ContentType = [System.Net.Http.Headers.MediaTypeHeaderValue]::Parse((Get-ContentType $File.Extension))
    $content.Add($fileContent, "file", $File.Name)
    $request = [System.Net.Http.HttpRequestMessage]::new([System.Net.Http.HttpMethod]::Post, "$Url/api/documents/upload")
    $request.Headers.Add("Idempotency-Key", "contest-dataset-$((Get-FileHash -Algorithm SHA256 -LiteralPath $File.FullName).Hash.ToLowerInvariant().Substring(0, 24))")
    $request.Content = $content
    $response = $client.SendAsync($request).GetAwaiter().GetResult()
    $raw = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
    if (-not $response.IsSuccessStatusCode) {
        throw "Upload failed for $($File.Name): HTTP $([int]$response.StatusCode) $raw"
    }
    return $raw | ConvertFrom-Json
}

function Wait-ContestIndexing([string]$Url, [string[]]$DocumentIds) {
    $targetIds = @($DocumentIds | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)
    if ($targetIds.Count -eq 0) {
        throw "No document IDs were collected for indexing."
    }
    if ($targetIds.Count -ne $DocumentIds.Count) {
        Write-Host "Index wait target has $($targetIds.Count) unique document IDs for $($DocumentIds.Count) attachment records; duplicate upload responses will be counted once." -ForegroundColor Yellow
    }

    $deadline = (Get-Date).AddMinutes($IngestTimeoutMinutes)
    $lastPrint = [datetime]::MinValue
    do {
        $payload = Invoke-RestMethod -Uri "$Url/api/documents" -TimeoutSec $TimeoutSeconds
        $documentsById = @{}
        foreach ($document in @($payload.documents)) {
            if ($document.document_id -and -not $documentsById.ContainsKey($document.document_id)) {
                $documentsById[$document.document_id] = $document
            }
        }
        $tracked = @($targetIds | Where-Object { $documentsById.ContainsKey($_) } | ForEach-Object { $documentsById[$_] })
        $missingIds = @($targetIds | Where-Object { -not $documentsById.ContainsKey($_) })
        $indexed = @($tracked | Where-Object { $_.status -eq "indexed" }).Count
        $activeDocs = @($tracked | Where-Object { $_.status -in @("uploaded", "index_queued", "indexing") })
        $failedDocs = @($tracked | Where-Object { $_.status -in @("index_failed", "source_missing", "deleting", "delete_failed") })
        $otherDocs = @($tracked | Where-Object { $_.status -notin @("indexed", "uploaded", "index_queued", "indexing", "index_failed", "source_missing", "deleting", "delete_failed") })
        if (((Get-Date) - $lastPrint).TotalSeconds -ge 5) {
            $queued = @($activeDocs | Where-Object { $_.status -eq "index_queued" }).Count
            $indexing = @($activeDocs | Where-Object { $_.status -eq "indexing" }).Count
            $uploaded = @($activeDocs | Where-Object { $_.status -eq "uploaded" }).Count
            Write-Host "Index progress: indexed=$indexed/$($targetIds.Count), queued=$queued, indexing=$indexing, uploaded=$uploaded, failed=$($failedDocs.Count), missing=$($missingIds.Count), other=$($otherDocs.Count)"
            $lastPrint = Get-Date
        }
        if ($indexed -eq $targetIds.Count) {
            Write-Host "Index completed: $indexed/$($targetIds.Count)"
            return
        }
        if ($activeDocs.Count -eq 0) {
            $examples = @()
            $examples += $failedDocs | Select-Object -First 10 | ForEach-Object { "$($_.filename) document_id=$($_.document_id) status=$($_.status) error=$($_.index_error)" }
            $remaining = 10 - $examples.Count
            if ($remaining -gt 0) {
                $examples += $otherDocs | Select-Object -First $remaining | ForEach-Object { "$($_.filename) document_id=$($_.document_id) status=$($_.status) error=$($_.index_error)" }
            }
            $remaining = 10 - $examples.Count
            if ($remaining -gt 0) {
                $examples += $missingIds | Select-Object -First $remaining | ForEach-Object { "document_id=$_ was not returned by /api/documents" }
            }
            if ($examples.Count -eq 0) {
                $examples = @("No queued, indexing, uploaded, failed, missing, or unexpected documents were visible; rerun the script or check /api/documents.")
            }
            throw "Contest indexing stopped before all target documents were indexed: indexed=$indexed/$($targetIds.Count), failed=$($failedDocs.Count), missing=$($missingIds.Count), other=$($otherDocs.Count).$([Environment]::NewLine)$($examples -join [Environment]::NewLine)"
        }
        Start-Sleep -Seconds 5
    } while ((Get-Date) -lt $deadline)
    throw ("Indexing timed out: {0} minutes elapsed before {1} target documents were indexed." -f $IngestTimeoutMinutes, $targetIds.Count)
}

function Get-QdrantPoints([string]$Collection) {
    try {
        $uri = $QdrantUrl + '/collections/' + $Collection
        $payload = Invoke-RestMethod -Uri $uri -TimeoutSec 10
        return [int]$payload.result.points_count
    } catch {
        return $null
    }
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker was not found. Install and start Docker Desktop first."
}

if ($AllowCpu) {
    $env:MODEL_DEVICE = "cpu"
    $env:REGUMATE_TORCH_FLAVOR = "cpu"
    if (-not $env:REGUMATE_APP_IMAGE) {
        $env:REGUMATE_APP_IMAGE = "regumate/app:contest-cpu"
    }
}

Write-Step "Check and start ReguMate"
$AppImage = if ($env:REGUMATE_APP_IMAGE) { $env:REGUMATE_APP_IMAGE } else { "regumate/app:contest-v3" }
docker image inspect $AppImage *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "$AppImage is missing. Building from source..." -ForegroundColor Yellow
    docker compose build app
    if ($LASTEXITCODE -ne 0) { throw "ReguMate app image build failed." }
}

powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_demo.ps1
if ($LASTEXITCODE -ne 0) { throw "ReguMate startup failed." }
Sync-ReguMateRuntimeUrls
Write-Host "Using ReguMate API: $BaseUrl"

Write-Step "Check GPU and contest data"
Assert-GpuReady $BaseUrl
$contestRoot = Join-Path $ProjectRoot "data\contest_dataset"
$attachments = Join-Path $contestRoot "dataset\nfra_page_attachments_500"
$qa = Join-Path $contestRoot "QA$([char]0x6570)$([char]0x636E).xlsx"
if (-not (Test-Path -LiteralPath $attachments)) {
    throw "Contest attachment directory was not found: $attachments"
}
if (-not (Test-Path -LiteralPath $qa)) {
    throw "Contest QA workbook was not found: $qa"
}
$files = @(Get-ChildItem -LiteralPath $attachments -File | Where-Object {
    $_.Extension.ToLowerInvariant() -in ".txt", ".md", ".doc", ".docx", ".pdf", ".xls", ".xlsx"
} | Sort-Object Name)
if ($files.Count -ne 500) {
    throw "Official attachment count is $($files.Count), expected 500."
}
Write-Host "Attachment check passed: 500 files"

if (-not $SkipWarmup) {
    Write-Step "Warm up models"
    & "$PSScriptRoot\warmup_models.ps1" -BaseUrl $BaseUrl -TimeoutSeconds ([Math]::Max($TimeoutSeconds, 300))
}

Write-Step "Upload contest_dataset knowledge base"
$ready = Invoke-RestMethod -Uri "$BaseUrl/api/health/ready" -TimeoutSec 30
$initialPoints = Get-QdrantPoints $ready.qdrant_collection
$forceReindex = $false
if ($null -ne $initialPoints) {
    Write-Host "Current Qdrant collection: $($ready.qdrant_collection), points=$initialPoints"
    if ($initialPoints -lt 10000) {
        $forceReindex = $true
        Write-Host "Current Qdrant collection has too few vector points; existing indexed contest documents will be requeued." -ForegroundColor Yellow
    }
} else {
    Write-Host "Current Qdrant collection: $($ready.qdrant_collection), points=unknown"
}
$existing = Get-ExistingDocuments $BaseUrl
$tracked = New-Object System.Collections.Generic.List[string]
$uploaded = 0
$skipped = 0
$requeued = 0
$index = 0
foreach ($file in $files) {
    $index += 1
    if ($existing.ContainsKey($file.Name)) {
        $document = $existing[$file.Name]
        $tracked.Add([string]$document.document_id)
        if ($document.status -eq "indexed" -and -not $forceReindex) {
            $skipped += 1
            Write-Host ("[{0:000}/500] Already indexed: {1}" -f $index, $file.Name)
            continue
        }
        if ($document.status -eq "indexed" -and $forceReindex) {
            Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/documents/$($document.document_id)/index" -TimeoutSec $TimeoutSeconds | Out-Null
            $requeued += 1
            Write-Host ("[{0:000}/500] Requeued indexed document for current Qdrant collection: {1}" -f $index, $file.Name)
            continue
        }
        if ($document.status -in @("index_failed", "uploaded", "index_queued", "indexing")) {
            Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/documents/$($document.document_id)/index" -TimeoutSec $TimeoutSeconds | Out-Null
            $requeued += 1
            Write-Host ("[{0:000}/500] Requeued existing document for indexing: {1} status={2}" -f $index, $file.Name, $document.status)
            continue
        }
    }
    $response = Send-ContestFile $BaseUrl $file
    $tracked.Add([string]$response.document_id)
    $uploaded += 1
    Write-Host ("[{0:000}/500] Uploaded: {1} -> {2}" -f $index, $file.Name, $response.document_id)
}
Write-Host "Upload stage completed: uploaded=$uploaded, skipped=$skipped, requeued=$requeued"

Wait-ContestIndexing $BaseUrl $tracked.ToArray()
$ready = Invoke-RestMethod -Uri "$BaseUrl/api/health/ready" -TimeoutSec 30
$points = Get-QdrantPoints $ready.qdrant_collection
if ($null -ne $points) {
    Write-Host "Current Qdrant collection points=$points"
    if ($points -lt 10000) {
        throw "Current Qdrant collection has too few vector points, points=$points expected_at_least=10000."
    }
}

Write-Host ""
Write-Host "contest_dataset knowledge-base upload completed." -ForegroundColor Green
