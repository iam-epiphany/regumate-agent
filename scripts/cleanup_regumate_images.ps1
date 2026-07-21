param(
    [string]$OldImageId = "",
    [string]$CurrentImage = "regumate/app:contest-v3"
)

$ErrorActionPreference = "Stop"

function Write-Skip([string]$Message) {
    if ($env:REGUMATE_VERBOSE_IMAGE_CLEANUP -eq "1") {
        Write-Host "[SKIP] $Message"
    }
}

function Write-Warn([string]$Message) {
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Get-ImageId([string]$Image) {
    if ([string]::IsNullOrWhiteSpace($Image)) { return "" }
    try {
        $id = docker image inspect $Image --format "{{.Id}}" 2>$null
    } catch {
        return ""
    }
    if ($LASTEXITCODE -ne 0) { return "" }
    return ([string]$id).Trim()
}

function Test-ReguMateAppImage([string]$ImageId) {
    if ([string]::IsNullOrWhiteSpace($ImageId)) { return $false }
    try {
        $role = docker image inspect $ImageId --format "{{ index .Config.Labels ""org.regumate.image.role"" }}" 2>$null
        if ($LASTEXITCODE -ne 0) { return $false }
        $title = docker image inspect $ImageId --format "{{ index .Config.Labels ""org.opencontainers.image.title"" }}" 2>$null
        if ($LASTEXITCODE -ne 0) { return $false }
    } catch {
        return $false
    }
    return ([string]$role).Trim() -eq "app" -and ([string]$title).Trim() -eq "ReguMate Agent"
}

function Remove-ImageIfPossible([string]$ImageId, [string]$Reason) {
    if ([string]::IsNullOrWhiteSpace($ImageId)) { return }
    $existingImageId = Get-ImageId $ImageId
    if ([string]::IsNullOrWhiteSpace($existingImageId)) {
        Write-Skip "Image $ImageId is already absent."
        return
    }
    Write-Host "Removing old ReguMate app image $existingImageId ($Reason)..."
    try {
        docker image rm $existingImageId *> $null
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "Could not remove image $existingImageId. It may still be used by an existing container."
        }
    } catch {
        Write-Warn "Could not remove image $existingImageId. It may still be used by an existing container."
    }
}

if ($env:REGUMATE_SKIP_IMAGE_CLEANUP -eq "1") {
    Write-Skip "REGUMATE_SKIP_IMAGE_CLEANUP=1; leaving Docker images untouched."
    exit 0
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Warn "Docker is unavailable; skipping ReguMate image cleanup."
    exit 0
}

$currentImageId = Get-ImageId $CurrentImage
if ([string]::IsNullOrWhiteSpace($currentImageId)) {
    Write-Warn "Current ReguMate image was not found: $CurrentImage"
}

$old = $OldImageId.Trim()
if ($old) {
    if ($currentImageId -and $old -eq $currentImageId) {
        Write-Skip "Old image ID is the current image; nothing to remove."
    } else {
        Remove-ImageIfPossible $old "previous app image"
    }
}

$danglingIds = @(docker image ls --filter "dangling=true" --quiet 2>$null | Where-Object {
    -not [string]::IsNullOrWhiteSpace($_)
} | Select-Object -Unique)
if ($LASTEXITCODE -ne 0 -or $danglingIds.Count -eq 0) {
    exit 0
}

foreach ($danglingId in $danglingIds) {
    $fullDanglingId = Get-ImageId $danglingId
    if ([string]::IsNullOrWhiteSpace($fullDanglingId)) { continue }
    if ($currentImageId -and $fullDanglingId -eq $currentImageId) { continue }
    if (Test-ReguMateAppImage $fullDanglingId) {
        Remove-ImageIfPossible $fullDanglingId "dangling ReguMate app image"
    }
}
