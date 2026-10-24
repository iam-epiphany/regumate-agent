param([switch]$Gpu)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $ProjectRoot
if (-not (Test-Path -LiteralPath '.env')) {
    throw 'Required root .env is missing. Restore the complete submission package.'
}
$images = Join-Path $ProjectRoot 'images\regumate-linux-amd64.tar'
if (-not (Test-Path -LiteralPath $images)) { throw 'Offline image archive is missing.' }
docker load -i $images
if ($LASTEXITCODE -ne 0) { throw 'Failed to load offline images.' }
$composeArgs = @('compose', '-f', 'docker-compose.offline.yml')
if ($Gpu) { $composeArgs += @('-f', 'docker-compose.offline.gpu.yml') }
& docker @composeArgs up -d --no-build --pull never
if ($LASTEXITCODE -ne 0) { throw 'Offline services did not start successfully.' }
& docker @composeArgs ps
Write-Host 'Use the SERVICE_PORT configured in .env. Default address: http://127.0.0.1:8000'
