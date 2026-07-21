$ErrorActionPreference = 'Continue'

$ProjectRoot = Split-Path -Parent $PSScriptRoot

foreach ($container in @('regumate-dev-frontend', 'regumate-dev-app', 'regumate-dev-qdrant')) {
    docker rm -f $container *> $null
}

$stateDir = Join-Path $ProjectRoot '.run-state'
foreach ($stateFile in @('docker-dev.json', 'runtime.json')) {
    $path = Join-Path $stateDir $stateFile
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
    }
}
