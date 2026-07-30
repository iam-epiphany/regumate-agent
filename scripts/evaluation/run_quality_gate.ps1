param(
    [ValidateSet("Quick", "Full")]
    [string]$Mode = "Quick"
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $ProjectRoot

python .\scripts\evaluation\run_quality_gate.py --mode $Mode
exit $LASTEXITCODE
