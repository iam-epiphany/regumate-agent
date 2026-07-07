@echo off
cd /d "%~dp0"
powershell -NoExit -ExecutionPolicy Bypass -Command ^
  "& { . .\.venv\Scripts\Activate.ps1; python -m pip install -r requirements.txt; cd frontend; if (-not (Test-Path node_modules)) { npm install }; npm run build; cd ..; $port = if (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue) { 8001 } else { 8000 }; Write-Host ('ReguMate is starting at http://127.0.0.1:' + $port + '/'); python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port $port }"
