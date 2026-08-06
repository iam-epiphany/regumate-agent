@echo off
setlocal
cd /d "%~dp0..\..\"

where powershell.exe >nul 2>nul
if not errorlevel 1 (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\..\scripts\stop_local_dev.ps1"
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0..\..\scripts\stop_docker_dev.ps1"
)

where docker.exe >nul 2>nul
if errorlevel 1 (
  echo [OK] Local ReguMate hot-reload processes were stopped if they were running.
  echo Docker was not found, so no Docker services were stopped.
  pause
  exit /b 0
)

echo Stopping ReguMate Docker services...
docker compose down
if errorlevel 1 (
  echo.
  echo [FAIL] ReguMate shutdown failed. You can also stop regumate-app and regumate-qdrant in Docker Desktop.
  pause
  exit /b 1
)

echo.
echo ReguMate services have been stopped.
echo Local data, models, Qdrant storage, and evaluation reports were not deleted.
pause
