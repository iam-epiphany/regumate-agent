@echo off
setlocal
cd /d "%~dp0"

set "BUILD_APP=0"
if /I "%~1"=="--build" set "BUILD_APP=1"
if /I "%~1"=="build" set "BUILD_APP=1"
if "%REGUMATE_BUILD_APP%"=="1" set "BUILD_APP=1"

where powershell.exe >nul 2>nul
if errorlevel 1 (
  echo [FAIL] PowerShell is required to start ReguMate.
  pause
  exit /b 1
)

if "%BUILD_APP%"=="1" (
  where docker.exe >nul 2>nul
  if errorlevel 1 (
    echo [FAIL] Docker is required to rebuild ReguMate.
    pause
    exit /b 1
  )

  echo Rebuilding ReguMate app image from current source...
  docker compose build app
  if errorlevel 1 (
    echo.
    echo [FAIL] ReguMate image rebuild failed.
    pause
    exit /b 1
  )
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_demo.ps1"
if errorlevel 1 (
  echo.
  echo ReguMate startup failed. Check the messages above, then run:
  echo   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\preflight.ps1
  pause
  exit /b 1
)

echo.
echo ReguMate is running at http://127.0.0.1:8000
pause
