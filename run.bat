@echo off
setlocal
cd /d "%~dp0"

where powershell.exe >nul 2>nul
if errorlevel 1 (
  echo [FAIL] PowerShell is required to start ReguMate.
  pause
  exit /b 1
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
