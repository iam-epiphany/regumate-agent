@echo off
setlocal EnableExtensions
cd /d "%~dp0"

where powershell.exe >nul 2>nul
if errorlevel 1 (
  echo [FAIL] PowerShell is required to start ReguMate Docker development mode.
  pause
  exit /b 1
)

set "BUILD_ARG="
if /I "%~1"=="--build" set "BUILD_ARG=-Build"
if /I "%~1"=="build" set "BUILD_ARG=-Build"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_docker_dev.ps1" %BUILD_ARG%
if errorlevel 1 (
  echo.
  echo [FAIL] ReguMate Docker development startup failed.
  echo Check Docker Desktop and container logs:
  echo   docker logs regumate-dev-app
  echo   docker logs regumate-dev-frontend
  pause
  exit /b 1
)

pause
exit /b 0
