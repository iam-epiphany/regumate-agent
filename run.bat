@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "APP_IMAGE=regumate/app:contest-v3"
set "BUILD_APP=0"
set "HAVE_APP_IMAGE=0"
if /I "%~1"=="--build" set "BUILD_APP=1"
if /I "%~1"=="build" set "BUILD_APP=1"
if "%REGUMATE_BUILD_APP%"=="1" set "BUILD_APP=1"

where docker.exe >nul 2>nul
if errorlevel 1 (
  echo [FAIL] Docker is required to start ReguMate.
  pause
  exit /b 1
)

docker image inspect "%APP_IMAGE%" >nul 2>nul
if errorlevel 1 (
  echo ReguMate app image is missing. It will be built from source.
  set "BUILD_APP=1"
) else (
  set "HAVE_APP_IMAGE=1"
)

where powershell.exe >nul 2>nul
if errorlevel 1 (
  echo [FAIL] PowerShell is required to start ReguMate.
  pause
  exit /b 1
)

if "%BUILD_APP%"=="1" (
  set "OLD_APP_IMAGE_ID="
  for /f "usebackq delims=" %%I in (`docker image inspect "%APP_IMAGE%" --format "{{.Id}}" 2^>nul`) do set "OLD_APP_IMAGE_ID=%%I"

  echo Rebuilding ReguMate app image from current source...
  docker compose build app
  if errorlevel 1 (
    echo.
    if "%HAVE_APP_IMAGE%"=="1" (
      echo [WARN] ReguMate image rebuild failed. Starting with existing image !OLD_APP_IMAGE_ID!.
      echo [WARN] Re-run run-dev.bat after network or Docker build issues are fixed to refresh the image.
    ) else (
      echo [FAIL] ReguMate image rebuild failed and no existing ReguMate app image is available.
      pause
      exit /b 1
    )
  ) else (
    set "NEW_APP_IMAGE_ID="
    for /f "usebackq delims=" %%I in (`docker image inspect "%APP_IMAGE%" --format "{{.Id}}" 2^>nul`) do set "NEW_APP_IMAGE_ID=%%I"
    if defined OLD_APP_IMAGE_ID if defined NEW_APP_IMAGE_ID if not "!OLD_APP_IMAGE_ID!"=="!NEW_APP_IMAGE_ID!" (
      echo Removing previous ReguMate app image !OLD_APP_IMAGE_ID!...
      docker image rm "!OLD_APP_IMAGE_ID!" >nul 2>nul
      if errorlevel 1 (
        echo [WARN] Previous ReguMate app image could not be removed. It may still be used by an existing container.
      )
    )
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

pause
