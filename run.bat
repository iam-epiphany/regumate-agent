@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "APP_IMAGE=regumate/app:contest-v3"
if exist "%~dp0.env" (
  for /f "tokens=1,* delims==" %%A in ('findstr /b "REGUMATE_APP_IMAGE=" "%~dp0.env" 2^>nul') do set "APP_IMAGE=%%B"
)
if defined REGUMATE_APP_IMAGE set "APP_IMAGE=%REGUMATE_APP_IMAGE%"
if not defined REGUMATE_BUILD_RETRY_LIMIT set "REGUMATE_BUILD_RETRY_LIMIT=5"
if not defined REGUMATE_BUILD_RETRY_DELAY_SECONDS set "REGUMATE_BUILD_RETRY_DELAY_SECONDS=10"
set "BUILD_RETRY_LIMIT=%REGUMATE_BUILD_RETRY_LIMIT%"
set "BUILD_RETRY_DELAY_SECONDS=%REGUMATE_BUILD_RETRY_DELAY_SECONDS%"
set "BUILD_APP=0"
set "HAVE_APP_IMAGE=0"
set "CLEANUP_OLD_APP_IMAGE_ID="
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
  call :build_app_with_retry
  if errorlevel 1 (
    echo.
    if "%HAVE_APP_IMAGE%"=="1" (
      echo [WARN] ReguMate image rebuild failed. Starting with existing image !OLD_APP_IMAGE_ID!.
      echo [WARN] Re-run rebuild-run.bat after network or Docker build issues are fixed to refresh the image.
    ) else (
      echo [FAIL] ReguMate image rebuild failed and no existing ReguMate app image is available.
      pause
      exit /b 1
    )
  ) else (
    set "NEW_APP_IMAGE_ID="
    for /f "usebackq delims=" %%I in (`docker image inspect "%APP_IMAGE%" --format "{{.Id}}" 2^>nul`) do set "NEW_APP_IMAGE_ID=%%I"
    if defined OLD_APP_IMAGE_ID if defined NEW_APP_IMAGE_ID if not "!OLD_APP_IMAGE_ID!"=="!NEW_APP_IMAGE_ID!" (
      set "CLEANUP_OLD_APP_IMAGE_ID=!OLD_APP_IMAGE_ID!"
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

if defined CLEANUP_OLD_APP_IMAGE_ID (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\cleanup_regumate_images.ps1" -OldImageId "!CLEANUP_OLD_APP_IMAGE_ID!" -CurrentImage "%APP_IMAGE%"
) else (
  powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\cleanup_regumate_images.ps1" -CurrentImage "%APP_IMAGE%"
)

pause
exit /b 0

:build_app_with_retry
set "BUILD_ATTEMPT=1"
:build_retry_loop
echo Build attempt !BUILD_ATTEMPT!/%BUILD_RETRY_LIMIT%...
docker compose build app
if not errorlevel 1 exit /b 0
if !BUILD_ATTEMPT! GEQ %BUILD_RETRY_LIMIT% (
  echo [FAIL] ReguMate app image build failed after %BUILD_RETRY_LIMIT% attempts.
  exit /b 1
)
echo [WARN] ReguMate app image build attempt !BUILD_ATTEMPT! failed. Retrying in %BUILD_RETRY_DELAY_SECONDS% seconds...
timeout /t %BUILD_RETRY_DELAY_SECONDS% /nobreak >nul
set /a BUILD_ATTEMPT+=1
goto build_retry_loop
