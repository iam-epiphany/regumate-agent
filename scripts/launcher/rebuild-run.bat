@echo off
setlocal EnableExtensions
cd /d "%~dp0..\..\"

if not defined REGUMATE_BUILD_RETRY_LIMIT set "REGUMATE_BUILD_RETRY_LIMIT=5"
if not defined REGUMATE_BUILD_RETRY_DELAY_SECONDS set "REGUMATE_BUILD_RETRY_DELAY_SECONDS=10"

echo Rebuilding ReguMate Docker app image, then starting the standard Docker runtime.
echo Retry policy: %REGUMATE_BUILD_RETRY_LIMIT% attempts, %REGUMATE_BUILD_RETRY_DELAY_SECONDS%s delay.
echo.
call "%~dp0..\..\scripts\launcher\run.bat" --build
exit /b %errorlevel%
