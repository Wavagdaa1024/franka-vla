@echo off
setlocal
cd /d "%~dp0.."

echo ===============================================================================
echo [NOTICE] Defaulting to Synchronous Agent (Stop-and-Go).
echo [TIP] To use Asynchronous RTC, run: scripts\run_async_agent.bat
echo ===============================================================================
call "%~dp0run_sync_agent.bat" %*
exit /b %errorlevel%
