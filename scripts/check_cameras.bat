@echo off
setlocal
cd /d "%~dp0.."

set "PY=C:\Users\74727\miniconda3\envs\lerobot\python.exe"

if not exist "%PY%" (
    echo [ERROR] Canonical lerobot Python was not found: %PY%
    exit /b 2
)

echo ===============================================================================
echo   FRANKA REALSENSE DUAL-CAMERA TEST AND PREVIEW
echo ===============================================================================
echo   Python Env:  %PY%
echo   Front Cam:   Serial 254322072252
echo   Wrist Cam:   Serial 348122070854
echo   Output Dir:  outputs\camera_preview
echo ===============================================================================

"%PY%" -u tests\check_cameras.py %*
exit /b %errorlevel%
