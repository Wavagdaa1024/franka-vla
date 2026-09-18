@echo off
setlocal
cd /d "%~dp0.."

set "PYTHONPATH=%~dp0..\src;%PYTHONPATH%"
set "PY=C:\Users\74727\miniconda3\envs\lerobot\python.exe"

if not exist "%PY%" (
    echo [ERROR] Canonical lerobot Python was not found: %PY%
    exit /b 2
)

echo ===============================================================================
echo   FRANKA REALSENSE CONTINUOUS DUAL-CAMERA STREAMER
echo ===============================================================================
echo   Python Env:  %PY%
echo   Front Cam:   Serial 254322072252
echo   Wrist Cam:   Serial 348122070854
echo   Resolution:  640x480 @ 30 FPS
echo   Web Stream:  http://0.0.0.0:5000 (LAN Accessible)
echo ===============================================================================

"%PY%" -u scripts\live_camera_stream.py %*
exit /b %errorlevel%
