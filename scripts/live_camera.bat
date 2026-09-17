@echo off
setlocal
cd /d "%~dp0.."

set "PY=C:\Users\74727\miniconda3\envs\lerobot\python.exe"

if not exist "%PY%" (
    echo [ERROR] Canonical lerobot Python was not found: %PY%
    exit /b 2
)

echo ===============================================================================
echo   FRANKA REALSENSE CONTINUOUS LIVE CAMERA STREAM
echo ===============================================================================
echo   Local Web URL:  http://localhost:8080
echo   LAN Web URL:    http://10.70.242.38:8080
echo   Desktop Window: OpenCV Side-by-Side Dual View (Press Q to exit)
echo ===============================================================================

"%PY%" -u scripts\live_camera_stream.py %*
exit /b %errorlevel%
