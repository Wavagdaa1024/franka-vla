@echo off
setlocal
cd /d "%~dp0"

set "PY=C:\Users\74727\miniconda3\envs\lerobot\python.exe"

if not exist "%PY%" (
    echo [ERROR] Python environment not found: %PY%
    pause
    exit /b 1
)

echo ===============================================================================
echo   FRANKA THIRD-PERSON CAMERA FAST ALIGNMENT AND HAND-EYE TOOL
echo ===============================================================================
echo   Target Camera: RealSense D435I (Serial: 254322072252)
echo   Target Marker: 60.0 mm ChArUco Board (4x4, Dict: 4x4_50)
echo   Local Web URL: http://localhost:8088
echo   LAN Web URL:   http://10.70.242.38:8088
echo   Desktop Window: OpenCV Preview Window (Press Q / ESC to exit)
echo   Controls:
echo     [S] / Web Button - Save current pose as Golden Baseline
echo     [G] / Web Button - Toggle Semi-transparent Ghost Overlay
echo     [R] / Web Button - Clear saved baseline
echo ===============================================================================

"%PY%" -u align_front_camera.py %*
if errorlevel 1 (
    echo.
    echo [ERROR] Program terminated with an error.
    pause
)
