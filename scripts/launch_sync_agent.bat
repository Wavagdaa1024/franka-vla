@echo off
setlocal
cd /d "%~dp0.."

set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
set "CUDA_VISIBLE_DEVICES=1"
set "HF_HUB_OFFLINE=1"
set "PY=C:\Users\74727\miniconda3\envs\lerobot\python.exe"

if not exist "%PY%" (
    echo [ERROR] Canonical lerobot Python was not found: %PY%
    exit /b 2
)

echo =========================================================================
echo   SYNCHRONOUS VLA AGENT LAUNCHER (PI0.5 + 5-DOF LOCK + FLOOR GUARD)
echo =========================================================================
echo   Mode:         SYNCHRONOUS (Stop-and-Go, 0 Motion Blur)
echo   Python:       %PY%
echo   Physical GPU: 1 (RTX 5090 32GB, GPU 0 STRICTLY ISOLATED)
echo   Floor Guard:  Z >= +7.0 mm
echo   Orientation:  Strict Vertical Downward (0.00 deg Tilt)
echo =========================================================================
"%PY%" -u scripts\sync_vla_agent.py %*
exit /b %errorlevel%
