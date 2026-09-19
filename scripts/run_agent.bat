@echo off
setlocal
cd /d "%~dp0.."

set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
set "CUDA_VISIBLE_DEVICES=1"
set "HF_HUB_OFFLINE=1"
set "PYTHONPATH=%~dp0..\src;%PYTHONPATH%"
set "PY=C:\Users\74727\miniconda3\envs\lerobot\python.exe"

if not exist "%PY%" (
    echo [ERROR] Canonical lerobot Python was not found: %PY%
    exit /b 2
)

echo ===============================================================================
echo   FRANKA VLA UNIFIED AGENT LAUNCHER (GPU 1, RTX 5090 32GB)
echo ===============================================================================
echo   Franka Host:  10.197.16.43:8765
echo   Action Mode:  jointpos
echo   Default Ckpt: pi05_lora_pure_flow_50k\latest.pt (Canonical 50k Pure Flow)
echo   Python Env:   %PY%
echo ===============================================================================

"%PY%" -u scripts\python\async_rtc_vla_agent.py %*
exit /b %errorlevel%
