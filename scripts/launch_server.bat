@echo off
setlocal
cd /d "%~dp0.."

set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
set "CUDA_VISIBLE_DEVICES=1"
set "HF_HUB_OFFLINE=1"
set "PYTHONPATH=%~dp0..\src;%PYTHONPATH%"
set "PY=C:\Users\74727\miniconda3\envs\lerobot\python.exe"

if not exist "%PY%" (
    echo [ERROR] Python was not found at %PY%
    exit /b 2
)

set "PORT=8088"
if not "%1"=="" set "PORT=%1"

set "CKPT=pure_flow"
if not "%2"=="" set "CKPT=%2"

echo ===========================================================================
echo   FRANKA Pi0.5 VLA HIGH-PERFORMANCE INFERENCE SERVICE LAUNCHER
echo ===========================================================================
echo   Target GPU:     Physical GPU 1 (RTX 5090 32GB, CUDA_VISIBLE_DEVICES=1)
echo   Service Port:   %PORT%
echo   Default Ckpt:   %CKPT%
echo ===========================================================================

"%PY%" -u scripts\python\vla_inference_server.py --port %PORT% --checkpoint %CKPT%
exit /b %errorlevel%
