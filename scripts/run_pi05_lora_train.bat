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

echo ===========================================================================
echo   PI0.5 MULTI-TASK LoRA FINE-TUNING LAUNCHER (GPU 1)
echo ===========================================================================
echo   Python: %PY%
echo   Physical GPU: 1 (logical cuda:0)
echo   Target: outputs\checkpoints\pi05_lora_multitask
echo ===========================================================================

"%PY%" -u scripts\train_pi05_lora.py %*
exit /b %errorlevel%
