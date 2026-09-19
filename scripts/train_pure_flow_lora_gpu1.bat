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

echo ===========================================================================
echo   PI0.5 PURE FLOW MATCHING LORA TRAINING (BASELINE, GPU 1)
echo ===========================================================================
echo   Target GPU:     1 (RTX 5090 32GB, STRICT PHYSICAL ISOLATION)
echo   Loss Mode:      pure_flow (Pure 8D Joint Flow Matching Loss)
echo   Datasets:       dataset\teleop_pick_cube_15hz_002
echo   Output:         outputs\checkpoints\pi05_lora_pure_flow
echo   Steps:          2000 (Save every 500 steps)
echo   Batch:          4 (effective 8 with grad accum 2)
echo ===========================================================================

"%PY%" -u scripts\python\train_pi05_lora.py --loss-mode pure_flow --dataset dataset\teleop_pick_cube_15hz_002 --output-dir outputs\checkpoints\pi05_lora_pure_flow --steps 2000 --save-freq 500 --eval-freq 250 %*
exit /b %errorlevel%
