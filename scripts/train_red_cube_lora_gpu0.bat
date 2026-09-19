@echo off
setlocal
cd /d "%~dp0.."

set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
set "CUDA_VISIBLE_DEVICES=0"
set "HF_HUB_OFFLINE=1"
set "PYTHONPATH=%~dp0..\src;%PYTHONPATH%"
set "PY=C:\Users\74727\miniconda3\envs\lerobot\python.exe"

if not exist "%PY%" (
    echo [ERROR] Canonical lerobot Python was not found: %PY%
    exit /b 2
)

echo ===========================================================================
echo   PI0.5 RED CUBE PURE JOINT LoRA TRAINING (GPU 0)
echo ===========================================================================
echo   Target GPU:     0 (RTX 5090 32GB, GPU 1 RESERVED FOR LIVE TESTING)
echo   Task Filter:    "red cube" (Pure Red Cube Training)
echo   Datasets:       dataset\teleop_pick_cube_15hz_001 dataset\teleop_pick_cube_15hz_002
echo   Output:         outputs\checkpoints\pi05_lora_red_cube
echo   Steps:          2000
echo   Batch:          4 (effective 8 with grad accum 2)
echo ===========================================================================

"%PY%" -u scripts\python\train_pi05_lora.py --dataset dataset\teleop_pick_cube_15hz_001 dataset\teleop_pick_cube_15hz_002 --task-filter "red cube" --output-dir outputs\checkpoints\pi05_lora_red_cube --steps 2000 --save-freq 500 --eval-freq 250 %*
exit /b %errorlevel%
