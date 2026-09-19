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
echo   PI0.5 DFK 7D CARTESIAN EE ACTION MSE LORA TRAINING (SCHEME, GPU 1)
echo ===========================================================================
echo   Target GPU:     1 (RTX 5090 32GB, STRICT PHYSICAL ISOLATION)
echo   Loss Mode:      cartesian_7d (7D EE Action MSE: [dx,dy,dz,drx,dry,drz,grip])
echo   Datasets:       dataset\teleop_pick_cube_15hz_002
echo   Output:         outputs\checkpoints\pi05_lora_cartesian_7d
echo   Steps:          2000 (Save every 500 steps)
echo   Batch:          4 (effective 8 with grad accum 2)
echo ===========================================================================

"%PY%" -u scripts\python\train_pi05_lora.py --loss-mode cartesian_7d --dataset dataset\teleop_pick_cube_15hz_002 --output-dir outputs\checkpoints\pi05_lora_cartesian_7d --steps 2000 --save-freq 500 --eval-freq 250 %*
exit /b %errorlevel%
