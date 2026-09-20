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
echo   PI0.5 NORTHWESTERN 16:9 CENTER-CROP LORA TRAINING 20K STEPS GPU 1
echo ===========================================================================
echo   Target GPU:     Physical GPU 1 - RTX 5090 32GB
echo   Image Crop:     16:9 Center Crop (640x360 -> Pad/Resize 224x224)
echo   Held-Out Test:  4 Full Episodes (3-5 Demonstrations Partitioned)
echo   Loss Mode:      pure_flow - 100 percent Clean Native Joint Flow Matching
echo   Datasets:       dataset\teleop_pick_cube_15hz_002
echo   Output:         outputs\checkpoints\pi05_lora_crop169_v2_20k
echo   Total Steps:    20000 - Checkpoints every 2500 steps
echo   Batch Size:     4 - Effective 8 with grad accum 2
echo ===========================================================================

"%PY%" -u scripts\python\train_pi05_lora.py --dataset dataset\teleop_pick_cube_15hz_002 --output-dir outputs\checkpoints\pi05_lora_crop169_v2_20k --image-crop 16_9 --steps 20000 --save-freq 2500 --eval-freq 2500 --num-val-episodes 4 --batch-size 4 --grad-accum 2 %*
exit /b %errorlevel%
