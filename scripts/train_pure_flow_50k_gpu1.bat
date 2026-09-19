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
echo   PI0.5 PURE 8D JOINT FLOW MATCHING LORA TRAINING 50K STEPS GPU 1
echo ===========================================================================
echo   Target GPU:     Physical GPU 1 - RTX 5090 32GB
echo   Loss Mode:      pure_flow - 100 percent Clean Native Joint Flow Matching
echo   Datasets:       dataset\teleop_pick_cube_15hz_002
echo   Output:         outputs\checkpoints\pi05_lora_pure_flow_50k
echo   State Noise:    0.0 rad - Clean Deterministic Joint Training
echo   Dropout:        LoRA Dropout p=0.05
echo   Resume:         Resuming from step_02500.pt
echo   Total Steps:    50000 - Checkpoints every 2500 steps
echo   Evaluation:     Every 2500 steps - Held-Out Test Set
echo   WandB:          Online - Real-Time Dashboard Synced
echo   Batch:          4 - Effective 8 with grad accum 2
echo ===========================================================================

"%PY%" -u scripts\python\train_pi05_lora.py --loss-mode pure_flow --dataset dataset\teleop_pick_cube_15hz_002 --output-dir outputs\checkpoints\pi05_lora_pure_flow_50k --resume outputs\checkpoints\pi05_lora_pure_flow_50k\step_02500.pt --state-noise 0.0 --lora-dropout 0.05 --steps 50000 --save-freq 2500 --eval-freq 2500 --val-ratio 0.15 --num-val-samples 25 --wandb --wandb-mode online %*
exit /b %errorlevel%
