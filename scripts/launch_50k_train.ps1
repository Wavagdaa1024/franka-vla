$ErrorActionPreference = "Stop"
Set-Location "C:\Users\74727\Desktop\project\VLA_franka"

$logFile = "C:\Users\74727\Desktop\project\VLA_franka\outputs\train_50k.log"
$errFile = "C:\Users\74727\Desktop\project\VLA_franka\outputs\train_50k.err.log"

if (Test-Path $logFile) { Remove-Item -Force $logFile }
if (Test-Path $errFile) { Remove-Item -Force $errFile }

$env:CUDA_DEVICE_ORDER    = "PCI_BUS_ID"
$env:CUDA_VISIBLE_DEVICES = "1"
$env:HF_HUB_OFFLINE       = "1"
$env:PYTHONPATH           = "C:\Users\74727\Desktop\project\VLA_franka;C:\Users\74727\Desktop\project\VLA_franka\src"

$argsList = @(
    "-u",
    "scripts\python\train_pi05_lora.py",
    "--dataset", "dataset\teleop_pick_cube_15hz_002",
    "--output-dir", "outputs\checkpoints\pi05_lora_pure_flow_50k",
    "--resume", "outputs\checkpoints\pi05_lora_pure_flow_50k\step_02500.pt",
    "--lora-dropout", "0.05",
    "--steps", "50000",
    "--save-freq", "5000",
    "--eval-freq", "2500",
    "--val-ratio", "0.15",
    "--num-val-samples", "25",
    "--wandb",
    "--wandb-project", "pi05-franka-vla",
    "--wandb-name", "pi05_pure_flow_dropout005_50k",
    "--wandb-mode", "online"
)

$proc = Start-Process -FilePath "C:\Users\74727\miniconda3\envs\lerobot\python.exe" `
    -ArgumentList $argsList `
    -WorkingDirectory "C:\Users\74727\Desktop\project\VLA_franka" `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError $errFile `
    -PassThru

Write-Host "[LAUNCH SUCCESS] Pi0.5 50k Pure Flow LoRA Training launched successfully!"
Write-Host "  PID:         $($proc.Id)"
Write-Host "  Log File:    $logFile"
Write-Host "  Error File:  $errFile"
