import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

PROJECT_ROOT = Path(r"C:\Users\74727\Desktop\project\VLA_franka")
sys.path.insert(0, str(PROJECT_ROOT))

from franka_teleop.pi05_engine.runtime import PI05Inference
from franka_teleop.pi05_engine.lora import inject_pi05_lora, load_lora_state_dict
from scripts.train_pi05_lora import build_multitask_dataset, evaluate_sample, DEFAULT_DATASET_DIRS

CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints" / "pi05_lora_multitask"
CHECKPOINT_BASE = PROJECT_ROOT / "checkpoints" / "pi05_droid_jointpos"
STATS_PATH = CHECKPOINT_BASE / "auxiliary" / "openpi_droid_jointpos_norm_stats.json"
TOKENIZER_PATH = CHECKPOINT_BASE / "auxiliary" / "paligemma_tokenizer.model"

print("=" * 85)
print("  MULTI-TASK LoRA CHECKPOINT LADDER EVALUATION MATRIX")
print("=" * 85)

# Load dataset
all_samples, cached_files, task_buckets = build_multitask_dataset(DEFAULT_DATASET_DIRS)
task_names = sorted(list(task_buckets.keys()))

# Pick fixed eval sample for each task
eval_samples = {}
for t_name in task_names:
    indices = task_buckets[t_name]
    chosen_idx = indices[len(indices) // 3]
    eval_samples[t_name] = all_samples[chosen_idx]

# Checkpoint ladder
ckpt_steps = [1000, 2000, 3000, 4000, 5000]

print("\n[Model] Loading base model on GPU 1...")
inference = PI05Inference.from_checkpoint(
    CHECKPOINT_BASE,
    stats_path=STATS_PATH,
    tokenizer_path=TOKENIZER_PATH,
    device="cuda:0",
    profile="droid_jointpos"
)
net = inference.network

# Inject LoRA architecture
print("[LoRA] Injecting LoRA architecture into base model...")
inject_pi05_lora(net, lang_rank=16, expert_rank=32)

results = {}

# Evaluate each checkpoint in ladder
for step in ckpt_steps:
    ckpt_file = CKPT_DIR / f"pi05_lora_multitask_step_{step:04d}.pt"
    if not ckpt_file.exists():
        print(f"Skipping missing checkpoint: {ckpt_file.name}")
        continue
    
    ckpt_data = torch.load(str(ckpt_file), map_location="cuda:0", weights_only=False)
    load_lora_state_dict(net, ckpt_data["state_dict"], strict=True)
    net.eval()
    
    results[step] = {}
    print(f"\n--- Evaluated Checkpoint Step {step} (Loss: {ckpt_data.get('loss', 0.0):.5f}) ---")
    step_maes = []
    step_mses = []
    for t_name in task_names:
        res = evaluate_sample(inference, eval_samples[t_name], cached_files)
        results[step][t_name] = res
        step_maes.append(res['joint_mae_deg'])
        step_mses.append(res['action_mse'])
        print(f"  {t_name:30s} | Action MSE: {res['action_mse']:.6f} | Joint MAE: {res['joint_mae_deg']:5.2f} deg (Hold: {res['hold_still_mae_deg']:5.2f} deg, {res['improve_pct']:+5.1f}%)")
    results[step]["__MEAN__"] = {
        "joint_mae_deg": float(np.mean(step_maes)),
        "action_mse": float(np.mean(step_mses)),
    }
    print(f"  >> OVERALL MEAN <<             | Action MSE: {results[step]['__MEAN__']['action_mse']:.6f} | Joint MAE: {results[step]['__MEAN__']['joint_mae_deg']:5.2f} deg")

print("\n" + "=" * 85)
print("  FINAL COMPARISON SUMMARY TABLE")
print("=" * 85)
header = f"{'Step':6s} | {'Mean MAE':10s} | {'Mean MSE':10s} | " + " | ".join(f"{t[:10]:10s}" for t in task_names)
print(header)
print("-" * len(header))
for step in ckpt_steps:
    if step not in results: continue
    r = results[step]
    row = f"{step:6d} | {r['__MEAN__']['joint_mae_deg']:9.2f}° | {r['__MEAN__']['action_mse']:9.6f} | "
    row += " | ".join(f"{r[t]['joint_mae_deg']:9.2f}°" for t in task_names)
    print(row)
print("=" * len(header))

# Save results JSON
out_json = CKPT_DIR / "evaluation_matrix.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({
        "checkpoints": {str(k): {t: {kk: vv for kk, vv in v.items() if not isinstance(vv, np.ndarray)} for t, v in data.items()} for k, data in results.items()}
    }, f, indent=2)
print(f"\n[Artifact] Saved detailed evaluation matrix to {out_json.name}")
