#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pi0.5 Architectural Benchmark: LoRA Fine-Tuning vs. Partial Layer Unfreezing vs. Zero-Shot Base.
Compares:
  1. Base (Zero-Shot, 0 trainable params)
  2. Expert-Only Partial Unfreeze (Round 1: Gemma-300M, 300M params, 1.62 GB)
  3. Vision + Expert Partial Unfreeze (Round 2: SigLIP top layers + Gemma-300M, ~350M params, 1.75 GB)
  4. Multi-Task LoRA Step 3000 (Northwestern Route: PaliGemma r=16 + Gemma-300M r=32, 10.87M params, 20.8 MB)
  5. Multi-Task LoRA Step 5000 (Northwestern Route: PaliGemma r=16 + Gemma-300M r=32, 10.87M params, 20.8 MB)
"""

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

CHECKPOINT_BASE = PROJECT_ROOT / "checkpoints" / "pi05_droid_jointpos"
STATS_PATH = CHECKPOINT_BASE / "auxiliary" / "openpi_droid_jointpos_norm_stats.json"
TOKENIZER_PATH = CHECKPOINT_BASE / "auxiliary" / "paligemma_tokenizer.model"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES = [
    {
        "name": "Base Zero-Shot",
        "type": "base",
        "path": None,
        "trainable_params_m": 0.0,
        "ckpt_size_mb": 0.0,
        "description": "Pre-trained pi05_droid_jointpos (no fine-tuning)",
    },
    {
        "name": "Round 1: Action Expert Unfreeze",
        "type": "full_partial",
        "path": PROJECT_ROOT / "outputs" / "checkpoints" / "pi05_jointpos_expert" / "action_expert_step_1500.pt",
        "trainable_params_m": 312.5,
        "ckpt_size_mb": 1619.9,
        "description": "Fine-tuned 300M Action Expert only (Vision & Language frozen)",
    },
    {
        "name": "Round 2: Vision + Expert Unfreeze",
        "type": "full_partial",
        "path": PROJECT_ROOT / "outputs" / "checkpoints" / "pi05_vision_tuned" / "action_expert_step_0900.pt",
        "trainable_params_m": 354.2,
        "ckpt_size_mb": 1751.3,
        "description": "Fine-tuned SigLIP Top 2 ViT layers + Projector + Action Expert",
    },
    {
        "name": "Round 3: Multi-Task LoRA (Step 3000)",
        "type": "lora",
        "path": PROJECT_ROOT / "outputs" / "checkpoints" / "pi05_lora_multitask" / "pi05_lora_multitask_step_3000.pt",
        "trainable_params_m": 10.87,
        "ckpt_size_mb": 20.83,
        "description": "PaliGemma-2B r=16 + Action Expert r=32 (Step 3000, 100% base frozen)",
    },
    {
        "name": "Round 3: Multi-Task LoRA (Step 5000)",
        "type": "lora",
        "path": PROJECT_ROOT / "outputs" / "checkpoints" / "pi05_lora_multitask" / "pi05_lora_multitask_step_5000.pt",
        "trainable_params_m": 10.87,
        "ckpt_size_mb": 20.83,
        "description": "PaliGemma-2B r=16 + Action Expert r=32 (Step 5000, 100% base frozen)",
    },
]


def load_model_for_candidate(cand):
    """Loads a fresh base model or injects/loads appropriate weights."""
    print(f"\n[Model Loader] Loading for candidate: {cand['name']}...")
    inference = PI05Inference.from_checkpoint(
        CHECKPOINT_BASE,
        stats_path=STATS_PATH,
        tokenizer_path=TOKENIZER_PATH,
        device="cuda:0",
        profile="droid_jointpos"
    )
    net = inference.network

    if cand["type"] == "base":
        net.eval()
        return inference

    if cand["type"] == "full_partial":
        ckpt_path = cand["path"]
        print(f"  Loading partial unfreeze weights from {ckpt_path.name}...")
        ckpt = torch.load(str(ckpt_path), map_location="cuda:0", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        net.load_state_dict(state_dict, strict=False)
        net.eval()
        return inference

    if cand["type"] == "lora":
        ckpt_path = cand["path"]
        print(f"  Injecting LoRA architecture and loading weights from {ckpt_path.name}...")
        inject_pi05_lora(net, lang_rank=16, expert_rank=32)
        ckpt = torch.load(str(ckpt_path), map_location="cuda:0", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)
        load_lora_state_dict(net, state_dict, strict=True)
        net.eval()
        return inference

    raise ValueError(f"Unknown candidate type: {cand['type']}")


def main():
    print("=" * 90)
    print("  PI0.5 COMPARATIVE BENCHMARK: LoRA vs PARTIAL LAYER UNFREEZING vs ZERO-SHOT")
    print("=" * 90)

    # 1. Load multi-task datasets and pick fixed evaluation samples
    all_samples, cached_files, task_buckets = build_multitask_dataset(DEFAULT_DATASET_DIRS)
    task_names = sorted(list(task_buckets.keys()))

    eval_samples = {}
    for t_name in task_names:
        indices = task_buckets[t_name]
        chosen_idx = indices[len(indices) // 3]
        eval_samples[t_name] = all_samples[chosen_idx]

    # Also pick specialized Reaching and Grasping frames for Red Cube
    red_cube_indices = task_buckets["pick and place the red cube"]
    reach_sample = None
    grasp_sample = None
    for idx in red_cube_indices:
        s = all_samples[idx]
        cf = cached_files[s["file_id"]]
        f_idx = s["frame_idx"]
        g_val = cf["states"][f_idx, 7]
        if reach_sample is None and f_idx >= 30 and g_val < 0.05:
            reach_sample = s
        if grasp_sample is None and f_idx >= 80 and g_val > 0.5:
            grasp_sample = s
    if reach_sample is None: reach_sample = all_samples[red_cube_indices[len(red_cube_indices)//4]]
    if grasp_sample is None: grasp_sample = all_samples[red_cube_indices[len(red_cube_indices)//2]]

    print(f"\n[*] Reaching Sample: Frame {reach_sample['frame_idx']} (Task: '{reach_sample['task']}')")
    print(f"[*] Grasping Sample: Frame {grasp_sample['frame_idx']} (Task: '{grasp_sample['task']}')")

    benchmark_results = []

    # 2. Evaluate each candidate
    for cand in CANDIDATES:
        if cand["path"] is not None and not cand["path"].exists():
            print(f"[Warning] Checkpoint file missing: {cand['path']}. Skipping.")
            continue

        inference = load_model_for_candidate(cand)

        # Evaluate reaching and grasping on red cube
        res_reach = evaluate_sample(inference, reach_sample, cached_files)
        res_grasp = evaluate_sample(inference, grasp_sample, cached_files)

        # Evaluate on all 5 tasks
        task_metrics = {}
        maes = []
        mses = []
        for t_name in task_names:
            r = evaluate_sample(inference, eval_samples[t_name], cached_files)
            task_metrics[t_name] = r
            maes.append(r["joint_mae_deg"])
            mses.append(r["action_mse"])

        cand_result = {
            "name": cand["name"],
            "type": cand["type"],
            "trainable_params_m": cand["trainable_params_m"],
            "ckpt_size_mb": cand["ckpt_size_mb"],
            "description": cand["description"],
            "reach_mae_deg": res_reach["joint_mae_deg"],
            "reach_mse": res_reach["action_mse"],
            "grasp_mae_deg": res_grasp["joint_mae_deg"],
            "grasp_mse": res_grasp["action_mse"],
            "overall_mean_mae_deg": float(np.mean(maes)),
            "overall_mean_mse": float(np.mean(mses)),
            "tasks": {t: {k: v for k, v in res.items() if not isinstance(v, np.ndarray)} for t, res in task_metrics.items()},
        }
        benchmark_results.append(cand_result)

        # Free GPU memory before loading next candidate
        del inference
        torch.cuda.empty_cache()

    # 3. Print Comprehensive Comparison Table
    print("\n" + "=" * 105)
    print("  COMPREHENSIVE BENCHMARK RESULTS TABLE")
    print("=" * 105)
    header = f"{'Model Configuration':34s} | {'Trainable':9s} | {'Size':8s} | {'5-Task Mean':11s} | {'Reach MAE':9s} | {'Grasp MAE':9s} | {'Red Cube':8s} | {'Carrot':8s}"
    print(header)
    print("-" * 105)
    for b in benchmark_results:
        red_mae = b["tasks"]["pick and place the red cube"]["joint_mae_deg"]
        carrot_mae = b["tasks"]["pick and place the carrot"]["joint_mae_deg"]
        row = (
            f"{b['name']:34s} | "
            f"{b['trainable_params_m']:7.1f} M | "
            f"{b['ckpt_size_mb']:6.1f}MB | "
            f"{b['overall_mean_mae_deg']:8.2f}°   | "
            f"{b['reach_mae_deg']:7.2f}°  | "
            f"{b['grasp_mae_deg']:7.2f}°  | "
            f"{red_mae:6.2f}°  | "
            f"{carrot_mae:6.2f}°"
        )
        print(row)
    print("=" * 105)

    # 4. Save JSON Report
    report_file = OUTPUT_DIR / "comparison_lora_vs_unfreeze.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\n[Report Saved] Benchmark JSON written to {report_file}")


if __name__ == "__main__":
    main()
