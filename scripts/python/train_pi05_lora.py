#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pi0.5 Franka Clean Pure Joint Flow Matching LoRA Fine-Tuning Pipeline.

Architecture & Formulation:
  - Base Model: Pi0.5 DROID JointPos (100% frozen, ~4.14B params).
  - Language Model (PaliGemma-2B): LoRA rank 16, alpha 32 on attention projections (q, k, v, o) with Dropout.
  - Action Expert (Gemma-300M): LoRA rank 32, alpha 64 on attention projections (q, k, v, o) with Dropout.
  - Action Space: Pure 8D joint-space relative displacement (7-DOF delta_q + 1-DOF binary gripper).
  - Loss Objective: Pure Continuous Flow Matching MSE: ||v_theta(x_t, t) - v_target||^2.
  - Zero state noise, zero artificial trajectory distortion.
  - Episode-level held-out validation set evaluation (10-step ODE Euler solver) on eval intervals.

Hardware:
  - Enforced physical GPU 1 (RTX 5090 32GB, CUDA_VISIBLE_DEVICES=1).
  - Physical GPU 0 strictly untouched (0MB).
"""

import os
import sys
import argparse
import json
import time
import random
import shutil
import subprocess
import hashlib
from pathlib import Path
from collections import defaultdict

# GPU Device Configuration (defaults to GPU 0 inside visible CUDA devices)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if "CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"].strip()
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import av
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.beta import Beta

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent if SCRIPT_DIR.name == "python" else SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from franka_teleop.pi05_engine.runtime import PI05Inference
from franka_teleop.pi05_engine.contracts import (read_checkpoint, checkpoint_state, validate_resume, ACTION_SEMANTICS, PRECISION)
from franka_teleop.pi05_engine.lora import (
    inject_pi05_lora,
    extract_lora_state_dict,
    load_lora_state_dict,
)
from franka_teleop.pi05_engine.utils import prepare_attention_masks_4d
from franka_teleop.tensor_utils import make_att_2d_masks
from franka_teleop.kinematics import FrankaDifferentiableKinematics

# Default Paths
REPO_ROOT = PROJECT_ROOT
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "pi05_droid_jointpos"
STATS_PATH = CHECKPOINT_DIR / "auxiliary" / "openpi_droid_jointpos_norm_stats.json"
TOKENIZER_PATH = CHECKPOINT_DIR / "auxiliary" / "paligemma_tokenizer.model"

DEFAULT_DATASET_DIRS = [
    REPO_ROOT / "dataset" / "teleop_pick_cube_15hz_002",
]
DEFAULT_OUTPUT_CKPT_DIR = REPO_ROOT / "outputs" / "checkpoints" / "pi05_lora_v2_20k"

CHUNK_SIZE = 15
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM = 2  # Effective batch size = 8
DEFAULT_STEPS = 20000
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_LOG_FREQ = 25
DEFAULT_SAVE_FREQ = 2500
DEFAULT_EVAL_FREQ = 2500
SEED = 42


def preload_video_frames(video_path: Path, crop_16_9: bool = False):
    """Preload all video frames into memory as RGB numpy arrays."""
    container = av.open(str(video_path))
    frames = []
    for frame in container.decode(video=0):
        arr = frame.to_ndarray(format="rgb24")
        if crop_16_9:
            # Northwestern / OpenPI DROID standard: Center-crop 4:3 (480x640) to 16:9 (360x640)
            h, w = arr.shape[:2]
            target_h = int(round(w * 9.0 / 16.0))
            if h > target_h:
                offset = (h - target_h) // 2
                arr = np.ascontiguousarray(arr[offset : offset + target_h, :])
            elif w > int(round(h * 16.0 / 9.0)):
                target_w = int(round(h * 16.0 / 9.0))
                offset = (w - target_w) // 2
                arr = np.ascontiguousarray(arr[:, offset : offset + target_w])
        frames.append(arr)
    container.close()
    return frames



def build_multitask_dataset(dataset_dirs, task_filter=None, val_ratio=0.15, num_val_episodes=4, val_seed=42, crop_16_9: bool = False):
    """
    Builds in-memory index of 15-step relative joint position chunks across all datasets:
      a_k = q_{t+k+1} - q_t = sum_{i=0}^k delta_q_{t+i} (for k = 0..14)
      gripper_k = action[t+k, 7] (0=OPEN, 1=CLOSED command)
    Implements deterministic episode-level split: held-out validation episodes have ZERO temporal leakage.
    """
    if isinstance(dataset_dirs, (str, Path)):
        dataset_dirs = [Path(dataset_dirs)]
    else:
        dataset_dirs = [Path(d) for d in dataset_dirs]

    t0 = time.perf_counter()
    cached_files = {}
    raw_samples = []

    for ds_idx, dataset_dir in enumerate(dataset_dirs):
        if not dataset_dir.exists():
            print(f"[Dataset {ds_idx+1}/{len(dataset_dirs)}] Skipping non-existent path: {dataset_dir.name}")
            continue

        print(f"[Dataset {ds_idx+1}/{len(dataset_dirs)}] Loading demonstrations from {dataset_dir.name}...")

        ep_tasks_map = {}
        episodes_meta_dir = dataset_dir / "meta" / "episodes"
        if episodes_meta_dir.is_dir():
            for ep_pq in sorted(episodes_meta_dir.rglob("*.parquet")):
                try:
                    t_ep = pq.read_table(ep_pq).to_pydict()
                    ep_indices = t_ep.get("episode_index", [])
                    task_vals = t_ep.get("tasks", [])
                    for ep_idx, task_val in zip(ep_indices, task_vals):
                        if isinstance(task_val, (list, np.ndarray)) and len(task_val) > 0:
                            ep_tasks_map[int(ep_idx)] = str(task_val[0])
                        elif isinstance(task_val, str):
                            ep_tasks_map[int(ep_idx)] = task_val
                except Exception as e:
                    print(f"  [Warning] Reading {ep_pq.name}: {e}")

        tasks_file = dataset_dir / "meta" / "tasks.parquet"
        task_map = {}
        if tasks_file.is_file():
            try:
                tasks_table = pq.read_table(str(tasks_file)).to_pydict()
                task_map = dict(zip(tasks_table.get("task_index", []), tasks_table.get("task", [])))
            except Exception:
                pass

        data_dir = dataset_dir / "data" / "chunk-000"
        video_front_dir = dataset_dir / "videos" / "observation.images.front" / "chunk-000"
        video_wrist_dir = dataset_dir / "videos" / "observation.images.wrist" / "chunk-000"

        parquet_files = sorted(data_dir.glob("file-*.parquet"))
        file_configs = []
        for pf in parquet_files:
            fid = int(pf.stem.split("-")[1])
            mp4_front = video_front_dir / f"file-{fid:03d}.mp4"
            mp4_wrist = video_wrist_dir / f"file-{fid:03d}.mp4"
            if mp4_front.exists() and mp4_wrist.exists():
                file_configs.append({
                    "id": fid,
                    "front_mp4": mp4_front,
                    "wrist_mp4": mp4_wrist,
                    "parquet": pf,
                })

        for cfg in file_configs:
            fid = cfg["id"]
            file_key = f"{dataset_dir.name}_{fid}"
            try:
                table = pq.read_table(str(cfg["parquet"]))
            except Exception:
                print(f"  Skipping unfinalized {dataset_dir.name}/file-{fid:03d}.parquet")
                continue

            print(f"  Loading video & parquet for {dataset_dir.name}/file-{fid:03d}...")
            front_frames = preload_video_frames(cfg["front_mp4"], crop_16_9=crop_16_9)
            wrist_frames = preload_video_frames(cfg["wrist_mp4"], crop_16_9=crop_16_9)

            states_raw = np.array([row.as_py() for row in table.column("observation.state")], dtype=np.float32)
            actions_raw = np.array([row.as_py() for row in table.column("action")], dtype=np.float32)
            ep_indices = np.array(table.column("episode_index").to_pylist(), dtype=np.int32)
            task_indices = np.array(table.column("task_index").to_pylist(), dtype=np.int32)

            cached_files[file_key] = {
                "front_frames": front_frames,
                "wrist_frames": wrist_frames,
                "states": states_raw,
                "actions": actions_raw,
                "episodes": ep_indices,
                "task_indices": task_indices,
                "length": len(states_raw)
            }

            ep_boundaries = defaultdict(list)
            for idx, ep in enumerate(ep_indices):
                ep_boundaries[ep].append(idx)

            added = 0
            for ep, indices in ep_boundaries.items():
                task_str = ep_tasks_map.get(int(ep), task_map.get(task_indices[indices[0]], "pick and place the red cube"))
                if task_filter and task_filter.lower() not in task_str.lower():
                    continue

                if len(indices) <= CHUNK_SIZE:
                    continue

                for local_i in range(len(indices) - CHUNK_SIZE):
                    f_idx = indices[local_i]
                    raw_samples.append({
                        "file_key": file_key,
                        "frame_idx": f_idx,
                        "task": task_str,
                        "episode": int(ep),
                        "global_ep_id": f"{dataset_dir.name}_ep{int(ep):04d}",
                    })
                    added += 1

            print(f"    -> Extracted {added:,} chunks for {dataset_dir.name}/file-{fid:03d}")

    if len(raw_samples) == 0:
        raise ValueError(f"No valid demonstration chunks found in specified dataset directories!")

    # Deterministic episode-level train / val split
    all_episodes = sorted(list(set(s["global_ep_id"] for s in raw_samples)))
    if num_val_episodes is not None and num_val_episodes > 0:
        rng = random.Random(val_seed)
        shuffled_eps = list(all_episodes)
        rng.shuffle(shuffled_eps)
        val_ep_set = set(shuffled_eps[:min(num_val_episodes, len(all_episodes) - 1)])
    elif val_ratio > 0.0 and len(all_episodes) >= 2:
        rng = random.Random(val_seed)
        num_val_eps = max(1, int(round(len(all_episodes) * val_ratio)))
        shuffled_eps = list(all_episodes)
        rng.shuffle(shuffled_eps)
        val_ep_set = set(shuffled_eps[:num_val_eps])
    else:
        val_ep_set = set()

    train_samples = [s for s in raw_samples if s["global_ep_id"] not in val_ep_set]
    val_samples = [s for s in raw_samples if s["global_ep_id"] in val_ep_set]

    train_task_buckets = defaultdict(list)
    for idx, s in enumerate(train_samples):
        train_task_buckets[s["task"]].append(idx)

    val_task_buckets = defaultdict(list)
    for idx, s in enumerate(val_samples):
        val_task_buckets[s["task"]].append(idx)

    print("\n" + "=" * 85)
    print("  DATASET PARTITION & LEAKAGE AUDIT")
    print("=" * 85)
    print(f"  Total Demonstrations Loaded: {len(raw_samples):,} chunks across {len(all_episodes)} episodes ({time.perf_counter()-t0:.2f}s)")
    print("  Partition Strategy:          Strict Episode-Level Isolation (0 temporal leakage)")
    print(f"  * Train Set:                 {len(train_samples):,} chunks ({len(train_samples)/len(raw_samples)*100:.1f}%) across {len(all_episodes)-len(val_ep_set)} episodes")
    print(f"  * Held-Out Validation Set:   {len(val_samples):,} chunks ({len(val_samples)/len(raw_samples)*100:.1f}%) across {len(val_ep_set)} episodes")
    print(f"  * Held-Out Episode IDs:      {sorted(list(val_ep_set))}")
    for t_name, idx_list in train_task_buckets.items():
        print(f"  * Train Task '{t_name}': {len(idx_list):,} chunks")
    for t_name, idx_list in val_task_buckets.items():
        print(f"  * Val Task '{t_name}':   {len(idx_list):,} chunks")
    print("=" * 85 + "\n")

    return train_samples, val_samples, cached_files, train_task_buckets, val_ep_set


def evaluate_sample(inference, s, cached_files, dfk=None):
    """
    Evaluates a single sample using the full multi-step ODE Euler solver.
    Computes true open-loop joint MAE, action MSE, and physical end-effector errors.
    """
    cf = cached_files[s["file_key"]]
    f_idx = s["frame_idx"]
    f_img = cf["front_frames"][f_idx]
    w_img = cf["wrist_frames"][f_idx]
    curr_state = cf["states"][f_idx]
    curr_q = curr_state[:7]

    future_states = cf["states"][f_idx + 1 : f_idx + 1 + CHUNK_SIZE]
    gt_delta_q = future_states[:, :7] - curr_q[None, :]
    future_actions = cf["actions"][f_idx : f_idx + CHUNK_SIZE]
    gt_gripper = future_actions[:, 7:8]
    gt_action_chunk = np.concatenate([gt_delta_q, gt_gripper], axis=1)

    front_t = torch.from_numpy(np.transpose(f_img, (2, 0, 1)))
    wrist_t = torch.from_numpy(np.transpose(w_img, (2, 0, 1)))
    obs = {
        "observation.images.base_0_rgb": front_t,
        "observation.images.left_wrist_0_rgb": wrist_t,
        "observation.state": curr_state,
    }
    # A stable per-sample noise tensor makes checkpoints comparable without
    # consuming training RNG state or depending on Python's randomized hash().
    identity = f"{s['file_key']}:{f_idx}:{s['task']}"
    seed = int.from_bytes(hashlib.sha256(identity.encode("utf-8")).digest()[:8], "little")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn((1, inference.config.chunk_size, inference.config.max_action_dim),
                        generator=generator, dtype=torch.float32)
    with torch.no_grad():
        pred_chunk = inference.predict_action_chunk(obs, s["task"], noise=noise).cpu().numpy()[0]

    action_mse = float(np.mean((pred_chunk - gt_action_chunk) ** 2))
    joint_mse = float(np.mean((pred_chunk[:, :7] - gt_delta_q) ** 2))
    gripper_mse = float(np.mean((pred_chunk[:, 7] - gt_gripper[:, 0]) ** 2))
    joint_mae_rad = float(np.mean(np.abs(pred_chunk[:, :7] - gt_delta_q)))
    joint_mae_deg = float(np.degrees(joint_mae_rad))
    hold_still_mae_rad = float(np.mean(np.abs(gt_delta_q)))
    hold_still_mae_deg = float(np.degrees(hold_still_mae_rad))
    improve_pct = float((hold_still_mae_rad - joint_mae_rad) / max(hold_still_mae_rad, 1e-6) * 100.0)

    ee_pos_err_mm = None
    ee_rot_err_deg = None
    if dfk is not None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        curr_q_t = torch.from_numpy(curr_q[None, :]).unsqueeze(1).float().to(device)
        pred_delta_q_t = torch.from_numpy(pred_chunk[None, :, :7]).float().to(device)
        gt_delta_q_t = torch.from_numpy(gt_delta_q[None, :, :7]).float().to(device)

        with torch.no_grad():
            pred_ee_6d = dfk.compute_relative_ee_action(curr_q_t, pred_delta_q_t)
            gt_ee_6d = dfk.compute_relative_ee_action(curr_q_t, gt_delta_q_t)
            ee_pos_err_mm = float(torch.norm(pred_ee_6d[..., :3] - gt_ee_6d[..., :3], dim=-1).mean().item() * 1000.0)
            ee_rot_err_deg = float(torch.rad2deg(torch.norm(pred_ee_6d[..., 3:6] - gt_ee_6d[..., 3:6], dim=-1)).mean().item())

    return {
        "task": s["task"],
        "action_mse": action_mse,
        "joint_mse": joint_mse,
        "gripper_mse": gripper_mse,
        "joint_mae_deg": joint_mae_deg,
        "hold_still_mae_deg": hold_still_mae_deg,
        "improve_pct": improve_pct,
        "ee_pos_err_mm": ee_pos_err_mm,
        "ee_rot_err_deg": ee_rot_err_deg,
        "pred_chunk": pred_chunk,
        "gt_chunk": gt_action_chunk,
        "frame_idx": f_idx,
    }


def evaluate_validation_set(inference, val_samples, cached_files, dfk=None, max_samples=25):
    """
    Evaluates a representative subset of held-out validation samples.
    Returns aggregated metrics for generalization tracking.
    """
    if not val_samples:
        return {}

    n = min(len(val_samples), max_samples)
    step_stride = max(1, len(val_samples) // n)
    chosen_samples = [val_samples[i * step_stride] for i in range(n)]

    results = []
    for s in chosen_samples:
        res = evaluate_sample(inference, s, cached_files, dfk=dfk)
        results.append(res)

    agg = {
        "val_samples_count": len(results),
        "action_mse": float(np.mean([r["action_mse"] for r in results])),
        "joint_mse": float(np.mean([r["joint_mse"] for r in results])),
        "gripper_mse": float(np.mean([r["gripper_mse"] for r in results])),
        "joint_mae_deg": float(np.mean([r["joint_mae_deg"] for r in results])),
        "hold_still_mae_deg": float(np.mean([r["hold_still_mae_deg"] for r in results])),
        "improve_pct": float(np.mean([r["improve_pct"] for r in results])),
    }

    pos_errs = [r["ee_pos_err_mm"] for r in results if r.get("ee_pos_err_mm") is not None]
    if pos_errs:
        agg["ee_pos_err_mm"] = float(np.mean(pos_errs))

    rot_errs = [r["ee_rot_err_deg"] for r in results if r.get("ee_rot_err_deg") is not None]
    if rot_errs:
        agg["ee_rot_err_deg"] = float(np.mean(rot_errs))

    return agg


def main():
    parser = argparse.ArgumentParser(description="Clean Pure Joint Flow Matching LoRA Fine-Tuning for Pi0.5 Franka.")
    parser.add_argument("--dataset", type=Path, nargs="+", default=DEFAULT_DATASET_DIRS, help="Path(s) to dataset(s)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_CKPT_DIR, help="Directory to save LoRA checkpoints")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Total training steps (default: 20000)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate for LoRA & projections (default: 1e-4)")
    parser.add_argument("--lang-rank", type=int, default=16, help="LoRA rank for PaliGemma-2B (default: 16)")
    parser.add_argument("--expert-rank", type=int, default=32, help="LoRA rank for Gemma-300M Expert (default: 32)")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA layer dropout probability (default: 0.05)")
    parser.add_argument("--resume", type=Path, default=None, help="Resume training from an existing LoRA checkpoint (.pt)")
    parser.add_argument("--warmup-steps", type=int, default=200, help="Linear warmup steps (default: 200)")
    parser.add_argument("--log-freq", type=int, default=DEFAULT_LOG_FREQ)
    parser.add_argument("--save-freq", type=int, default=DEFAULT_SAVE_FREQ)
    parser.add_argument("--eval-freq", type=int, default=DEFAULT_EVAL_FREQ)
    parser.add_argument("--task-filter", type=str, default=None, help="Filter dataset by task substring (e.g. 'red cube')")
    parser.add_argument("--preflight-only", action="store_true", help="Run empirical benchmark and exit")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Ratio of episodes held out for test/validation (default: 0.15)")
    parser.add_argument("--num-val-episodes", type=int, default=4, help="Number of held-out demonstration episodes reserved for test set (default: 4)")
    parser.add_argument("--val-seed", type=int, default=42, help="Random seed for train/val episode split (default: 42)")
    parser.add_argument("--num-val-samples", type=int, default=25, help="Number of held-out test chunks evaluated on eval steps (default: 25)")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases real-time tracking")
    parser.add_argument("--wandb-project", type=str, default="pi05-franka-vla", help="W&B project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run display name")
    parser.add_argument("--wandb-id", type=str, default=None, help="W&B run unique ID for resuming the same run")
    parser.add_argument("--loss-mode", type=str, default="pure_flow", help="Legacy compatibility flag (always 'pure_flow')")
    parser.add_argument("--state-noise", type=float, default=0.0, help="Legacy compatibility flag (state noise is strictly disabled 0.0)")
    parser.add_argument("--wandb-mode", type=str, choices=("online", "offline", "disabled"), default="online", help="W&B mode (default: online)")
    parser.add_argument("--image-crop", type=str, choices=("16_9", "none"), default=None, help="Crop mode; new training defaults to none, resume inherits saved metadata")
    parser.add_argument("--resume-mode", choices=("strict", "migrate"), default="strict",
                        help="strict: same training contract; migrate: weights only, reset optimizer and step")
    args = parser.parse_args()
    resume_ckpt = read_checkpoint(args.resume) if args.resume else None
    if args.image_crop is None:
        saved_crop = resume_ckpt.get("image_crop") if resume_ckpt else None
        if resume_ckpt is not None and args.resume_mode == "migrate" and saved_crop is None:
            raise ValueError("Legacy migration requires explicit --image-crop none or 16_9")
        args.image_crop = saved_crop or "none"
    if resume_ckpt is not None:
        validate_resume(resume_ckpt, args)
        if args.resume_mode == "migrate" and args.output_dir.resolve() == args.resume.parent.resolve():
            raise ValueError("Migration requires a new output directory")
        old_id = resume_ckpt.get("args", {}).get("wandb_id")
        if args.resume_mode == "migrate" and args.wandb_id and args.wandb_id == old_id:
            raise ValueError("Migration requires a new W&B run ID")
    if (resume_ckpt is None or args.resume_mode == "migrate") and any(args.output_dir.glob("*.pt")):
        raise ValueError("New training/migration output directory already contains checkpoints")
    try:
        source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        source_dirty = bool(subprocess.check_output(["git", "diff", "--name-only", "HEAD"], cwd=REPO_ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        source_revision, source_dirty = "unknown", True


    print("=" * 85)
    print(f"  PI0.5 CLEAN PURE JOINT FLOW MATCHING LORA PIPELINE (CROP: {args.image_crop.upper()})")
    print("=" * 85)
    print("[*] Loss Mode:       [PURE_FLOW] 100% Native OpenPI Joint Flow Matching Loss")
    print(f"[*] Image Transform: [{args.image_crop.upper()}] Preprocessing mode")
    print("[*] Datasets ({}):".format(len(args.dataset)))
    for d in args.dataset:
        print(f"      - {d}")
    print(f"[*] Output Dir:      {args.output_dir}")
    print(f"[*] Steps:           {args.steps:,} (Warmup: {args.warmup_steps}, SaveFreq: {args.save_freq:,}, EvalFreq: {args.eval_freq:,})")
    print(f"[*] Batch Size:      {args.batch_size} (Grad Accum: {args.grad_accum}, Effective Batch: {args.batch_size*args.grad_accum})")
    print(f"[*] Learning Rate:   {args.lr:.2e}")
    print(f"[*] PaliGemma LoRA:  Rank={args.lang_rank}, Alpha={args.lang_rank*2}, Dropout={args.lora_dropout}")
    print(f"[*] Action Expert:   Rank={args.expert_rank}, Alpha={args.expert_rank*2}, Dropout={args.lora_dropout}")
    print(f"[*] Action Space:    15-step cumulative relative joint displacement a[k] = q[t+k+1] - q[t]")
    print(f"[*] State Noise:     DISABLED (Clean Ground Truth Joint Teleop)")
    print(f"[*] Held-Out Test:   {args.num_val_episodes} episodes (seed={args.val_seed}, {args.num_val_samples} eval samples)")
    print(f"[*] W&B Tracking:    {'ENABLED (Mode: ' + args.wandb_mode + ')' if args.wandb else 'DISABLED'}")
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "1")
    print(f"[*] Hardware:        Physical GPU {gpu_id} (RTX 5090 32GB, CUDA_VISIBLE_DEVICES={gpu_id})")
    print("-" * 85)

    # 1. Dataset build with episode-level train / val partition
    train_samples, val_samples, cached_files, train_task_buckets, val_ep_set = build_multitask_dataset(
        args.dataset,
        task_filter=args.task_filter,
        val_ratio=args.val_ratio,
        num_val_episodes=args.num_val_episodes,
        val_seed=args.val_seed,
        crop_16_9=(args.image_crop == "16_9")
    )

    if resume_ckpt is not None:
        validate_resume(resume_ckpt, args, val_ep_set)

    task_names = sorted(list(train_task_buckets.keys()))

    # Initialize W&B if requested
    wandb_run = None
    if args.wandb:
        import wandb
        run_name = args.wandb_name or f"pi05_pure_flow_{args.steps // 1000}k"
        init_kwargs = {
            "project": args.wandb_project,
            "name": run_name,
            "config": vars(args),
            "mode": args.wandb_mode,
        }
        if args.wandb_id:
            init_kwargs["id"] = args.wandb_id
            init_kwargs["resume"] = "allow"
        try:
            wandb_run = wandb.init(**init_kwargs)
            run_url = getattr(wandb_run, "url", None)
            print(f"[W&B] Initialized run '{run_name}' (ID: {wandb_run.id}) in project '{args.wandb_project}' (Mode: {args.wandb_mode})")
            if run_url:
                print(f"[W&B URL] Real-Time Cloud Dashboard: {run_url}")
        except Exception as e:
            print(f"[W&B Warning] Online init failed: {e}. Falling back to offline mode...")
            init_kwargs["mode"] = "offline"
            wandb_run = wandb.init(**init_kwargs)
            print(f"[W&B] Initialized offline run '{run_name}'.")

    # 2. Model initialization
    print(f"\n[Model] Initializing Pi0.5 base model on GPU {gpu_id}...")
    t0 = time.perf_counter()
    inference = PI05Inference.from_checkpoint(
        CHECKPOINT_DIR,
        stats_path=STATS_PATH,
        tokenizer_path=TOKENIZER_PATH,
        device="cuda:0",
        profile="droid_jointpos",
        crop_mode=args.image_crop,
        trainable_fp32=True
    )
    print(f"[Model] Base model loaded in {time.perf_counter()-t0:.2f}s.")

    net = inference.network
    processor = inference.processor

    # Ensure trainable projections are strictly float32 for high numerical precision & stability
    net.action_in_proj.to(dtype=torch.float32)
    net.action_out_proj.to(dtype=torch.float32)
    net.time_mlp_in.to(dtype=torch.float32)
    net.time_mlp_out.to(dtype=torch.float32)
    net.action_in_proj.requires_grad_(True)
    net.action_out_proj.requires_grad_(True)
    net.time_mlp_in.requires_grad_(True)
    net.time_mlp_out.requires_grad_(True)

    # 3. Inject LoRA adapters with Dropout (initialized in float32)
    print(f"\n[LoRA] Injecting LoRA adapters (Language r={args.lang_rank}, Expert r={args.expert_rank}, Dropout={args.lora_dropout:.2f})...")
    lang_loras, expert_loras = inject_pi05_lora(
        net,
        lang_rank=args.lang_rank,
        lang_alpha=float(args.lang_rank * 2),
        expert_rank=args.expert_rank,
        expert_alpha=float(args.expert_rank * 2),
        lora_dropout=args.lora_dropout,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    )

    # Parameter Audit
    trainable_params = [p for p in net.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in net.parameters())
    frozen_count = total_count - trainable_count

    print("=" * 85)
    print("  PARAMETER ISOLATION AUDIT")
    print("=" * 85)
    print(f"  PaliGemma LoRAs:      {len(lang_loras)} linear layers ({sum(p.numel() for l in lang_loras.values() for p in [l.lora_A, l.lora_B])/1e6:,.2f} M params)")
    print(f"  Action Expert LoRAs:  {len(expert_loras)} linear layers ({sum(p.numel() for l in expert_loras.values() for p in [l.lora_A, l.lora_B])/1e6:,.2f} M params)")
    print(f"  Action Projections:   {sum(p.numel() for m in [net.action_in_proj, net.action_out_proj, net.time_mlp_in, net.time_mlp_out] for p in m.parameters())/1e6:,.2f} M params)")
    print("-" * 85)
    print(f"  [SUMMARY] Trainable params: {trainable_count:,} ({trainable_count/1e6:,.2f} M, {trainable_count/total_count*100:.2f}%)")
    print(f"  [SUMMARY] Frozen params:    {frozen_count:,} ({frozen_count/1e6:,.2f} M, {frozen_count/total_count*100:.2f}%)")
    print(f"  [SUMMARY] Total params:     {total_count:,} ({total_count/1e6:,.2f} M)")
    print("=" * 85)

    # 4. Optimizer & True Linear Warmup + Cosine Annealing Scheduler
    import math
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=DEFAULT_WEIGHT_DECAY, betas=(0.9, 0.95))

    def lr_lambda(current_step: int) -> float:
        if current_step < args.warmup_steps:
            return float(current_step + 1) / float(max(1, args.warmup_steps))
        progress = float(current_step - args.warmup_steps) / float(max(1, args.steps - args.warmup_steps))
        progress = min(1.0, max(0.0, progress))
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = 1e-6 / args.lr
        return max(min_ratio, cos_factor)

    start_step = 1
    if resume_ckpt is not None:
        load_lora_state_dict(net, checkpoint_state(resume_ckpt), strict=True)
        if args.resume_mode == "strict":
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
            start_step = int(resume_ckpt["step"]) + 1
            if start_step > args.steps:
                raise ValueError("Checkpoint already reached requested total steps")
        else:
            print("[Migration] Loaded weights only; optimizer/scheduler/step reset for new experiment")

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    if resume_ckpt is not None and args.resume_mode == "strict":
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        # LambdaLR constructor adjusts lr; restore the saved current lr afterwards.
        for group, saved in zip(optimizer.param_groups, resume_ckpt["optimizer_state_dict"]["param_groups"]):
            group["lr"] = saved["lr"]
        print(f"[Resume] Step {start_step}, crop={args.image_crop}, FP32 contract validated")
    del resume_ckpt

    beta_dist = Beta(
        torch.tensor(getattr(net.config, "time_sampling_beta_alpha", 1.5), device="cuda:0"),
        torch.tensor(getattr(net.config, "time_sampling_beta_beta", 1.0), device="cuda:0")
    )

    # Differentiable Kinematics strictly for periodic validation evaluations
    dfk = FrankaDifferentiableKinematics(device="cuda:0")

    # Balanced epoch-shuffled sampling (No chunk starvation, pristine demonstration joints)
    class TaskBucketSampler:
        def __init__(self, sample_indices, seed=42):
            self.sample_indices = list(sample_indices)
            self.indices = list(sample_indices)
            self.rng = random.Random(seed)
            self.rng.shuffle(self.indices)
            self.ptr = 0

        def next_idx(self):
            if self.ptr >= len(self.indices):
                self.rng.shuffle(self.indices)
                self.ptr = 0
            idx = self.indices[self.ptr]
            self.ptr += 1
            return idx

    task_samplers = {t: TaskBucketSampler(train_task_buckets[t], seed=SEED + i) for i, t in enumerate(task_names)}

    def sample_balanced_batch(batch_size):
        sampled = []
        for _ in range(batch_size):
            t_choice = random.choice(task_names)
            idx_choice = task_samplers[t_choice].next_idx()
            sampled.append(train_samples[idx_choice])

        front_list, wrist_list, state_list, action_list, tasks_list = [], [], [], [], []
        for s in sampled:
            cf = cached_files[s["file_key"]]
            f_idx = s["frame_idx"]

            f_img = cf["front_frames"][f_idx]
            w_img = cf["wrist_frames"][f_idx]
            curr_state = cf["states"][f_idx]  # Pristine clean state
            curr_q = curr_state[:7]

            future_states = cf["states"][f_idx + 1 : f_idx + 1 + CHUNK_SIZE]
            delta_q = future_states[:, :7] - curr_q[None, :]
            future_actions = cf["actions"][f_idx : f_idx + CHUNK_SIZE]
            gripper = future_actions[:, 7:8]
            action_chunk = np.concatenate([delta_q, gripper], axis=1)

            front_list.append(torch.from_numpy(np.transpose(f_img, (2, 0, 1))))
            wrist_list.append(torch.from_numpy(np.transpose(w_img, (2, 0, 1))))
            state_list.append(curr_state)
            action_list.append(action_chunk)
            tasks_list.append(s["task"])

        return (
            torch.stack(front_list),
            torch.stack(wrist_list),
            np.array(state_list, dtype=np.float32),
            np.array(action_list, dtype=np.float32),
            tasks_list,
        )

    # Pure Flow Matching Loss computation (Zero per-batch pseudo-Euler DFK overhead)
    def compute_loss(front_t, wrist_t, state_t, action_t, batch_tasks, cur_batch_size):
        obs = {
            "observation.images.base_0_rgb": front_t,
            "observation.images.left_wrist_0_rgb": wrist_t,
            "observation.state": state_t,
        }
        prepared = processor.prepare(obs, batch_tasks, device="cuda:0")

        actions_gpu = torch.from_numpy(action_t).to("cuda:0", dtype=torch.float32)
        norm_actions_8d = processor._transform(actions_gpu, "action", "ACTION")

        target_actions = torch.zeros(cur_batch_size, CHUNK_SIZE, 32, device="cuda:0", dtype=torch.float32)
        target_actions[:, :, :8] = norm_actions_8d

        noise = torch.randn_like(target_actions)
        time_beta = beta_dist.sample((cur_batch_size,))
        time_steps = time_beta * 0.999 + 0.001
        t_expanded = time_steps.view(-1, 1, 1)

        x_t = t_expanded * noise + (1.0 - t_expanded) * target_actions
        v_target = noise - target_actions

        proj_dtype = net.action_in_proj.weight.dtype
        x_t = x_t.to(dtype=proj_dtype)
        v_target = v_target.to(dtype=proj_dtype)

        # Frozen Vision Tower: compute image embeddings without gradients
        with torch.no_grad():
            img_embs = []
            pad_masks = []
            att_masks = []
            for img, img_mask in zip(prepared.images, prepared.image_masks, strict=True):
                img_emb = net.paligemma_with_expert.embed_image(img)
                bsize, num_img_embs = img_emb.shape[:2]
                img_embs.append(img_emb)
                pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
                att_masks += [0] * num_img_embs

        # Language tokens: track gradients through PaliGemma LoRA with Dropout
        lang_emb = net.paligemma_with_expert.embed_language_tokens(prepared.tokens)
        img_embs.append(lang_emb)
        pad_masks.append(prepared.token_mask)
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        prefix_embs = torch.cat(img_embs, dim=1)
        prefix_pad_masks = torch.cat(pad_masks, dim=1)
        att_masks_t = torch.tensor(att_masks, dtype=torch.bool, device=prefix_pad_masks.device)
        prefix_att_masks = att_masks_t[None, :].expand(prefix_pad_masks.shape[0], len(att_masks))

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = prepare_attention_masks_4d(prefix_att_2d_masks)
        net.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"

        # Forward through PaliGemma with LoRA
        _, past_key_values = net.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # Forward through Action Expert with LoRA
        v_pred = net.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=time_steps.to(dtype=proj_dtype),
        )

        # Mathematically exact Flow Matching Loss on the 8D action space
        flow_loss = F.mse_loss(v_pred[:, :, :8], v_target[:, :, :8])
        return flow_loss

    # PREFLIGHT BENCHMARK MODE
    if args.preflight_only:
        print("\n" + "=" * 85)
        print("  EMPIRICAL PREFLIGHT BENCHMARK (Warmup + 5 Trial Steps)")
        print("=" * 85)
        torch.cuda.reset_peak_memory_stats("cuda:0")
        net.train()

        optimizer.zero_grad()
        f_t, w_t, s_t, a_t, t_list = sample_balanced_batch(args.batch_size)
        w_loss = compute_loss(f_t, w_t, s_t, a_t, t_list, args.batch_size)
        w_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()

        trial_steps = 5
        t_bench_start = time.perf_counter()
        for i in range(trial_steps):
            optimizer.zero_grad()
            step_flow_val = 0.0
            for _ in range(args.grad_accum):
                f_t, w_t, s_t, a_t, t_list = sample_balanced_batch(args.batch_size)
                f_l = compute_loss(f_t, w_t, s_t, a_t, t_list, args.batch_size)
                (f_l / args.grad_accum).backward()
                step_flow_val += f_l.item() / args.grad_accum
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            torch.cuda.synchronize()
            print(f"    Trial step {i+1}/{trial_steps}: FlowLoss={step_flow_val:.4f}")
        t_bench_end = time.perf_counter()

        elapsed_bench = t_bench_end - t_bench_start
        step_time_s = elapsed_bench / trial_steps
        steps_per_sec = trial_steps / elapsed_bench
        peak_vram_gb = torch.cuda.max_memory_allocated("cuda:0") / (1024 ** 3)
        total_vram_gb = torch.cuda.get_device_properties("cuda:0").total_memory / (1024 ** 3)
        est_total_min = (args.steps * step_time_s) / 60.0

        print("-" * 85)
        print(f"  [*] Trainable params: {trainable_count:,} ({trainable_count/1e6:,.2f} M)")
        print(f"  [*] Frozen params:    {frozen_count:,} ({frozen_count/1e6:,.2f} M)")
        print(f"  [*] Peak VRAM:        {peak_vram_gb:.2f} GB / {total_vram_gb:.2f} GB ({peak_vram_gb/total_vram_gb*100:.1f}%)")
        print(f"  [*] Step Latency:     {step_time_s*1000:.1f} ms / step ({steps_per_sec:.2f} steps/sec)")
        print(f"  [*] Estimated Time:   {est_total_min:.1f} minutes for {args.steps} steps")
        print("=" * 85)
        print("\n[Preflight Passed] Clean Pure Flow Matching benchmark passed. Ready for full execution.")
        return 0

    # Baseline Eval on held-out validation set
    print("\n" + "=" * 85)
    print("  INITIAL EVALUATION (HELD-OUT VALIDATION SET, 10-STEP EULER ODE)")
    print("=" * 85)
    net.eval()
    if val_samples:
        init_val_agg = evaluate_validation_set(inference, val_samples, cached_files, dfk=dfk, max_samples=args.num_val_samples)
        print(f"  Samples Evaluated: {init_val_agg.get('val_samples_count', 0)} from {len(val_ep_set)} held-out episodes")
        print(f"  * Baseline Action MSE: {init_val_agg.get('action_mse', 0.0):.6f}")
        print(f"  * Baseline Joint MAE:  {init_val_agg.get('joint_mae_deg', 0.0):.2f}° (Hold-Still Baseline: {init_val_agg.get('hold_still_mae_deg', 0.0):.2f}°)")
        print(f"  * Baseline Gripper MSE:{init_val_agg.get('gripper_mse', 0.0):.4f}")
        if 'ee_pos_err_mm' in init_val_agg:
            print(f"  * Baseline EE PosErr:  {init_val_agg['ee_pos_err_mm']:.1f} mm")
            print(f"  * Baseline EE RotErr:  {init_val_agg['ee_rot_err_deg']:.2f}°")
        if wandb_run:
            wandb.log({
                "val/action_mse": init_val_agg.get("action_mse", 0.0),
                "val/joint_mae_deg": init_val_agg.get("joint_mae_deg", 0.0),
                "val/hold_still_mae_deg": init_val_agg.get("hold_still_mae_deg", 0.0),
                "val/improve_pct": init_val_agg.get("improve_pct", 0.0),
                "val/gripper_mse": init_val_agg.get("gripper_mse", 0.0),
                "val/ee_pos_err_mm": init_val_agg.get("ee_pos_err_mm", 0.0),
                "val/ee_rot_err_deg": init_val_agg.get("ee_rot_err_deg", 0.0),
            }, step=start_step - 1)
    print("=" * 85)

    # 5. Training Loop
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Training] Commencing multi-task LoRA fine-tuning (Steps {start_step:,} to {args.steps:,})...")
    print(f"[*] Checkpoints will be saved to: {args.output_dir}")

    net.train()
    flow_losses = []
    t_start = time.perf_counter()

    for step in range(start_step, args.steps + 1):
        optimizer.zero_grad()
        accum_flow = 0.0

        for _ in range(args.grad_accum):
            f_t, w_t, s_t, a_t, t_list = sample_balanced_batch(args.batch_size)
            f_l = compute_loss(f_t, w_t, s_t, a_t, t_list, args.batch_size)
            (f_l / args.grad_accum).backward()
            accum_flow += f_l.item() / args.grad_accum

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        flow_losses.append(accum_flow)

        # Periodic train logging
        if step % args.log_freq == 0:
            avg_flow = np.mean(flow_losses[-args.log_freq:])
            cur_lr = scheduler.get_last_lr()[0]
            elapsed = time.perf_counter() - t_start
            steps_per_sec = (step - start_step + 1) / max(elapsed, 1e-4)
            print(f"  [Step {step:05d}/{args.steps:05d}] FlowLoss: {avg_flow:.4f} | LR: {cur_lr:.2e} | Speed: {steps_per_sec:.2f} s/s | Elapsed: {elapsed/60:.1f}m")

            if wandb_run:
                wandb.log({
                    "train/flow_loss": avg_flow,
                    "train/lr": cur_lr,
                    "train/speed_sps": steps_per_sec,
                }, step=step)

        # Held-out validation evaluation with 10-step ODE Euler solver
        if step % args.eval_freq == 0 and val_samples:
            net.eval()
            print(f"\n  ===========================================================================")
            print(f"  >>> HELD-OUT VALIDATION SET EVALUATION @ Step {step:05d} (10-Step Euler) <<<")
            print(f"  ===========================================================================")
            val_agg = evaluate_validation_set(inference, val_samples, cached_files, dfk=dfk, max_samples=args.num_val_samples)
            print(f"  Samples Evaluated: {val_agg.get('val_samples_count', 0)} from {len(val_ep_set)} held-out episodes")
            print(f"  * Val Action MSE:  {val_agg.get('action_mse', 0.0):.6f}")
            print(f"  * Val Joint MAE:   {val_agg.get('joint_mae_deg', 0.0):.2f}° (Hold-Still: {val_agg.get('hold_still_mae_deg', 0.0):.2f}°)")
            print(f"  * Val vs Hold:     +{val_agg.get('improve_pct', 0.0):.1f}% improvement")
            print(f"  * Val Gripper MSE: {val_agg.get('gripper_mse', 0.0):.4f}")
            if 'ee_pos_err_mm' in val_agg:
                print(f"  * Val EE PosErr:   {val_agg['ee_pos_err_mm']:.1f} mm")
                print(f"  * Val EE RotErr:   {val_agg['ee_rot_err_deg']:.2f}°")
            print(f"  ===========================================================================\n")

            if wandb_run:
                wandb.log({
                    "val/action_mse": val_agg.get("action_mse", 0.0),
                    "val/samples_evaluated": val_agg.get("val_samples_count", 0),
                    "val/pool_size": len(val_samples),
                    "val/joint_mae_deg": val_agg.get("joint_mae_deg", 0.0),
                    "val/hold_still_mae_deg": val_agg.get("hold_still_mae_deg", 0.0),
                    "val/improve_pct": val_agg.get("improve_pct", 0.0),
                    "val/gripper_mse": val_agg.get("gripper_mse", 0.0),
                    "val/ee_pos_err_mm": val_agg.get("ee_pos_err_mm", 0.0),
                    "val/ee_rot_err_deg": val_agg.get("ee_rot_err_deg", 0.0),
                }, step=step)
            net.train()

        # Checkpoint saving
        if step % args.save_freq == 0 or step == args.steps:
            ckpt_name = f"step_{step:05d}.pt"
            ckpt_path = args.output_dir / ckpt_name
            state_dict = extract_lora_state_dict(net)
            save_payload = {
                "step": step,
                "flow_loss": float(np.mean(flow_losses[-50:])),
                "state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "lang_rank": args.lang_rank,
                "expert_rank": args.expert_rank,
                "lora_dropout": args.lora_dropout,
                "args": vars(args),
                "val_episodes": list(val_ep_set),
                "action_semantics": ACTION_SEMANTICS,
                "image_crop": args.image_crop,
                "precision": PRECISION,
                "source_revision": source_revision,
                "source_dirty": source_dirty,
                "validation_noise": "sha256_sample_identity_cpu_v1",
                "lang_alpha": float(args.lang_rank * 2),
                "expert_alpha": float(args.expert_rank * 2),
                "timestamp": time.time(),
            }
            torch.save(save_payload, ckpt_path)
            latest_path = args.output_dir / "latest.pt"
            try:
                shutil.copyfile(str(ckpt_path), str(latest_path))
            except Exception:
                pass
            ckpt_size_mb = ckpt_path.stat().st_size / (1024 * 1024)
            print(f"  [Checkpoint] Saved LoRA checkpoint to {ckpt_name} ({ckpt_size_mb:.2f} MB)")

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 85)
    print(f"  LoRA FINE-TUNING COMPLETED in {total_time/60:.1f} minutes!")
    print(f"  Checkpoints saved in: {args.output_dir}")
    print("=" * 85)

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
