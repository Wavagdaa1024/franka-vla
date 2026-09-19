#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pi0.5 Franka Multi-Task LoRA Fine-Tuning Pipeline.
Implements Northwestern University IDEAS Lab methodology with dual ablation modes:
  1. [pure_flow]    : Pure 8D joint-space Flow Matching Loss (OpenPI Baseline).
  2. [cartesian_7d] : Forward-inferred 7D End-Effector action MSE Loss ([dx, dy, dz, drx, dry, drz, gripper]).

Features:
  - Episode-level train/test split: strictly isolates held-out episodes for validation (zero temporal leakage).
  - Weights & Biases (wandb) integration: real-time logging of train flow loss, physical errors, LR, and held-out validation metrics.
  - Multi-sample held-out validation evaluation during training.
  - Periodic checkpointing every N steps (step_05000.pt, step_10000.pt, ... step_50000.pt + latest.pt).

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
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from franka_teleop.pi05_engine.runtime import PI05Inference
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
DEFAULT_OUTPUT_CKPT_DIR = REPO_ROOT / "outputs" / "checkpoints" / "pi05_lora_pure_flow_50k"

CHUNK_SIZE = 15
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM = 2  # Effective batch size = 8
DEFAULT_STEPS = 50000
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_LOG_FREQ = 25
DEFAULT_SAVE_FREQ = 5000
DEFAULT_EVAL_FREQ = 2500
SEED = 42


def preload_video_frames(video_path: Path):
    """Preload all video frames into memory as RGB numpy arrays."""
    container = av.open(str(video_path))
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return frames


def build_multitask_dataset(dataset_dirs, task_filter=None, val_ratio=0.15, val_seed=42):
    """
    Builds in-memory index of 15-step relative joint position chunks across all datasets:
      a_k = q_{t+k+1} - q_t = sum_{i=0}^k delta_q_{t+i} (for k = 0..14)
      gripper_k = gripper_{t+k+1} (0=OPEN, 1=CLOSED)
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

        # Load episode-level tasks mapping from meta/episodes/**/*.parquet
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

        # Fallback to tasks.parquet if available
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
            front_frames = preload_video_frames(cfg["front_mp4"])
            wrist_frames = preload_video_frames(cfg["wrist_mp4"])

            states_raw = np.array([row.as_py() for row in table.column("observation.state")], dtype=np.float32)
            ep_indices = np.array(table.column("episode_index").to_pylist(), dtype=np.int32)
            task_indices = np.array(table.column("task_index").to_pylist(), dtype=np.int32)

            cached_files[file_key] = {
                "front_frames": front_frames,
                "wrist_frames": wrist_frames,
                "states": states_raw,
                "episodes": ep_indices,
                "task_indices": task_indices,
                "length": len(states_raw)
            }

            # Group indices by episode
            ep_boundaries = defaultdict(list)
            for idx, ep in enumerate(ep_indices):
                ep_boundaries[ep].append(idx)

            # Sample valid chunks
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
    if val_ratio > 0.0 and len(all_episodes) >= 2:
        rng = random.Random(val_seed)
        num_val_eps = max(1, int(round(len(all_episodes) * val_ratio)))
        shuffled_eps = list(all_episodes)
        rng.shuffle(shuffled_eps)
        val_ep_set = set(shuffled_eps[:num_val_eps])
    else:
        val_ep_set = set()

    train_samples = [s for s in raw_samples if s["global_ep_id"] not in val_ep_set]
    val_samples = [s for s in raw_samples if s["global_ep_id"] in val_ep_set]

    # Index training chunks by task for balanced sampling
    train_task_buckets = defaultdict(list)
    for idx, s in enumerate(train_samples):
        train_task_buckets[s["task"]].append(idx)

    # Index validation chunks by task
    val_task_buckets = defaultdict(list)
    for idx, s in enumerate(val_samples):
        val_task_buckets[s["task"]].append(idx)

    print(f"\n" + "=" * 85)
    print(f"  DATASET PARTITION & LEAKAGE AUDIT")
    print(f"=" * 85)
    print(f"  Total Demonstrations Loaded: {len(raw_samples):,} chunks across {len(all_episodes)} episodes ({time.perf_counter()-t0:.2f}s)")
    print(f"  Partition Strategy:          Strict Episode-Level Isolation (0 temporal leakage)")
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
    """Evaluates a single sample and computes detailed joint & Cartesian metrics."""
    cf = cached_files[s["file_key"]]
    f_idx = s["frame_idx"]
    f_img = cf["front_frames"][f_idx]
    w_img = cf["wrist_frames"][f_idx]
    curr_state = cf["states"][f_idx]
    curr_q = curr_state[:7]

    future_states = cf["states"][f_idx + 1 : f_idx + 1 + CHUNK_SIZE]
    gt_delta_q = future_states[:, :7] - curr_q[None, :]
    gt_gripper = future_states[:, 7:8]
    gt_action_chunk = np.concatenate([gt_delta_q, gt_gripper], axis=1)

    front_t = torch.from_numpy(np.transpose(f_img, (2, 0, 1)))
    wrist_t = torch.from_numpy(np.transpose(w_img, (2, 0, 1)))
    obs = {
        "observation.images.base_0_rgb": front_t,
        "observation.images.left_wrist_0_rgb": wrist_t,
        "observation.state": curr_state
    }
    with torch.no_grad():
        pred_chunk = inference.predict_action_chunk(obs, s["task"]).cpu().numpy()[0]

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
        curr_q_t = torch.from_numpy(curr_q[None, :]).unsqueeze(1).float().to("cuda:0" if torch.cuda.is_available() else "cpu")
        pred_delta_q_t = torch.from_numpy(pred_chunk[None, :, :7]).float().to("cuda:0" if torch.cuda.is_available() else "cpu")
        gt_delta_q_t = torch.from_numpy(gt_delta_q[None, :, :7]).float().to("cuda:0" if torch.cuda.is_available() else "cpu")

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
    Returns aggregated metrics for logging and generalization tracking.
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
    parser = argparse.ArgumentParser(description="Multi-Task LoRA Fine-Tuning for Pi0.5 Franka.")
    parser.add_argument("--loss-mode", type=str, choices=("pure_flow", "cartesian_7d"), default="pure_flow",
                        help="Loss objective: 'pure_flow' (8D Joint Flow Matching Loss) or 'cartesian_7d' (DFK 7D EE Action MSE Loss)")
    parser.add_argument("--dataset", type=Path, nargs="+", default=DEFAULT_DATASET_DIRS, help="Path(s) to dataset(s)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_CKPT_DIR, help="Directory to save LoRA checkpoints")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Total training steps (default: 50000)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate for LoRA & projections (default: 1e-4)")
    parser.add_argument("--lang-rank", type=int, default=16, help="LoRA rank for PaliGemma-2B (default: 16)")
    parser.add_argument("--expert-rank", type=int, default=32, help="LoRA rank for Gemma-300M Expert (default: 32)")
    parser.add_argument("--state-noise", type=float, default=0.008, help="Gaussian noise std added to state (default: 0.008 rad)")
    parser.add_argument("--warmup-steps", type=int, default=200, help="Linear warmup steps (default: 200)")
    parser.add_argument("--log-freq", type=int, default=DEFAULT_LOG_FREQ)
    parser.add_argument("--save-freq", type=int, default=DEFAULT_SAVE_FREQ)
    parser.add_argument("--eval-freq", type=int, default=DEFAULT_EVAL_FREQ)
    parser.add_argument("--task-filter", type=str, default=None, help="Filter dataset by task substring (e.g. 'red cube')")
    parser.add_argument("--preflight-only", action="store_true", help="Run empirical benchmark and exit")
    # Validation & WandB options
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Ratio of episodes held out for test/validation (default: 0.15)")
    parser.add_argument("--val-seed", type=int, default=42, help="Random seed for train/val episode split (default: 42)")
    parser.add_argument("--num-val-samples", type=int, default=25, help="Number of held-out test chunks evaluated on eval steps (default: 25)")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases real-time tracking")
    parser.add_argument("--wandb-project", type=str, default="pi05-franka-vla", help="W&B project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run display name")
    parser.add_argument("--wandb-mode", type=str, choices=("online", "offline", "disabled"), default="offline", help="W&B mode (default: offline)")
    args = parser.parse_args()

    # Automatically align default output directory if not explicitly overridden
    if args.output_dir == DEFAULT_OUTPUT_CKPT_DIR and args.loss_mode == "cartesian_7d":
        args.output_dir = REPO_ROOT / "outputs" / "checkpoints" / "pi05_lora_cartesian_7d"

    print("=" * 85)
    print("  PI0.5 MULTI-TASK LoRA FINE-TUNING PIPELINE (NORTHWESTERN IDEAS LAB ROUTE)")
    print("=" * 85)
    print(f"[*] Loss Mode:       [{args.loss_mode.upper()}] {'(Pure 8D Joint Flow Matching Loss)' if args.loss_mode == 'pure_flow' else '(DFK 7D EE Action MSE Loss [dx,dy,dz,drx,dry,drz,grip])'}")
    print(f"[*] Datasets ({len(args.dataset)}):")
    for d in args.dataset:
        print(f"      - {d}")
    print(f"[*] Output Dir:      {args.output_dir}")
    print(f"[*] Steps:           {args.steps:,} (Warmup: {args.warmup_steps}, SaveFreq: {args.save_freq:,}, EvalFreq: {args.eval_freq:,})")
    print(f"[*] Batch Size:      {args.batch_size} (Grad Accum: {args.grad_accum}, Effective Batch: {args.batch_size*args.grad_accum})")
    print(f"[*] Learning Rate:   {args.lr:.2e}")
    print(f"[*] PaliGemma LoRA:  Rank={args.lang_rank}, Alpha={args.lang_rank*2} (Attention projections)")
    print(f"[*] Action Expert:   Rank={args.expert_rank}, Alpha={args.expert_rank*2} (Attention projections)")
    print(f"[*] Action Space:    15-step cumulative relative joint displacement a[k] = q[t+k+1] - q[t]")
    print(f"[*] State Noise:     {args.state_noise:.4f} rad (Anti-Trajectory Memorization)")
    print(f"[*] Val Partition:   {args.val_ratio*100:.1f}% held-out episodes (seed={args.val_seed}, {args.num_val_samples} eval samples)")
    print(f"[*] W&B Tracking:    {'ENABLED (Mode: ' + args.wandb_mode + ')' if args.wandb else 'DISABLED'}")
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "1")
    print(f"[*] Hardware:        Physical GPU {gpu_id} (RTX 5090 32GB, CUDA_VISIBLE_DEVICES={gpu_id})")
    print("-" * 85)

    # 1. Dataset build with episode-level train / val partition
    train_samples, val_samples, cached_files, train_task_buckets, val_ep_set = build_multitask_dataset(
        args.dataset,
        task_filter=args.task_filter,
        val_ratio=args.val_ratio,
        val_seed=args.val_seed
    )
    task_names = sorted(list(train_task_buckets.keys()))

    # Initialize W&B if requested
    wandb_run = None
    if args.wandb:
        import wandb
        run_name = args.wandb_name or f"pi05_{args.loss_mode}_{args.steps // 1000}k"
        try:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=vars(args),
                mode=args.wandb_mode,
            )
            print(f"[W&B] Initialized run '{run_name}' in project '{args.wandb_project}' (Mode: {args.wandb_mode})")
        except Exception as e:
            print(f"[W&B Warning] Online init failed: {e}. Falling back to offline mode...")
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=vars(args),
                mode="offline",
            )
            print(f"[W&B] Initialized offline run '{run_name}'.")

    # 2. Model initialization
    print(f"\n[Model] Initializing Pi0.5 base model on GPU {gpu_id}...")
    t0 = time.perf_counter()
    inference = PI05Inference.from_checkpoint(
        CHECKPOINT_DIR,
        stats_path=STATS_PATH,
        tokenizer_path=TOKENIZER_PATH,
        device="cuda:0",
        profile="droid_jointpos"
    )
    print(f"[Model] Base model loaded in {time.perf_counter()-t0:.2f}s.")

    net = inference.network
    processor = inference.processor

    # 3. Inject LoRA adapters
    print(f"\n[LoRA] Injecting LoRA adapters (Language r={args.lang_rank}, Expert r={args.expert_rank})...")
    lang_loras, expert_loras = inject_pi05_lora(
        net,
        lang_rank=args.lang_rank,
        lang_alpha=float(args.lang_rank * 2),
        expert_rank=args.expert_rank,
        expert_alpha=float(args.expert_rank * 2),
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
    print(f"  Action Projections:   {sum(p.numel() for m in [net.action_in_proj, net.action_out_proj, net.time_mlp_in, net.time_mlp_out] for p in m.parameters())/1e6:,.2f} M params")
    print("-" * 85)
    print(f"  [SUMMARY] Trainable params: {trainable_count:,} ({trainable_count/1e6:,.2f} M, {trainable_count/total_count*100:.2f}%)")
    print(f"  [SUMMARY] Frozen params:    {frozen_count:,} ({frozen_count/1e6:,.2f} M, {frozen_count/total_count*100:.2f}%)")
    print(f"  [SUMMARY] Total params:     {total_count:,} ({total_count/1e6:,.2f} M)")
    print("=" * 85)

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=DEFAULT_WEIGHT_DECAY, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-6)

    beta_dist = Beta(
        torch.tensor(getattr(net.config, "time_sampling_beta_alpha", 1.5), device="cuda:0"),
        torch.tensor(getattr(net.config, "time_sampling_beta_beta", 1.0), device="cuda:0")
    )

    # Initialize PyTorch Differentiable Kinematics on GPU for physical tracking
    print("\n[Kinematics] Initializing PyTorch Franka Differentiable Kinematics (DFK) on cuda:0...")
    dfk = FrankaDifferentiableKinematics(device="cuda:0")

    # Balanced batch sampling from training partition
    def sample_balanced_batch(batch_size):
        sampled = []
        for _ in range(batch_size):
            t_choice = random.choice(task_names)
            idx_choice = random.choice(train_task_buckets[t_choice])
            sampled.append(train_samples[idx_choice])

        front_list, wrist_list, state_list, action_list, tasks_list = [], [], [], [], []
        for s in sampled:
            cf = cached_files[s["file_key"]]
            f_idx = s["frame_idx"]

            f_img = cf["front_frames"][f_idx]
            w_img = cf["wrist_frames"][f_idx]
            curr_state = cf["states"][f_idx].copy()
            curr_q = curr_state[:7]

            # Anti-memorization jitter
            if args.state_noise > 0:
                curr_state[:7] += np.random.normal(0, args.state_noise, size=7).astype(np.float32)

            future_states = cf["states"][f_idx + 1 : f_idx + 1 + CHUNK_SIZE]
            delta_q = future_states[:, :7] - curr_q[None, :]
            gripper = future_states[:, 7:8]
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

        # Vision Tower is frozen: compute image embeddings without tracking gradients
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

        # Language tokens: track gradients through PaliGemma LoRA
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

        # Forward through PaliGemma with LoRA (gradients enabled)
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

        flow_loss = F.mse_loss(v_pred[:, :, :8], v_target[:, :, :8])

        # Forward-infer clean action via 1-step ODE integration
        pred_norm_actions_8d = (noise[:, :, :8] - v_pred[:, :, :8]).float()
        pred_actions_8d = processor._transform(pred_norm_actions_8d, "action", "ACTION", inverse=True)
        pred_delta_q = pred_actions_8d[:, :, :7]
        pred_gripper = pred_actions_8d[:, :, 7:8]

        gt_delta_q = actions_gpu[:, :, :7]
        gt_gripper = actions_gpu[:, :, 7:8]

        curr_q_tensor = torch.as_tensor(state_t[:, :7], device="cuda:0", dtype=torch.float32).unsqueeze(1)

        pos_err_mm = 0.0
        rot_err_deg = 0.0
        grip_err = 0.0

        if args.loss_mode == "pure_flow":
            total_loss = flow_loss
            loss_7d = torch.tensor(0.0, device="cuda:0")

            with torch.no_grad():
                pred_ee_6d = dfk.compute_relative_ee_action(curr_q_tensor, pred_delta_q)
                gt_ee_6d = dfk.compute_relative_ee_action(curr_q_tensor, gt_delta_q)
                pos_err_mm = float(torch.norm(pred_ee_6d[..., :3] - gt_ee_6d[..., :3], dim=-1).mean().item() * 1000.0)
                rot_err_deg = float(torch.rad2deg(torch.norm(pred_ee_6d[..., 3:6] - gt_ee_6d[..., 3:6], dim=-1)).mean().item())
                grip_err = float(torch.abs(pred_gripper - gt_gripper).mean().item())

        elif args.loss_mode == "cartesian_7d":
            pred_ee_6d = dfk.compute_relative_ee_action(curr_q_tensor, pred_delta_q)
            pred_7d = torch.cat([pred_ee_6d, pred_gripper], dim=-1)

            with torch.no_grad():
                gt_ee_6d = dfk.compute_relative_ee_action(curr_q_tensor, gt_delta_q)
                gt_7d = torch.cat([gt_ee_6d, gt_gripper], dim=-1)

            loss_7d = F.mse_loss(pred_7d, gt_7d)
            total_loss = loss_7d

            with torch.no_grad():
                pos_err_mm = float(torch.norm(pred_ee_6d[..., :3] - gt_ee_6d[..., :3], dim=-1).mean().item() * 1000.0)
                rot_err_deg = float(torch.rad2deg(torch.norm(pred_ee_6d[..., 3:6] - gt_ee_6d[..., 3:6], dim=-1)).mean().item())
                grip_err = float(torch.abs(pred_gripper - gt_gripper).mean().item())

        return total_loss, flow_loss, loss_7d, pos_err_mm, rot_err_deg, grip_err

    # PREFLIGHT MODE
    if args.preflight_only:
        print("\n" + "=" * 85)
        print(f"  EMPIRICAL PREFLIGHT BENCHMARK (Warmup + 5 Trial Steps) [{args.loss_mode.upper()}]")
        print("=" * 85)
        torch.cuda.reset_peak_memory_stats("cuda:0")
        net.train()

        # Warmup step
        optimizer.zero_grad()
        f_t, w_t, s_t, a_t, t_list = sample_balanced_batch(args.batch_size)
        w_loss, _, _, _, _, _ = compute_loss(f_t, w_t, s_t, a_t, t_list, args.batch_size)
        w_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()

        trial_steps = 5
        t_bench_start = time.perf_counter()
        for i in range(trial_steps):
            optimizer.zero_grad()
            step_loss_val = 0.0
            step_flow_val = 0.0
            step_7d_val = 0.0
            step_pos_err_val = 0.0
            step_rot_err_val = 0.0
            step_grip_err_val = 0.0
            for _ in range(args.grad_accum):
                f_t, w_t, s_t, a_t, t_list = sample_balanced_batch(args.batch_size)
                tot_l, f_l, l7d, p_err, r_err, g_err = compute_loss(f_t, w_t, s_t, a_t, t_list, args.batch_size)
                (tot_l / args.grad_accum).backward()
                step_loss_val += tot_l.item() / args.grad_accum
                step_flow_val += f_l.item() / args.grad_accum
                step_7d_val += l7d.item() / args.grad_accum
                step_pos_err_val += p_err / args.grad_accum
                step_rot_err_val += r_err / args.grad_accum
                step_grip_err_val += g_err / args.grad_accum
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            torch.cuda.synchronize()
            loss_detail = f"FlowLoss={step_flow_val:.4f}" if args.loss_mode == "pure_flow" else f"7D-MSE={step_7d_val:.5f}"
            print(f"    Trial step {i+1}/{trial_steps}: Loss={step_loss_val:.5f} ({loss_detail}) | PosErr={step_pos_err_val:.1f}mm | RotErr={step_rot_err_val:.2f}° | GripErr={step_grip_err_val:.3f}")
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
        print(f"\n[Preflight Passed] Empirical benchmark passed for [{args.loss_mode.upper()}]. Ready for full execution.")
        return 0

    # Initial Baseline Eval on held-out validation set
    print("\n" + "=" * 85)
    print("  INITIAL ZERO-SHOT EVALUATION (HELD-OUT VALIDATION SET)")
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
            }, step=0)
    print("=" * 85)

    # 5. Training Loop
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Training] Commencing {args.steps:,} steps of multi-task LoRA fine-tuning...")
    print(f"[*] Checkpoints will be saved to: {args.output_dir}")

    net.train()
    losses = []
    flow_losses = []
    l7d_losses = []
    pos_errors = []
    rot_errors = []
    grip_errors = []
    t_start = time.perf_counter()

    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_flow = 0.0
        accum_7d = 0.0
        accum_pos_err = 0.0
        accum_rot_err = 0.0
        accum_grip_err = 0.0

        for _ in range(args.grad_accum):
            f_t, w_t, s_t, a_t, t_list = sample_balanced_batch(args.batch_size)
            tot_l, f_l, l7d, p_err, r_err, g_err = compute_loss(f_t, w_t, s_t, a_t, t_list, args.batch_size)
            (tot_l / args.grad_accum).backward()
            accum_loss += tot_l.item() / args.grad_accum
            accum_flow += f_l.item() / args.grad_accum
            accum_7d += l7d.item() / args.grad_accum
            accum_pos_err += p_err / args.grad_accum
            accum_rot_err += r_err / args.grad_accum
            accum_grip_err += g_err / args.grad_accum

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        losses.append(accum_loss)
        flow_losses.append(accum_flow)
        l7d_losses.append(accum_7d)
        pos_errors.append(accum_pos_err)
        rot_errors.append(accum_rot_err)
        grip_errors.append(accum_grip_err)

        # Periodic train logging
        if step % args.log_freq == 0:
            avg_loss = np.mean(losses[-args.log_freq:])
            avg_flow = np.mean(flow_losses[-args.log_freq:])
            avg_7d = np.mean(l7d_losses[-args.log_freq:])
            avg_pos_err = np.mean(pos_errors[-args.log_freq:])
            avg_rot_err = np.mean(rot_errors[-args.log_freq:])
            avg_grip_err = np.mean(grip_errors[-args.log_freq:])
            cur_lr = scheduler.get_last_lr()[0]
            elapsed = time.perf_counter() - t_start
            steps_per_sec = step / elapsed
            loss_str = f"FlowLoss: {avg_flow:.4f}" if args.loss_mode == "pure_flow" else f"7D-MSE: {avg_7d:.5f}"
            print(f"  [Step {step:05d}/{args.steps:05d}] {loss_str} | PosErr: {avg_pos_err:.1f}mm | RotErr: {avg_rot_err:.2f}° | GripErr: {avg_grip_err:.3f} | LR: {cur_lr:.2e} | Speed: {steps_per_sec:.2f} s/s | Elapsed: {elapsed/60:.1f}m")

            if wandb_run:
                wandb.log({
                    "train/loss": avg_loss,
                    "train/flow_loss": avg_flow,
                    "train/pos_err_mm": avg_pos_err,
                    "train/rot_err_deg": avg_rot_err,
                    "train/grip_err": avg_grip_err,
                    "train/lr": cur_lr,
                    "train/speed_sps": steps_per_sec,
                }, step=step)

        # Held-out validation evaluation
        if step % args.eval_freq == 0 and val_samples:
            net.eval()
            print(f"\n  ===========================================================================")
            print(f"  >>> HELD-OUT VALIDATION SET EVALUATION @ Step {step:05d} <<<")
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
                "loss_mode": args.loss_mode,
                "loss": float(np.mean(losses[-50:])),
                "flow_loss": float(np.mean(flow_losses[-50:])),
                "cartesian_7d_loss": float(np.mean(l7d_losses[-50:])),
                "pos_err_mm": float(np.mean(pos_errors[-50:])),
                "rot_err_deg": float(np.mean(rot_errors[-50:])),
                "grip_err": float(np.mean(grip_errors[-50:])),
                "state_dict": state_dict,
                "lang_rank": args.lang_rank,
                "expert_rank": args.expert_rank,
                "args": vars(args),
                "val_episodes": list(val_ep_set),
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
    print(f"  LoRA FINE-TUNING [{args.loss_mode.upper()}] COMPLETED in {total_time/60:.1f} minutes!")
    print(f"  Checkpoints saved in: {args.output_dir}")
    print("=" * 85)

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
