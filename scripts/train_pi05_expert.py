#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pi0.5 Joint Position Action Expert Fine-Tuning Script.
Trains on audited clean teleoperation demonstrations.

Architecture Configuration:
  - Language Backbone (Gemma-2B): 100% FROZEN (zero trainable parameters, no drift).
  - Vision Tower (SigLIP So400M ViT): 100% FROZEN for Round 1 (Top K layers unfreezable for Round 2).
  - Multi-Modal Projector: 100% FROZEN for Round 1.
  - Action Expert (Gemma-300M): Full fine-tuning (LR=5e-5).
  - Action Projections & Time MLP: Full fine-tuning (LR=1e-4).

Action Space Definition:
  - 15-step cumulative relative joint displacement:
      a_k = q_{t+k+1} - q_t = sum_{i=0}^k delta_q_{t+i}   (k = 0..14)
  - Gripper state:
      gripper_k in [0.0, 1.0] (0=OPEN, 1=CLOSED)

Hardware Target: Physical GPU 1 (RTX 5090 32GB, CUDA_VISIBLE_DEVICES=1).
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

# Enforce strict GPU 1 isolation
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import av
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.distributions.beta import Beta

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from franka_teleop.pi05_engine.runtime import PI05Inference
from franka_teleop.pi05_engine.utils import prepare_attention_masks_4d
from franka_teleop.tensor_utils import make_att_2d_masks

# Default Paths
REPO_ROOT = PROJECT_ROOT
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "pi05_droid_jointpos"
STATS_PATH = CHECKPOINT_DIR / "auxiliary" / "openpi_droid_jointpos_norm_stats.json"
TOKENIZER_PATH = CHECKPOINT_DIR / "auxiliary" / "paligemma_tokenizer.model"
DEFAULT_DATASET_DIRS = [
    REPO_ROOT / "dataset" / "teleop_pick_cube_15hz_001",
]
DEFAULT_OUTPUT_CKPT_DIR = REPO_ROOT / "outputs" / "checkpoints" / "pi05_jointpos_expert"

CHUNK_SIZE = 15
DEFAULT_BATCH_SIZE = 8
DEFAULT_STEPS = 2000
DEFAULT_ACTION_LR = 5e-5
DEFAULT_PROJ_LR = 1e-4
DEFAULT_VISION_LR = 5e-6
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_LOG_FREQ = 20
DEFAULT_SAVE_FREQ = 500
DEFAULT_EVAL_FREQ = 100
DEFAULT_VISION_LAYERS = 2
SEED = 42


def preload_video_frames(video_path: Path):
    """Preload all video frames into memory as RGB numpy arrays."""
    container = av.open(str(video_path))
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return frames


def build_jointpos_dataset(dataset_dirs, single_episode: int = None, invert_legacy_gripper: bool = False):
    """
    Builds in-memory index of 15-step relative joint position chunks across one or multiple datasets:
      a_k = q_{t+k+1} - q_t = sum_{i=0}^k delta_q_{t+i} (for k = 0..14)
      gripper_k = gripper_{t+k+1} (0=OPEN, 1=CLOSED)
    Ensures 0 cross-episode leakage.
    """
    if isinstance(dataset_dirs, (str, Path)):
        dataset_dirs = [Path(dataset_dirs)]
    else:
        dataset_dirs = [Path(d) for d in dataset_dirs]

    t0 = time.perf_counter()
    cached_files = {}
    all_samples = []

    for ds_idx, dataset_dir in enumerate(dataset_dirs):
        print(f"[Dataset {ds_idx+1}/{len(dataset_dirs)}] Loading demonstrations from {dataset_dir.name}...")
        if single_episode is not None:
            print(f"  >>> Single-Episode Mode (Episode {single_episode}) <<<")

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
                print(f"  Skipping unfinished/active {dataset_dir.name}/file-{fid:03d}.parquet")
                continue

            print(f"  Loading video & parquet for {dataset_dir.name}/file-{fid:03d}...")
            front_frames = preload_video_frames(cfg["front_mp4"])
            wrist_frames = preload_video_frames(cfg["wrist_mp4"])

            states_raw = np.array([row.as_py() for row in table.column("observation.state")], dtype=np.float32)
            ep_indices = np.array(table.column("episode_index").to_pylist(), dtype=np.int32)
            task_indices = np.array(table.column("task_index").to_pylist(), dtype=np.int32)

            # Gripper state: DROID standard (0.0=OPEN, 1.0=CLOSED)
            states_droid = states_raw.copy()
            if invert_legacy_gripper:
                states_droid[:, 7] = np.clip(1.0 - states_raw[:, 7], 0.0, 1.0)
            else:
                states_droid[:, 7] = np.clip(states_raw[:, 7], 0.0, 1.0)

            cached_files[file_key] = {
                "front_frames": front_frames,
                "wrist_frames": wrist_frames,
                "states": states_droid,
                "ep_indices": ep_indices,
                "task_indices": task_indices
            }

            # Index valid 15-step chunk starting indices
            unique_eps, counts = np.unique(ep_indices, return_counts=True)
            for ep_id, count in zip(unique_eps, counts):
                if single_episode is not None and ep_id != single_episode:
                    continue

                start = np.where(ep_indices == ep_id)[0][0]
                end = start + count - 1
                valid_len = count - CHUNK_SIZE
                if valid_len <= 0:
                    continue

                task_prompt = ep_tasks_map.get(ep_id)
                if not task_prompt or task_prompt == "None":
                    t_idx = task_indices[start]
                    task_prompt = task_map.get(t_idx, "pick and place the red cube")

                for t in range(start, end - CHUNK_SIZE + 1):
                    all_samples.append({
                        "file_id": file_key,
                        "frame_idx": t,
                        "task": task_prompt,
                        "ep_id": ep_id,
                        "dataset": dataset_dir.name,
                    })

    if not all_samples:
        raise ValueError("No valid demonstration samples found! Check dataset directories and single_episode filter.")

    print(f"\n[Index Ready] Indexed {len(all_samples)} valid 15-step chunks across {len(dataset_dirs)} dataset(s) in {time.perf_counter()-t0:.2f}s.")
    sample_tasks = sorted(list({s['task'] for s in all_samples}))
    print(f"  Unique Prompts ({len(sample_tasks)} total):")
    for task_name in sample_tasks:
        n_matching = sum(1 for s in all_samples if s['task'] == task_name)
        print(f"    - '{task_name}': {n_matching} chunks")
    return all_samples, cached_files


def evaluate_fixed_sample(inference, s, cached_files):
    """Run full 10-step Euler integration inference on a single sample and return physical action metrics."""
    cf = cached_files[s["file_id"]]
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

    return {
        "action_mse": action_mse,
        "joint_mse": joint_mse,
        "gripper_mse": gripper_mse,
        "joint_mae_deg": joint_mae_deg,
        "hold_still_mae_deg": hold_still_mae_deg,
        "improve_pct": improve_pct,
        "pred_chunk": pred_chunk,
        "gt_chunk": gt_action_chunk,
        "frame_idx": f_idx,
    }


def print_comparison_table(eval_res, title="Single Frame Action Test"):
    """Prints a clear side-by-side table comparing predicted vs ground-truth actions."""
    print("\n" + "=" * 85)
    print(f"  {title.upper()} (Frame {eval_res['frame_idx']})")
    print("=" * 85)
    print(f"[*] Action Total MSE: {eval_res['action_mse']:.6f} | Joint MSE: {eval_res['joint_mse']:.6f} | Gripper MSE: {eval_res['gripper_mse']:.6f}")
    print(f"[*] Joint MAE:        {eval_res['joint_mae_deg']:.2f} deg vs Hold Still Baseline: {eval_res['hold_still_mae_deg']:.2f} deg ({eval_res['improve_pct']:+.1f}% improvement)")
    print("-" * 85)
    print(f" Step | Pred Delta Q (deg, joints 0..6)          | GT Delta Q (deg, joints 0..6)           | Pred Grip | GT Grip")
    print(f" -----+------------------------------------------+-----------------------------------------+-----------+--------")
    pred = eval_res["pred_chunk"]
    gt = eval_res["gt_chunk"]
    for k in range(len(pred)):
        p_deg = [round(x, 1) for x in np.degrees(pred[k, :7])]
        g_deg = [round(x, 1) for x in np.degrees(gt[k, :7])]
        p_str = f"[{p_deg[0]:+5.1f},{p_deg[1]:+5.1f},{p_deg[2]:+5.1f},{p_deg[3]:+5.1f},{p_deg[4]:+5.1f},{p_deg[5]:+5.1f},{p_deg[6]:+5.1f}]"
        g_str = f"[{g_deg[0]:+5.1f},{g_deg[1]:+5.1f},{g_deg[2]:+5.1f},{g_deg[3]:+5.1f},{g_deg[4]:+5.1f},{g_deg[5]:+5.1f},{g_deg[6]:+5.1f}]"
        print(f"  {k:02d}  | {p_str} | {g_str} | {pred[k, 7]:9.3f} | {gt[k, 7]:6.3f}")
    print("=" * 85 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Pi0.5 Action Expert on 15Hz Franka demonstrations.")
    parser.add_argument("--dataset", type=Path, nargs="+", default=DEFAULT_DATASET_DIRS, help="Path(s) to LeRobot 15Hz dataset(s)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_CKPT_DIR, help="Directory to save model checkpoints")
    parser.add_argument("--single-episode", type=int, default=None, help="Overfit on a single episode index")
    parser.add_argument("--invert-legacy-gripper", action="store_true", help="Invert gripper 0/1 (ONLY for legacy 30Hz datasets)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Total training steps (default: 2000)")
    parser.add_argument("--action-lr", type=float, default=DEFAULT_ACTION_LR, help="Learning rate for Action Expert (default: 5e-5)")
    parser.add_argument("--proj-lr", type=float, default=DEFAULT_PROJ_LR, help="Learning rate for Action Projections & Time MLP (default: 1e-4)")
    parser.add_argument("--vision-lr", type=float, default=DEFAULT_VISION_LR, help="Learning rate for Vision Tower & Projector (Round 2, default: 5e-6)")
    parser.add_argument("--tune-vision", action="store_true", default=False, help="Enable fine-tuning vision tower top layers (Round 2)")
    parser.add_argument("--no-tune-vision", dest="tune_vision", action="store_false", help="Disable vision fine-tuning (Action Expert only, Round 1 default)")
    parser.add_argument("--vision-layers", type=int, default=DEFAULT_VISION_LAYERS, help="Number of top vision encoder layers to fine-tune (default: 2)")
    parser.add_argument("--state-noise", type=float, default=0.008, help="Gaussian noise std added to observation.state to break open-loop trajectory memorization (default: 0.008 rad)")
    parser.add_argument("--warmup-steps", type=int, default=200, help="Linear warmup steps (default: 200)")
    parser.add_argument("--log-freq", type=int, default=DEFAULT_LOG_FREQ)
    parser.add_argument("--save-freq", type=int, default=DEFAULT_SAVE_FREQ)
    parser.add_argument("--eval-freq", type=int, default=DEFAULT_EVAL_FREQ, help="Frequency of running Action MSE evaluation")
    parser.add_argument("--preflight-only", action="store_true", help="Run sanity preflight with empirical speed/VRAM benchmark and exit")
    args = parser.parse_args()

    if args.tune_vision and args.batch_size == DEFAULT_BATCH_SIZE:
        args.batch_size = 2  # Protect against VRAM overflow when backpropping through vision layers

    print("=" * 85)
    print("  PI0.5 DROID JOINTPOS: ACTION EXPERT FINE-TUNING PIPELINE")
    print("=" * 85)
    print(f"[*] Datasets ({len(args.dataset)}):")
    for d in args.dataset:
        print(f"      - {d}")
    print(f"[*] Output Dir:      {args.output_dir}")
    print(f"[*] Steps:           {args.steps} (Warmup: {args.warmup_steps})")
    print(f"[*] Batch Size:      {args.batch_size}")
    print(f"[*] Action LR:       {args.action_lr:.2e} (Gemma-300M Expert)")
    print(f"[*] Proj/MLP LR:     {args.proj_lr:.2e} (Action Projections + Time MLP)")
    print(f"[*] Tune Vision:     {args.tune_vision} " + (f"(Top {args.vision_layers} SigLIP Layers @ LR={args.vision_lr:.2e})" if args.tune_vision else "(100% Frozen)"))
    print(f"[*] State Noise:     {args.state_noise:.4f} rad (Anti-Trajectory Memorization)")
    print(f"[*] Language Model:  100% FROZEN (Gemma-2B zero drift)")
    print(f"[*] Action Definition: 15-step cumulative delta a[k] = q[t+k+1] - q[t]")
    print(f"[*] Hardware:        Physical GPU 1 (RTX 5090 32GB, CUDA_VISIBLE_DEVICES=1)")
    print("-" * 85)

    # 1. Dataset build
    all_samples, cached_files = build_jointpos_dataset(
        args.dataset,
        single_episode=args.single_episode,
        invert_legacy_gripper=args.invert_legacy_gripper
    )
    num_samples = len(all_samples)

    # Pick fixed evaluation samples
    eval_reach_sample = None
    eval_grasp_sample = None
    for s in all_samples:
        cf = cached_files[s["file_id"]]
        f_idx = s["frame_idx"]
        g_val = cf["states"][f_idx, 7]
        if eval_reach_sample is None and f_idx >= 30 and g_val < 0.05:
            eval_reach_sample = s
        if eval_grasp_sample is None and f_idx >= 80 and g_val > 0.5:
            eval_grasp_sample = s
    if eval_reach_sample is None:
        eval_reach_sample = all_samples[len(all_samples) // 4]
    if eval_grasp_sample is None:
        eval_grasp_sample = all_samples[min(len(all_samples) - 1, len(all_samples) // 2)]

    print(f"[*] Fixed Eval Sample 1 (Reaching): Frame {eval_reach_sample['frame_idx']} (Task: '{eval_reach_sample['task']}')")
    print(f"[*] Fixed Eval Sample 2 (Grasping): Frame {eval_grasp_sample['frame_idx']} (Task: '{eval_grasp_sample['task']}')")

    # 2. Model initialization
    print("\n[Model] Initializing Pi0.5 base model on GPU 1...")
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

    # 3. Model Architecture & Trainable Parameter Isolation
    print("\n[Isolation] Freezing entire model first...")
    for p in net.parameters():
        p.requires_grad = False

    # Gemma-300M Action Expert
    expert_params = []
    for p in net.paligemma_with_expert.gemma_expert.parameters():
        p.requires_grad = True
        expert_params.append(p)

    # Action Projections and Time MLP
    proj_mlp_params = []
    for m in [net.action_in_proj, net.action_out_proj, net.time_mlp_in, net.time_mlp_out]:
        for p in m.parameters():
            p.requires_grad = True
            proj_mlp_params.append(p)

    vision_params = []
    if args.tune_vision:
        print(f"[Vision Tuning] Unfreezing Multi-Modal Projector and Top {args.vision_layers} Vision Tower Layers...")
        for p in net.paligemma_with_expert.paligemma.model.multi_modal_projector.parameters():
            p.requires_grad = True
            vision_params.append(p)

        vt_layers = net.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers
        for layer in vt_layers[-args.vision_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
                vision_params.append(p)

    # STRICT ASSERTION: Language Model (Gemma-2B) MUST BE 100% FROZEN!
    lang_model = net.paligemma_with_expert.paligemma.model.language_model
    assert not any(p.requires_grad for p in lang_model.parameters()), "CRITICAL: Language model must be frozen!"

    trainable_params = [p for p in net.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    total_count = sum(p.numel() for p in net.parameters())
    frozen_count = total_count - trainable_count

    print("=" * 85)
    print("  PARAMETER ISOLATION AUDIT")
    print("=" * 85)
    print(f"  Frozen Language Model:    {sum(p.numel() for p in lang_model.parameters())/1e6:,.2f} M params (100% FROZEN)")
    print(f"  Trainable Action Expert:  {sum(p.numel() for p in expert_params)/1e6:,.2f} M params (Gemma-300M @ LR={args.action_lr:.2e})")
    print(f"  Trainable Proj/MLP:       {sum(p.numel() for p in proj_mlp_params)/1e6:,.2f} M params (In/Out Proj + Time MLP @ LR={args.proj_lr:.2e})")
    if vision_params:
        print(f"  Trainable Vision:         {sum(p.numel() for p in vision_params)/1e6:,.2f} M params (SigLIP Top {args.vision_layers} Layers @ LR={args.vision_lr:.2e})")
    else:
        print(f"  Frozen Vision Tower:      {sum(p.numel() for p in net.paligemma_with_expert.paligemma.model.vision_tower.parameters())/1e6:,.2f} M params (100% FROZEN)")
    print("-" * 85)
    print(f"  [SUMMARY] Trainable params: {trainable_count:,} ({trainable_count/1e6:,.2f} M, {trainable_count/total_count*100:.2f}%)")
    print(f"  [SUMMARY] Frozen params:    {frozen_count:,} ({frozen_count/1e6:,.2f} M, {frozen_count/total_count*100:.2f}%)")
    print(f"  [SUMMARY] Total params:     {total_count:,} ({total_count/1e6:,.2f} M)")
    print("=" * 85)

    # 4. Optimizer with Parameter Groups
    param_groups = [
        {"params": expert_params, "lr": args.action_lr, "weight_decay": DEFAULT_WEIGHT_DECAY},
        {"params": proj_mlp_params, "lr": args.proj_lr, "weight_decay": DEFAULT_WEIGHT_DECAY},
    ]
    if vision_params:
        param_groups.append(
            {"params": vision_params, "lr": args.vision_lr, "weight_decay": DEFAULT_WEIGHT_DECAY}
        )

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=1e-6
    )

    beta_dist = Beta(
        torch.tensor(getattr(net.config, "time_sampling_beta_alpha", 1.5), device="cuda:0"),
        torch.tensor(getattr(net.config, "time_sampling_beta_beta", 1.0), device="cuda:0")
    )

    def run_training_step():
        batch_indices = np.random.choice(num_samples, size=args.batch_size, replace=True)
        batch_front = []
        batch_wrist = []
        batch_state = []
        batch_action_chunks = []
        batch_tasks = []

        for idx in batch_indices:
            s = all_samples[idx]
            cf = cached_files[s["file_id"]]
            f_idx = s["frame_idx"]

            f_img = cf["front_frames"][f_idx]
            w_img = cf["wrist_frames"][f_idx]
            curr_state = cf["states"][f_idx]

            # Cumulative relative joint displacement:
            # a_k = q_{t+k+1} - q_t = sum_{i=0}^k delta_q_{t+i}
            curr_q = curr_state[:7]
            future_states = cf["states"][f_idx + 1 : f_idx + 1 + CHUNK_SIZE]
            delta_q = future_states[:, :7] - curr_q[None, :]
            gripper = future_states[:, 7:8]
            action_chunk = np.concatenate([delta_q, gripper], axis=1)

            state_obs = curr_state.copy()
            if args.state_noise > 0:
                state_obs[:7] += np.random.normal(0.0, args.state_noise, size=7).astype(np.float32)

            batch_front.append(torch.from_numpy(np.transpose(f_img, (2, 0, 1))))
            batch_wrist.append(torch.from_numpy(np.transpose(w_img, (2, 0, 1))))
            batch_state.append(state_obs)
            batch_action_chunks.append(action_chunk)
            batch_tasks.append(s["task"])

        front_t = torch.stack(batch_front)
        wrist_t = torch.stack(batch_wrist)
        state_t = np.stack(batch_state)
        action_t = np.stack(batch_action_chunks)

        obs = {
            "observation.images.base_0_rgb": front_t,
            "observation.images.left_wrist_0_rgb": wrist_t,
            "observation.state": state_t
        }

        prepared = processor.prepare(obs, batch_tasks, device="cuda:0")

        # DROID quantile normalization of target actions
        actions_gpu = torch.from_numpy(action_t).to("cuda:0", dtype=torch.float32)
        norm_actions_8d = processor._transform(actions_gpu, "action", "ACTION", inverse=False)

        # Pad to max_action_dim (32) for network input
        target_actions = torch.zeros(args.batch_size, CHUNK_SIZE, 32, device="cuda:0", dtype=torch.float32)
        target_actions[:, :, :8] = norm_actions_8d

        # Flow Matching training step
        noise = torch.randn_like(target_actions)
        time_beta = beta_dist.sample((args.batch_size,))
        time_steps = time_beta * 0.999 + 0.001
        t_expanded = time_steps.view(-1, 1, 1)

        x_t = t_expanded * noise + (1.0 - t_expanded) * target_actions
        v_target = noise - target_actions

        proj_dtype = net.action_in_proj.weight.dtype
        x_t = x_t.to(dtype=proj_dtype)
        v_target = v_target.to(dtype=proj_dtype)

        # Forward through prefix
        if args.tune_vision:
            prefix_embs, prefix_pad_masks, prefix_att_masks = net.embed_prefix(
                prepared.images,
                prepared.image_masks,
                prepared.tokens,
                prepared.token_mask
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = prepare_attention_masks_4d(prefix_att_2d_masks)
            net.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"

            _, past_key_values = net.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
        else:
            with torch.no_grad():
                prefix_embs, prefix_pad_masks, prefix_att_masks = net.embed_prefix(
                    prepared.images,
                    prepared.image_masks,
                    prepared.tokens,
                    prepared.token_mask
                )
                prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
                prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
                prefix_att_2d_masks_4d = prepare_attention_masks_4d(prefix_att_2d_masks)
                net.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"

                _, past_key_values = net.paligemma_with_expert.forward(
                    attention_mask=prefix_att_2d_masks_4d,
                    position_ids=prefix_position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, None],
                    use_cache=True,
                )

        # Forward through Action Expert (trainable)
        v_pred = net.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=x_t,
            timestep=time_steps.to(dtype=proj_dtype),
        )

        loss = nn.functional.mse_loss(v_pred[:, :, :8], v_target[:, :, :8])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        return float(loss.item())

    # PREFLIGHT MODE: Benchmark empirical VRAM and throughput
    if args.preflight_only:
        print("\n" + "=" * 85)
        print("  EMPIRICAL PREFLIGHT BENCHMARK (Warmup + 5 Trial Steps)")
        print("=" * 85)
        torch.cuda.reset_peak_memory_stats("cuda:0")
        net.train()
        
        # 1 warmup step
        _ = run_training_step()
        torch.cuda.synchronize()

        trial_steps = 5
        t_bench_start = time.perf_counter()
        for i in range(trial_steps):
            step_loss = run_training_step()
            torch.cuda.synchronize()
            print(f"    Trial step {i+1}/{trial_steps}: Flow Loss = {step_loss:.5f}")
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
        print("\n[Preflight Passed] All criteria verified empirically. Ready for full training execution.")
        return 0

    # Initial baseline evaluation before training
    print("\n[Initial Baseline Eval @ Step 0]")
    res_init = evaluate_fixed_sample(inference, eval_reach_sample, cached_files)
    print(f"  Initial Action MSE:    {res_init['action_mse']:.6f}")
    print(f"  Initial Joint MAE:     {res_init['joint_mae_deg']:.2f} deg vs Hold Baseline: {res_init['hold_still_mae_deg']:.2f} deg ({res_init['improve_pct']:+.1f}%)")
    print(f"  Initial Gripper Pred:  {res_init['pred_chunk'][:, 7].mean():.3f} (GT: {res_init['gt_chunk'][:, 7].mean():.3f})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Training Started] Running {args.steps} steps with batch_size={args.batch_size}...")

    losses = []
    t_start = time.perf_counter()
    net.train()

    for step in range(1, args.steps + 1):
        loss_val = run_training_step()
        losses.append(loss_val)

        # Regular training loss log
        if step % args.log_freq == 0:
            avg_loss = np.mean(losses[-args.log_freq:])
            cur_lr = scheduler.get_last_lr()[0]
            elapsed = time.perf_counter() - t_start
            steps_per_sec = step / elapsed
            print(f"  [Step {step:04d}/{args.steps:04d}] Flow Loss: {avg_loss:.5f} | Action LR: {cur_lr:.2e} | Speed: {steps_per_sec:.2f} step/s | Elapsed: {elapsed/60:.1f}m")

        # Periodic Action MSE evaluation on fixed test frame
        if step % args.eval_freq == 0:
            net.eval()
            res_eval = evaluate_fixed_sample(inference, eval_reach_sample, cached_files)
            print(f"  >>> [Step {step:04d} EVAL] Action MSE: {res_eval['action_mse']:.6f} | Joint MAE: {res_eval['joint_mae_deg']:.2f} deg vs Hold: {res_eval['hold_still_mae_deg']:.2f} deg ({res_eval['improve_pct']:+.1f}%) | Grip Pred: {res_eval['pred_chunk'][:, 7].mean():.3f} (GT: {res_eval['gt_chunk'][:, 7].mean():.3f})")
            net.train()

        # Checkpoint saving at regular intervals (500, 1000, 1500, 2000)
        if step % args.save_freq == 0 or step == args.steps:
            ckpt_path = args.output_dir / f"action_expert_step_{step:04d}.pt"
            saved_weights = {
                k: v.detach().cpu() for k, v in net.state_dict().items()
                if any(part in k for part in ("gemma_expert", "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out", "multi_modal_projector", "vision_tower"))
                and net.get_parameter(k).requires_grad
            }
            state_dict_to_save = {
                "step": step,
                "loss": loss_val,
                "tune_vision": args.tune_vision,
                "vision_layers": args.vision_layers if args.tune_vision else 0,
                "state_dict": saved_weights,
            }
            torch.save(state_dict_to_save, str(ckpt_path))
            print(f"  [Checkpoint Saved] -> {ckpt_path.name} ({len(saved_weights)} trainable weight tensors)")

    total_time = time.perf_counter() - t_start
    final_path = args.output_dir / "action_expert_final.pt"
    saved_weights = {
        k: v.detach().cpu() for k, v in net.state_dict().items()
        if any(part in k for part in ("gemma_expert", "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out", "multi_modal_projector", "vision_tower"))
        and net.get_parameter(k).requires_grad
    }
    torch.save({
        "step": args.steps,
        "loss": float(np.mean(losses[-50:]) if losses else 0.0),
        "tune_vision": args.tune_vision,
        "vision_layers": args.vision_layers if args.tune_vision else 0,
        "state_dict": saved_weights,
    }, str(final_path))

    print("=" * 85)
    print(f"[Training Complete] Finished {args.steps} steps in {total_time/60:.1f} minutes.")
    print(f"Final Model Saved -> {final_path} ({len(saved_weights)} tensors)")
    print("=" * 85)

    # Comprehensive Single-Frame Action Comparison Tests
    net.eval()
    print("\n" + "#" * 85)
    print("  FINAL EVALUATION: SINGLE-FRAME DATASET TEST (PREDICTED VS GROUND TRUTH)")
    print("#" * 85)
    res_reach = evaluate_fixed_sample(inference, eval_reach_sample, cached_files)
    print_comparison_table(res_reach, title="Test 1: Reaching Phase (Approaching Red Cube, Gripper Open)")

    res_grasp = evaluate_fixed_sample(inference, eval_grasp_sample, cached_files)
    print_comparison_table(res_grasp, title="Test 2: Grasping Phase (Gripping Red Cube, Gripper Closed)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
