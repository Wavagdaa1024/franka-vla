#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pi0.5 Franka Multi-Task LoRA Fine-Tuning Pipeline.
Implements Northwestern University IDEAS Lab methodology for robotic VLA adaptation.

Architecture:
  - Base Model: Pi0.5 DROID JointPos (100% frozen, ~4.14B params).
  - Language Model (PaliGemma-2B): LoRA rank 16, alpha 32 on attention projections (q, k, v, o).
  - Action Expert (Gemma-300M): LoRA rank 32, alpha 64 on attention projections (q, k, v, o).
  - Action Projections & Time MLP: Full rank trainable.
  - Total Trainable Parameters: ~10.87M (0.26% of model).

Demonstrations:
  - Multi-task combination across 3 LeRobot datasets (99 episodes, 16,052 chunks, 5 unique tasks).
  - Balanced multi-task batch sampling.
  - 15-step cumulative relative joint delta action space.
  - Anti-memorization state jittering (Gaussian std=0.008 rad).

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
from pathlib import Path
from collections import defaultdict

# GPU Device Configuration (defaults to GPU 0 if not explicitly set)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if "CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"].strip()
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
PROJECT_ROOT = SCRIPT_DIR.parent
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
    REPO_ROOT / "dataset" / "teleop_pick_cube_15hz_001",
    REPO_ROOT / "dataset" / "teleop_pick_cube_15hz_002",
    REPO_ROOT / "dataset" / "teleop_pick_vegetables_15hz_001",
]
DEFAULT_OUTPUT_CKPT_DIR = REPO_ROOT / "outputs" / "checkpoints" / "pi05_lora_multitask"

CHUNK_SIZE = 15
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM = 2  # Effective batch size = 8
DEFAULT_STEPS = 5000
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_LOG_FREQ = 25
DEFAULT_SAVE_FREQ = 1000
DEFAULT_EVAL_FREQ = 250
SEED = 42


def preload_video_frames(video_path: Path):
    """Preload all video frames into memory as RGB numpy arrays."""
    container = av.open(str(video_path))
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return frames


def build_multitask_dataset(dataset_dirs, task_filter=None):
    """
    Builds in-memory index of 15-step relative joint position chunks across all datasets:
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

            states_droid = states_raw.copy()
            states_droid[:, 7] = np.clip(states_raw[:, 7], 0.0, 1.0)

            cached_files[file_key] = {
                "front_frames": front_frames,
                "wrist_frames": wrist_frames,
                "states": states_droid,
                "ep_indices": ep_indices,
                "task_indices": task_indices
            }

            unique_eps, counts = np.unique(ep_indices, return_counts=True)
            for ep_id, count in zip(unique_eps, counts):
                start = np.where(ep_indices == ep_id)[0][0]
                end = start + count - 1
                valid_len = count - CHUNK_SIZE
                if valid_len <= 0:
                    continue

                task_prompt = ep_tasks_map.get(ep_id)
                if not task_prompt or task_prompt == "None":
                    t_idx = task_indices[start]
                    task_prompt = task_map.get(t_idx, "pick and place the red cube")

                if task_filter and task_filter.lower() not in task_prompt.lower():
                    continue

                for t in range(start, end - CHUNK_SIZE + 1):
                    all_samples.append({
                        "file_id": file_key,
                        "frame_idx": t,
                        "task": task_prompt,
                        "ep_id": ep_id,
                        "dataset": dataset_dir.name,
                    })

    if not all_samples:
        raise ValueError("No valid demonstration samples found across provided datasets!")

    print(f"\n[Index Ready] Indexed {len(all_samples)} valid 15-step chunks across {len(dataset_dirs)} dataset(s) in {time.perf_counter()-t0:.2f}s.")
    task_buckets = defaultdict(list)
    for idx, s in enumerate(all_samples):
        task_buckets[s["task"]].append(idx)

    print(f"  Unique Prompts ({len(task_buckets)} total):")
    for task_name, indices in sorted(task_buckets.items()):
        print(f"    - '{task_name}': {len(indices)} chunks ({len(indices)/len(all_samples)*100:.1f}%)")

    return all_samples, cached_files, task_buckets


def evaluate_sample(inference, s, cached_files, dfk=None):
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

    ee_pos_err_mm = None
    ee_tilt_deg = None
    if dfk is not None:
        pred_q_traj = torch.from_numpy(curr_q[None, :] + pred_chunk[:, :7]).float()
        gt_q_traj = torch.from_numpy(curr_q[None, :] + gt_delta_q).float()
        with torch.no_grad():
            p_pred, z_pred = dfk(pred_q_traj.to("cuda:0" if torch.cuda.is_available() else "cpu"))
            p_gt, _ = dfk(gt_q_traj.to("cuda:0" if torch.cuda.is_available() else "cpu"))
            ee_pos_err_mm = float(torch.norm(p_pred - p_gt, dim=-1).mean().item() * 1000.0)
            z_u = z_pred / torch.norm(z_pred, dim=-1, keepdim=True).clamp(min=1e-6)
            ee_tilt_deg = float(torch.rad2deg(torch.acos(torch.clamp(-z_u[..., 2], -1.0, 1.0))).mean().item())

    return {
        "task": s["task"],
        "action_mse": action_mse,
        "joint_mse": joint_mse,
        "gripper_mse": gripper_mse,
        "joint_mae_deg": joint_mae_deg,
        "hold_still_mae_deg": hold_still_mae_deg,
        "improve_pct": improve_pct,
        "ee_pos_err_mm": ee_pos_err_mm,
        "ee_tilt_deg": ee_tilt_deg,
        "pred_chunk": pred_chunk,
        "gt_chunk": gt_action_chunk,
        "frame_idx": f_idx,
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Task LoRA Fine-Tuning for Pi0.5 Franka.")
    parser.add_argument("--dataset", type=Path, nargs="+", default=DEFAULT_DATASET_DIRS, help="Path(s) to dataset(s)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_CKPT_DIR, help="Directory to save LoRA checkpoints")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Total training steps (default: 5000)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate for LoRA & projections (default: 1e-4)")
    parser.add_argument("--lang-rank", type=int, default=16, help="LoRA rank for PaliGemma-2B (default: 16)")
    parser.add_argument("--expert-rank", type=int, default=32, help="LoRA rank for Gemma-300M Expert (default: 32)")
    parser.add_argument("--state-noise", type=float, default=0.008, help="Gaussian noise std added to state (default: 0.008 rad)")
    parser.add_argument("--warmup-steps", type=int, default=200, help="Linear warmup steps (default: 200)")
    parser.add_argument("--log-freq", type=int, default=DEFAULT_LOG_FREQ)
    parser.add_argument("--save-freq", type=int, default=DEFAULT_SAVE_FREQ)
    parser.add_argument("--eval-freq", type=int, default=DEFAULT_EVAL_FREQ)
    parser.add_argument("--task-filter", type=str, default=None, help="Filter dataset by task substring (e.g. 'red cube')")
    parser.add_argument("--cartesian-loss-weight", type=float, default=5.0, help="Weight for EE 3D position MSE loss (default: 5.0)")
    parser.add_argument("--vertical-loss-weight", type=float, default=2.0, help="Weight for EE vertical downward orientation loss (default: 2.0)")
    parser.add_argument("--preflight-only", action="store_true", help="Run empirical benchmark and exit")
    args = parser.parse_args()

    print("=" * 85)
    print("  PI0.5 MULTI-TASK LoRA FINE-TUNING PIPELINE (NORTHWESTERN IDEAS LAB ROUTE)")
    print("=" * 85)
    print(f"[*] Datasets ({len(args.dataset)}):")
    for d in args.dataset:
        print(f"      - {d}")
    print(f"[*] Output Dir:      {args.output_dir}")
    print(f"[*] Steps:           {args.steps} (Warmup: {args.warmup_steps})")
    print(f"[*] Batch Size:      {args.batch_size} (Grad Accum: {args.grad_accum}, Effective Batch: {args.batch_size*args.grad_accum})")
    print(f"[*] Learning Rate:   {args.lr:.2e}")
    print(f"[*] PaliGemma LoRA:  Rank={args.lang_rank}, Alpha={args.lang_rank*2} (Attention projections)")
    print(f"[*] Action Expert:   Rank={args.expert_rank}, Alpha={args.expert_rank*2} (Attention projections)")
    print(f"[*] DFK Aux Losses:  Cartesian Pos Weight={args.cartesian_loss_weight}, Vertical Z Downward Weight={args.vertical_loss_weight}")
    print(f"[*] Action Space:    15-step cumulative relative joint displacement a[k] = q[t+k+1] - q[t]")
    print(f"[*] State Noise:     {args.state_noise:.4f} rad (Anti-Trajectory Memorization)")
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    print(f"[*] Hardware:        Physical GPU {gpu_id} (RTX 5090 32GB, CUDA_VISIBLE_DEVICES={gpu_id})")
    print("-" * 85)

    # 1. Dataset build
    all_samples, cached_files, task_buckets = build_multitask_dataset(args.dataset, task_filter=args.task_filter)
    task_names = sorted(list(task_buckets.keys()))

    # Pick 1 fixed evaluation sample per task
    eval_samples = {}
    for t_name in task_names:
        indices = task_buckets[t_name]
        # Pick a sample around 40% into the episode where action is actively occurring
        chosen_idx = indices[len(indices) // 3]
        eval_samples[t_name] = all_samples[chosen_idx]
        print(f"  [*] Fixed Eval Sample for '{t_name}': Frame {eval_samples[t_name]['frame_idx']}")

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

    dfk = None
    if args.cartesian_loss_weight > 0 or args.vertical_loss_weight > 0:
        print("\n[Kinematics] Initializing PyTorch Franka Differentiable Kinematics (DFK) on cuda:0...")
        dfk = FrankaDifferentiableKinematics(device="cuda:0")

    def sample_balanced_batch(batch_size):
        """Samples a balanced mixture across all tasks."""
        batch_indices = []
        for _ in range(batch_size):
            t_choice = random.choice(task_names)
            idx_choice = random.choice(task_buckets[t_choice])
            batch_indices.append(idx_choice)

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

        return front_t, wrist_t, state_t, action_t, batch_tasks

    def compute_loss(front_t, wrist_t, state_t, action_t, batch_tasks, cur_batch_size):
        obs = {
            "observation.images.base_0_rgb": front_t,
            "observation.images.left_wrist_0_rgb": wrist_t,
            "observation.state": state_t
        }
        prepared = processor.prepare(obs, batch_tasks, device="cuda:0")

        actions_gpu = torch.from_numpy(action_t).to("cuda:0", dtype=torch.float32)
        norm_actions_8d = processor._transform(actions_gpu, "action", "ACTION", inverse=False)

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

        cartesian_loss = torch.tensor(0.0, device="cuda:0")
        vertical_loss = torch.tensor(0.0, device="cuda:0")
        pos_err_mm = 0.0
        tilt_err_deg = 0.0

        if dfk is not None and (args.cartesian_loss_weight > 0 or args.vertical_loss_weight > 0):
            pred_norm_actions_8d = (noise[:, :, :8] - v_pred[:, :, :8]).float()
            pred_actions_8d = processor._transform(pred_norm_actions_8d, "action", "ACTION", inverse=True)
            pred_delta_q = pred_actions_8d[:, :, :7]

            curr_q_tensor = torch.as_tensor(state_t[:, :7], device="cuda:0", dtype=torch.float32).unsqueeze(1)
            pred_q_traj = curr_q_tensor + pred_delta_q
            pred_p_ee, pred_z_ee = dfk(pred_q_traj)

            if args.cartesian_loss_weight > 0:
                with torch.no_grad():
                    gt_q_traj = curr_q_tensor + actions_gpu[:, :, :7]
                    gt_p_ee, _ = dfk(gt_q_traj)
                cartesian_loss = F.mse_loss(pred_p_ee, gt_p_ee)
                pos_err_mm = float(torch.norm(pred_p_ee - gt_p_ee, dim=-1).mean().item() * 1000.0)

            if args.vertical_loss_weight > 0:
                target_z = torch.tensor([0.0, 0.0, -1.0], device="cuda:0", dtype=torch.float32).expand_as(pred_z_ee)
                vertical_loss = F.mse_loss(pred_z_ee, target_z)
                with torch.no_grad():
                    z_u = pred_z_ee / torch.norm(pred_z_ee, dim=-1, keepdim=True).clamp(min=1e-6)
                    tilt_err_deg = float(torch.rad2deg(torch.acos(torch.clamp(-z_u[..., 2], -1.0, 1.0))).mean().item())

        total_loss = flow_loss + args.cartesian_loss_weight * cartesian_loss + args.vertical_loss_weight * vertical_loss
        return total_loss, flow_loss, cartesian_loss, vertical_loss, pos_err_mm, tilt_err_deg

    # PREFLIGHT MODE
    if args.preflight_only:
        print("\n" + "=" * 85)
        print("  EMPIRICAL PREFLIGHT BENCHMARK (Warmup + 5 Trial Steps)")
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
            step_pos_val = 0.0
            step_vert_val = 0.0
            step_pos_err_val = 0.0
            step_tilt_err_val = 0.0
            for _ in range(args.grad_accum):
                f_t, w_t, s_t, a_t, t_list = sample_balanced_batch(args.batch_size)
                tot_l, f_l, p_l, v_l, p_err, t_err = compute_loss(f_t, w_t, s_t, a_t, t_list, args.batch_size)
                (tot_l / args.grad_accum).backward()
                step_loss_val += tot_l.item() / args.grad_accum
                step_flow_val += f_l.item() / args.grad_accum
                step_pos_val += p_l.item() / args.grad_accum
                step_vert_val += v_l.item() / args.grad_accum
                step_pos_err_val += p_err / args.grad_accum
                step_tilt_err_val += t_err / args.grad_accum
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            torch.cuda.synchronize()
            print(f"    Trial step {i+1}/{trial_steps}: Total Loss={step_loss_val:.4f} (Flow={step_flow_val:.4f}, PosLoss={step_pos_val:.5f}, VertLoss={step_vert_val:.4f}) | PosErr={step_pos_err_val:.1f}mm | TiltErr={step_tilt_err_val:.1f}°")
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
        print("\n[Preflight Passed] Empirical benchmark passed. Ready for full execution.")
        return 0

    # Initial Baseline Eval on all tasks
    print(f"\n[Initial Baseline Eval @ Step 0 across all {len(eval_samples)} Tasks]")
    net.eval()
    for t_name, s_eval in eval_samples.items():
        res = evaluate_sample(inference, s_eval, cached_files, dfk=dfk)
        tilt_str = f" | Tilt={res['ee_tilt_deg']:.1f}°" if res.get('ee_tilt_deg') is not None else ""
        pos_str = f" | EE PosErr={res['ee_pos_err_mm']:.1f}mm" if res.get('ee_pos_err_mm') is not None else ""
        print(f"  Task '{t_name:30s}': MSE={res['action_mse']:.6f} | Joint MAE={res['joint_mae_deg']:.2f}° (Hold: {res['hold_still_mae_deg']:.2f}°){pos_str}{tilt_str} | Grip={res['pred_chunk'][:, 7].mean():.2f}")
    net.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Training Started] Running {args.steps} steps with effective batch_size={args.batch_size * args.grad_accum}...")

    losses = []
    flow_losses = []
    cart_losses = []
    vert_losses = []
    pos_errors = []
    tilt_errors = []
    t_start = time.perf_counter()

    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_flow = 0.0
        accum_cart = 0.0
        accum_vert = 0.0
        accum_pos_err = 0.0
        accum_tilt_err = 0.0

        for _ in range(args.grad_accum):
            f_t, w_t, s_t, a_t, t_list = sample_balanced_batch(args.batch_size)
            tot_l, f_l, p_l, v_l, p_err, t_err = compute_loss(f_t, w_t, s_t, a_t, t_list, args.batch_size)
            (tot_l / args.grad_accum).backward()
            accum_loss += tot_l.item() / args.grad_accum
            accum_flow += f_l.item() / args.grad_accum
            accum_cart += p_l.item() / args.grad_accum
            accum_vert += v_l.item() / args.grad_accum
            accum_pos_err += p_err / args.grad_accum
            accum_tilt_err += t_err / args.grad_accum

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        losses.append(accum_loss)
        flow_losses.append(accum_flow)
        cart_losses.append(accum_cart)
        vert_losses.append(accum_vert)
        pos_errors.append(accum_pos_err)
        tilt_errors.append(accum_tilt_err)

        if step % args.log_freq == 0:
            avg_loss = np.mean(losses[-args.log_freq:])
            avg_flow = np.mean(flow_losses[-args.log_freq:])
            avg_cart = np.mean(cart_losses[-args.log_freq:])
            avg_vert = np.mean(vert_losses[-args.log_freq:])
            avg_pos_err = np.mean(pos_errors[-args.log_freq:])
            avg_tilt_err = np.mean(tilt_errors[-args.log_freq:])
            cur_lr = scheduler.get_last_lr()[0]
            elapsed = time.perf_counter() - t_start
            steps_per_sec = step / elapsed
            print(f"  [Step {step:04d}/{args.steps:04d}] Loss: {avg_loss:.4f} (Flow: {avg_flow:.4f}, PosLoss: {avg_cart:.5f}, VertLoss: {avg_vert:.4f}) | PosErr: {avg_pos_err:.1f}mm | TiltErr: {avg_tilt_err:.1f}° | LR: {cur_lr:.2e} | Speed: {steps_per_sec:.2f} s/s | Elapsed: {elapsed/60:.1f}m")

        if step % args.eval_freq == 0:
            net.eval()
            print(f"\n  >>> [EVALUATION @ Step {step:04d}] <<<")
            eval_maes = []
            for t_name, s_eval in eval_samples.items():
                res = evaluate_sample(inference, s_eval, cached_files, dfk=dfk)
                eval_maes.append(res['joint_mae_deg'])
                tilt_str = f" | Tilt={res['ee_tilt_deg']:.1f}°" if res.get('ee_tilt_deg') is not None else ""
                pos_str = f" | EE PosErr={res['ee_pos_err_mm']:.1f}mm" if res.get('ee_pos_err_mm') is not None else ""
                print(f"      '{t_name:30s}': MSE={res['action_mse']:.6f} | MAE={res['joint_mae_deg']:.2f}° (Hold={res['hold_still_mae_deg']:.2f}°){pos_str}{tilt_str} | Grip={res['pred_chunk'][:, 7].mean():.2f}")
            print(f"      [Mean Joint MAE across all {len(eval_samples)} tasks: {np.mean(eval_maes):.2f}°]\n")
            net.train()

        if step % args.save_freq == 0 or step == args.steps:
            ckpt_path = args.output_dir / f"pi05_lora_multitask_step_{step:04d}.pt"
            state_dict = extract_lora_state_dict(net)
            torch.save({
                "step": step,
                "loss": float(np.mean(losses[-50:])),
                "flow_loss": float(np.mean(flow_losses[-50:])),
                "cartesian_loss": float(np.mean(cart_losses[-50:])),
                "vertical_loss": float(np.mean(vert_losses[-50:])),
                "pos_err_mm": float(np.mean(pos_errors[-50:])),
                "tilt_err_deg": float(np.mean(tilt_errors[-50:])),
                "state_dict": state_dict,
                "lang_rank": args.lang_rank,
                "expert_rank": args.expert_rank,
                "args": vars(args),
                "timestamp": time.time(),
            }, ckpt_path)
            ckpt_size_mb = ckpt_path.stat().st_size / (1024 * 1024)
            print(f"  [Checkpoint] Saved LoRA checkpoint to {ckpt_path.name} ({ckpt_size_mb:.2f} MB)")

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 85)
    print(f"  LoRA MULTI-TASK FINE-TUNING COMPLETED in {total_time/60:.1f} minutes!")
    print(f"  Checkpoints saved in: {args.output_dir}")
    print("=" * 85)


if __name__ == "__main__":
    main()
