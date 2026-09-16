#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate all checkpoints (Base, Step 500, 1000, 1500, 2000) on an entire demonstration episode.
Computes Action Total MSE, Joint MSE, Gripper MSE, and Joint MAE (deg).
Runs on GPU 1.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import av
import pyarrow.parquet as pq
import torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

PROJECT_ROOT = Path(r"C:\Users\74727\Desktop\project\VLA_franka")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from franka_teleop.pi05_engine.runtime import PI05Inference

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "pi05_droid_jointpos"
STATS_PATH = CHECKPOINT_DIR / "auxiliary" / "openpi_droid_jointpos_norm_stats.json"
TOKENIZER_PATH = CHECKPOINT_DIR / "auxiliary" / "paligemma_tokenizer.model"
DATASET_DIR = PROJECT_ROOT / "dataset" / "teleop_pick_cube_15hz_001"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "checkpoints" / "pi05_jointpos_expert"

CHUNK_SIZE = 15


def preload_video_frames(video_path: Path):
    container = av.open(str(video_path))
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return frames


def main():
    target_ep = 0
    num_eval_frames = 15

    print("=" * 85)
    print(f"  CROSS-CHECKPOINT BENCHMARK: OFFLINE TRAJECTORY EVALUATION")
    print("=" * 85)
    print(f"[*] Target Episode:  Episode {target_ep} ({num_eval_frames} evaluation points)")
    print(f"[*] Hardware:        Physical GPU 1 (RTX 5090 32GB)")
    print("-" * 85)

    # 1. Load Data
    data_dir = DATASET_DIR / "data" / "chunk-000"
    video_front_dir = DATASET_DIR / "videos" / "observation.images.front" / "chunk-000"
    video_wrist_dir = DATASET_DIR / "videos" / "observation.images.wrist" / "chunk-000"

    target_pf = data_dir / "file-000.parquet"
    mp4_front = video_front_dir / "file-000.mp4"
    mp4_wrist = video_wrist_dir / "file-000.mp4"

    print("[1/3] Loading parquet and preloading video frames into memory...")
    table = pq.read_table(str(target_pf))
    front_frames = preload_video_frames(mp4_front)
    wrist_frames = preload_video_frames(mp4_wrist)

    states_raw = np.array([row.as_py() for row in table.column("observation.state")], dtype=np.float32)
    ep_indices = np.array(table.column("episode_index").to_pylist(), dtype=np.int32)
    task_indices = np.array(table.column("task_index").to_pylist(), dtype=np.int32)

    ep_mask = (ep_indices == target_ep)
    start_idx = np.where(ep_mask)[0][0]
    ep_len = int(np.sum(ep_mask))
    print(f"  Episode {target_ep} Length: {ep_len} frames ({ep_len/15.0:.1f}s)")

    eval_frame_indices = np.linspace(start_idx, start_idx + ep_len - CHUNK_SIZE - 1, num_eval_frames, dtype=int)

    # 2. Load Base Model
    print("\n[2/3] Loading Base Pi0.5 model on GPU 1...")
    t0 = time.perf_counter()
    inference = PI05Inference.from_checkpoint(
        CHECKPOINT_DIR,
        stats_path=STATS_PATH,
        tokenizer_path=TOKENIZER_PATH,
        device="cuda:0",
        profile="droid_jointpos"
    )
    print(f"  Base model loaded in {time.perf_counter()-t0:.2f}s.")
    net = inference.network
    net.eval()

    # Cache base action weights
    base_action_weights = {
        k: v.clone() for k, v in net.state_dict().items()
        if any(part in k for part in ("gemma_expert", "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"))
    }

    # 3. Checkpoints to Evaluate
    checkpoints = [
        ("Base (Step 0)", None),
        ("Step 500", OUTPUT_DIR / "action_expert_step_0500.pt"),
        ("Step 1000", OUTPUT_DIR / "action_expert_step_1000.pt"),
        ("Step 1500", OUTPUT_DIR / "action_expert_step_1500.pt"),
        ("Step 2000", OUTPUT_DIR / "action_expert_step_2000.pt"),
    ]

    results = []

    print("\n[3/3] Evaluating checkpoints sequentially...")
    task_prompt = "pick and place the red cube"

    for label, ckpt_path in checkpoints:
        if ckpt_path is None:
            net.load_state_dict(base_action_weights, strict=False)
        else:
            if not ckpt_path.exists():
                print(f"  [Skipping] {ckpt_path.name} not found.")
                continue
            ckpt = torch.load(str(ckpt_path), map_location="cuda:0", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            net.load_state_dict(state_dict, strict=False)

        action_mses = []
        joint_mses = []
        gripper_mses = []
        joint_maes_deg = []
        hold_maes_deg = []

        t_eval_start = time.perf_counter()
        for f_idx in eval_frame_indices:
            f_img = front_frames[f_idx]
            w_img = wrist_frames[f_idx]
            curr_state = states_raw[f_idx].copy()
            curr_q = curr_state[:7]

            future_states = states_raw[f_idx + 1 : f_idx + 1 + CHUNK_SIZE]
            gt_delta_q = future_states[:, :7] - curr_q[None, :]
            gt_gripper = np.clip(future_states[:, 7:8], 0.0, 1.0)
            gt_action_chunk = np.concatenate([gt_delta_q, gt_gripper], axis=1)

            front_t = torch.from_numpy(np.transpose(f_img, (2, 0, 1)))
            wrist_t = torch.from_numpy(np.transpose(w_img, (2, 0, 1)))
            obs = {
                "observation.images.base_0_rgb": front_t,
                "observation.images.left_wrist_0_rgb": wrist_t,
                "observation.state": curr_state
            }

            with torch.no_grad():
                pred_chunk = inference.predict_action_chunk(obs, task_prompt).cpu().numpy()[0]

            act_mse = float(np.mean((pred_chunk - gt_action_chunk) ** 2))
            j_mse = float(np.mean((pred_chunk[:, :7] - gt_delta_q) ** 2))
            g_mse = float(np.mean((pred_chunk[:, 7] - gt_gripper[:, 0]) ** 2))
            j_mae_deg = float(np.degrees(np.mean(np.abs(pred_chunk[:, :7] - gt_delta_q))))
            h_mae_deg = float(np.degrees(np.mean(np.abs(gt_delta_q))))

            action_mses.append(act_mse)
            joint_mses.append(j_mse)
            gripper_mses.append(g_mse)
            joint_maes_deg.append(j_mae_deg)
            hold_maes_deg.append(h_mae_deg)

        eval_time = time.perf_counter() - t_eval_start
        mean_act_mse = np.mean(action_mses)
        mean_j_mse = np.mean(joint_mses)
        mean_g_mse = np.mean(gripper_mses)
        mean_j_mae_deg = np.mean(joint_maes_deg)
        mean_h_mae_deg = np.mean(hold_maes_deg)
        improve_pct = (mean_h_mae_deg - mean_j_mae_deg) / max(mean_h_mae_deg, 1e-6) * 100.0

        results.append({
            "label": label,
            "action_mse": mean_act_mse,
            "joint_mse": mean_j_mse,
            "gripper_mse": mean_g_mse,
            "joint_mae_deg": mean_j_mae_deg,
            "hold_mae_deg": mean_h_mae_deg,
            "improve_pct": improve_pct,
            "eval_time": eval_time
        })
        print(f"  -> {label:<15} | Action MSE: {mean_act_mse:.6f} | Joint MAE: {mean_j_mae_deg:.2f}° (vs Hold {mean_h_mae_deg:.2f}°, {improve_pct:+.1f}%) | Grip MSE: {mean_g_mse:.5f}")

    print("\n" + "=" * 95)
    print(f"  CROSS-CHECKPOINT PERFORMANCE COMPARISON SUMMARY (EPISODE {target_ep})")
    print("=" * 95)
    print(f" {'Checkpoint':<16} | {'Action Total MSE':<16} | {'Joint MSE':<12} | {'Gripper MSE':<12} | {'Joint MAE':<10} | {'Improvement':<12}")
    print("-" * 95)
    for r in results:
        print(f" {r['label']:<16} | {r['action_mse']:<16.6f} | {r['joint_mse']:<12.6f} | {r['gripper_mse']:<12.6f} | {r['joint_mae_deg']:<8.2f}° | {r['improve_pct']:+9.1f}%")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()
