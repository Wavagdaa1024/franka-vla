#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pi0.5 Joint Position Offline Trajectory Evaluation & Sanity Check Script.
Implements Sanity Check Step 3:
  - Directly compares predicted action chunks vs ground-truth recorded teleop chunks.
  - Prints side-by-side comparison of 7 joint deltas + 1 gripper value.
  - Analyzes numerical scale, sign orientation, MAE/RMSE errors, and gripper match.
Target: Physical GPU 1 (RTX 5090 32GB, CUDA_VISIBLE_DEVICES=1).
"""

import os
import sys
import argparse
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

# Add src to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MYCODE_SRC = PROJECT_ROOT / "src"
if str(MYCODE_SRC) not in sys.path:
    sys.path.insert(0, str(MYCODE_SRC))

from franka_teleop.pi05_engine.runtime import PI05Inference

REPO_ROOT = PROJECT_ROOT
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "pi05_droid_jointpos"
STATS_PATH = CHECKPOINT_DIR / "auxiliary" / "openpi_droid_jointpos_norm_stats.json"
TOKENIZER_PATH = CHECKPOINT_DIR / "auxiliary" / "paligemma_tokenizer.model"
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset" / "teleop_pick_cube_15hz_002"
DEFAULT_CKPT = REPO_ROOT / "outputs" / "checkpoints" / "pi05_lora_pure_flow_50k" / "latest.pt"

CKPT_ALIASES = {
    "pure_flow": REPO_ROOT / "outputs" / "checkpoints" / "pi05_lora_pure_flow_50k" / "latest.pt",
    "pure_flow_latest": REPO_ROOT / "outputs" / "checkpoints" / "pi05_lora_pure_flow_50k" / "latest.pt",
    "pure_flow_50k": REPO_ROOT / "outputs" / "checkpoints" / "pi05_lora_pure_flow_50k" / "latest.pt",
    "pure_flow_50000": REPO_ROOT / "outputs" / "checkpoints" / "pi05_lora_pure_flow_50k" / "step_50000.pt",
    "pure_flow_2500": REPO_ROOT / "outputs" / "checkpoints" / "pi05_lora_pure_flow_50k" / "step_02500.pt",
}

CHUNK_SIZE = 15


def preload_video_frames(video_path: Path):
    container = av.open(str(video_path))
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return frames


def evaluate_checkpoint(dataset_dir: Path, ckpt_path=None, test_episode=0, num_eval_frames=5, print_details=True):
    print("=" * 85)
    print("  PI0.5 DROID JOINTPOS OFFLINE TRAJECTORY EVALUATION (SANITY CHECK STEP 3)")
    print("=" * 85)
    print(f"[*] Dataset:       {dataset_dir}")
    print(f"[*] Checkpoint:    {ckpt_path if ckpt_path else 'Zero-Shot (Base Pretrained π0.5)'}")
    print(f"[*] Target Ep:     Episode {test_episode}")
    print(f"[*] Eval Samples:  {num_eval_frames} frames across episode")
    print("-" * 85)

    # 1. Load Model
    print("[Model] Loading Pi0.5 droid_jointpos on GPU 1...")
    t0 = time.perf_counter()
    inference = PI05Inference.from_checkpoint(
        CHECKPOINT_DIR,
        stats_path=STATS_PATH,
        tokenizer_path=TOKENIZER_PATH,
        device="cuda:0",
        profile="droid_jointpos"
    )
    print(f"[Model] Base model loaded in {time.perf_counter()-t0:.2f}s.")

    if ckpt_path is None:
        ckpt_path = DEFAULT_CKPT if DEFAULT_CKPT.exists() else None
    elif str(ckpt_path).strip().lower() in CKPT_ALIASES:
        ckpt_path = CKPT_ALIASES[str(ckpt_path).strip().lower()]
    else:
        p = Path(ckpt_path)
        if not p.exists() and (REPO_ROOT / "outputs" / "checkpoints" / ckpt_path).exists():
            ckpt_path = REPO_ROOT / "outputs" / "checkpoints" / ckpt_path
        else:
            ckpt_path = p

    if ckpt_path and Path(ckpt_path).exists():
        resolved_ckpt = Path(ckpt_path)
        print(f"[Model] Loading weights from {resolved_ckpt.name}...")
        ckpt = torch.load(str(resolved_ckpt), map_location="cuda:0", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        is_lora = any("lora_" in k for k in state_dict.keys()) or "lora" in str(resolved_ckpt).lower()
        if is_lora:
            from franka_teleop.pi05_engine.lora import inject_pi05_lora, load_lora_state_dict
            lang_r = ckpt.get("lang_rank", 16)
            exp_r = ckpt.get("expert_rank", 32)
            has_lora = hasattr(inference.network.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj, "lora_A")
            if not has_lora:
                print(f"[Model] Injecting LoRA architecture (Lang r={lang_r}, Expert r={exp_r})...")
                inject_pi05_lora(inference.network, lang_rank=lang_r, expert_rank=exp_r)
            load_lora_state_dict(inference.network, state_dict, strict=True)
            step_info = ckpt.get('step', '?')
            loss_info = ckpt.get('loss', 0.0)
            print(f"[Model OK] LoRA multi-task weights loaded! (Step: {step_info}, Loss: {loss_info})")
        else:
            missing, unexpected = inference.network.load_state_dict(state_dict, strict=False)
            trainable_names = {name for name, _ in inference.network.named_parameters()
                               if any(part in name for part in ("gemma_expert", "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"))}
            missing_trainable = sorted(trainable_names.intersection(missing))
            if missing_trainable or unexpected:
                print(f"[Warning] Mismatch: missing={len(missing_trainable)}, unexpected={len(unexpected)}")
            else:
                step_info = ckpt.get('step', '?')
                loss_info = ckpt.get('loss', 0.0)
                print(f"[Model OK] Fine-tuned weights loaded cleanly! (Step: {step_info}, Loss: {loss_info:.5f})")

    inference.network.eval()

    # 2. Load Episode & Video
    episodes_meta_dir = dataset_dir / "meta" / "episodes"
    ep_task = "pick and place the object"
    if episodes_meta_dir.is_dir():
        for ep_pq in sorted(episodes_meta_dir.rglob("*.parquet")):
            try:
                ep_rows = [row for row in pq.read_table(ep_pq).to_pylist()
                           if int(row["episode_index"]) == test_episode]
                if ep_rows:
                    task_val = ep_rows[0].get("tasks")
                    if isinstance(task_val, (list, np.ndarray)) and len(task_val) > 0:
                        ep_task = str(task_val[0])
                    elif isinstance(task_val, str):
                        ep_task = task_val
                    break
            except Exception:
                pass

    data_dir = dataset_dir / "data" / "chunk-000"
    video_front_dir = dataset_dir / "videos" / "observation.images.front" / "chunk-000"
    video_wrist_dir = dataset_dir / "videos" / "observation.images.wrist" / "chunk-000"

    # Find the parquet file containing target episode
    target_pf = None
    target_table = None
    for pf in sorted(data_dir.glob("file-*.parquet")):
        try:
            tbl = pq.read_table(pf)
            ep_indices = tbl.column("episode_index").to_pylist()
            if test_episode in ep_indices:
                target_pf = pf
                target_table = tbl
                break
        except Exception:
            continue

    if target_pf is None:
        raise ValueError(f"Episode {test_episode} not found in {dataset_dir}!")

    fid = int(target_pf.stem.split("-")[1])
    mp4_front = video_front_dir / f"file-{fid:03d}.mp4"
    mp4_wrist = video_wrist_dir / f"file-{fid:03d}.mp4"
    print(f"\n[Data] Target Episode {test_episode} found in file-{fid:03d}.parquet")
    print(f"       Task Prompt: '{ep_task}'")
    print(f"       Preloading video streams...")
    front_frames = preload_video_frames(mp4_front)
    wrist_frames = preload_video_frames(mp4_wrist)

    states_raw = np.array([row.as_py() for row in target_table.column("observation.state")], dtype=np.float32)
    ep_indices = np.array(target_table.column("episode_index").to_pylist(), dtype=np.int32)
    
    # Locate frames for test_episode
    ep_frame_mask = (ep_indices == test_episode)
    ep_start = np.where(ep_frame_mask)[0][0]
    ep_len = np.sum(ep_frame_mask)
    print(f"       Episode Length: {ep_len} frames ({ep_len/15.0:.2f}s)")

    eval_indices = np.linspace(ep_start, ep_start + ep_len - CHUNK_SIZE - 1, num_eval_frames, dtype=int)
    all_maes = []
    all_rmses = []
    all_grip_acc = []

    print("\n" + "=" * 85)
    print("  GROUND TRUTH vs PREDICTED ACTION ALIGNMENT INSPECTION")
    print("=" * 85)

    for step_i, f_idx in enumerate(eval_indices):
        rel_idx = f_idx - ep_start
        progress = rel_idx / ep_len * 100.0

        f_img = front_frames[f_idx]
        w_img = wrist_frames[f_idx]
        curr_state = states_raw[f_idx]

        # Ground Truth Chunk
        curr_q = curr_state[:7]
        gt_future = states_raw[f_idx + 1 : f_idx + 1 + CHUNK_SIZE]
        gt_delta_q = gt_future[:, :7] - curr_q[None, :]
        # Existing parquet files use 0=CLOSED, larger=OPEN; compare in
        # pi05/DROID semantics (0=OPEN, 1=CLOSED).
        gt_gripper = np.clip(gt_future[:, 7], 0.0, 1.0)

        # Model Inference
        obs = {
            "observation.images.base_0_rgb": torch.from_numpy(np.transpose(f_img, (2, 0, 1))).unsqueeze(0),
            "observation.images.left_wrist_0_rgb": torch.from_numpy(np.transpose(w_img, (2, 0, 1))).unsqueeze(0),
            "observation.state": curr_state[None, :]
        }

        with torch.no_grad():
            pred_action_chunk = inference.predict_action_chunk(obs, ep_task).numpy()

        pred_chunk = pred_action_chunk[0] if pred_action_chunk.ndim == 3 else pred_action_chunk
        pred_delta_q = pred_chunk[:, :7]
        pred_gripper = pred_chunk[:, 7]

        # Error metrics
        joint_mae = np.mean(np.abs(pred_delta_q - gt_delta_q))
        joint_rmse = np.sqrt(np.mean((pred_delta_q - gt_delta_q) ** 2))
        grip_acc = np.mean((pred_gripper > 0.5) == (gt_gripper > 0.5)) * 100.0

        all_maes.append(joint_mae)
        all_rmses.append(joint_rmse)
        all_grip_acc.append(grip_acc)

        print(f"\n--- [Eval Sample #{step_i+1}/{num_eval_frames}] Frame {rel_idx:03d}/{ep_len} ({progress:4.1f}% Progress) ---")
        print(f"  Summary: Joint MAE = {joint_mae:.4f} rad ({np.degrees(joint_mae):.2f}°), RMSE = {joint_rmse:.4f} rad, Gripper Match = {grip_acc:.1f}%")

        if print_details:
            print("  Detailed Step-by-Step Action Horizon (Steps 1..15):")
            print("  Horizon | GT Delta q (rad) [J1..J7] & Grip | PRED Delta q (rad) [J1..J7] & Grip | Abs Err")
            print("  " + "-" * 80)
            for k in range(min(5, CHUNK_SIZE)):  # Print first 5 steps of horizon for clarity
                gt_str = " ".join(f"{x:+0.3f}" for x in gt_delta_q[k]) + f" | {gt_gripper[k]:0.2f}"
                pr_str = " ".join(f"{x:+0.3f}" for x in pred_delta_q[k]) + f" | {pred_gripper[k]:0.2f}"
                err_q = np.abs(pred_delta_q[k] - gt_delta_q[k])
                err_str = f"{np.mean(err_q):.4f}"
                print(f"   t+{k+1:02d}   | {gt_str} | {pr_str} | {err_str}")
            if CHUNK_SIZE > 5:
                k = CHUNK_SIZE - 1
                gt_str = " ".join(f"{x:+0.3f}" for x in gt_delta_q[k]) + f" | {gt_gripper[k]:0.2f}"
                pr_str = " ".join(f"{x:+0.3f}" for x in pred_delta_q[k]) + f" | {pred_gripper[k]:0.2f}"
                err_q = np.abs(pred_delta_q[k] - gt_delta_q[k])
                err_str = f"{np.mean(err_q):.4f}"
                print(f"   ...    | ...                                       | ...                                       | ...")
                print(f"   t+{k+1:02d}   | {gt_str} | {pr_str} | {err_str}")

        # Sanity check diagnostics
        scale_ratio = np.mean(np.abs(pred_delta_q)) / (np.mean(np.abs(gt_delta_q)) + 1e-6)
        if scale_ratio < 0.1:
            print("  [WARNING] Prediction scale is severely collapsed (< 0.1x GT) -> Check normalization/denorm!")
        elif scale_ratio > 10.0:
            print("  [WARNING] Prediction scale is severely inflated (> 10x GT) -> Check units/scaling!")
        else:
            print(f"  [CHECK PASSED] Action magnitude scale is healthy (Ratio: {scale_ratio:.2f}x GT).")

    mean_mae = np.mean(all_maes)
    mean_rmse = np.mean(all_rmses)
    mean_grip = np.mean(all_grip_acc)

    print("\n" + "=" * 85)
    print("  OVERALL EVALUATION SUMMARY")
    print("=" * 85)
    print(f"[*] Mean Joint MAE:      {mean_mae:.4f} rad ({np.degrees(mean_mae):.2f}°)")
    print(f"[*] Mean Joint RMSE:     {mean_rmse:.4f} rad ({np.degrees(mean_rmse):.2f}°)")
    print(f"[*] Mean Gripper Match:  {mean_grip:.1f}%")
    print("=" * 85)
    return {
        "mean_mae": mean_mae,
        "mean_rmse": mean_rmse,
        "mean_grip": mean_grip
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Pi0.5 jointpos checkpoint against ground truth actions.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR, help="Path to dataset directory")
    parser.add_argument("--checkpoint", default="pure_flow", help="Path to checkpoint (.pt) or alias (pure_flow, pure_flow_50k, etc.)")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to evaluate (default: 0)")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of frames across episode to evaluate")
    args = parser.parse_args()

    evaluate_checkpoint(
        dataset_dir=args.dataset,
        ckpt_path=args.checkpoint,
        test_episode=args.episode,
        num_eval_frames=args.num_samples,
        print_details=True
    )


if __name__ == "__main__":
    main()
