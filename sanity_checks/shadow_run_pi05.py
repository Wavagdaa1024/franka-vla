#!/usr/bin/env python3
"""
M3 Step 2: Pi0.5 Shadow Run Benchmark on GPU 1.
Strict isolation: Physical GPU 1 only. Never touches GPU 0.
Supports both Joint Position (jointpos) and Joint Velocity (droid) action spaces.
Never commands physical robot motion (shadow evaluation only).
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

# Enforce strict GPU 1 isolation BEFORE importing torch
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import numpy as np
from PIL import Image

# Add src to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MYCODE_SRC = PROJECT_ROOT / "src"
if str(MYCODE_SRC) not in sys.path:
    sys.path.insert(0, str(MYCODE_SRC))

from midterm_robot.vla.pi05.runtime import PI05Inference
from midterm_robot.vla.pi05.config import PI05Config
from midterm_robot.vla.live_guards import validate_joint_velocity_chunk, validate_joint_position_chunk


def format_bytes(b: int) -> str:
    return f"{b / (1024**3):.2f} GB"


def main():
    parser = argparse.ArgumentParser(description="Pi0.5 Shadow Run Benchmark")
    parser.add_argument("--profile", choices=["jointpos", "droid"], default="jointpos",
                        help="Action profile: jointpos (Joint Position, default) or droid (Joint Velocity)")
    parser.add_argument("--checkpoint", default=None, help="Custom path to checkpoint directory")
    args = parser.parse_args()

    print("=" * 70)
    print(f"M3 Step 2: Pi0.5 Shadow Run ({args.profile.upper()} MODE on GPU 1)")
    print("=" * 70)

    # 1. Device and Safety Checks
    assert torch.cuda.is_available(), "CUDA is not available!"
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(f"[Device Check] CUDA_VISIBLE_DEVICES = {visible_devices}")
    assert visible_devices == "1", f"Expected CUDA_VISIBLE_DEVICES='1', got '{visible_devices}'"

    device_count = torch.cuda.device_count()
    print(f"[Device Check] PyTorch sees {device_count} GPU(s)")
    assert device_count == 1, f"Expected exactly 1 visible GPU, got {device_count}"

    device_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory
    print(f"[Device Check] Target Device: cuda:0 -> {device_name} ({format_bytes(total_mem)})")
    
    target_device = "cuda:0"
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    # 2. Asset Paths
    REPO_ROOT = PROJECT_ROOT
    if args.profile == "jointpos":
        CHECKPOINT_DIR = Path(args.checkpoint) if args.checkpoint else REPO_ROOT / "checkpoints" / "pi05_droid_jointpos"
        STATS_PATH = CHECKPOINT_DIR / "auxiliary" / "openpi_droid_jointpos_norm_stats.json"
        TOKENIZER_PATH = CHECKPOINT_DIR / "auxiliary" / "paligemma_tokenizer.model"
        profile_name = "droid_jointpos"
    else:
        CHECKPOINT_DIR = Path(args.checkpoint) if args.checkpoint else REPO_ROOT / "checkpoints" / "pi05_droid"
        STATS_PATH = CHECKPOINT_DIR / "auxiliary" / "openpi_droid_norm_stats.json"
        TOKENIZER_PATH = CHECKPOINT_DIR / "auxiliary" / "paligemma_tokenizer.model"
        profile_name = "droid"
    
    candidates = [
        PROJECT_ROOT / "outputs" / "camera_snapshots",
        PROJECT_ROOT / "target_mode" / "outputs" / "camera_snapshots",
    ]
    snap_dir = next((c for c in candidates if (c / "front_camera.jpg").is_file()), candidates[0])
    FRONT_CAM_PATH = snap_dir / "front_camera.jpg"
    WRIST_CAM_PATH = snap_dir / "wrist_camera.jpg"
    
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON = OUTPUT_DIR / f"m3_shadow_run_{args.profile}.json"

    print(f"[Assets] Checkpoint: {CHECKPOINT_DIR}")
    print(f"[Assets] Stats:      {STATS_PATH}")
    print(f"[Assets] Tokenizer:  {TOKENIZER_PATH}")
    print(f"[Assets] Front Cam:  {FRONT_CAM_PATH}")
    print(f"[Assets] Wrist Cam:  {WRIST_CAM_PATH}")

    assert CHECKPOINT_DIR.is_dir(), f"Checkpoint dir not found: {CHECKPOINT_DIR}"
    assert STATS_PATH.is_file(), f"Stats file not found: {STATS_PATH}"
    assert TOKENIZER_PATH.is_file(), f"Tokenizer file not found: {TOKENIZER_PATH}"
    assert FRONT_CAM_PATH.is_file(), f"Front camera image not found: {FRONT_CAM_PATH}"
    assert WRIST_CAM_PATH.is_file(), f"Wrist camera image not found: {WRIST_CAM_PATH}"

    # 3. Load Model and Measure Timings & Memory
    print("-" * 70)
    print(f"Loading Pi0.5 ({profile_name}) Model to GPU 1...")
    t0 = time.perf_counter()
    vram_before_load = torch.cuda.memory_allocated(0)

    model = PI05Inference.from_checkpoint(
        CHECKPOINT_DIR,
        stats_path=STATS_PATH,
        tokenizer_path=TOKENIZER_PATH,
        device=target_device,
        profile=profile_name
    )

    torch.cuda.synchronize()
    t_load = time.perf_counter() - t0
    vram_after_load = torch.cuda.memory_allocated(0)
    vram_reserved = torch.cuda.memory_reserved(0)

    print(f"[Load Complete] Load Time: {t_load:.2f} s")
    print(f"[VRAM Allocated]: {format_bytes(vram_after_load)} (Net added: {format_bytes(vram_after_load - vram_before_load)})")
    print(f"[VRAM Reserved]:  {format_bytes(vram_reserved)}")

    # 4. Prepare Observations
    print("-" * 70)
    print("Preparing Real Camera Observations & Franka Initial State...")

    front_img = Image.open(FRONT_CAM_PATH).convert("RGB")
    wrist_img = Image.open(WRIST_CAM_PATH).convert("RGB")
    print(f"[Input Cameras] Front: {front_img.size} ({front_img.mode}), Wrist: {wrist_img.size} ({wrist_img.mode})")

    front_chw = torch.from_numpy(np.transpose(np.array(front_img), (2, 0, 1)))  # uint8, (3, H, W)
    wrist_chw = torch.from_numpy(np.transpose(np.array(wrist_img), (2, 0, 1)))  # uint8, (3, H, W)

    # Franka standard ready joint configuration (radians) + gripper (0 = closed, 1 = open)
    franka_ready_state = np.array(
        [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398, 0.0],
        dtype=np.float32
    )
    print(f"[Input Robot State] Shape: {franka_ready_state.shape}, Values: {franka_ready_state.tolist()}")

    observations = {
        "observation.images.base_0_rgb": front_chw,
        "observation.images.left_wrist_0_rgb": wrist_chw,
        "observation.state": franka_ready_state,
    }

    # 5. Benchmark Tasks
    tasks_to_test = [
        ("Blue_Cube_Pick", "pick and place the blue cube"),
        ("Blue_Cube_SubSkill", "pick up the blue block"),
        ("Onion_Full", "pick up the purple onion and place it in the brown basket"),
        ("Pepper_Full", "pick up the green pepper and place it in the brown basket"),
    ]

    # Warmup Run
    print("-" * 70)
    print("Executing Warmup Inference on GPU 1...")
    torch.manual_seed(42)
    _ = model.predict_action_chunk(observations, "warmup pick")
    torch.cuda.synchronize()
    print("[Warmup Done] CUDA kernels initialized.")

    # Main Inference Runs
    print("-" * 70)
    print(f"Executing Shadow Run Benchmark Across Instructions ({args.profile.upper()})...")
    results = {
        "device": device_name,
        "gpu_id": 1,
        "profile": args.profile,
        "vram_allocated_model": format_bytes(vram_after_load),
        "vram_reserved_model": format_bytes(vram_reserved),
        "model_load_time_seconds": round(t_load, 2),
        "tasks": []
    }

    for task_tag, task_prompt in tasks_to_test:
        print(f"\n---> Testing Task: [{task_tag}] '{task_prompt}'")
        latencies = []
        chunks = []

        # Run 3 iterations per task to get stable latency metrics
        for it in range(3):
            torch.cuda.reset_peak_memory_stats(0)
            torch.cuda.synchronize()
            t_start = time.perf_counter()

            if it == 0:
                torch.manual_seed(42)
            action_chunk = model.predict_action_chunk(observations, task_prompt)
            torch.cuda.synchronize()
            t_dur = (time.perf_counter() - t_start) * 1000.0  # ms
            latencies.append(t_dur)
            chunks.append(action_chunk.numpy())

        # Analyze first run (deterministic)
        main_chunk = chunks[0]  # shape (1, 15, 8)
        actions_seq = main_chunk[0]  # (15, 8)

        is_finite = bool(np.isfinite(actions_seq).all())
        chunk_shape = list(main_chunk.shape)
        peak_vram = torch.cuda.max_memory_allocated(0)

        gripper_seq = actions_seq[:, 7].tolist()

        if args.profile == "jointpos":
            delta_q = actions_seq[:, :7]
            q_current = franka_ready_state[:7]
            q_targets = q_current[None, :] + delta_q
            q_targets_valid, gripper_valid = validate_joint_position_chunk(
                q_targets, actions_seq[:, 7], current_q=q_current
            )
            step_0_targets = q_targets_valid[0].tolist()
            step_14_targets = q_targets_valid[-1].tolist()
            total_delta = (q_targets_valid[-1] - q_current).tolist()

            print(f"     Latencies (3 runs): {[f'{x:.1f}ms' for x in latencies]}")
            print(f"     Avg Latency:        {np.mean(latencies):.1f} ms (Min: {np.min(latencies):.1f} ms)")
            print(f"     Peak VRAM:          {format_bytes(peak_vram)}")
            print(f"     Target Joints q0:   {[round(q, 3) for q in step_0_targets]}")
            print(f"     Target Joints q14:  {[round(q, 3) for q in step_14_targets]}")
            print(f"     Target Delta (14):  {[round(d, 3) for d in total_delta]} rad")
            print(f"     Gripper Trajectory: {[round(g, 3) for g in gripper_seq]}")

            task_record = {
                "task_tag": task_tag,
                "task_prompt": task_prompt,
                "chunk_shape": chunk_shape,
                "is_finite": is_finite,
                "latency_runs_ms": [round(x, 1) for x in latencies],
                "latency_mean_ms": round(float(np.mean(latencies)), 1),
                "peak_vram": format_bytes(peak_vram),
                "step_0_target_q": [round(x, 4) for x in step_0_targets],
                "step_14_target_q": [round(x, 4) for x in step_14_targets],
                "total_delta_q": [round(x, 4) for x in total_delta],
                "gripper_sequence": [round(x, 4) for x in gripper_seq],
            }
        else:
            joint_means = np.mean(actions_seq[:, :7], axis=0).tolist()
            total_q_delta = (actions_seq[-1, :7] - actions_seq[0, :7]).tolist()
            print(f"     Latencies (3 runs): {[f'{x:.1f}ms' for x in latencies]}")
            print(f"     Avg Latency:        {np.mean(latencies):.1f} ms")
            print(f"     Mean dq (vel):      {[round(x, 3) for x in joint_means]} rad/s")
            print(f"     Gripper Trajectory: {[round(g, 3) for g in gripper_seq]}")

            task_record = {
                "task_tag": task_tag,
                "task_prompt": task_prompt,
                "chunk_shape": chunk_shape,
                "is_finite": is_finite,
                "latency_mean_ms": round(float(np.mean(latencies)), 1),
                "peak_vram": format_bytes(peak_vram),
                "joint_means": [round(x, 4) for x in joint_means],
                "gripper_sequence": [round(x, 4) for x in gripper_seq],
            }

        results["tasks"].append(task_record)

    # 6. Save Report Artifacts
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 70)
    print(f"[SUCCESS] Shadow run results saved to {OUTPUT_JSON}")
    print("=" * 70)


if __name__ == "__main__":
    main()
