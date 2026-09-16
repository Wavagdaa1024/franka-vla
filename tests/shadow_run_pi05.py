#!/usr/bin/env python3
"""
VLA_franka: Pi0.5 DROID Standalone Shadow Run on GPU 1.
Strict isolation: Physical GPU 1 only. Never touches GPU 0.
Never commands physical robot motion (shadow evaluation only).
"""

import os
import sys
import time
import json
from pathlib import Path

# Enforce strict GPU 1 isolation BEFORE importing torch
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import numpy as np
from PIL import Image

# Import engine from local package
from franka_teleop.pi05_engine.runtime import PI05Inference
from franka_teleop.pi05_engine.config import PI05Config


def format_bytes(b: int) -> str:
    return f"{b / (1024**3):.2f} GB"


def main():
    print("=" * 70)
    print("VLA_franka: Pi0.5 DROID Standalone Shadow Run (GPU 1 Isolation)")
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
    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent
    CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "pi05_droid_jointpos"
    STATS_PATH = CHECKPOINT_DIR / "auxiliary" / "openpi_droid_jointpos_norm_stats.json"
    TOKENIZER_PATH = CHECKPOINT_DIR / "auxiliary" / "paligemma_tokenizer.model"
    
    FRONT_CAM_PATH = SCRIPT_DIR / "assets" / "front_camera.jpg"
    WRIST_CAM_PATH = SCRIPT_DIR / "assets" / "wrist_camera.jpg"
    
    OUTPUT_DIR = REPO_ROOT / "outputs"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON = OUTPUT_DIR / "m3_shadow_run.json"

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
    print("Loading Pi0.5 DROID Jointpos Model to GPU 1...")
    t0 = time.perf_counter()
    vram_before_load = torch.cuda.memory_allocated(0)

    model = PI05Inference.from_checkpoint(
        CHECKPOINT_DIR,
        stats_path=STATS_PATH,
        tokenizer_path=TOKENIZER_PATH,
        device=target_device,
        profile="droid_jointpos"
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
    print("Preparing Camera Observations & Franka Initial State...")

    front_img = Image.open(FRONT_CAM_PATH).convert("RGB")
    wrist_img = Image.open(WRIST_CAM_PATH).convert("RGB")
    print(f"[Input Cameras] Front: {front_img.size} ({front_img.mode}), Wrist: {wrist_img.size} ({wrist_img.mode})")

    front_chw = torch.from_numpy(np.transpose(np.array(front_img), (2, 0, 1)))
    wrist_chw = torch.from_numpy(np.transpose(np.array(wrist_img), (2, 0, 1)))

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
        ("M0_E1_Full", "pick up the purple onion and place it in the brown basket"),
        ("M0_SubSkill_Pick", "pick up the purple onion"),
        ("M0_SubSkill_Place", "place into the brown basket"),
        ("M0_E2_Full", "pick up the green pepper and place it in the brown basket"),
    ]

    print("-" * 70)
    print(f"Executing Shadow Runs on GPU 1 across {len(tasks_to_test)} instructions...")

    results = []
    
    # Warm-up run
    print("[Warm-up] Executing 1 warm-up inference...")
    _ = model.predict_action_chunk(observations, "warm up run")
    torch.cuda.synchronize()
    print("[Warm-up] Complete.")

    for task_id, task_text in tasks_to_test:
        print(f"\n---> Task [{task_id}]: \"{task_text}\"")
        torch.cuda.reset_peak_memory_stats(0)
        
        t_infer_start = time.perf_counter()
        action_chunk = model.predict_action_chunk(observations, task_text)
        torch.cuda.synchronize()
        infer_latency_ms = (time.perf_counter() - t_infer_start) * 1000.0

        vram_peak = torch.cuda.max_memory_allocated(0)
        
        # Verify shape and finite values
        actions_np = action_chunk.cpu().numpy()  # shape: (1, 15, 8)
        assert actions_np.shape == (1, 15, 8), f"Unexpected shape {actions_np.shape}"
        assert np.isfinite(actions_np).all(), "Output contains NaN or Inf!"
        
        chunk_0 = actions_np[0]  # (15, 8)
        first_step = chunk_0[0]
        last_step = chunk_0[-1]
        
        # Gripper values (index 7)
        g_first = float(first_step[7])
        g_last = float(last_step[7])
        
        print(f"     Latency:     {infer_latency_ms:.1f} ms  (Chunk size: 15 steps @ 30Hz ~ 500ms)")
        print(f"     Peak VRAM:   {format_bytes(vram_peak)}")
        print(f"     Step 0  dq:  [{', '.join(f'{v:+.3f}' for v in first_step[:7])}], grip: {g_first:.2f}")
        print(f"     Step 14 dq:  [{', '.join(f'{v:+.3f}' for v in last_step[:7])}], grip: {g_last:.2f}")
        
        results.append({
            "task_id": task_id,
            "instruction": task_text,
            "latency_ms": round(infer_latency_ms, 2),
            "peak_vram_gb": round(vram_peak / (1024**3), 2),
            "action_shape": list(actions_np.shape),
            "all_finite": bool(np.isfinite(actions_np).all()),
            "first_step_actions": [round(float(x), 4) for x in first_step],
            "last_step_actions": [round(float(x), 4) for x in last_step],
        })

    # 6. Overall Metrics
    latencies = [r["latency_ms"] for r in results]
    avg_latency = np.mean(latencies)
    rtf = avg_latency / 500.0

    print("\n" + "=" * 70)
    print("SHADOW RUN SUMMARY (PHYSICAL GPU 1)")
    print("=" * 70)
    print(f"[*] Total Test Cases:      {len(results)}/4")
    print(f"[*] Average Latency:       {avg_latency:.1f} ms")
    print(f"[*] Real-Time Factor (RTF):{rtf:.2f} (Target < 1.0; computation is {1.0/rtf:.1f}x faster than physical execution)")
    print(f"[*] Model Peak Memory:     {format_bytes(torch.cuda.max_memory_reserved(0))}")
    print(f"[*] Numerical Safety:      100% Finite (0 NaN / 0 Inf)")
    print("=" * 70)

    summary_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device_name,
        "visible_devices": visible_devices,
        "load_time_s": round(t_load, 2),
        "vram_load_gb": round(vram_after_load / (1024**3), 2),
        "vram_peak_gb": round(torch.cuda.max_memory_reserved(0) / (1024**3), 2),
        "avg_latency_ms": round(float(avg_latency), 2),
        "rtf": round(float(rtf), 3),
        "tests": results
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"[Report] Saved summary to {OUTPUT_JSON}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
