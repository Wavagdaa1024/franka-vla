#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Asynchronous RTC (Real-Time Chunking) VLA Inference Agent.
Runs on Windows Server (Strict GPU 1 Isolation: RTX 5090).

Key Capabilities:
  1. Asynchronous Prefetch Response: Responds in ~100-150ms to Franka prefetch requests,
     enabling Receding Horizon Blending (zero stop-and-go stutter).
  2. Task-Priority Nullspace Orientation Stabilization:
     Preserves exact 3D position trajectory while keeping gripper tool vector z_ee
     perpendicular to table [0, 0, -1], eliminating cumulative drift.
  3. Hard Table Floor Limit: Enforces Z >= +0.0070m (+7.0mm) collision protection.
  4. RealSense Dual RGB Streamer (Front S/N 254322072252, Wrist S/N 348122070854).
"""

import os
import sys

# Strict GPU 1 isolation
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import time
import json
import socket
import struct
import threading
from typing import Optional
from pathlib import Path

import torch
import numpy as np
import pyrealsense2 as rs

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from franka_teleop.pi05_engine.runtime import PI05Inference
from franka_teleop.live_guards import validate_joint_velocity_chunk, validate_joint_position_chunk
from franka_teleop.kinematics import (
    forward_kinematics,
    correct_chunk_nullspace,
    DEFAULT_Z_FLOOR
)

FRONT_SERIAL = "254322072252"
WRIST_SERIAL = "348122070854"
DEFAULT_TARGET_HOST = "10.197.16.43"
DEFAULT_TARGET_PORT = 8765
DEFAULT_TASK = "pick and place the red cube"
DEFAULT_LORA_CKPT = PROJECT_ROOT / "outputs" / "checkpoints" / "pi05_lora_multitask" / "pi05_lora_multitask_step_5000.pt"


class DualRealSenseStreamer:
    """Streams dual RealSense cameras in background threads with timestamp matching."""
    def __init__(self, front_serial: str, wrist_serial: str, fps: int = 15):
        self.front_serial = front_serial
        self.wrist_serial = wrist_serial
        self.fps = fps
        self.frames = {"front": None, "wrist": None}
        self.timestamps = {"front": 0.0, "wrist": 0.0}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.threads = []

    def _worker(self, serial: str, role: str):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, self.fps)

        try:
            pipeline.start(config)
            # Warm up auto-exposure
            for _ in range(15):
                pipeline.wait_for_frames(3000)
            print(f"[Camera {role.upper()}] Online and streaming at {self.fps} fps (S/N: {serial}).")

            while not self.stop_event.is_set():
                try:
                    frames = pipeline.wait_for_frames(1000)
                    cf = frames.get_color_frame()
                    if cf:
                        data = np.asanyarray(cf.get_data())
                        with self.lock:
                            self.frames[role] = data
                            self.timestamps[role] = time.time()
                except Exception:
                    pass
        except Exception as e:
            print(f"[Camera {role.upper()} Fatal] {e}")
        finally:
            try:
                pipeline.stop()
            except Exception:
                pass

    def start(self) -> bool:
        self.stop_event.clear()
        t_front = threading.Thread(target=self._worker, args=(self.front_serial, "front"), daemon=True)
        t_wrist = threading.Thread(target=self._worker, args=(self.wrist_serial, "wrist"), daemon=True)
        self.threads = [t_front, t_wrist]
        t_front.start()
        t_wrist.start()

        print("[Cameras] Waiting for initial frames...")
        t0 = time.time()
        while time.time() - t0 < 10.0:
            with self.lock:
                if self.frames["front"] is not None and self.frames["wrist"] is not None:
                    print("[Cameras] Dual streams online and verified.")
                    return True
            time.sleep(0.1)
        print("[Cameras Warning] Warmup timeout, continuing with available frames.")
        return False

    def get_frames(self, max_age_s: float = 0.25, max_skew_s: float = 0.1):
        with self.lock:
            now = time.time()
            if (self.frames["front"] is None or (now - self.timestamps["front"]) > max_age_s or
                self.frames["wrist"] is None or (now - self.timestamps["wrist"]) > max_age_s or
                abs(self.timestamps["front"] - self.timestamps["wrist"]) > max_skew_s):
                return None, None
            return self.frames["front"].copy(), self.frames["wrist"].copy()

    def stop(self):
        self.stop_event.set()
        for t in self.threads:
            t.join(timeout=2.0)


def send_json(conn: socket.socket, obj: dict):
    payload = json.dumps(obj).encode("utf-8")
    conn.sendall(struct.pack("!I", len(payload)) + payload)


def recv_json(conn: socket.socket) -> Optional[dict]:
    header = conn.recv(4)
    if not header or len(header) < 4:
        return None
    size = struct.unpack("!I", header)[0]
    data = bytearray()
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return json.loads(data.decode("utf-8"))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Async RTC VLA Inference Agent for Franka Panda")
    parser.add_argument("--host", default=DEFAULT_TARGET_HOST, help="Franka Linux host IP (default: 10.197.16.43)")
    parser.add_argument("--port", type=int, default=DEFAULT_TARGET_PORT, help="Franka Linux host port (default: 8765)")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task instruction prompt")
    parser.add_argument("--profile", choices=["jointpos", "droid"], default="jointpos",
                        help="Action profile: jointpos (Joint Position, default) or droid (Joint Velocity)")
    parser.add_argument("--model-dir", default=None, help="Custom path to base model weights checkpoint directory")
    parser.add_argument("--checkpoint", default=None, help="Path to fine-tuned or LoRA checkpoint .pt")
    parser.add_argument("--z-floor", type=float, default=DEFAULT_Z_FLOOR,
                        help=f"Minimum safe table Z height in meters (default: {DEFAULT_Z_FLOOR}m = +7.0mm)")
    parser.add_argument("--kp-rot", type=float, default=5.0,
                        help="Nullspace gripper orientation vertical alignment gain (default: 5.0)")
    parser.add_argument("--disable-nullspace", action="store_true", default=False,
                        help="Disable nullspace orientation stabilization (ablation only)")
    parser.add_argument("--flip-lr", action="store_true", default=False,
                        help="Invert Joint 0 (base yaw) for camera perspective matching")
    parser.add_argument("--fps", type=int, default=15, help="Control loop frequency in Hz (default: 15)")
    parser.add_argument("--live", action="store_true", help="Explicitly allow camera/network policy streaming")
    args = parser.parse_args()

    if not args.live:
        parser.error("BLOCKED: live VLA requires explicit --live flag after controller safety approval")

    print("=" * 80)
    print("  ASYNC RTC VLA INFERENCE AGENT (STRICT GPU 1 ISOLATION: RTX 5090)")
    print("=" * 80)
    print(f"[*] Franka Host:            {args.host}:{args.port}")
    print(f"[*] Task Instruction:       '{args.task}'")
    print(f"[*] Action Profile:         {args.profile}")
    print(f"[*] Table Floor Limit:      Z >= {args.z_floor * 1000.0:.1f} mm")
    print(f"[*] Nullspace Stabilizer:   {'DISABLED' if args.disable_nullspace else f'ACTIVE (kp_rot={args.kp_rot})'}")
    print(f"[*] Invert Joint 0:         {args.flip_lr}")

    # 1. Start RealSense Streams
    print("\n[1/3] Initializing Dual RealSense Streams...")
    cams = DualRealSenseStreamer(FRONT_SERIAL, WRIST_SERIAL, fps=args.fps)
    cams.start()

    # 2. Load Model onto GPU 1
    if args.profile == "jointpos":
        CHECKPOINT_DIR = Path(args.model_dir) if args.model_dir else PROJECT_ROOT / "checkpoints" / "pi05_droid_jointpos"
        STATS_PATH = CHECKPOINT_DIR / "auxiliary" / "openpi_droid_jointpos_norm_stats.json"
        TOKENIZER_PATH = CHECKPOINT_DIR / "auxiliary" / "paligemma_tokenizer.model"
        profile_name = "droid_jointpos"
    else:
        CHECKPOINT_DIR = Path(args.model_dir) if args.model_dir else PROJECT_ROOT / "checkpoints" / "pi05_droid"
        STATS_PATH = CHECKPOINT_DIR / "auxiliary" / "openpi_droid_norm_stats.json"
        TOKENIZER_PATH = CHECKPOINT_DIR / "auxiliary" / "paligemma_tokenizer.model"
        profile_name = "droid"

    print(f"\n[2/3] Loading Pi0.5 ({profile_name}) from {CHECKPOINT_DIR.name} on GPU 1 (RTX 5090)...")
    t0 = time.perf_counter()
    model = PI05Inference.from_checkpoint(
        CHECKPOINT_DIR,
        stats_path=STATS_PATH,
        tokenizer_path=TOKENIZER_PATH,
        device="cuda:0",
        profile=profile_name
    )
    print(f"[Model OK] Base model loaded in {time.perf_counter() - t0:.2f}s.")

    ckpt_path = Path(args.checkpoint) if args.checkpoint else (
        DEFAULT_LORA_CKPT if DEFAULT_LORA_CKPT.exists() else None
    )
    if ckpt_path and ckpt_path.exists():
        print(f"[Model] Loading weights from {ckpt_path.name}...")
        ckpt = torch.load(str(ckpt_path), map_location="cuda:0", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        is_lora = any("lora_" in k for k in state_dict.keys()) or "lora" in str(ckpt_path).lower()
        if is_lora:
            from franka_teleop.pi05_engine.lora import inject_pi05_lora, load_lora_state_dict
            lang_r = ckpt.get("lang_rank", 16)
            exp_r = ckpt.get("expert_rank", 32)
            print(f"[Model] Injecting LoRA architecture (Lang r={lang_r}, Expert r={exp_r})...")
            inject_pi05_lora(model.network, lang_rank=lang_r, expert_rank=exp_r)
            load_lora_state_dict(model.network, state_dict, strict=True)
            print(f"[Model OK] LoRA Multi-Task Adapters loaded! (Step: {ckpt.get('step', '?')}, Loss: {ckpt.get('loss', 0.0):.4f})")
        else:
            missing, unexpected = model.network.load_state_dict(state_dict, strict=False)
            trainable_names = {name for name, _ in model.network.named_parameters()
                               if any(part in name for part in ("gemma_expert", "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"))}
            missing_trainable = sorted(trainable_names.intersection(missing))
            allowed_prefixes = ("gemma_expert", "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out", "multi_modal_projector", "vision_tower")
            unexpected_critical = [k for k in unexpected if not any(p in k for p in allowed_prefixes)]
            if missing_trainable or unexpected_critical:
                raise RuntimeError(f"checkpoint mismatch: missing_trainable={missing_trainable}, unexpected={unexpected_critical}")
            has_vision = any("vision_tower" in k or "multi_modal_projector" in k for k in state_dict)
            vision_tag = " (+Vision Tuned)" if has_vision else ""
            print(f"[Model OK] Fine-tuned Action Expert{vision_tag} loaded! (File: {ckpt_path.name}, Step: {ckpt.get('step', '?')}, Loss: {ckpt.get('loss', 0.0):.4f})")
    else:
        print(f"[Model Warning] Checkpoint {ckpt_path} not found. Running base pre-trained weights.")

    # 3. Connection & Inference Loop to Franka Controller
    while True:
        print(f"\n[3/3] Connecting to Franka Controller at {args.host}:{args.port}...")
        sock = None
        while True:
            try:
                sock = socket.create_connection((args.host, args.port), timeout=3.0)
                sock.settimeout(None)
                torch.cuda.empty_cache()
                print(f"[Network] Connected to Franka Controller! Awaiting prefetch requests...")
                break
            except Exception as e:
                print(f"  [Waiting for Franka] Please ensure closed_loop_franka.py is running on {args.host} ({e})")
                time.sleep(2.0)

        loop_cnt = 0
        try:
            while True:
                msg = recv_json(sock)
                if msg is None:
                    print("[Network] Franka Controller disconnected. Reconnecting...")
                    break

                loop_cnt += 1
                t_start = time.perf_counter()

                raw_q = np.array(msg["q"][:7], dtype=np.float32)
                raw_grip = float(msg.get("gripper", 0.0))  # 0.0 = OPEN, 1.0 = CLOSED
                robot_state = np.zeros(8, dtype=np.float32)
                robot_state[:7] = raw_q
                robot_state[7] = raw_grip

                img_f, img_w = cams.get_frames()
                if img_f is None or img_w is None:
                    print(f"  [RTC #{loop_cnt:03d} Warning] RealSense frames stale/missing! Sending hold chunk.")
                    resp = {
                        "type": "action_chunk",
                        "loop": loop_cnt,
                        "action_mode": args.profile,
                        "latency_ms": 0.0,
                        "joint_positions": [raw_q.tolist()] * 15,
                        "joint_velocities": [[0.0] * 7] * 15,
                        "gripper": [raw_grip] * 15,
                        "max_tilt_deg": 0.0,
                        "z_clamped_count": 0
                    }
                    send_json(sock, resp)
                    continue

                front_chw = torch.from_numpy(np.transpose(img_f, (2, 0, 1)))
                wrist_chw = torch.from_numpy(np.transpose(img_w, (2, 0, 1)))

                obs = {
                    "observation.images.base_0_rgb": front_chw,
                    "observation.images.left_wrist_0_rgb": wrist_chw,
                    "observation.state": robot_state,
                }

                # 15-step action chunk inference
                action_chunk = model.predict_action_chunk(obs, args.task).numpy()[0]
                t_infer = (time.perf_counter() - t_start) * 1000.0

                action_chunk[:, 7] = np.clip(action_chunk[:, 7], 0.0, 1.0)

                if args.profile == "jointpos":
                    delta_q = action_chunk[:, :7].copy()
                    if args.flip_lr:
                        delta_q[:, 0] = -delta_q[:, 0]
                    q_raw_targets = raw_q[None, :] + delta_q

                    # Apply Nullspace Orientation Stabilization & Hard Floor Guard
                    if not args.disable_nullspace:
                        q_processed, max_tilt_deg, z_clamped = correct_chunk_nullspace(
                            current_q=raw_q,
                            chunk_q=q_raw_targets,
                            dt=1.0 / float(args.fps),
                            z_floor=args.z_floor,
                            kp_rot=args.kp_rot
                        )
                    else:
                        q_processed = q_raw_targets
                        max_tilt_deg = 0.0
                        z_clamped = 0

                    q_valid, g_valid = validate_joint_position_chunk(
                        q_processed, action_chunk[:, 7], current_q=raw_q, z_floor=args.z_floor
                    )

                    # Velocity fallback
                    dt_ctrl = 1.0 / float(args.fps)
                    dq_fallback = np.zeros_like(q_valid)
                    dq_fallback[0] = (q_valid[0] - raw_q) / dt_ctrl
                    for s in range(1, len(q_valid)):
                        dq_fallback[s] = (q_valid[s] - q_valid[s - 1]) / dt_ctrl

                    p_curr = forward_kinematics(raw_q)[:3, 3]
                    p_target = forward_kinematics(q_valid[-1])[:3, 3]
                    dp = p_target - p_curr

                    resp = {
                        "type": "action_chunk",
                        "loop": loop_cnt,
                        "action_mode": "joint_position",
                        "latency_ms": round(t_infer, 1),
                        "joint_positions": q_valid.tolist(),
                        "joint_velocities": dq_fallback.tolist(),
                        "gripper": g_valid.tolist(),
                        "max_tilt_deg": round(max_tilt_deg, 2),
                        "z_clamped_count": z_clamped
                    }
                    send_json(sock, resp)
                    clamp_str = f" | [FLOOR CLAMP x{z_clamped}]" if z_clamped > 0 else ""
                    print(f"  [RTC #{loop_cnt:03d}] Infer: {t_infer:5.1f}ms | Tilt Max: {max_tilt_deg:4.2f}° | Target Δp: [{dp[0]:+.3f}, {dp[1]:+.3f}, {dp[2]:+.3f}]m | Grip: {g_valid[0]:.2f}{clamp_str}")

                else:
                    # Velocity mode (droid)
                    joint_vels_raw = action_chunk[:, :7].copy()
                    if args.flip_lr:
                        joint_vels_raw[:, 0] = -joint_vels_raw[:, 0]
                    v_valid, g_valid = validate_joint_velocity_chunk(joint_vels_raw, action_chunk[:, 7])
                    resp = {
                        "type": "action_chunk",
                        "loop": loop_cnt,
                        "action_mode": "joint_velocity",
                        "latency_ms": round(t_infer, 1),
                        "joint_velocities": v_valid.tolist(),
                        "gripper": g_valid.tolist(),
                        "max_tilt_deg": 0.0,
                        "z_clamped_count": 0
                    }
                    send_json(sock, resp)
                    mean_v = np.mean(v_valid, axis=0)
                    print(f"  [RTC #{loop_cnt:03d}] Infer: {t_infer:5.1f}ms | Mean dq: [{mean_v[0]:+.3f}, {mean_v[1]:+.3f}, {mean_v[2]:+.3f}, {mean_v[3]:+.3f}, {mean_v[4]:+.3f}, {mean_v[5]:+.3f}, {mean_v[6]:+.3f}] | Grip: {g_valid[0]:.2f}")

        except (ConnectionError, OSError) as e:
            print(f"[Network] Franka disconnected: {e}. Re-waiting for connection...")
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

    cams.stop()


if __name__ == "__main__":
    main()
