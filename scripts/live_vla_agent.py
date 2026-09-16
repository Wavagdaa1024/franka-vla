#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live VLA Inference Agent for Closed-Loop Manipulation.
Runs on Windows Server (Strict GPU 1 Isolation, RTX 5090).
Streams dual RealSense cameras (30 fps RGB8 in dedicated threads),
connects outbound to Franka Controller (10.197.16.43:8765),
supports loading fine-tuned Action Expert checkpoints (--checkpoint),
and provides continuous 15-step native DROID joint velocity chunks.
"""

import os
import sys

# Enforce strict GPU 1 isolation
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import time
import json
import socket
import struct
import threading
from pathlib import Path

import torch
import numpy as np
import pyrealsense2 as rs

# Add mycode/src to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from franka_teleop.pi05_engine.runtime import PI05Inference
from franka_teleop.live_guards import validate_joint_velocity_chunk, validate_joint_position_chunk

FRONT_SERIAL = "254322072252"
WRIST_SERIAL = "348122070854"
DEFAULT_TARGET_HOST = "10.197.16.43"
DEFAULT_TARGET_PORT = 8765
DEFAULT_TASK = "pick and place the red cube"


class DualRealSenseStreamer:
    """Streams dual RealSense cameras in dedicated background threads."""
    def __init__(self, front_serial, wrist_serial, fps: int = 15):
        self.front_serial = front_serial
        self.wrist_serial = wrist_serial
        self.fps = fps
        self.frames = {"front": None, "wrist": None}
        self.timestamps = {"front": 0.0, "wrist": 0.0}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.threads = []

    def _worker(self, serial, role):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, self.fps)

        try:
            pipeline.start(config)
            for _ in range(20):
                frames = pipeline.wait_for_frames(3000)
            print(f"[Camera {role}] Online and streaming at {self.fps} fps (S/N: {serial}).")

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
            print(f"[Camera {role} Fatal] {e}")
        finally:
            try:
                pipeline.stop()
            except Exception:
                pass

    def start(self):
        self.stop_event.clear()
        t_front = threading.Thread(target=self._worker, args=(self.front_serial, "front"), daemon=True)
        t_wrist = threading.Thread(target=self._worker, args=(self.wrist_serial, "wrist"), daemon=True)
        self.threads = [t_front, t_wrist]
        t_front.start()
        t_wrist.start()

        print("[Cameras] Waiting for first frame from both cameras...")
        t0 = time.time()
        while time.time() - t0 < 10.0:
            with self.lock:
                if self.frames["front"] is not None and self.frames["wrist"] is not None:
                    print("[Cameras] Dual streams verified ready.")
                    return True
            time.sleep(0.1)
        print("[Cameras Warning] Warmup timeout, continuing with available frames.")
        return False

    def get_frames(self, max_age_s=0.2, max_skew_s=0.1):
        with self.lock:
            now = time.time()
            if (self.frames["front"] is None or (now - self.timestamps["front"]) > max_age_s or
                self.frames["wrist"] is None or (now - self.timestamps["wrist"]) > max_age_s or
                abs(self.timestamps["front"] - self.timestamps["wrist"]) > max_skew_s):
                return None, None
            f = self.frames["front"].copy()
            w = self.frames["wrist"].copy()
        return f, w

    def stop(self):
        self.stop_event.set()
        for t in self.threads:
            t.join(timeout=2.0)


def send_json(conn, obj):
    payload = json.dumps(obj).encode("utf-8")
    conn.sendall(struct.pack("!I", len(payload)) + payload)


def recv_json(conn):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_TARGET_HOST, help="Franka Linux host IP")
    parser.add_argument("--port", type=int, default=DEFAULT_TARGET_PORT, help="Franka Linux host port")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task instruction prompt")
    parser.add_argument("--profile", choices=["jointpos", "droid"], default="jointpos",
                        help="Action profile: jointpos (Joint Position, default) or droid (Joint Velocity)")
    parser.add_argument("--model-dir", default=None, help="Custom path to model weights checkpoint directory")
    parser.add_argument("--checkpoint", default=None, help="Path to fine-tuned action expert .pt checkpoint")
    parser.add_argument("--flip-lr", action="store_true", default=False,
                        help="Invert Joint 0 (base yaw) for frontal/opposite-facing camera perspective")
    parser.add_argument("--fps", type=int, default=15, help="Camera and control frequency in Hz (default: 15)")
    parser.add_argument("--live", action="store_true", help="Explicitly allow camera/network policy streaming")
    args = parser.parse_args()
    if not args.live:
        parser.error("BLOCKED: live VLA is disabled by default; pass --live only after controller safety approval")

    print("=" * 75)
    print(f"  LIVE VLA GPU INFERENCE AGENT ({args.profile.upper()} MODE @ {args.fps}Hz)")
    print("=" * 75)
    print(f"[*] Franka Host:       {args.host}:{args.port}")
    print(f"[*] Task Instruction:  '{args.task}'")
    print(f"[*] Action Mode:       {args.profile}")
    print(f"[*] Policy FPS:        {args.fps} Hz")
    if args.checkpoint:
        print(f"[*] Fine-Tuned Checkpoint: {args.checkpoint}")

    # 1. Start Dual Cameras
    cams = DualRealSenseStreamer(FRONT_SERIAL, WRIST_SERIAL, fps=args.fps)
    cams.start()

    # 2. Load Model onto GPU 1
    REPO_ROOT = PROJECT_ROOT
    if args.profile == "jointpos":
        CHECKPOINT_DIR = Path(args.model_dir) if args.model_dir else REPO_ROOT / "checkpoints" / "pi05_droid_jointpos"
        STATS_PATH = CHECKPOINT_DIR / "auxiliary" / "openpi_droid_jointpos_norm_stats.json"
        TOKENIZER_PATH = CHECKPOINT_DIR / "auxiliary" / "paligemma_tokenizer.model"
        profile_name = "droid_jointpos"
    else:
        CHECKPOINT_DIR = Path(args.model_dir) if args.model_dir else REPO_ROOT / "checkpoints" / "pi05_droid"
        STATS_PATH = CHECKPOINT_DIR / "auxiliary" / "openpi_droid_norm_stats.json"
        TOKENIZER_PATH = CHECKPOINT_DIR / "auxiliary" / "paligemma_tokenizer.model"
        profile_name = "droid"

    print(f"[Model] Loading Pi0.5 ({profile_name}) from {CHECKPOINT_DIR.name} on GPU 1 (RTX 5090)...")
    t0 = time.perf_counter()
    model = PI05Inference.from_checkpoint(
        CHECKPOINT_DIR,
        stats_path=STATS_PATH,
        tokenizer_path=TOKENIZER_PATH,
        device="cuda:0",
        profile=profile_name
    )
    print(f"[Model] Base model loaded in {time.perf_counter()-t0:.2f}s.")

    # Load fine-tuned weights if provided or default to latest audited checkpoint (for droid velocity mode)
    if args.checkpoint:
        ckpt_target = Path(args.checkpoint)
        if ckpt_target.exists():
            print(f"[Model] Loading weights from {ckpt_target.name}...")
            ckpt = torch.load(str(ckpt_target), map_location="cuda:0", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            is_lora = any("lora_" in k for k in state_dict.keys())
            if is_lora:
                from franka_teleop.pi05_engine.lora import inject_pi05_lora, load_lora_state_dict
                lang_rank = ckpt.get("lang_rank", 16)
                expert_rank = ckpt.get("expert_rank", 32)
                print(f"[Model] Injecting LoRA architecture (Lang r={lang_rank}, Expert r={expert_rank})...")
                inject_pi05_lora(model.network, lang_rank=lang_rank, expert_rank=expert_rank)
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
                print(f"[Model OK] Fine-tuned Action Expert{vision_tag} loaded! (File: {ckpt_target.name}, Step: {ckpt.get('step', '?')}, Loss: {ckpt.get('loss', 0.0):.4f})")
        else:
            raise FileNotFoundError(f"Fine-tuned checkpoint not found: {ckpt_target}")

    # 3. Reconnect Loop to Franka Controller
    while True:
        print(f"\n[Network] Connecting to Franka Controller at {args.host}:{args.port}...")
        sock = None
        while True:
            try:
                sock = socket.create_connection((args.host, args.port), timeout=3.0)
                sock.settimeout(None)
                torch.cuda.empty_cache()
                print(f"[Network] Connected successfully to Franka Controller! GPU cache cleared. Awaiting start...")
                break
            except Exception as e:
                print(f"  [Waiting for Franka] Please start 'closed_loop_franka_server.py' on Franka PC ({e})")
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
                raw_grip = float(msg.get("gripper", 0.0))  # DROID standard: 0.0 = OPEN, 1.0 = CLOSED
                robot_state = np.zeros(8, dtype=np.float32)
                robot_state[:7] = raw_q
                robot_state[7] = raw_grip

                img_f, img_w = cams.get_frames()
                if img_f is None or img_w is None:
                    print(f"  [Loop #{loop_cnt:03d} Warning] RealSense frames stale/missing! Sending hold command.")
                    resp = {
                        "type": "action_chunk",
                        "loop": loop_cnt,
                        "action_mode": "joint_position" if args.profile == "jointpos" else "joint_velocity",
                        "latency_ms": 0.0,
                        "joint_positions": [raw_q.tolist()] * 15,
                        "joint_velocities": [[0.0] * 7] * 15,
                        "gripper": [raw_grip] * 15
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

                # Pi0.5 inference: outputs (15, 8)
                action_chunk = model.predict_action_chunk(obs, args.task).numpy()[0]
                t_infer = (time.perf_counter() - t_start) * 1000.0

                action_chunk[:, 7] = np.clip(action_chunk[:, 7], 0.0, 1.0)

                if args.profile == "jointpos":
                    delta_q = action_chunk[:, :7].copy()
                    if args.flip_lr:
                        delta_q[:, 0] = -delta_q[:, 0]
                    q_targets = raw_q[None, :] + delta_q
                    q_targets_valid, gripper_cmds_valid = validate_joint_position_chunk(
                        q_targets, action_chunk[:, 7], current_q=raw_q
                    )
                    # Step-to-step velocity fallback
                    dt_control = 1.0 / float(args.fps)
                    dq_fallback = np.zeros_like(q_targets_valid)
                    dq_fallback[0] = (q_targets_valid[0] - raw_q) / dt_control
                    for s_i in range(1, len(q_targets_valid)):
                        dq_fallback[s_i] = (q_targets_valid[s_i] - q_targets_valid[s_i - 1]) / dt_control

                    target_delta = q_targets_valid[-1] - raw_q
                    resp = {
                        "type": "action_chunk",
                        "loop": loop_cnt,
                        "action_mode": "joint_position",
                        "latency_ms": round(t_infer, 1),
                        "joint_positions": q_targets_valid.tolist(),
                        "joint_velocities": dq_fallback.tolist(),
                        "gripper": gripper_cmds_valid.tolist()
                    }
                    send_json(sock, resp)
                    flip_tag = " [FLIP-LR]" if args.flip_lr else ""
                    print(f"  [Loop #{loop_cnt:03d}{flip_tag}] Infer: {t_infer:5.1f}ms | Target Δq: [{target_delta[0]:+.3f}, {target_delta[1]:+.3f}, {target_delta[2]:+.3f}, {target_delta[3]:+.3f}, {target_delta[4]:+.3f}, {target_delta[5]:+.3f}, {target_delta[6]:+.3f}] rad | Grip: {gripper_cmds_valid[0]:.2f}")
                else:
                    joint_vels_raw = action_chunk[:, :7].copy()
                    if args.flip_lr:
                        joint_vels_raw[:, 0] = -joint_vels_raw[:, 0]
                    joint_vel_array, gripper_array = validate_joint_velocity_chunk(joint_vels_raw, action_chunk[:, 7])
                    joint_vels = joint_vel_array.tolist()
                    gripper_cmds = gripper_array.tolist()
                    mean_dq = np.mean(joint_vel_array, axis=0)

                    resp = {
                        "type": "action_chunk",
                        "loop": loop_cnt,
                        "action_mode": "joint_velocity",
                        "latency_ms": round(t_infer, 1),
                        "joint_velocities": joint_vels,
                        "gripper": gripper_cmds
                    }
                    send_json(sock, resp)
                    print(f"  [Loop #{loop_cnt:03d}] Infer: {t_infer:5.1f}ms | Mean dq: [{mean_dq[0]:+.3f}, {mean_dq[1]:+.3f}, {mean_dq[2]:+.3f}, {mean_dq[3]:+.3f}, {mean_dq[4]:+.3f}, {mean_dq[5]:+.3f}, {mean_dq[6]:+.3f}] rad/s | Grip: {gripper_cmds[0]:.2f}")

        except (ConnectionError, OSError) as e:
            print(f"[Network Notice] Connection closed: {e}. Re-waiting for Franka...")
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

    cams.stop()


if __name__ == "__main__":
    main()
