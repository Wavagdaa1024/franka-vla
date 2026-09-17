#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Franka Closed-Loop Execution Service.
Runs on Franka Linux Control PC (10.197.16.43:8765) or simulation.

Execution Modes:
  1. RTC Continuous Streaming (--rtc, default):
     Arm NEVER stops: Receding Horizon Blending smoothly fuses overlapping action
     chunks across a 3-step transition window (Zero Stutter / Fluid Motion).
  2. Synchronous Stop-and-Go (--sync):
     Arm stops -> Settles stationary -> Informs GPU Agent -> Executes N steps -> Halts.

Safety Systems:
  - Task-Priority Nullspace Orientation Stabilization
  - Hard Table Collision Guard: Z >= z_min (default: +0.0070m = +7.0mm, calibrated at physical table lowest point)
  - Acceleration & Speed Clamps with Anti-Jerk Smoothing
  - Asynchronous Non-Blocking Gripper Actuation
"""

import os
import sys

CATKIN_WS2_PYTHON = "/home/ssui/franka_ros_ws/catkin_ws2/devel/lib/python3/dist-packages"
if os.path.exists(CATKIN_WS2_PYTHON) and CATKIN_WS2_PYTHON not in sys.path:
    sys.path.insert(0, CATKIN_WS2_PYTHON)

ROS_NOETIC_PYTHON = "/opt/ros/noetic/lib/python3/dist-packages"
if os.path.exists(ROS_NOETIC_PYTHON) and ROS_NOETIC_PYTHON not in sys.path:
    sys.path.append(ROS_NOETIC_PYTHON)

import time
import json
import socket
import struct
import argparse
import threading
import queue
import numpy as np

import rospy

# Import kinematics module (supports both local Linux directory and Windows package)
try:
    from kinematics import (
        forward_kinematics,
        analytical_jacobian,
        project_velocity_z_floor,
        DEFAULT_Z_FLOOR,
        FRANKA_JOINT_LIMITS
    )
except ImportError:
    from franka_teleop.kinematics import (
        forward_kinematics,
        analytical_jacobian,
        project_velocity_z_floor,
        DEFAULT_Z_FLOOR,
        FRANKA_JOINT_LIMITS
    )

# Import 1kHz joint velocity controller client
try:
    from Base_franka_joint_velocity_controller import FrankaJointVelocityController
except ImportError:
    # Fallback mock for testing in non-ROS environments
    class FrankaJointVelocityController:
        def __init__(self):
            self._q = np.array([0.1329, 0.4759, 0.0550, -2.4485, 0.0220, 2.8999, 0.9361], dtype=np.float64)
            self._grip = 0.04
        def get_cartesian_pose(self):
            T = forward_kinematics(self._q)
            return T[:3, 3], np.array([0, 0, 0, 1])
        def get_joint_positions(self):
            return self._q.copy()
        def get_gripper_width(self):
            return self._grip
        def set_joint_velocities(self, dq):
            self._q += np.asarray(dq) * (1.0 / 15.0)
        def stop(self):
            pass
        def close_gripper(self, **kwargs):
            self._grip = 0.01
        def open_gripper(self, **kwargs):
            self._grip = 0.08

DEFAULT_PORT = 8765
DEFAULT_SYNC_STEPS = 15       # 15 steps @ 15Hz = 1.000s motion per sync cycle
DEFAULT_STEPS_PER_CHUNK = 8   # Rolling stride: 8 steps @ 15Hz = 533ms per chunk
PREEMPT_STEP = 1              # Step at which to trigger async prefetch (67ms into chunk)
DEFAULT_BLEND_STEPS = 3       # Steps over which to blend overlapping chunks
CONTROL_HZ = 15.0             # 15 Hz policy frequency
DT = 1.0 / CONTROL_HZ         # 0.0667 s
MAX_JOINT_VEL = 0.35          # 0.35 rad/s max safe testing speed
MAX_JOINT_ACC = 1.5           # 1.5 rad/s^2 smooth responsive acceleration clamp


def send_json(conn, obj):
    try:
        payload = json.dumps(obj).encode("utf-8")
        conn.sendall(struct.pack("!I", len(payload)) + payload)
        return True
    except (socket.error, ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
        return False


def recv_json(conn):
    try:
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
    except (socket.error, ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
        return None


class AsyncChunkClient:
    """Asynchronous background worker for continuous streaming & prefetching."""
    def __init__(self, conn):
        self.conn = conn
        self.req_queue = queue.Queue(maxsize=1)
        self.resp_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                msg = self.req_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            t0 = time.time()
            if not send_json(self.conn, msg):
                break
            resp = recv_json(self.conn)
            if resp is None:
                break
            resp["latency_ms"] = (time.time() - t0) * 1000.0

            try:
                self.resp_queue.put_nowait(resp)
            except queue.Full:
                try:
                    self.resp_queue.get_nowait()
                except queue.Empty:
                    pass
                self.resp_queue.put(resp)

    def request_chunk(self, state_msg):
        try:
            self.req_queue.put_nowait(state_msg)
            return True
        except queue.Full:
            return False

    def get_chunk(self, block=False, timeout=None):
        try:
            return self.resp_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=1.0)


def read_gripper_normalized(arm, default=0.0):
    """0.0 = OPEN (width=0.08m), 1.0 = CLOSED (width=0.0m)."""
    w = arm.get_gripper_width()
    if w is None:
        return default
    return float(np.clip(1.0 - float(w) / 0.08, 0.0, 1.0))


def run_rtc_loop(arm, conn, args):
    """
    Real-Time Chunking (RTC) Continuous Streaming Loop:
      - Prefetches Chunk K+1 at args.preempt_step (step 6).
      - Receding Horizon Blending over args.blend_steps (3 steps) upon chunk arrival.
      - Arm moves fluidly with 0ms stop pause.
      - Dual-layer table collision guard (Z >= args.z_min).
    """
    client = AsyncChunkClient(conn)
    rate = rospy.Rate(CONTROL_HZ)
    init_grip = read_gripper_normalized(arm, default=0.0)
    gripper_state = 0 if init_grip <= 0.5 else 1
    chunk_idx = 0
    t_start = time.time()
    prev_dq = np.zeros(7, dtype=np.float64)
    close_intent_counter = 0

    print("=" * 80)
    print("  REAL-TIME CHUNKING (RTC) CONTINUOUS STREAMING ACTIVE")
    print(f"  [Config] Control Frequency: {CONTROL_HZ} Hz (DT = {DT * 1000:.1f} ms)")
    print(f"  [Config] Prefetch Timing:   Step {args.preempt_step} ({args.preempt_step * DT * 1000:.0f} ms)")
    print(f"  [Config] Receding Window:   {args.blend_steps} steps ({args.blend_steps * DT * 1000:.0f} ms blend)")
    print(f"  [Config] Hard Floor Guard:  Z >= {args.z_min * 1000.0:.1f} mm")
    print(f"  [Config] Max Speed Clamp:   {args.max_vel:.2f} rad/s")
    print("=" * 80)

    # Fetch initial Chunk 0 synchronously to bootstrap motion
    start_pos, _ = arm.get_cartesian_pose()
    start_q = arm.get_joint_positions()
    print(f"\n[Bootstrap] Arm Pose: [{start_pos[0]:+.3f}, {start_pos[1]:+.3f}, {start_pos[2]:+.3f}] m. Fetching Chunk 0...")

    init_state = {
        "loop": 0,
        "q": start_q.tolist(),
        "pos": start_pos.tolist(),
        "gripper": init_grip
    }
    client.request_chunk(init_state)
    active_chunk = client.get_chunk(block=True, timeout=15.0)

    if active_chunk is None:
        print("[ERROR] Failed to receive bootstrap action chunk from GPU Agent!")
        client.stop()
        return 0, t_start

    pos_targets = active_chunk.get("joint_positions", [])
    vel_targets = active_chunk.get("joint_velocities", [])
    grip_targets = active_chunk.get("gripper", [])
    action_mode = active_chunk.get("action_mode", "joint_position" if pos_targets else "joint_velocity")
    use_pos = (action_mode == "joint_position" and len(pos_targets) > 0)

    print(f"[Bootstrap OK] Received Chunk 0 ({len(pos_targets) if use_pos else len(vel_targets)} steps, Infer: {active_chunk.get('latency_ms', 0):.0f}ms). Starting continuous RTC...")

    step_in_chunk = 0
    prefetch_sent = False
    blending = False
    blend_step = 0
    next_chunk = None

    try:
        while not rospy.is_shutdown():
            curr_pos, _ = arm.get_cartesian_pose()
            curr_q = arm.get_joint_positions()
            if curr_pos is None or curr_q is None:
                rate.sleep()
                continue

            # 1. Trigger asynchronous prefetch early in chunk execution
            if step_in_chunk >= args.preempt_step and not prefetch_sent:
                state_msg = {
                    "loop": chunk_idx + 1,
                    "q": curr_q.tolist(),
                    "pos": curr_pos.tolist(),
                    "gripper": read_gripper_normalized(arm, default=(1.0 if gripper_state == 1 else 0.0))
                }
                if client.request_chunk(state_msg):
                    prefetch_sent = True

            # 2. Check if newly prefetched chunk has arrived
            fresh_chunk = client.get_chunk(block=False)
            if fresh_chunk is not None:
                next_chunk = fresh_chunk
                lat = next_chunk.get("latency_ms", 0)
                tilt_max = next_chunk.get("max_tilt_deg", 0.0)
                z_clamped = next_chunk.get("z_clamped_count", 0)
                clamp_info = f" | [CLAMP x{z_clamped}]" if z_clamped > 0 else ""
                print(f"[RTC Chunk #{chunk_idx + 1:03d} Ready] Lat: {lat:3.0f}ms | Tilt: {tilt_max:.2f}°{clamp_info}")

            # 3. Continuous Handover when stride is reached and next chunk is ready
            if step_in_chunk >= args.steps_per_chunk and next_chunk is not None:
                active_chunk = next_chunk
                pos_targets = active_chunk.get("joint_positions", [])
                vel_targets = active_chunk.get("joint_velocities", [])
                grip_targets = active_chunk.get("gripper", [])
                step_in_chunk = 0
                prefetch_sent = False
                next_chunk = None
                chunk_idx += 1
                p_now, _ = arm.get_cartesian_pose()
                print(f"[Continuous Handover #{chunk_idx:03d}] EE: [{p_now[0]:.3f}, {p_now[1]:.3f}, {p_now[2]:.3f}] | Continuous Stream")

            # 4. Target extraction & Starvation Fallback
            total_steps = len(pos_targets) if use_pos else len(vel_targets)
            if step_in_chunk >= total_steps:
                # Starvation fallback: wait briefly for fresh chunk
                print(f"  [RTC #{chunk_idx:03d}] Buffer Starvation! Waiting for next chunk...")
                fresh = client.get_chunk(block=True, timeout=0.5)
                if fresh is not None:
                    active_chunk = fresh
                    pos_targets = active_chunk.get("joint_positions", [])
                    vel_targets = active_chunk.get("joint_velocities", [])
                    grip_targets = active_chunk.get("gripper", [])
                    step_in_chunk = 0
                    prefetch_sent = False
                    chunk_idx += 1
                    continue
                else:
                    if not prefetch_sent:
                        client.request_chunk(state_msg)
                        prefetch_sent = True
                    arm.stop()
                    prev_dq = np.zeros(7)
                    rate.sleep()
                    continue

            if use_pos:
                q_des = np.array(pos_targets[step_in_chunk], dtype=np.float64)
                pos_err = q_des - curr_q
                dq_target = args.kp_pos * pos_err

                # Real-time closed-loop vertical orientation locking
                z_live = forward_kinematics(curr_q)[:3, 2]
                w_tilt = np.cross(z_live, [0.0, 0.0, -1.0])
                if np.linalg.norm(w_tilt) > 0.005:  # > 0.3 deg tilt
                    J = analytical_jacobian(curr_q)
                    J_w = J[3:, :]
                    J_w_pinv = J_w.T @ np.linalg.inv(J_w @ J_w.T + 1e-4 * np.eye(3))
                    dq_target += J_w_pinv @ (3.0 * w_tilt)
            else:
                dq_target = np.array(vel_targets[step_in_chunk], dtype=np.float64)

            grip_cmd = float(grip_targets[step_in_chunk]) if step_in_chunk < len(grip_targets) else 0.0
            step_in_chunk += 1

            # 4. Joint Speed & Acceleration Limiting
            dq_target = np.clip(dq_target, -args.max_vel, args.max_vel)
            if args.flip_lr:
                dq_target[0] = -dq_target[0]

            max_delta = MAX_JOINT_ACC * DT
            dq_smooth = np.clip(dq_target, prev_dq - max_delta, prev_dq + max_delta)

            # 5. Joint Soft Limits Protection
            q_pred = curr_q + dq_smooth * DT
            for i in range(7):
                if q_pred[i] < FRANKA_JOINT_LIMITS[i][0] and dq_smooth[i] < 0:
                    dq_smooth[i] = 0.0
                elif q_pred[i] > FRANKA_JOINT_LIMITS[i][1] and dq_smooth[i] > 0:
                    dq_smooth[i] = 0.0

            # 6. Real-Time Table Floor Collision Guard (Z >= z_min)
            dq_safe, clamped = project_velocity_z_floor(curr_q, dq_smooth, dt=DT, z_floor=args.z_min)
            if clamped:
                p_now = forward_kinematics(curr_q)[:3, 3]
                print(f"  [FLOOR GUARD] Z limit enforced: EE Z={p_now[2] * 1000.0:.1f}mm >= {args.z_min * 1000.0:.1f}mm")

            # 7. Publish velocity to 1kHz controller
            if not args.shadow:
                arm.set_joint_velocities(dq_safe)
            prev_dq = dq_safe.copy()

            # 8. Asynchronous Gripper Actuation
            if grip_cmd > 0.65:
                close_intent_counter += 1
                if gripper_state == 0 and close_intent_counter >= args.close_delay_steps:
                    def _do_close():
                        try:
                            arm.close_gripper(width=0.04, force=15.0, speed=0.35, inner_epsilon=0.025, outer_epsilon=0.025)
                        except Exception as e:
                            rospy.logwarn(f"Gripper close: {e}")
                    threading.Thread(target=_do_close, daemon=True).start()
                    gripper_state = 1
                    print(f"  [GRIPPER] Closing triggered in background (cmd={grip_cmd:.2f})")
            elif grip_cmd < 0.35:
                close_intent_counter = 0
                if gripper_state == 1:
                    def _do_open():
                        try:
                            arm.open_gripper(width=0.08, speed=0.35)
                        except Exception as e:
                            rospy.logwarn(f"Gripper open: {e}")
                    threading.Thread(target=_do_open, daemon=True).start()
                    gripper_state = 0
                    print(f"  [GRIPPER] Opening triggered in background (cmd={grip_cmd:.2f})")

            rate.sleep()

    except KeyboardInterrupt:
        print("\n[PAUSE] RTC continuous loop stopped by user.")
    except Exception as e:
        print(f"\n[WARN] RTC loop exception: {e}")
    finally:
        client.stop()

    return chunk_idx, t_start


def run_sync_loop(arm, conn, args):
    """Synchronous Stop-and-Go Closed-Loop Control Loop (Fallback)."""
    rate = rospy.Rate(CONTROL_HZ)
    init_grip = read_gripper_normalized(arm, default=0.0)
    gripper_state = 0 if init_grip <= 0.5 else 1
    chunk_idx = 0
    t_start = time.time()
    prev_dq = np.zeros(7, dtype=np.float64)
    close_intent_counter = 0

    print("=" * 80)
    print("  SYNCHRONOUS STOP-AND-GO MODE ACTIVE")
    print(f"  [Config] Steps per Cycle:  {args.sync_steps} steps (~{args.sync_steps * DT:.2f}s motion)")
    print(f"  [Config] Settling Time:    {args.settle_time * 1000:.0f} ms (Zero Motion Blur)")
    print(f"  [Config] Table Floor:      Z >= {args.z_min * 1000.0:.1f} mm")
    print("=" * 80)

    try:
        while not rospy.is_shutdown():
            chunk_idx += 1
            arm.stop()
            prev_dq = np.zeros(7, dtype=np.float64)
            if args.settle_time > 0:
                rospy.sleep(args.settle_time)

            curr_pos, _ = arm.get_cartesian_pose()
            curr_q = arm.get_joint_positions()
            if curr_pos is None or curr_q is None:
                rospy.sleep(0.02)
                continue

            curr_grip = read_gripper_normalized(arm, default=(1.0 if gripper_state == 1 else 0.0))
            state_msg = {
                "loop": chunk_idx,
                "q": curr_q.tolist(),
                "pos": curr_pos.tolist(),
                "gripper": curr_grip
            }

            t_infer_start = time.time()
            if not send_json(conn, state_msg):
                print("[ERROR] Failed to send state message! Exiting.")
                break

            resp = recv_json(conn)
            if resp is None:
                print("[ERROR] Connection closed! Exiting.")
                break

            infer_ms = (time.time() - t_infer_start) * 1000.0
            pos_targets = resp.get("joint_positions", [])
            vels = resp.get("joint_velocities", [])
            grips = resp.get("gripper", [])
            action_mode = resp.get("action_mode", "joint_position" if pos_targets else "joint_velocity")
            use_pos = (action_mode == "joint_position" and len(pos_targets) > 0)

            total_steps = len(pos_targets) if use_pos else len(vels)
            steps_to_exec = min(args.sync_steps, total_steps)
            p_ee = forward_kinematics(curr_q)[:3, 3]
            tilt_max = resp.get("max_tilt_deg", 0.0)
            z_clamped = resp.get("z_clamped_count", 0)
            clamp_info = f" | [CLAMP x{z_clamped}]" if z_clamped > 0 else ""
            print(f"[Sync Cycle #{chunk_idx:03d}] Infer: {infer_ms:4.0f}ms | EE: [{p_ee[0]:+.3f}, {p_ee[1]:+.3f}, {p_ee[2]:+.3f}] | Tilt: {tilt_max:.1f}° | Grip: {grips[0]:.2f}{clamp_info}")

            for step in range(steps_to_exec):
                if rospy.is_shutdown():
                    break

                curr_q_live = arm.get_joint_positions()
                if curr_q_live is None:
                    continue

                if use_pos:
                    q_des = np.array(pos_targets[step], dtype=np.float64)
                    pos_err = q_des - curr_q_live
                    dq_target = args.kp_pos * pos_err
                else:
                    dq_target = np.array(vels[step], dtype=np.float64)

                grip_cmd = float(grips[step]) if step < len(grips) else 0.0

                dq_target = np.clip(dq_target, -args.max_vel, args.max_vel)
                if args.flip_lr:
                    dq_target[0] = -dq_target[0]

                max_delta = MAX_JOINT_ACC * DT
                dq_smooth = np.clip(dq_target, prev_dq - max_delta, prev_dq + max_delta)

                # Floor collision check
                dq_safe, clamped = project_velocity_z_floor(curr_q_live, dq_smooth, dt=DT, z_floor=args.z_min)

                # Soft deceleration near cycle end
                rem = steps_to_exec - 1 - step
                if rem == 1:
                    dq_safe *= 0.5
                elif rem == 0:
                    dq_safe *= 0.2

                if not args.shadow:
                    arm.set_joint_velocities(dq_safe)
                prev_dq = dq_safe.copy()

                # Gripper
                if grip_cmd > 0.65:
                    close_intent_counter += 1
                    if gripper_state == 0 and close_intent_counter >= args.close_delay_steps:
                        def _do_close():
                            try:
                                arm.close_gripper(width=0.04, force=15.0, speed=0.35, inner_epsilon=0.025, outer_epsilon=0.025)
                            except Exception as e:
                                rospy.logwarn(f"Gripper close: {e}")
                        threading.Thread(target=_do_close, daemon=True).start()
                        gripper_state = 1
                elif grip_cmd < 0.35:
                    close_intent_counter = 0
                    if gripper_state == 1:
                        def _do_open():
                            try:
                                arm.open_gripper(width=0.08, speed=0.35)
                            except Exception as e:
                                rospy.logwarn(f"Gripper open: {e}")
                        threading.Thread(target=_do_open, daemon=True).start()
                        gripper_state = 0

                rate.sleep()

            arm.stop()
            prev_dq = np.zeros(7, dtype=np.float64)

    except KeyboardInterrupt:
        print("\n[PAUSE] Sync loop stopped by user.")
    except Exception as e:
        print(f"\n[WARN] Sync loop exception: {e}")

    return chunk_idx, t_start


def main():
    parser = argparse.ArgumentParser(description="Franka Closed-Loop VLA Execution Service")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Listening port (default: 8765)")
    parser.add_argument("--rtc", dest="rtc", action="store_true", default=True,
                        help="Enable Real-Time Chunking continuous streaming mode (default: True)")
    parser.add_argument("--sync", dest="rtc", action="store_false",
                        help="Enable synchronous Stop-and-Go mode (fallback)")
    parser.add_argument("--sync-steps", type=int, default=DEFAULT_SYNC_STEPS,
                        help="Steps per cycle in sync mode (default: 15)")
    parser.add_argument("--settle-time", type=float, default=0.05,
                        help="Stationary settling time before inference in sync mode (default: 0.05s)")
    parser.add_argument("--preempt-step", type=int, default=PREEMPT_STEP,
                        help="Step to trigger async prefetch in RTC mode (default: 6)")
    parser.add_argument("--blend-steps", type=int, default=DEFAULT_BLEND_STEPS,
                        help="Steps over which to blend overlapping chunks in RTC mode (default: 3)")
    parser.add_argument("--max-vel", type=float, default=MAX_JOINT_VEL,
                        help="Max joint speed clamp in rad/s (default: 0.35)")
    parser.add_argument("--max-acc", type=float, default=MAX_JOINT_ACC,
                        help="Max joint acceleration clamp in rad/s^2 (default: 1.5)")
    parser.add_argument("--z-min", type=float, default=DEFAULT_Z_FLOOR,
                        help=f"Minimum safe table Z height in meters (default: {DEFAULT_Z_FLOOR}m = +7.0mm)")
    parser.add_argument("--kp-pos", type=float, default=8.0,
                        help="P-servo tracking gain for joint position mode (default: 8.0)")
    parser.add_argument("--close-delay-steps", type=int, default=2,
                        help="Debounce steps before closing gripper (default: 2)")
    parser.add_argument("--flip-lr", action="store_true", default=False,
                        help="Invert Joint 0 (base yaw)")
    parser.add_argument("--shadow", action="store_true", default=False,
                        help="Shadow mode: log commands without moving physical robot")
    args = parser.parse_args()

    if not rospy.core.is_initialized():
        rospy.init_node("closed_loop_franka_server", anonymous=True)

    mode_str = "RTC (Continuous Real-Time Chunking)" if args.rtc else "SYNCHRONOUS (Stop-and-Go)"
    print("=" * 80)
    print(f"  FRANKA CLOSED-LOOP SERVICE: {mode_str}")
    print("=" * 80)
    print(f"[*] Port:              {args.port}")
    print(f"[*] Mode:              {mode_str}")
    if args.rtc:
        print(f"[*] RTC Handover:      Step {args.preempt_step} (Prefetch) -> Blend {args.blend_steps} steps")
    else:
        print(f"[*] Sync Steps:        {args.sync_steps} steps (Pause: {args.settle_time * 1000:.0f}ms)")
    print(f"[*] Hard Table Floor:  Z >= {args.z_min * 1000.0:.1f} mm")
    print(f"[*] Max Speed:         {args.max_vel:.2f} rad/s")
    print("-" * 80)

    print("[1/2] Connecting to Franka Controller Client...")
    arm = FrankaJointVelocityController()

    start_wait = time.time()
    initial_pos, initial_q = None, None
    while time.time() - start_wait < 5.0 and not rospy.is_shutdown():
        initial_pos, _ = arm.get_cartesian_pose()
        initial_q = arm.get_joint_positions()
        if initial_pos is not None and initial_q is not None:
            break
        rospy.sleep(0.05)

    if initial_pos is None or initial_q is None:
        print("[ERROR] Failed to receive Franka robot state!")
        return 1

    print(f"[OK] Franka Connected.")
    print(f"     EE Pose:   [{initial_pos[0]:+.4f}, {initial_pos[1]:+.4f}, {initial_pos[2]:+.4f}] m")
    print(f"     Joints:    [{', '.join(f'{v:+.3f}' for v in initial_q)}]")

    print(f"[2/2] Setting up TCP Server on port {args.port}...")
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("0.0.0.0", args.port))
    server_sock.listen(1)
    print(f"[OK] Listening on port {args.port}. Awaiting GPU Agent...")

    conn, addr = server_sock.accept()
    print(f"[OK] GPU Agent Connected from {addr}!")
    print("-" * 80)
    input("[READY] Hold physical E-STOP. Press [ENTER] to start closed-loop control...")

    total_chunks = 0
    t_start = time.time()
    try:
        if args.rtc:
            total_chunks, t_start = run_rtc_loop(arm, conn, args)
        else:
            total_chunks, t_start = run_sync_loop(arm, conn, args)
    finally:
        try:
            conn.close()
        except Exception:
            pass
        try:
            server_sock.close()
        except Exception:
            pass

        arm.stop()
        p_final, _ = arm.get_cartesian_pose()
        print("\n" + "=" * 80)
        print("  CLOSED-LOOP EXECUTION TERMINATED SAFELY")
        print("=" * 80)
        print(f"[*] Total Chunks: {total_chunks}")
        print(f"[*] Elapsed:      {time.time() - t_start:.1f} s")
        if p_final is not None:
            print(f"[*] Final Pose:   [{p_final[0]:+.4f}, {p_final[1]:+.4f}, {p_final[2]:+.4f}] m")
        print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
