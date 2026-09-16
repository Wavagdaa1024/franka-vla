#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Franka Closed-Loop Execution Service.
Runs on Franka Linux Control PC (10.197.16.43:8765).

Modes Supported:
  1. Synchronous Stop-and-Go (--sync, default):
     Arm stops -> Settles 50ms (zero motion blur) -> Samples exact state -> 
     Requests chunk from GPU Agent -> Executes N steps -> Arm stops smoothly -> Repeats.
     Completely eliminates camera motion blur and 300ms observation latency.

  2. Asynchronous Continuous Double-Buffer (--async-mode):
     Arm never stops: background worker prefetches chunk K+1 at step 5 of 10.
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
import math
import socket
import struct
import argparse
import threading
import queue
import numpy as np

import rospy
from std_msgs.msg import Float64MultiArray

# Import the 1kHz joint velocity controller client
from Base_franka_joint_velocity_controller import FrankaJointVelocityController

DEFAULT_PORT = 8765
DEFAULT_SYNC_STEPS = 10      # 10 steps @ 30Hz = 0.333s motion per sync cycle
DEFAULT_STEPS_PER_CHUNK = 12 # 12 steps @ 30Hz = 0.400s per chunk in async mode
PREEMPT_STEP = 6             # Step at which to trigger async prefetch
CONTROL_HZ = 30.0            # 30 Hz policy frequency (aligned with 30Hz teleop dataset)
DT = 1.0 / CONTROL_HZ        # 0.0333 s
MAX_JOINT_VEL = 0.35         # 0.35 rad/s max safe testing speed
MAX_JOINT_ACC = 2.5          # 2.5 rad/s^2 smooth responsive acceleration clamp
Z_MIN = 0.055                # Safe floor: 5.5 cm above Franka base (Anti-collision)
Z_MAX = 0.65

# Franka Panda soft joint limits (with 0.05 rad buffer)
JOINT_LIMITS = [
    (-2.84, 2.84),
    (-1.71, 1.71),
    (-2.84, 2.84),
    (-3.02, -0.12),
    (-2.84, 2.84),
    (0.03, 3.70),
    (-2.84, 2.84)
]


# Franka Craig-modified DH Forward Kinematics for Z-Height Safety Guard
def rot_x(alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])

def trans_x(a):
    T = np.eye(4)
    T[0, 3] = a
    return T

def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

def trans_z(d):
    T = np.eye(4)
    T[2, 3] = d
    return T

def franka_fk(q, ee_offset=0.107):
    mdh = [
        (0, 0.333, 0),
        (0, 0, -np.pi/2),
        (0, 0.316, np.pi/2),
        (0.0825, 0, np.pi/2),
        (-0.0825, 0.384, -np.pi/2),
        (0, 0, np.pi/2),
        (0.088, 0, np.pi/2),
        (0, ee_offset, 0)
    ]
    T = np.eye(4)
    for i in range(7):
        a, d, alpha = mdh[i]
        theta = q[i]
        T = T @ (rot_x(alpha) @ trans_x(a) @ rot_z(theta) @ trans_z(d))
    T = T @ (rot_x(mdh[7][2]) @ trans_x(mdh[7][0]) @ trans_z(mdh[7][1]))
    return T


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
    """Asynchronous background worker for continuous double-buffer streaming."""
    def __init__(self, conn):
        self.conn = conn
        self.req_queue = queue.Queue(maxsize=1)
        self.resp_queue = queue.Queue(maxsize=1)
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
    """
    DROID / OpenPI standard:
      0.0 = OPEN (width > 0.035m)
      1.0 = CLOSED
    """
    w = arm.get_gripper_width()
    if w is None:
        return default
    return 0.0 if w > 0.035 else 1.0


def run_sync_loop(arm, conn, args):
    """
    Synchronous Stop-and-Go Closed-Loop Control Loop:
      1. Settle stationary (50ms) -> No motion blur
      2. Sample exact stationary joint state and gripper width
      3. Request chunk from Windows GPU Agent and block-wait for inference
      4. Execute N steps with safety guards
      5. Repeat smoothly
    """
    rate = rospy.Rate(CONTROL_HZ)
    init_grip = read_gripper_normalized(arm, default=0.0)
    gripper_state = 0 if init_grip <= 0.5 else 1  # 0=OPEN, 1=CLOSED
    chunk_idx = 0
    t_start = time.time()
    prev_dq = np.zeros(7, dtype=np.float64)
    close_intent_counter = 0

    print("=" * 75)
    print("  SYNCHRONOUS STOP-AND-GO MODE ACTIVE")
    print(f"  [Config] Sync Steps:       {args.sync_steps} steps (~{args.sync_steps * DT:.2f}s motion per cycle)")
    print(f"  [Config] Settling Time:    {args.settle_time * 1000:.0f} ms (Zero Motion Blur)")
    print(f"  [Config] Speed Clamp:      {args.max_vel:.2f} rad/s")
    print(f"  [Config] Table Floor:      Z >= {args.z_min * 100:.1f} cm")
    print(f"  [Config] Initial Gripper:  {'OPEN' if gripper_state == 1 else 'CLOSED'} (width={init_grip:.2f})")
    print("=" * 75)

    try:
        while not rospy.is_shutdown():
            chunk_idx += 1

            # Step 1: Ensure arm is completely halted & settle stationary
            arm.stop()
            prev_dq = np.zeros(7, dtype=np.float64)
            if args.settle_time > 0:
                rospy.sleep(args.settle_time)

            # Step 2: Read exact stationary hardware state
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

            # Step 3: Synchronously request action chunk from Windows GPU Agent
            t_infer_start = time.time()
            if not send_json(conn, state_msg):
                print("[ERROR] Failed to send state message to GPU Agent! Exiting.")
                break

            resp = recv_json(conn)
            if resp is None:
                print("[ERROR] Connection closed or invalid response from GPU Agent! Exiting.")
                break

            infer_ms = (time.time() - t_infer_start) * 1000.0
            pos_targets = resp.get("joint_positions", [])
            vels = resp.get("joint_velocities", [])
            grips = resp.get("gripper", [])
            action_mode = resp.get("action_mode", "joint_position" if pos_targets else "joint_velocity")
            use_pos = (action_mode == "joint_position" and len(pos_targets) > 0)

            if not use_pos and (not vels or len(vels) == 0):
                print(f"  [Sync #{chunk_idx:03d}] Warning: empty chunk received. Holding position.")
                continue

            # Step 4: Execute up to args.sync_steps steps smoothly
            total_steps_in_chunk = len(pos_targets) if use_pos else len(vels)
            steps_to_exec = min(args.sync_steps, total_steps_in_chunk)
            p_ee, _ = arm.get_cartesian_pose()
            mode_tag = "POS-SERVO" if use_pos else "VEL-OPEN"
            print(f"[Sync Cycle #{chunk_idx:03d} | {mode_tag}] Infer: {infer_ms:4.0f}ms | EE: [{p_ee[0]:+.3f}, {p_ee[1]:+.3f}, {p_ee[2]:+.3f}] | Grip: {grips[0]:.2f} | Exec: {steps_to_exec} steps")

            interrupted = False
            for step in range(steps_to_exec):
                if rospy.is_shutdown():
                    break

                curr_q_live = arm.get_joint_positions()
                if use_pos and curr_q_live is not None:
                    q_des = np.array(pos_targets[step], dtype=np.float64)
                    pos_err = q_des - curr_q_live
                    dq_target = args.kp_pos * pos_err
                else:
                    dq_target = np.array(vels[step], dtype=np.float64)

                grip_cmd = float(grips[step]) if step < len(grips) else 0.0

                # 1. Clamp target speed
                dq_target = np.clip(dq_target, -args.max_vel, args.max_vel)
                if args.flip_lr:
                    dq_target[0] = -dq_target[0]

                # 2. Acceleration smoothing
                max_delta = MAX_JOINT_ACC * DT
                dq_smooth = np.clip(dq_target, prev_dq - max_delta, prev_dq + max_delta)

                # 3. Soft joint limit safety guard
                if curr_q_live is not None:
                    q_pred = curr_q_live + dq_smooth * DT
                    for i in range(7):
                        if q_pred[i] < JOINT_LIMITS[i][0] and dq_smooth[i] < 0:
                            dq_smooth[i] = 0.0
                        elif q_pred[i] > JOINT_LIMITS[i][1] and dq_smooth[i] > 0:
                            dq_smooth[i] = 0.0

                    # 4. Z_MIN Table Anti-Collision Safety Guard
                    ee_pred = franka_fk(curr_q_live + dq_smooth * DT)[:3, 3]
                    if ee_pred[2] < args.z_min:
                        dq_smooth = np.zeros(7, dtype=np.float64)
                        print(f"  [FLOOR GUARD] Z limit reached ({ee_pred[2]:.3f}m < {args.z_min:.2f}m). Clamped velocity to 0.")

                # Publish velocity to controller
                if not args.shadow:
                    arm.set_joint_velocities(dq_smooth)
                else:
                    if step == 0:
                        print(f"  [SHADOW DRY-RUN] dq: {[round(x, 3) for x in dq_smooth]}")
                prev_dq = dq_smooth.copy()

                # Gripper control logic (DROID standard: 0.0 = OPEN, 1.0 = CLOSED)
                # Non-blocking trigger: NEVER stop arm or break chunk execution!
                if grip_cmd > 0.65:
                    # Model wants to CLOSE gripper
                    close_intent_counter += 1
                    if gripper_state == 0 and close_intent_counter >= args.close_delay_steps:
                        def _do_close():
                            try:
                                arm.close_gripper(width=0.04, force=15.0, speed=0.1, inner_epsilon=0.025, outer_epsilon=0.025)
                            except Exception as e:
                                rospy.logwarn(f"Gripper close: {e}")
                        threading.Thread(target=_do_close, daemon=True).start()
                        gripper_state = 1
                        print(f"  [GRIPPER] Closed in background (cmd={grip_cmd:.2f})")
                elif grip_cmd < 0.35:
                    # Model wants to OPEN gripper
                    close_intent_counter = 0
                    if gripper_state == 1:
                        def _do_open():
                            try:
                                arm.open_gripper(width=0.08, speed=0.1)
                            except Exception as e:
                                rospy.logwarn(f"Gripper open: {e}")
                        threading.Thread(target=_do_open, daemon=True).start()
                        gripper_state = 0
                        print(f"  [GRIPPER] Opened in background (cmd={grip_cmd:.2f})")

                rate.sleep()

            # Step 5: Stop arm at the end of the executed chunk steps
            arm.stop()
            prev_dq = np.zeros(7, dtype=np.float64)

    except KeyboardInterrupt:
        print("\n[PAUSE] Synchronous loop stopped by user.")
    except Exception as e:
        print(f"\n[WARN] Synchronous loop exception: {e}")

    return chunk_idx, t_start


def run_async_loop(arm, conn, args):
    """
    Asynchronous Continuous Double-Buffering Loop (Arm Never Stops).
    Prefetches chunk K+1 at PREEMPT_STEP while executing chunk K.
    """
    client = AsyncChunkClient(conn)
    rate = rospy.Rate(CONTROL_HZ)
    init_grip = read_gripper_normalized(arm, default=1.0)
    gripper_state = 1 if init_grip > 0.5 else 0
    chunk_idx = 0
    t_start = time.time()
    prev_dq = np.zeros(7, dtype=np.float64)
    close_intent_counter = 0

    print("=" * 75)
    print("  ASYNCHRONOUS CONTINUOUS STREAMING MODE ACTIVE (DOUBLE BUFFER)")
    print(f"  [Config] Handover Steps:   {args.steps_per_chunk} steps (~{args.steps_per_chunk * DT:.2f}s per chunk)")
    print(f"  [Config] Prefetch Step:    {args.preempt_step}")
    print(f"  [Config] Speed Clamp:      {args.max_vel:.2f} rad/s")
    print(f"  [Config] Table Floor:      Z >= {args.z_min * 100:.1f} cm")
    print("=" * 75)

    start_pos, _ = arm.get_cartesian_pose()
    start_q = arm.get_joint_positions()

    # Fetch initial chunk
    print(f"\n[Bootstrap] Gripper state: {'OPEN' if gripper_state == 1 else 'CLOSED'} (width={init_grip:.2f}). Fetching initial chunk...")
    init_state = {
        "loop": 0,
        "q": start_q.tolist(),
        "pos": start_pos.tolist(),
        "gripper": init_grip
    }
    client.request_chunk(init_state)
    active_chunk = client.get_chunk(block=True, timeout=10.0)

    if active_chunk is None:
        print("[ERROR] Failed to receive initial action chunk from Windows GPU Agent!")
        client.stop()
        return 0, t_start

    active_vels = active_chunk.get("joint_velocities", [])
    active_grips = active_chunk.get("gripper", [])
    print(f"[Bootstrap OK] Received initial chunk ({len(active_vels)} steps, infer: {active_chunk.get('latency_ms', 0):.0f}ms). Starting streaming...")

    step_in_chunk = 0
    prefetch_sent = False

    try:
        while not rospy.is_shutdown():
            if step_in_chunk >= len(active_vels):
                print(f"  [Buffer Starvation] Waiting for fresh chunk...")
                fresh = client.get_chunk(block=True, timeout=0.8)
                if fresh is not None:
                    active_chunk = fresh
                    active_vels = active_chunk.get("joint_velocities", [])
                    active_grips = active_chunk.get("gripper", [])
                    step_in_chunk = 0
                    prefetch_sent = False
                    chunk_idx += 1
                else:
                    arm.stop()
                    prev_dq = np.zeros(7)
                    rate.sleep()
                    continue

            curr_pos, _ = arm.get_cartesian_pose()
            curr_q = arm.get_joint_positions()
            if curr_pos is None or curr_q is None:
                rate.sleep()
                continue

            if step_in_chunk == args.preempt_step and not prefetch_sent:
                state_msg = {
                    "loop": chunk_idx + 1,
                    "q": curr_q.tolist(),
                    "pos": curr_pos.tolist(),
                    "gripper": read_gripper_normalized(arm, default=(1.0 if gripper_state == 1 else 0.0))
                }
                client.request_chunk(state_msg)
                prefetch_sent = True

            if step_in_chunk >= args.steps_per_chunk:
                next_chunk = client.get_chunk(block=False)
                if next_chunk is not None:
                    chunk_idx += 1
                    active_chunk = next_chunk
                    active_vels = active_chunk.get("joint_velocities", [])
                    active_grips = active_chunk.get("gripper", [])
                    lat = active_chunk.get("latency_ms", 0)
                    step_in_chunk = 0
                    prefetch_sent = False
                    p_now, _ = arm.get_cartesian_pose()
                    print(f"[Chunk #{chunk_idx:03d} Handover] Infer: {lat:3.0f}ms | EE: [{p_now[0]:.3f}, {p_now[1]:.3f}, {p_now[2]:.3f}] | Continuous Stream")

            dq_target = np.array(active_vels[step_in_chunk], dtype=np.float64)
            grip_cmd = float(active_grips[step_in_chunk]) if step_in_chunk < len(active_grips) else 0.0

            dq_target = np.clip(dq_target, -args.max_vel, args.max_vel)
            if args.flip_lr:
                dq_target[0] = -dq_target[0]

            max_delta = MAX_JOINT_ACC * DT
            dq_smooth = np.clip(dq_target, prev_dq - max_delta, prev_dq + max_delta)

            curr_q_live = arm.get_joint_positions()
            if curr_q_live is not None:
                q_pred = curr_q_live + dq_smooth * DT
                for i in range(7):
                    if q_pred[i] < JOINT_LIMITS[i][0] and dq_smooth[i] < 0:
                        dq_smooth[i] = 0.0
                    elif q_pred[i] > JOINT_LIMITS[i][1] and dq_smooth[i] > 0:
                        dq_smooth[i] = 0.0

                ee_pred = franka_fk(curr_q_live + dq_smooth * DT)[:3, 3]
                if ee_pred[2] < args.z_min:
                    dq_smooth = np.zeros(7, dtype=np.float64)
                    print(f"  [FLOOR GUARD] Z limit reached ({ee_pred[2]:.3f}m < {args.z_min:.2f}m). Clamped velocity to 0.")

            arm.set_joint_velocities(dq_smooth)
            prev_dq = dq_smooth.copy()

            # Gripper control logic (DROID standard: 0.0 = OPEN, 1.0 = CLOSED)
            # Non-blocking trigger: NEVER stop arm or break chunk execution!
            if grip_cmd > 0.65:
                # Model wants to CLOSE gripper (grasp)
                close_intent_counter += 1
                if gripper_state == 0 and close_intent_counter >= args.close_delay_steps:
                    def _do_close_async():
                        try:
                            arm.close_gripper(width=0.04, force=15.0, speed=0.1, inner_epsilon=0.025, outer_epsilon=0.025)
                        except Exception as e:
                            rospy.logwarn(f"Gripper close: {e}")
                    threading.Thread(target=_do_close_async, daemon=True).start()
                    gripper_state = 1
                    print(f"  [GRIPPER] Closed in background (cmd={grip_cmd:.2f})")
            elif grip_cmd < 0.35:
                # Model wants to OPEN gripper (release)
                close_intent_counter = 0
                if gripper_state == 1:
                    def _do_open_async():
                        try:
                            arm.open_gripper(width=0.08, speed=0.1)
                        except Exception as e:
                            rospy.logwarn(f"Gripper open: {e}")
                    threading.Thread(target=_do_open_async, daemon=True).start()
                    gripper_state = 0
                    print(f"  [GRIPPER] Opened in background (cmd={grip_cmd:.2f})")

            step_in_chunk += 1
            rate.sleep()

    except KeyboardInterrupt:
        print("\n[PAUSE] Asynchronous loop stopped by user.")
    except Exception as e:
        print(f"\n[WARN] Asynchronous loop exception: {e}")
    finally:
        client.stop()

    return chunk_idx, t_start


def main():
    parser = argparse.ArgumentParser(description="Franka Closed-Loop Joint Velocity Execution Service")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Listening port (default: 8765)")
    parser.add_argument("--sync", dest="sync", action="store_true", default=True, help="Enable synchronous Stop-and-Go mode (default: True)")
    parser.add_argument("--async-mode", dest="sync", action="store_false", help="Enable asynchronous continuous double-buffer mode")
    parser.add_argument("--sync-steps", type=int, default=DEFAULT_SYNC_STEPS, help="Chunk steps to run per cycle in sync mode (default: 10 @ 30Hz = 0.33s)")
    parser.add_argument("--settle-time", type=float, default=0.05, help="Stationary settling time in seconds before inference (default: 0.05s = 50ms)")
    parser.add_argument("--steps-per-chunk", type=int, default=DEFAULT_STEPS_PER_CHUNK, help="Steps before handover in async mode (default: 12)")
    parser.add_argument("--preempt-step", type=int, default=PREEMPT_STEP, help="Step to prefetch in async mode (default: 6)")
    parser.add_argument("--max-vel", type=float, default=MAX_JOINT_VEL, help="Max joint speed clamp in rad/s (default: 0.35)")
    parser.add_argument("--max-acc", type=float, default=MAX_JOINT_ACC, help="Max joint acceleration clamp in rad/s^2 (default: 2.5)")
    parser.add_argument("--z-min", type=float, default=Z_MIN, help="Minimum safe Z height in meters (default: 0.055)")
    parser.add_argument("--flip-lr", action="store_true", default=False, help="Invert Joint 1 (base yaw, default: False)")
    parser.add_argument("--close-delay-steps", type=int, default=2, help="Debounce steps before closing gripper (default: 2 steps = ~0.067s @ 30Hz)")
    parser.add_argument("--kp-pos", type=float, default=8.0, help="P-servo tracking gain for joint position mode in rad/s per rad error (default: 8.0)")
    parser.add_argument("--shadow", action="store_true", default=False, help="Shadow mode: log tracking commands without publishing physical robot velocities")
    args = parser.parse_args()

    if not rospy.core.is_initialized():
        rospy.init_node("closed_loop_franka_server", anonymous=True)

    mode_str = "SYNCHRONOUS (Stop-and-Go)" if args.sync else "ASYNCHRONOUS (Continuous Double-Buffer)"
    print("=" * 75)
    print(f"  FRANKA CLOSED-LOOP SERVICE: {mode_str}")
    print("=" * 75)
    print(f"[*] Listening Port:        {args.port}")
    print(f"[*] Execution Mode:        {mode_str}")
    if args.sync:
        print(f"[*] Steps per Cycle:       {args.sync_steps} steps (~{args.sync_steps * DT:.2f}s motion)")
        print(f"[*] Settling Pause:        {args.settle_time * 1000:.0f} ms")
    else:
        print(f"[*] Handover Timing:       Step {args.preempt_step} (Prefetch) -> Step {args.steps_per_chunk} (Handover)")
    print(f"[*] Max Joint Speed:       {args.max_vel:.2f} rad/s")
    print(f"[*] Safe Table Floor:      Z >= {args.z_min * 100:.1f} cm")
    print("-" * 75)

    # 1. Connect to Franka Joint Velocity Controller Client
    print("[1/3] Connecting to Franka Joint Velocity Controller Client...")
    arm = FrankaJointVelocityController()

    start_wait = time.time()
    initial_pos, initial_quat = None, None
    initial_q = None
    while time.time() - start_wait < 5.0 and not rospy.is_shutdown():
        initial_pos, initial_quat = arm.get_cartesian_pose()
        initial_q = arm.get_joint_positions()
        if initial_pos is not None and initial_q is not None:
            break
        rospy.sleep(0.05)

    if initial_pos is None or initial_q is None:
        print("[ERROR] Failed to receive Franka robot state within 5.0s!")
        return 1

    start_pos = initial_pos.copy()
    start_q = initial_q.copy()
    print(f"[OK] Franka Connected.")
    print(f"     Initial EE Pose: x={start_pos[0]:+.4f}, y={start_pos[1]:+.4f}, z={start_pos[2]:+.4f} m")
    print(f"     Initial Joints:  [{', '.join(f'{v:+.3f}' for v in start_q)}]")

    # 2. Setup Listening Socket
    print(f"[2/3] Setting up TCP Server on port {args.port}...")
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("0.0.0.0", args.port))
    server_sock.listen(1)
    print(f"[OK] Server listening on port {args.port}. Waiting for Windows GPU Agent to connect...")

    conn, addr = server_sock.accept()
    print(f"[OK] Windows GPU Agent Connected from {addr}!")

    print("-" * 75)
    input("[READY] Hold physical E-STOP. Press [ENTER] to start closed-loop control...")

    # Refresh live hardware pose immediately upon user ENTER
    live_pos, _ = arm.get_cartesian_pose()
    live_q = arm.get_joint_positions()
    if live_pos is not None and live_q is not None:
        start_pos = live_pos.copy()
        start_q = live_q.copy()
        print(f"[OK] Starting Pose: x={start_pos[0]:+.4f}, y={start_pos[1]:+.4f}, z={start_pos[2]:+.4f} m")

    total_chunks = 0
    t_start = time.time()

    try:
        if args.sync:
            total_chunks, t_start = run_sync_loop(arm, conn, args)
        else:
            total_chunks, t_start = run_async_loop(arm, conn, args)
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
        print("\n" + "=" * 75)
        print("  CLOSED-LOOP EXECUTION TERMINATED SAFELY")
        print("=" * 75)
        print(f"[*] Total Cycles/Chunks: {total_chunks}")
        print(f"[*] Elapsed Time:        {time.time() - t_start:.1f} s")
        if p_final is not None:
            print(f"[*] Final Position:      x={p_final[0]:+.4f}, y={p_final[1]:+.4f}, z={p_final[2]:+.4f} m")
        print("[*] Status:              Arm stopped safely (holding position).")
        print("=" * 75)

    return 0


if __name__ == "__main__":
    sys.exit(main())
