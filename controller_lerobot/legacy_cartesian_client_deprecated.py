#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Closed-Loop VLA Client for Franka Control Computer.
Connects to live Windows Policy Server (10.70.242.38:8765),
streams Franka state, receives real-time visual-servoing action chunks,
and executes smooth closed-loop manipulation with strict safety bounds.
"""

import sys
import time
import math
import json
import struct
import socket
import argparse
import numpy as np
import rospy

from Base_franka_cartesian_impedance_controller import FrankaCartesianImpedanceController

# Safety Limits
DEFAULT_SERVER_HOST = "10.70.242.38"
DEFAULT_SERVER_PORT = 8765

MAX_SPEED_MPS = 0.08       # Max velocity: 80 mm/s (safe and smooth)
CONTROL_HZ = 15.0          # 15 Hz
DT = 1.0 / CONTROL_HZ      # ~0.0667 s
MAX_STEP_DIST_M = MAX_SPEED_MPS * DT  # ~5.3 mm per step

# Workspace Safety Bounding Box (in panda_link0 frame)
# Hard bounds: robot end-effector is strictly forbidden from leaving this box!
X_MIN, X_MAX = 0.20, 0.70   # Table forward depth: 20cm to 70cm from base
Y_MIN, Y_MAX = -0.35, +0.35 # Table lateral width: -35cm to +35cm
Z_MIN, Z_MAX = 0.08, 0.65   # Table height: >= 8cm (NEVER hit table surface), <= 65cm


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


def clamp_workspace(pos: np.ndarray) -> np.ndarray:
    clamped = pos.copy()
    clamped[0] = np.clip(clamped[0], X_MIN, X_MAX)
    clamped[1] = np.clip(clamped[1], Y_MIN, Y_MAX)
    clamped[2] = np.clip(clamped[2], Z_MIN, Z_MAX)
    return clamped


def parse_args():
    parser = argparse.ArgumentParser(description="Closed-Loop Franka VLA Client")
    parser.add_argument("--server", default=DEFAULT_SERVER_HOST, help="Policy server host IP")
    parser.add_argument("--port", type=int, default=DEFAULT_SERVER_PORT, help="Policy server port")
    parser.add_argument("--steps-per-chunk", type=int, default=10, help="Steps to execute per chunk (default: 10/15 = 0.67s)")
    parser.add_argument("--max-loops", type=int, default=40, help="Maximum closed-loop iterations (default: 40 ~ 30s)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not rospy.core.is_initialized():
        rospy.init_node("closed_loop_vla_client", anonymous=True)

    print("=" * 70)
    print("  FRANKA CLOSED-LOOP REAL-TIME VLA CLIENT")
    print("=" * 70)
    print(f"[*] Policy Server:          {args.server}:{args.port}")
    print(f"[*] Task Instruction:      'pick up the blue block and place it in the brown basket'")
    print(f"[*] Max Speed Limit:        {MAX_SPEED_MPS*1000:.1f} mm/s")
    print(f"[*] Safe Workspace Box:     X=[{X_MIN*100:.0f}, {X_MAX*100:.0f}] cm")
    print(f"                            Y=[{Y_MIN*100:.0f}, {Y_MAX*100:.0f}] cm")
    print(f"                            Z=[{Z_MIN*100:.0f}, {Z_MAX*100:.0f}] cm (Hard floor >= 8cm)")
    print("-" * 70)

    # 1. Connect to Franka ROS Controller
    print("[1/3] Connecting to Franka Cartesian Impedance Controller...")
    arm = FrankaCartesianImpedanceController()

    start_wait = time.time()
    initial_pos, initial_quat = None, None
    while time.time() - start_wait < 5.0 and not rospy.is_shutdown():
        initial_pos, initial_quat = arm.get_cartesian_pose()
        if initial_pos is not None and initial_quat is not None:
            break
        rospy.sleep(0.05)

    if initial_pos is None or initial_quat is None:
        print("[ERROR] Failed to receive Franka Cartesian pose within 5.0s!")
        return 1

    start_pos = initial_pos.copy()
    fixed_quat = initial_quat.copy()
    print(f"[OK] Franka Connected. Current Pose: x={start_pos[0]:+.4f}, y={start_pos[1]:+.4f}, z={start_pos[2]:+.4f}")

    # 2. Connect to Policy Server
    print(f"[2/3] Connecting to Policy Server at {args.server}:{args.port}...")
    try:
        sock = socket.create_connection((args.server, args.port), timeout=5.0)
        sock.settimeout(10.0)
        print("[OK] Connected to Policy Server.")
    except Exception as e:
        print(f"[ERROR] Connection to Policy Server failed: {e}")
        print("        Ensure 'live_vla_server.py' is running on the Windows server.")
        return 1

    # 3. User Confirmation Prompt
    print("-" * 70)
    input("[SAFETY READY] Hold the physical E-STOP button. Press [ENTER] to start closed-loop control...")

    rate = rospy.Rate(CONTROL_HZ)
    current_target = start_pos.copy()
    gripper_state = 0  # 0: closed, 1: open

    total_loops = 0
    t_start = time.time()

    try:
        while not rospy.is_shutdown() and total_loops < args.max_loops:
            total_loops += 1

            # A. Get current robot state
            curr_pos, _ = arm.get_cartesian_pose()
            curr_q = arm.get_joint_positions()
            curr_grip_width = arm.get_gripper_width()

            if curr_pos is None or curr_q is None:
                rospy.logwarn("Missing state, waiting...")
                rate.sleep()
                continue

            # B. Send state to server
            state_msg = {
                "loop": total_loops,
                "q": curr_q.tolist(),
                "pos": curr_pos.tolist(),
                "gripper": 1.0 if gripper_state == 1 else 0.0
            }
            t_send = time.time()
            send_json(sock, state_msg)

            # C. Receive action chunk
            resp = recv_json(sock)
            if resp is None:
                print("[WARN] Server closed connection or sent null response.")
                break

            latency = (time.time() - t_send) * 1000.0
            waypoints = resp.get("waypoints", [])
            steps_to_run = min(len(waypoints), args.steps_per_chunk)

            # Anchor for this chunk
            chunk_anchor = curr_pos.copy()

            # D. Execute waypoints with smooth speed clamp
            for s_idx in range(steps_to_run):
                if rospy.is_shutdown():
                    break

                dx, dy, dz, grip_cmd = waypoints[s_idx]

                # Desired position
                desired = chunk_anchor + np.array([dx, dy, dz], dtype=np.float64)

                # Workspace bounding box clamp
                desired = clamp_workspace(desired)

                # Step-by-step velocity clamp
                step_disp = desired - current_target
                step_dist = np.linalg.norm(step_disp)
                if step_dist > MAX_STEP_DIST_M:
                    desired = current_target + (step_disp / step_dist) * MAX_STEP_DIST_M

                current_target = desired.copy()

                # Publish to Franka
                arm.set_cartesian_pose(current_target, fixed_quat)

                # Handle gripper
                if grip_cmd > 0.5 and gripper_state == 0:
                    try:
                        arm.open_gripper(width=0.08, speed=0.1)
                        gripper_state = 1
                    except Exception as e:
                        rospy.logwarn(f"Gripper open: {e}")
                elif grip_cmd <= 0.4 and gripper_state == 1:
                    try:
                        arm.close_gripper(width=0.04, force=10.0, speed=0.1)
                        gripper_state = 0
                    except Exception as e:
                        rospy.logwarn(f"Gripper close: {e}")

                rate.sleep()

            # Telemetry printout
            actual_p, _ = arm.get_cartesian_pose()
            grip_desc = "OPEN (8cm)" if gripper_state == 1 else "CLOSE/GRASP"
            print(f"  [Loop #{total_loops:02d}] RTT: {latency:4.0f}ms | Pos: [{actual_p[0]:.3f}, {actual_p[1]:.3f}, {actual_p[2]:.3f}] | Grip: {grip_desc}")

    except KeyboardInterrupt:
        print("\n[PAUSE] Closed-loop execution paused by user.")

    finally:
        # Safe hold
        sock.close()
        pos_final, _ = arm.get_cartesian_pose()
        hold = pos_final if pos_final is not None else current_target
        arm.set_cartesian_pose(hold, fixed_quat)

        print("\n" + "=" * 70)
        print("  CLOSED-LOOP EXECUTION TERMINATED")
        print("=" * 70)
        print(f"[*] Total Loops:        {total_loops}")
        print(f"[*] Total Duration:     {time.time() - t_start:.1f} s")
        print(f"[*] Final Position:     x={hold[0]:+.4f}, y={hold[1]:+.4f}, z={hold[2]:+.4f}")
        print("[*] Status:             Holding position safely.")
        print("=" * 70)

        # Optional Return Prompt
        try:
            choice = input("\n[POST-CHECK] Enter 'r' to return to start pose, or any key to stay: ").strip().lower()
            if choice == "r":
                print("[RETURN] Returning smoothly to initial pose...")
                ret_hz = 20.0
                ret_steps = int(2.5 * ret_hz)
                ret_rate = rospy.Rate(ret_hz)
                for step in range(ret_steps + 1):
                    if rospy.is_shutdown():
                        break
                    s = 0.5 * (1.0 - math.cos(math.pi * float(step) / float(ret_steps)))
                    wp = hold + s * (start_pos - hold)
                    arm.set_cartesian_pose(wp, fixed_quat)
                    ret_rate.sleep()
                arm.set_cartesian_pose(start_pos, fixed_quat)
                print("[RETURN] Back at initial pose.")
        except (KeyboardInterrupt, EOFError):
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
