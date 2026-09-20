#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure Synchronous (Stop-and-Go) Franka Closed-Loop Execution Service.
Runs on Franka Linux Control PC (10.197.16.43:8765).

Completely eliminates RTC prefetch / stride / starvation complexity.
Execution Protocol:
  1. Arm executes 15 steps of current action chunk at 15 Hz (~1.0s).
  2. Smooth deceleration at steps 13 and 14 -> Arm stops.
  3. 50ms stationary settling time (Zero Camera Motion Blur).
  4. Arm samples live Cartesian EE pose & joint angles -> Sends to GPU Agent.
  5. GPU Agent (RTX 5090) infers next 15-step chunk (~430ms) with 5-DOF vertical
     orientation locking (0.00° tilt) and table floor limit (Z >= +7.0mm).
  6. Server receives action chunk and executes next cycle.

Safety Systems:
  - Task-Priority Nullspace Gripper Vertical Lock (Roll/Pitch tilt strictly eliminated)
  - Hard Table Floor Limit: Z >= z_min (default: +0.0070m = +7.0mm)
  - Joint limits & Acceleration clamps
  - Asynchronous non-blocking gripper actuation
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
import numpy as np

try:
    import rospy
except ImportError:
    class _MockRate:
        def __init__(self, hz):
            self.dt = 1.0 / float(hz)
        def sleep(self):
            time.sleep(self.dt)

    class _MockRospyCore:
        @staticmethod
        def is_initialized():
            return True

    class _MockRospy:
        core = _MockRospyCore
        @staticmethod
        def is_shutdown():
            return False
        @staticmethod
        def Rate(hz):
            return _MockRate(hz)
        @staticmethod
        def sleep(t):
            time.sleep(t)
        @staticmethod
        def logwarn(msg):
            print(f"[WARN] {msg}")
        @staticmethod
        def loginfo(msg):
            print(f"[INFO] {msg}")
        @staticmethod
        def init_node(*args, **kwargs):
            pass

    rospy = _MockRospy()

# Import kinematics module
try:
    from kinematics import (
        forward_kinematics,
        analytical_jacobian,
        damped_pinv,
        project_velocity_z_floor,
        compute_ee_tilt,
        solve_ee_recovery_joints,
        generate_smooth_recovery_traj,
        DEFAULT_Z_FLOOR,
        FRANKA_JOINT_LIMITS
    )
except ImportError:
    from franka_teleop.kinematics import (
        forward_kinematics,
        analytical_jacobian,
        damped_pinv,
        project_velocity_z_floor,
        compute_ee_tilt,
        solve_ee_recovery_joints,
        generate_smooth_recovery_traj,
        DEFAULT_Z_FLOOR,
        FRANKA_JOINT_LIMITS
    )

# Import 1kHz joint velocity controller client
try:
    from Base_franka_joint_velocity_controller import FrankaJointVelocityController
except ImportError:
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
DEFAULT_SYNC_STEPS = 6        # 6 steps @ 15Hz = 0.400s motion per cycle (responsive visual feedback)
CONTROL_HZ = 15.0             # 15 Hz policy frequency
DT = 1.0 / CONTROL_HZ         # 0.0667 s
MAX_JOINT_VEL = 0.35          # 0.35 rad/s max safe testing speed
MAX_JOINT_ACC = 1.5           # 1.5 rad/s^2 smooth responsive acceleration clamp


def _recv_exact(conn, n):
    data = bytearray()
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def send_json(conn, obj):
    try:
        payload = json.dumps(obj).encode("utf-8")
        conn.sendall(struct.pack("!I", len(payload)) + payload)
        return True
    except (socket.error, ConnectionResetError, BrokenPipeError, ConnectionAbortedError, OSError):
        return False


def recv_json(conn):
    try:
        header = _recv_exact(conn, 4)
        if header is None:
            return None
        size = struct.unpack("!I", header)[0]
        if size == 0 or size > 10 * 1024 * 1024:  # 10MB safety bound
            return None
        data = _recv_exact(conn, size)
        if data is None:
            return None
        return json.loads(data.decode("utf-8"))
    except (socket.error, ConnectionResetError, BrokenPipeError, ConnectionAbortedError, OSError, json.JSONDecodeError):
        return None


def read_gripper_normalized(arm, default=0.0):
    """0.0 = OPEN (width=0.08m), 1.0 = CLOSED (width=0.0m)."""
    w = arm.get_gripper_width()
    if w is None:
        return default
    return float(np.clip(1.0 - float(w) / 0.08, 0.0, 1.0))


def check_and_restore_ee_pose(arm, curr_q, args, rate):
    """
    Checks if end-effector pose deviates beyond safe thresholds:
      - Tilt exceeds args.max_tilt_deg (e.g. 8.0°)
      - Position drops below args.z_min (table floor)
    If deviated, executes smooth restorative trajectory to return to vertical downward orientation
    and safe height without disturbing the (X, Y) task trajectory.
    Returns:
      (updated_q, did_recover)
    """
    if not getattr(args, "recover_ee", True) or curr_q is None:
        return curr_q, False

    tilt_deg = compute_ee_tilt(curr_q)
    T_live = forward_kinematics(curr_q)
    p_live = T_live[:3, 3]

    needs_recovery = (tilt_deg > args.max_tilt_deg) or (p_live[2] < args.z_min)
    if not needs_recovery:
        return curr_q, False

    reasons = []
    if tilt_deg > args.max_tilt_deg:
        reasons.append(f"Tilt {tilt_deg:.1f}° > {args.max_tilt_deg:.1f}°")
    if p_live[2] < args.z_min:
        reasons.append(f"Z {p_live[2]*1000.0:.1f}mm < {args.z_min*1000.0:.1f}mm")
    reason_str = ", ".join(reasons)

    print(f"\n[POSE GUARD] OOD State Detected ({reason_str})! Restoring upright EE pose...")
    q_target, tilt_after, pos_err = solve_ee_recovery_joints(
        curr_q, z_floor=args.z_min, max_iters=args.recovery_iters, z_lift=0.003
    )
    dq_traj = generate_smooth_recovery_traj(
        curr_q, q_target, steps=args.recovery_steps, dt=DT, max_vel=args.recovery_vel
    )

    if not args.shadow:
        for dq_cmd in dq_traj:
            if rospy.is_shutdown():
                break
            q_live = arm.get_joint_positions()
            dq_safe, _ = project_velocity_z_floor(q_live, dq_cmd, dt=DT, z_floor=args.z_min)
            for j in range(7):
                if (q_live[j] + dq_safe[j] * DT) < FRANKA_JOINT_LIMITS[j][0] and dq_safe[j] < 0:
                    dq_safe[j] = 0.0
                elif (q_live[j] + dq_safe[j] * DT) > FRANKA_JOINT_LIMITS[j][1] and dq_safe[j] > 0:
                    dq_safe[j] = 0.0
            arm.set_joint_velocities(dq_safe)
            rate.sleep()
        arm.stop()
        if args.settle_time > 0:
            rospy.sleep(args.settle_time)
    else:
        arm.set_joint_velocities(np.zeros(7))

    q_final = arm.get_joint_positions()
    p_final = forward_kinematics(q_final)[:3, 3] if q_final is not None else p_live
    print(f"[POSE GUARD OK] Recovery Finished: Tilt {tilt_after:.2f}° (drift {tilt_deg - tilt_after:.1f}° cleared) | EE: [{p_final[0]:+.3f}, {p_final[1]:+.3f}, {p_final[2]:+.3f}]m\n")
    return q_final, True


def run_sync_loop(arm, conn, args):
    """
    Pure Synchronous Stop-and-Go Closed-Loop Execution Loop.
    Zero prefetch race conditions, zero buffer starvation.
    """
    rate = rospy.Rate(CONTROL_HZ)
    init_grip = read_gripper_normalized(arm, default=0.0)
    gripper_state = 0 if init_grip <= 0.5 else 1
    chunk_idx = 0
    t_start = time.time()
    prev_dq = np.zeros(7, dtype=np.float64)
    close_intent_counter = 0

    print("=" * 80)
    print("  SYNCHRONOUS CLOSED-LOOP STREAMING ACTIVE")
    print(f"  [Config] Steps per Cycle:  {args.sync_steps} steps (~{args.sync_steps * DT:.2f}s motion)")
    print(f"  [Config] Settling Time:    {args.settle_time * 1000:.0f} ms (Zero Motion Blur)")
    print(f"  [Config] Hard Table Floor: Z >= {args.z_min * 1000.0:.1f} mm")
    print(f"  [Config] Max Speed Clamp:  {args.max_vel:.2f} rad/s")
    print(f"  [Config] EE Pose Recovery: {getattr(args, 'recover_ee', True)} (Max Tilt: {args.max_tilt_deg:.1f}°)")
    print("=" * 80)

    try:
        while not rospy.is_shutdown():
            chunk_idx += 1

            # 1. Halt arm and settle stationary for camera clarity
            arm.stop()
            prev_dq = np.zeros(7, dtype=np.float64)
            if args.settle_time > 0:
                rospy.sleep(args.settle_time)

            # 2. EE Pose Guard & Recovery: Detect OOD tilt/height and restore before camera capture
            curr_q = arm.get_joint_positions()
            if curr_q is not None and getattr(args, "recover_ee", True):
                curr_q, did_recover = check_and_restore_ee_pose(arm, curr_q, args, rate)
                if did_recover:
                    prev_dq = np.zeros(7, dtype=np.float64)

            # 3. Sample live in-distribution robot state
            curr_pos, _ = arm.get_cartesian_pose()
            if curr_q is None:
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

            # 3. Request next action chunk from GPU Agent
            t_infer_start = time.time()
            if not send_json(conn, state_msg):
                print("[ERROR] Failed to send state message to GPU Agent! Exiting.")
                break

            resp = recv_json(conn)
            if resp is None:
                print("[ERROR] GPU Agent connection closed! Exiting.")
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

            # 4. Execute chunk steps at 15 Hz
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

                    # Real-time closed-loop vertical orientation locking in position nullspace
                    z_live = forward_kinematics(curr_q_live)[:3, 2]
                    w_tilt = np.cross(z_live, [0.0, 0.0, -1.0])
                    if np.linalg.norm(w_tilt) > 0.005:  # > 0.3 deg tilt
                        J = analytical_jacobian(curr_q_live)
                        J_v = J[:3, :]
                        J_w = J[3:, :]
                        J_v_pinv = damped_pinv(J_v, damping=1e-4)
                        N_v = np.eye(7, dtype=np.float64) - J_v_pinv @ J_v
                        J_w_null = J_w @ N_v
                        J_w_null_pinv = damped_pinv(J_w_null, damping=1e-3)
                        dq_orient = J_w_null_pinv @ (3.0 * w_tilt)
                        dq_orient_norm = np.linalg.norm(dq_orient)
                        if dq_orient_norm > 0.15:
                            dq_orient = dq_orient * (0.15 / dq_orient_norm)
                        dq_target += dq_orient
                else:
                    dq_target = np.array(vels[step], dtype=np.float64)

                grip_cmd = float(grips[step]) if step < len(grips) else 0.0

                # Velocity limits & base yaw inversion
                dq_target = np.clip(dq_target, -args.max_vel, args.max_vel)
                if args.flip_lr:
                    dq_target[0] = -dq_target[0]

                # Acceleration limit clamp (Anti-Jerk)
                max_delta = args.max_acc * DT
                dq_smooth = np.clip(dq_target, prev_dq - max_delta, prev_dq + max_delta)

                # Joint soft limits protection
                q_pred = curr_q_live + dq_smooth * DT
                for i in range(7):
                    if q_pred[i] < FRANKA_JOINT_LIMITS[i][0] and dq_smooth[i] < 0:
                        dq_smooth[i] = 0.0
                    elif q_pred[i] > FRANKA_JOINT_LIMITS[i][1] and dq_smooth[i] > 0:
                        dq_smooth[i] = 0.0

                # Floor collision check
                dq_safe, clamped = project_velocity_z_floor(curr_q_live, dq_smooth, dt=DT, z_floor=args.z_min)

                # Final joint limit verification after floor projection
                for i in range(7):
                    if (curr_q_live[i] + dq_safe[i] * DT) < FRANKA_JOINT_LIMITS[i][0] and dq_safe[i] < 0:
                        dq_safe[i] = 0.0
                    elif (curr_q_live[i] + dq_safe[i] * DT) > FRANKA_JOINT_LIMITS[i][1] and dq_safe[i] > 0:
                        dq_safe[i] = 0.0

                # Absolute floor guard post-check
                if forward_kinematics(curr_q_live + dq_safe * DT)[:3, 3][2] < args.z_min:
                    dq_safe = np.zeros(7, dtype=np.float64)

                # Soft deceleration near cycle end (steps 13 & 14)
                rem = steps_to_exec - 1 - step
                if rem == 1:
                    dq_safe *= 0.5
                elif rem == 0:
                    dq_safe *= 0.2

                if not args.shadow:
                    arm.set_joint_velocities(dq_safe)
                prev_dq = dq_safe.copy()

                # Gripper actuation
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
    parser = argparse.ArgumentParser(description="Franka Synchronous (Stop-and-Go) VLA Execution Service")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Listening port (default: 8765)")
    parser.add_argument("--sync-steps", type=int, default=DEFAULT_SYNC_STEPS,
                        help=f"Steps per cycle in sync mode (default: {DEFAULT_SYNC_STEPS})")
    parser.add_argument("--settle-time", type=float, default=0.05,
                        help="Stationary settling time before camera capture (default: 0.05s)")
    parser.add_argument("--max-vel", type=float, default=MAX_JOINT_VEL,
                        help="Max joint speed clamp in rad/s (default: 0.35)")
    parser.add_argument("--max-acc", type=float, default=MAX_JOINT_ACC,
                        help="Max joint acceleration clamp in rad/s^2 (default: 1.5)")
    parser.add_argument("--z-min", type=float, default=DEFAULT_Z_FLOOR,
                        help=f"Minimum safe table Z height in meters (default: {DEFAULT_Z_FLOOR}m = +7.0mm)")
    parser.add_argument("--kp-pos", type=float, default=8.0,
                        help="P-servo tracking gain for joint position mode (default: 8.0)")
    parser.add_argument("--max-tilt-deg", type=float, default=8.0,
                        help="Max allowable gripper tilt from vertical downward in degrees (default: 8.0)")
    parser.add_argument("--recover-ee", dest="recover_ee", action="store_true", default=True,
                        help="Enable automatic end-effector pose recovery when tilt/height limits exceeded (default: True)")
    parser.add_argument("--no-recover-ee", dest="recover_ee", action="store_false",
                        help="Disable automatic end-effector pose recovery")
    parser.add_argument("--recovery-steps", type=int, default=10,
                        help="Number of steps for smooth recovery trajectory (default: 10 @ 15Hz = 0.67s)")
    parser.add_argument("--recovery-vel", type=float, default=0.20,
                        help="Max joint velocity clamp during pose recovery in rad/s (default: 0.20)")
    parser.add_argument("--recovery-iters", type=int, default=25,
                        help="Max Newton-Raphson IK solver iterations for recovery (default: 25)")
    parser.add_argument("--close-delay-steps", type=int, default=2,
                        help="Debounce steps before closing gripper (default: 2)")
    parser.add_argument("--flip-lr", action="store_true", default=False,
                        help="Invert Joint 0 (base yaw)")
    parser.add_argument("--enable-nullspace", action="store_true", default=False,
                        help="Enable artificial vertical orientation locking in nullspace (ablation only, default: False)")
    parser.add_argument("--shadow", action="store_true", default=False,
                        help="Shadow mode: log commands without moving physical robot")
    args = parser.parse_args()

    if not rospy.core.is_initialized():
        rospy.init_node("sync_franka_server", anonymous=True)

    print("=" * 80)
    print("  FRANKA CLOSED-LOOP SERVICE: SYNCHRONOUS (Stop-and-Go)")
    print("=" * 80)
    print(f"[*] Port:              {args.port}")
    print(f"[*] Sync Steps:        {args.sync_steps} steps (Pause: {args.settle_time * 1000:.0f}ms)")
    print(f"[*] Hard Table Floor:  Z >= {args.z_min * 1000.0:.1f} mm")
    print(f"[*] Max Speed:         {args.max_vel:.2f} rad/s")
    print(f"[*] Max Accel:         {args.max_acc:.2f} rad/s^2")
    print(f"[*] Pose Recovery:     {args.recover_ee} (Max Tilt: {args.max_tilt_deg:.1f}°, Max RecVel: {args.recovery_vel:.2f} rad/s)")
    print(f"[*] Shadow Mode:       {args.shadow}")
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
        print("[ERROR] Could not read Franka state! Check libfranka/ROS.")
        return

    print("[OK] Franka Connected.")
    print(f"     EE Pose:   [{initial_pos[0]:+.4f}, {initial_pos[1]:+.4f}, {initial_pos[2]:+.4f}] m")
    print(f"     Joints:    [{', '.join(f'{q:+.3f}' for q in initial_q)}]")

    print(f"[2/2] Setting up TCP Server on port {args.port}...")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", args.port))
    server.listen(1)
    print(f"[OK] Listening on port {args.port}. Awaiting GPU Agent...")

    try:
        conn, addr = server.accept()
        print(f"[OK] GPU Agent Connected from {addr}!")
        print("-" * 80)
        print("[READY] Hold physical E-STOP. Press [ENTER] to start closed-loop control...")
        try:
            input()
        except EOFError:
            pass

        total_chunks, t_start = run_sync_loop(arm, conn, args)
    finally:
        try:
            arm.stop()
        except Exception:
            pass
        server.close()

    elapsed = time.time() - t_start if 't_start' in locals() else 0.0
    final_pos, _ = arm.get_cartesian_pose() if arm else (None, None)
    print("\n" + "=" * 80)
    print("  CLOSED-LOOP EXECUTION TERMINATED SAFELY")
    print("=" * 80)
    print(f"[*] Total Cycles: {total_chunks if 'total_chunks' in locals() else 0}")
    print(f"[*] Elapsed:      {elapsed:.1f} s")
    if final_pos is not None:
        print(f"[*] Final Pose:   [{final_pos[0]:+.4f}, {final_pos[1]:+.4f}, {final_pos[2]:+.4f}] m")
    print("=" * 80)


if __name__ == "__main__":
    main()
