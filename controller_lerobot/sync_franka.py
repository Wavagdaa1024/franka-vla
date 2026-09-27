#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synchronous Franka execution with fresh-state settling and bounded tracking.
Commands are held for one monotonic control period. Independent safety guards
remain authoritative; inference is rejected when the observation has drifted.
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
import uuid

try:
    from rollout_recording import Recorder, file_sha256, matches_request
    from control_safety import hold_period, proportional_limit, checked_snapshot, wait_until_settled
    from timestamped_feedback import TimestampedFeedback
except ImportError:
    from franka_teleop.rollout_recording import Recorder, file_sha256, matches_request
    from franka_teleop.control_safety import hold_period, proportional_limit, checked_snapshot, wait_until_settled
    from franka_teleop.timestamped_feedback import TimestampedFeedback
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
        compute_ee_orientation_error,
        compute_so3_error_vector,
        DEFAULT_EE_ORIENTATION,
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
        compute_ee_orientation_error,
        compute_so3_error_vector,
        DEFAULT_EE_ORIENTATION,
        solve_ee_recovery_joints,
        generate_smooth_recovery_traj,
        DEFAULT_Z_FLOOR,
        FRANKA_JOINT_LIMITS
    )

# Import 1kHz joint velocity controller client
try:
    from Base_franka_joint_velocity_controller import FrankaJointVelocityController as _HardwareController
    class FrankaJointVelocityController(TimestampedFeedback, _HardwareController):
        pass
except ImportError:
    class FrankaJointVelocityController:
        def __init__(self):
            raise RuntimeError("Hardware controller adapter unavailable; refusing live execution")


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
      - 3D SO(3) rotation error exceeds args.max_tilt_deg (e.g. 8.0°) - covers Roll, Pitch, and Yaw!
      - Position drops below args.z_min (table floor)
    If deviated, executes smooth restorative trajectory to return to canonical orientation
    (all 3 rotational DOFs restored, zero tilt and zero yaw twist) and safe height.
    Returns:
      (updated_q, did_recover)
    """
    if not getattr(args, "recover_ee", True) or curr_q is None:
        return curr_q, False

    so3_err_deg, tilt_deg = compute_ee_orientation_error(curr_q, DEFAULT_EE_ORIENTATION)
    T_live = forward_kinematics(curr_q)
    p_live = T_live[:3, 3]

    needs_recovery = (so3_err_deg > args.max_tilt_deg) or (p_live[2] < args.z_min)
    if not needs_recovery:
        return curr_q, False

    reasons = []
    if so3_err_deg > args.max_tilt_deg:
        reasons.append(f"SO(3) error {so3_err_deg:.1f}° > {args.max_tilt_deg:.1f}° (Tilt: {tilt_deg:.1f}°)")
    if p_live[2] < args.z_min:
        reasons.append(f"Z {p_live[2]*1000.0:.1f}mm < {args.z_min*1000.0:.1f}mm")
    reason_str = ", ".join(reasons)

    print(f"\n[POSE GUARD] OOD 3D Orientation Detected ({reason_str})! Restoring all 3 DOFs (Roll, Pitch, Yaw)...")
    q_target, so3_after, pos_err = solve_ee_recovery_joints(
        curr_q, z_floor=args.z_min, max_iters=args.recovery_iters, z_lift=0.003
    )
    dq_traj = generate_smooth_recovery_traj(
        curr_q, q_target, steps=args.recovery_steps, dt=DT, max_vel=args.recovery_vel
    )

    if not args.shadow:
        for dq_cmd in dq_traj:
            if rospy.is_shutdown():
                break
            q_live = checked_snapshot(arm, max_age=getattr(args, "feedback_max_age", 0.1))["q"]
            dq_safe, _ = project_velocity_z_floor(q_live, dq_cmd, dt=DT, z_floor=args.z_min)
            for j in range(7):
                if (q_live[j] + dq_safe[j] * DT) < FRANKA_JOINT_LIMITS[j][0] and dq_safe[j] < 0:
                    dq_safe[j] = 0.0
                elif (q_live[j] + dq_safe[j] * DT) > FRANKA_JOINT_LIMITS[j][1] and dq_safe[j] > 0:
                    dq_safe[j] = 0.0
            command_started = time.monotonic()
            arm.set_joint_velocities(dq_safe)
            hold_period(command_started, DT, shutdown=rospy.is_shutdown)
        arm.stop()
        if args.settle_time > 0:
            rospy.sleep(args.settle_time)
    else:
        arm.set_joint_velocities(np.zeros(7))

    q_final = arm.get_joint_positions()
    p_final = forward_kinematics(q_final)[:3, 3] if q_final is not None else p_live
    so3_final, tilt_final = compute_ee_orientation_error(q_final, DEFAULT_EE_ORIENTATION)
    print(f"[POSE GUARD OK] 3D Recovery Complete: SO(3) Error {so3_final:.2f}° (drift {so3_err_deg - so3_final:.1f}° cleared), Tilt {tilt_final:.2f}° | EE: [{p_final[0]:+.3f}, {p_final[1]:+.3f}, {p_final[2]:+.3f}]m\n")
    return q_final, True


def run_sync_loop(arm, conn, args):
    """
    Pure Synchronous Stop-and-Go Closed-Loop Execution Loop.
    Zero prefetch race conditions, zero buffer starvation.
    """
    recorder = Recorder(getattr(args, "record_dir", None),
                        dict(role="controller", options=vars(args), source_sha256=file_sha256(__file__)))
    session_id = uuid.uuid4().hex
    rate = None  # Motion periods start at each command, never before inference.
    init_grip = read_gripper_normalized(arm, default=0.0)
    gripper_state = 0 if init_grip <= 0.5 else 1
    chunk_idx = 0
    t_start = time.time()
    prev_dq = np.zeros(7, dtype=np.float64)
    close_intent_counter = 0

    print("=" * 80)
    print("  SYNCHRONOUS CLOSED-LOOP STREAMING ACTIVE")
    print(f"  [Config] Steps per Cycle:  {args.sync_steps} steps (~{args.sync_steps * DT:.2f}s motion)")
    print(f"  [Config] Settling Time:    {args.settle_time * 1000:.0f} ms (minimum; verified with feedback)")
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
            settled = wait_until_settled(arm, minimum=max(0.1, args.settle_time),
                timeout=getattr(args, "settle_timeout", 2.0),
                max_age=getattr(args, "feedback_max_age", 0.1), shutdown=rospy.is_shutdown)
            recorder.record("settled", session_id=session_id, loop=chunk_idx, feedback=settled)

            # 2. EE Pose Guard & Recovery: Detect OOD tilt/height and restore before camera capture
            curr_q = settled["q"].copy()
            recovery_before = None if curr_q is None else curr_q.copy()
            did_recover = False
            if curr_q is not None and getattr(args, "recover_ee", True):
                curr_q, did_recover = check_and_restore_ee_pose(arm, curr_q, args, rate)
                if did_recover:
                    prev_dq = np.zeros(7, dtype=np.float64)
                    settled = wait_until_settled(arm, minimum=max(0.1, args.settle_time),
                        timeout=getattr(args, "settle_timeout", 2.0),
                        max_age=getattr(args, "feedback_max_age", 0.1), shutdown=rospy.is_shutdown)
                    curr_q = settled["q"].copy()

            recorder.record("recovery_check", session_id=session_id, loop=chunk_idx,
                q_before=recovery_before, q_after=curr_q, triggered=did_recover,
                orientation_error_deg=None if recovery_before is None else
                    compute_ee_orientation_error(recovery_before, DEFAULT_EE_ORIENTATION))

            # 3. Sample live in-distribution robot state
            curr_pos = forward_kinematics(curr_q)[:3, 3]
            if curr_q is None:
                curr_q = arm.get_joint_positions()
            if curr_pos is None or curr_q is None:
                rospy.sleep(0.02)
                continue

            curr_grip = read_gripper_normalized(arm, default=(1.0 if gripper_state == 1 else 0.0))
            state_msg = {
                "loop": chunk_idx,
                "session_id": session_id,
                "protocol_version": 2,
                "feedback": {k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in settled.items()},
                "q": curr_q.tolist(),
                "pos": curr_pos.tolist(),
                "gripper": curr_grip
            }

            # 3. Request next action chunk from GPU Agent
            recorder.record("request", **state_msg)
            t_infer_start = time.time()
            if not send_json(conn, state_msg):
                print("[ERROR] Failed to send state message to GPU Agent! Exiting.")
                break

            resp = recv_json(conn)
            if resp is None:
                print("[ERROR] GPU Agent connection closed! Exiting.")
                break

            if not matches_request(state_msg, resp):
                recorder.record("response_rejected", session_id=session_id, loop=chunk_idx,
                                reason="session_or_loop_mismatch", response=resp)
                arm.stop()
                print("[ERROR] Response correlation mismatch; update both sync endpoints.")
                break
            if resp.get("hold", False):
                arm.stop()
                close_intent_counter = 0
                recorder.record("hold", session_id=session_id, loop=chunk_idx)
                rospy.sleep(DT)
                continue
            execution_state = checked_snapshot(arm, max_age=getattr(args, "feedback_max_age", 0.1))
            drift_q = float(np.max(np.abs(execution_state["q"] - curr_q)))
            drift_xyz = float(np.linalg.norm(forward_kinematics(execution_state["q"])[:3,3] - curr_pos))
            if (execution_state["source_stamp"] <= settled["source_stamp"]
                    or np.max(np.abs(execution_state["dq"])) > 0.01
                    or drift_q > 0.003 or drift_xyz > 0.003):
                arm.stop()
                close_intent_counter = 0
                print(f"[HOLD] Observation drift: {drift_xyz*1000:.2f}mm, {drift_q:.4f}rad; reobserving.")
                recorder.record("observation_rejected", session_id=session_id, loop=chunk_idx,
                    drift_q_rad=drift_q, drift_xyz_m=drift_xyz, feedback=execution_state)
                continue
            recorder.record("response", session_id=session_id, loop=chunk_idx, response=resp)
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

                before = checked_snapshot(arm, max_age=getattr(args, "feedback_max_age", 0.1))
                curr_q_live = before["q"]
                if step and before["source_stamp"] <= previous_source_stamp:
                    raise RuntimeError("No new robot feedback for next control step")
                previous_source_stamp = before["source_stamp"]

                trace = dict(session_id=session_id, loop=chunk_idx, step=step,
                             q_before=curr_q_live.copy(), q_read_monotonic=time.monotonic(),
                             shadow=args.shadow, feedback_before=before)
                dq_orient = np.zeros(7, dtype=np.float64)
                if use_pos:
                    q_des = np.array(pos_targets[step], dtype=np.float64)
                    pos_err = q_des - curr_q_live
                    dq_target = args.kp_pos * pos_err
                    trace["q_target"] = q_des.copy()
                    trace["dq_tracking"] = dq_target.copy()

                    # Real-time closed-loop 3-DOF orientation locking in position nullspace (Roll, Pitch, Yaw)
                    # Gated strictly by --enable-nullspace for unbiased 7-DOF trajectory execution
                    if getattr(args, "enable_nullspace", False):
                        R_live = forward_kinematics(curr_q_live)[:3, :3]
                        w_rot = compute_so3_error_vector(R_live, DEFAULT_EE_ORIENTATION)
                        if np.linalg.norm(w_rot) > 0.008:  # > 0.5 deg SO(3) 3D error
                            J = analytical_jacobian(curr_q_live)
                            J_v = J[:3, :]
                            J_w = J[3:, :]
                            J_v_pinv = damped_pinv(J_v, damping=1e-4)
                            N_v = np.eye(7, dtype=np.float64) - J_v_pinv @ J_v
                            J_w_null = J_w @ N_v
                            J_w_null_pinv = damped_pinv(J_w_null, damping=1e-3)
                            dq_orient = J_w_null_pinv @ (3.0 * w_rot)
                            dq_orient_norm = np.linalg.norm(dq_orient)
                            if dq_orient_norm > 0.15:
                                dq_orient = dq_orient * (0.15 / dq_orient_norm)
                            dq_target += dq_orient
                else:
                    dq_target = np.array(vels[step], dtype=np.float64)

                trace.setdefault("dq_tracking", dq_target.copy())
                trace["dq_orientation"] = dq_orient.copy()
                trace["dq_before_limits"] = dq_target.copy()
                grip_cmd = float(grips[step]) if step < len(grips) else 0.0
                trace["gripper_intent"] = grip_cmd

                # One scalar preserves coordinated joint ratios. Infeasible turns
                # end the chunk; do not keep advancing targets while braking.
                if args.flip_lr:
                    dq_target[0] = -dq_target[0]
                dq_smooth, limiter = proportional_limit(dq_target, prev_dq,
                    args.max_vel, args.max_acc, DT)
                trace["limiter"] = limiter
                trace["dq_after_speed_flip"] = dq_target * limiter["speed_scale"]
                trace["dq_after_acceleration"] = dq_smooth.copy()
                if limiter["braking"]:
                    arm.stop()
                    close_intent_counter = 0
                    print(f"[HOLD] Cycle {chunk_idx} step {step}: direction infeasible under acceleration limit; reobserving.")
                    recorder.record("tracking_infeasible", **trace)
                    break

                # Joint soft limits protection
                q_pred = curr_q_live + dq_smooth * DT
                for i in range(7):
                    if q_pred[i] < FRANKA_JOINT_LIMITS[i][0] and dq_smooth[i] < 0:
                        dq_smooth[i] = 0.0
                    elif q_pred[i] > FRANKA_JOINT_LIMITS[i][1] and dq_smooth[i] > 0:
                        dq_smooth[i] = 0.0

                trace["dq_after_joint_limits"] = dq_smooth.copy()

                # Floor collision check
                dq_safe, clamped = project_velocity_z_floor(curr_q_live, dq_smooth, dt=DT, z_floor=args.z_min)

                trace["dq_after_floor_projection"] = dq_safe.copy()
                trace["floor_projection_active"] = bool(clamped)

                # Final joint limit verification after floor projection
                for i in range(7):
                    if (curr_q_live[i] + dq_safe[i] * DT) < FRANKA_JOINT_LIMITS[i][0] and dq_safe[i] < 0:
                        dq_safe[i] = 0.0
                    elif (curr_q_live[i] + dq_safe[i] * DT) > FRANKA_JOINT_LIMITS[i][1] and dq_safe[i] > 0:
                        dq_safe[i] = 0.0

                trace["dq_after_final_joint_limits"] = dq_safe.copy()
                trace["floor_hard_stop"] = False

                # Absolute floor guard post-check
                if forward_kinematics(curr_q_live + dq_safe * DT)[:3, 3][2] < args.z_min:
                    dq_safe = np.zeros(7, dtype=np.float64)
                    trace["floor_hard_stop"] = True

                trace["dq_after_floor_stop"] = dq_safe.copy()

                # Soft deceleration near cycle end (steps 13 & 14)
                if getattr(args, "end_decel", "step") == "step":
                    rem = steps_to_exec - 1 - step
                    if rem == 1:
                        dq_safe *= 0.5
                    elif rem == 0:
                        dq_safe *= 0.2

                trace["dq_command"] = dq_safe.copy()
                trace["command_monotonic"] = time.monotonic()
                if not args.shadow:
                    arm.set_joint_velocities(dq_safe)
                trace["command_sent"] = not args.shadow
                trace["command_return_monotonic"] = time.monotonic()
                prev_dq = dq_safe.copy()

                # Gripper actuation (configurable sensitivity threshold, default: 0.35)
                grip_thresh = getattr(args, "grip_thresh", 0.35)
                if grip_cmd > grip_thresh:
                    close_intent_counter += 1
                    if not args.shadow and gripper_state == 0 and close_intent_counter >= args.close_delay_steps:
                        grip_context = dict(session_id=session_id, loop=chunk_idx, step=step, command="close")
                        recorder.record("gripper_request", **grip_context)
                        def _do_close(context=grip_context.copy()):
                            try:
                                recorder.record("gripper_call", **context)
                                result = arm.close_gripper(width=0.04, force=15.0, speed=0.35, inner_epsilon=0.025, outer_epsilon=0.025)
                                recorder.record("gripper_return", **context, result=repr(result))
                            except Exception as e:
                                recorder.record("gripper_error", **context, error=repr(e))
                                rospy.logwarn(f"Gripper close: {e}")
                        threading.Thread(target=_do_close, daemon=True).start()
                        gripper_state = 1
                elif grip_cmd < (grip_thresh * 0.5):
                    close_intent_counter = 0
                    if not args.shadow and gripper_state == 1:
                        grip_context = dict(session_id=session_id, loop=chunk_idx, step=step, command="open")
                        recorder.record("gripper_request", **grip_context)
                        def _do_open(context=grip_context.copy()):
                            try:
                                recorder.record("gripper_call", **context)
                                result = arm.open_gripper(width=0.08, speed=0.35)
                                recorder.record("gripper_return", **context, result=repr(result))
                            except Exception as e:
                                recorder.record("gripper_error", **context, error=repr(e))
                                rospy.logwarn(f"Gripper open: {e}")
                        threading.Thread(target=_do_open, daemon=True).start()
                        gripper_state = 0

                recorder.record("control_step", **trace)
                hold_period(trace["command_monotonic"], DT, shutdown=rospy.is_shutdown)
                after = checked_snapshot(arm, max_age=getattr(args, "feedback_max_age", 0.1))
                q_after = after["q"]
                recorder.record("feedback_read", session_id=session_id, loop=chunk_idx, step=step,
                    q_after=q_after, gripper_width=arm.get_gripper_width(),
                    feedback=after, sensor_timestamp_available=True)

            arm.stop()
            prev_dq = np.zeros(7, dtype=np.float64)

    except KeyboardInterrupt:
        print("\n[PAUSE] Sync loop stopped by user.")
    except Exception as e:
        print(f"\n[WARN] Sync loop exception: {e}")

    finally:
        arm.stop()
        recorder.close()
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
    parser.add_argument("--grip-thresh", type=float, default=0.35,
                        help="Gripper closing activation threshold in [0, 1] (default: 0.35)")
    parser.add_argument("--flip-lr", action="store_true", default=False,
                        help="Invert Joint 0 (base yaw)")
    parser.add_argument("--enable-nullspace", action="store_true", default=False,
                        help="Enable artificial vertical orientation locking in nullspace (ablation only, default: False)")
    parser.add_argument("--end-decel", type=str, choices=("step", "none"), default="step",
                        help="Cycle end deceleration mode (default: step)")
    parser.add_argument("--shadow", action="store_true", default=False,
                        help="Shadow mode: log commands without moving physical robot")
    parser.add_argument("--feedback-max-age", type=float, default=0.1,
                        help="Maximum host age of timestamped feedback in seconds")
    parser.add_argument("--settle-timeout", type=float, default=2.0,
                        help="Maximum time to confirm stationary fresh feedback")
    parser.add_argument("--record-dir", default=None, help="Opt-in bounded asynchronous execution trace")
    args = parser.parse_args()
    if not (0 < args.feedback_max_age <= 0.5 and args.settle_timeout >= max(.1, args.settle_time)
            and args.max_vel > 0 and args.max_acc > 0 and args.sync_steps > 0):
        parser.error("Invalid feedback, settling or motion limits")

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
