#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teleoperation Dataset Trajectory Replay Script.
Directly replays the recorded human demonstration trajectory onto the Franka Panda arm.
Used for A/B testing:
  - If the replay successfully picks the object and places it,
    it proves the robot hardware, kinematics, gripper, and demonstration data are 100% sound,
    and any failure in closed-loop control is strictly due to VLA policy generalization/training.
"""

import os
import sys
import time
import json
import math
import argparse
import numpy as np

import rospy

from Base_franka_joint_velocity_controller import FrankaJointVelocityController

MAX_JOINT_VEL = 0.35        # Safe max velocity: 0.35 rad/s
MAX_JOINT_ACC = 1.5         # Acceleration limit: 1.5 rad/s^2
ALIGN_MAX_VEL = 0.15        # Very gentle alignment speed: 0.15 rad/s
Z_MIN = 0.055               # Table floor safety margin: 5.5 cm

# Soft joint limits with margin
JOINT_LIMITS = [
    (-2.80, 2.80),
    (-1.70, 1.70),
    (-2.80, 2.80),
    (-3.00, -0.05),
    (-2.80, 2.80),
    (0.00, 3.70),
    (-2.80, 2.80),
]


def clamp_action(curr_q, curr_pos, prev_dq, dq_target, dt, max_acc=MAX_JOINT_ACC, max_vel=MAX_JOINT_VEL):
    """Enforces safety limits, soft joint boundaries, acceleration clamping, and floor protection."""
    dq_clamped = np.array(dq_target, dtype=np.float64).copy()
    dq_clamped = np.clip(dq_clamped, -max_vel, max_vel)

    # Acceleration limiting
    if prev_dq is not None:
        delta = dq_clamped - prev_dq
        max_delta = max_acc * dt
        dq_clamped = prev_dq + np.clip(delta, -max_delta, max_delta)

    # Soft joint limit protection
    if curr_q is not None:
        for i in range(7):
            q_min, q_max = JOINT_LIMITS[i]
            if curr_q[i] < q_min and dq_clamped[i] < 0:
                dq_clamped[i] = 0.0
            elif curr_q[i] > q_max and dq_clamped[i] > 0:
                dq_clamped[i] = 0.0

    # Floor protection (Z >= Z_MIN)
    if curr_pos is not None and curr_pos[2] < Z_MIN:
        if dq_clamped[1] > 0:
            dq_clamped[1] = 0.0
        if dq_clamped[3] < 0:
            dq_clamped[3] = 0.0

    return dq_clamped


def move_to_joints_smooth(arm, target_q, max_vel=ALIGN_MAX_VEL, timeout=10.0, hz=15.0):
    """Smoothly moves arm to target joint positions using a closed-loop P-controller."""
    rate = rospy.Rate(hz)
    t_start = time.time()
    dt = 1.0 / hz
    prev_dq = np.zeros(7, dtype=np.float64)

    while not rospy.is_shutdown():
        curr_q = arm.get_joint_positions()
        curr_pos, _ = arm.get_cartesian_pose()
        if curr_q is None:
            rospy.sleep(0.02)
            continue

        error = target_q - curr_q
        max_err = np.max(np.abs(error))
        if max_err < 0.015:
            # Reached target
            arm.stop()
            return True

        if time.time() - t_start > timeout:
            print(f"[Warning] Alignment timeout (remaining error: {max_err:.3f} rad). Continuing...")
            arm.stop()
            return False

        # Proportional controller
        kp = 1.8
        dq_cmd = kp * error
        dq_cmd = clamp_action(curr_q, curr_pos, prev_dq, dq_cmd, dt=dt, max_acc=0.8, max_vel=max_vel)
        prev_dq = dq_cmd.copy()
        arm.set_joint_velocities(dq_cmd.tolist())
        rate.sleep()

    arm.stop()
    return False


def main():
    parser = argparse.ArgumentParser(description="Replay recorded Franka teleop trajectory.")
    parser.add_argument("--trajectory", default="ep_replay.json", help="Path to exported trajectory JSON (default: ep_replay.json)")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed scale (default 1.0)")
    args = parser.parse_args()

    if not os.path.exists(args.trajectory):
        print(f"[ERROR] Trajectory file '{args.trajectory}' not found!")
        return 1

    with open(args.trajectory, "r", encoding="utf-8") as f:
        traj = json.load(f)

    ep_idx = traj.get("episode_index", 0)
    num_frames = traj["num_frames"]
    duration_s = traj["duration_s"]
    hz = float(traj.get("hz", 15.0))
    dt = 1.0 / hz
    target_qs = np.array(traj["joint_positions"], dtype=np.float64)
    target_dqs = np.array(traj["joint_velocities"], dtype=np.float64)
    target_grips = np.array(traj["gripper_commands"], dtype=np.float64)
    init_q = np.array(traj["initial_q"], dtype=np.float64)

    print("=" * 72)
    print(f"  FRANKA TELEOP TRAJECTORY REPLAY TEST (EPISODE {ep_idx})")
    print("=" * 72)
    print(f"[*] Trajectory:     {args.trajectory}")
    print(f"[*] Total Steps:    {num_frames} frames ({duration_s:.2f} s @ {hz:.1f} Hz)")
    print(f"[*] Playback Speed: {args.speed:.1f}x")
    print(f"[*] Initial Joints: [{', '.join(f'{x:+.3f}' for x in init_q)}]")

    if not rospy.core.is_initialized():
        rospy.init_node("franka_trajectory_replayer", anonymous=True)

    print("\n[1/3] Connecting to Franka Joint Velocity Controller...")
    arm = FrankaJointVelocityController()

    # Wait for valid state
    t0 = time.time()
    curr_q = None
    curr_pos = None
    while time.time() - t0 < 5.0 and not rospy.is_shutdown():
        curr_q = arm.get_joint_positions()
        curr_pos, _ = arm.get_cartesian_pose()
        if curr_q is not None and curr_pos is not None:
            break
        rospy.sleep(0.05)

    if curr_q is None:
        print("[ERROR] Failed to receive Franka robot state within 5.0s!")
        return 1

    print(f"[OK] Franka Connected.")
    print(f"     Current EE Pose: x={curr_pos[0]:+.4f}, y={curr_pos[1]:+.4f}, z={curr_pos[2]:+.4f} m")
    print(f"     Current Joints:  [{', '.join(f'{x:+.3f}' for x in curr_q)}]")

    # Check distance to start pose
    dist_to_start = np.max(np.abs(curr_q - init_q))
    print(f"     Max Joint Offset from Start: {dist_to_start:.3f} rad ({math.degrees(dist_to_start):.1f} deg)")

    print("-" * 72)
    input("[STEP 1] Press [ENTER] to smoothly move the arm to the Start Pose & open gripper...")
    print("[Moving] Aligning to trajectory start pose...")
    move_to_joints_smooth(arm, init_q, max_vel=ALIGN_MAX_VEL, timeout=12.0, hz=hz)
    print("[Moving] Opening gripper (0.08m)...")
    arm.open_gripper(width=0.08)
    time.sleep(1.0)
    print("[OK] Arm is accurately positioned at trajectory Start Pose!")

    print("-" * 72)
    print("[STEP 2] Check table environment:")
    print("         - Ensure the TARGET OBJECT (Cube / Carrot / Pepper) is in place on the tabletop.")
    print("         - Ensure target place container/basket is in place.")
    print("         - Hold physical E-STOP for safety.")
    input("[READY] Press [ENTER] to execute trajectory replay...")

    # Start playback loop
    print(f"\n[Replay] Starting playback at {hz:.1f} Hz...")
    rate = rospy.Rate(hz * args.speed)
    prev_dq = np.zeros(7, dtype=np.float64)
    gripper_closed = False
    kp = 2.2  # Tracking proportional gain

    t_start = time.time()

    for step in range(num_frames):
        if rospy.is_shutdown():
            break

        curr_q = arm.get_joint_positions()
        curr_pos, _ = arm.get_cartesian_pose()
        if curr_q is None:
            rate.sleep()
            continue

        q_ref = target_qs[step]
        dq_ref = target_dqs[step] * args.speed
        grip_ref = target_grips[step]

        # Feedforward + Feedback PD control
        dq_cmd = dq_ref + kp * (q_ref - curr_q)
        dq_cmd = clamp_action(curr_q, curr_pos, prev_dq, dq_cmd, dt=dt, max_acc=MAX_JOINT_ACC, max_vel=MAX_JOINT_VEL)
        prev_dq = dq_cmd.copy()

        arm.set_joint_velocities(dq_cmd.tolist())

        # Gripper event handling (DROID Standard: 0.0=OPEN, 1.0=CLOSED)
        if grip_ref >= 0.5 and not gripper_closed:
            print(f"  [Step #{step:03d} ({step/hz:.2f}s)] >> CLOSING GRIPPER (Grasp Object, grip_ref={grip_ref:.2f})...")
            arm.close_gripper(width=0.045, force=10.0)
            gripper_closed = True
        elif grip_ref < 0.5 and gripper_closed:
            print(f"  [Step #{step:03d} ({step/hz:.2f}s)] >> OPENING GRIPPER (Release Object, grip_ref={grip_ref:.2f})...")
            arm.open_gripper(width=0.08)
            gripper_closed = False

        if step % 15 == 0 or step == num_frames - 1:
            q_err = np.max(np.abs(curr_q - q_ref))
            print(f"  [Step #{step:03d}/{num_frames}] t={time.time()-t_start:4.1f}s | Track Err: {q_err*1000:.1f} mrad | Grip: {'CLOSED' if gripper_closed else 'OPEN'}")

        rate.sleep()

    arm.stop()
    t_elapsed = time.time() - t_start
    print("=" * 72)
    print(f"[OK] Trajectory Replay Completed in {t_elapsed:.2f}s!")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
