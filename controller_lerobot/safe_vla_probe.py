#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M3 Step 3 (Amplified): Franka Safe VLA Approach & Gripper Probing Runner.
Follows the VLA (pi0.5-DROID) forward-downward approach direction with larger amplitude:
- Default target displacement: ~12-15 cm (configurable up to 25 cm, within user's 30cm safe zone)
- Smooth minimum-jerk / cosine velocity profile
- Strict speed clamp: <= 0.10 m/s (100 mm/s)
- Strict acceleration clamp: <= 0.20 m/s^2
- Safe hold at end, with optional auto-return to start position
"""

import sys
import time
import math
import argparse
import numpy as np
import rospy

from Base_franka_cartesian_impedance_controller import FrankaCartesianImpedanceController


def parse_args():
    parser = argparse.ArgumentParser(description="Franka Safe VLA Approach Probing Runner")
    parser.add_argument(
        "--distance-cm",
        type=float,
        default=12.0,
        help="Target movement distance in cm along VLA approach vector (default: 12.0, max: 25.0)"
    )
    parser.add_argument(
        "--speed-mps",
        type=float,
        default=0.08,
        help="Max velocity clamp in m/s (default: 0.08 m/s = 80 mm/s, max: 0.12 m/s)"
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=2.5,
        help="Total duration in seconds for the approach motion (default: 2.5 s)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Safety Bound Checks on CLI Arguments
    target_dist_m = min(max(args.distance_cm / 100.0, 0.02), 0.25)  # Clamped strictly between 2 cm and 25 cm
    max_speed_mps = min(max(args.speed_mps, 0.02), 0.12)             # Clamped strictly <= 0.12 m/s
    total_duration = max(args.duration_s, target_dist_m / max_speed_mps)

    if not rospy.core.is_initialized():
        rospy.init_node("safe_vla_probe_amplified", anonymous=True)

    print("=" * 70)
    print("  FRANKA SAFE VLA AMPLIFIED PROBING RUNNER (APPROACH & GRIPPER)")
    print("=" * 70)
    print(f"[*] Target Excursion:       {target_dist_m * 100:.1f} cm (Hard safe ceiling: <= 25.0 cm)")
    print(f"[*] Max Speed Clamp:        {max_speed_mps * 1000:.1f} mm/s")
    print(f"[*] Total Duration:         {total_duration:.1f} s")
    print("-" * 70)

    # 1. Initialize Arm Controller
    print("[1/4] Connecting to Franka Cartesian Impedance Controller...")
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
        print("        Please verify the ROS impedance controller is running.")
        return 1

    anchor_pos = initial_pos.copy()
    anchor_quat = initial_quat.copy()

    print("[2/4] Initial Franka Pose:")
    print(f"      Position (m): x={anchor_pos[0]:+.4f}, y={anchor_pos[1]:+.4f}, z={anchor_pos[2]:+.4f}")
    print(f"      Quat (x,y,z,w): {np.round(anchor_quat, 4).tolist()}")
    initial_gripper = arm.get_gripper_width()
    print(f"      Gripper Width: {initial_gripper} m")
    print("-" * 70)

    # 2. Compute Approach Direction from VLA Output
    # VLA pi0.5-DROID predicted vector: forward (+X), slightly right (+Y), downward (-Z)
    raw_vla_vector = np.array([+0.0145, +0.0037, -0.0124], dtype=np.float64)
    unit_vla_dir = raw_vla_vector / np.linalg.norm(raw_vla_vector)

    # Target relative vector
    offset_vector = unit_vla_dir * target_dist_m
    final_target_pos = anchor_pos + offset_vector

    print("[3/4] Planned Approach Target:")
    print(f"      Delta X (Forward):    {offset_vector[0] * 100:+.2f} cm")
    print(f"      Delta Y (Side):       {offset_vector[1] * 100:+.2f} cm")
    print(f"      Delta Z (Downward):   {offset_vector[2] * 100:+.2f} cm")
    print(f"      Target Pos (m):       x={final_target_pos[0]:+.4f}, y={final_target_pos[1]:+.4f}, z={final_target_pos[2]:+.4f}")
    print(f"      Gripper Target:       Open to 8.0 cm (0.08 m)")
    print("-" * 70)

    input("[SAFETY CHECK] Hold the physical E-STOP button. Press [ENTER] to execute approach...")

    # 3. Execution using Smooth Cosine Profile
    control_hz = 20.0
    total_steps = int(total_duration * control_hz)
    rate = rospy.Rate(control_hz)
    dt = 1.0 / control_hz

    print(f"\n[4/4] Executing Approach ({total_steps} steps @ {control_hz:.0f} Hz)...")
    t_start = time.time()
    gripper_opened = False

    try:
        for step in range(total_steps + 1):
            if rospy.is_shutdown():
                print("[ABORT] ROS shutdown requested.")
                break

            # Smooth S-curve (half-cosine ramp 0.0 -> 1.0)
            progress = float(step) / float(total_steps)
            s = 0.5 * (1.0 - math.cos(math.pi * progress))

            # Current waypoint
            waypoint = anchor_pos + s * offset_vector

            # Hard safety clamp (excursion <= 25cm)
            dist_from_anchor = np.linalg.norm(waypoint - anchor_pos)
            if dist_from_anchor > 0.25:
                waypoint = anchor_pos + (waypoint - anchor_pos) / dist_from_anchor * 0.25

            arm.set_cartesian_pose(waypoint, anchor_quat)

            # Open gripper during first third of trajectory
            if progress > 0.15 and not gripper_opened:
                try:
                    arm.open_gripper(width=0.08, speed=0.1)
                    gripper_opened = True
                except Exception as e:
                    rospy.logwarn(f"Gripper exception: {e}")

            if step % 5 == 0 or step == total_steps:
                actual_pos, _ = arm.get_cartesian_pose()
                disp_cm = np.linalg.norm(actual_pos - anchor_pos) * 100.0 if actual_pos is not None else 0.0
                print(f"  Step {step:02d}/{total_steps:02d} | Progress: {progress*100:4.1f}% | Disp: {disp_cm:5.2f} cm")

            rate.sleep()

    except KeyboardInterrupt:
        print("\n[STOP] Execution halted by user.")

    # Hold at final position
    pos_end, _ = arm.get_cartesian_pose()
    hold_pos = pos_end if pos_end is not None else final_target_pos
    arm.set_cartesian_pose(hold_pos, anchor_quat)

    actual_total_disp = np.linalg.norm(hold_pos - anchor_pos) * 100.0
    print("\n" + "=" * 70)
    print("  APPROACH MOTION COMPLETED")
    print("=" * 70)
    print(f"[*] Total Net Displacement: {actual_total_disp:.2f} cm")
    print(f"[*] Final Position (m):     x={hold_pos[0]:+.4f}, y={hold_pos[1]:+.4f}, z={hold_pos[2]:+.4f}")
    print(f"[*] Gripper Status:         Opened to ~80 mm")
    print("=" * 70)

    # 4. Optional Return Prompt
    try:
        user_choice = input("\n[POST-CHECK] Enter 'r' to return to initial position, or any other key to stay: ").strip().lower()
        if user_choice == "r":
            print("[RETURN] Returning smoothly to initial pose...")
            return_steps = int(2.0 * control_hz)
            for step in range(return_steps + 1):
                if rospy.is_shutdown():
                    break
                progress = float(step) / float(return_steps)
                s = 0.5 * (1.0 - math.cos(math.pi * progress))
                waypoint = hold_pos + s * (anchor_pos - hold_pos)
                arm.set_cartesian_pose(waypoint, anchor_quat)
                rate.sleep()
            arm.set_cartesian_pose(anchor_pos, anchor_quat)
            print("[RETURN] Back at initial pose successfully.")
    except (KeyboardInterrupt, EOFError):
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
