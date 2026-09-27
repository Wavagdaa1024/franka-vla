#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoothly move Franka arm to the exact average demonstration initial pose."""
import time
import numpy as np
import rospy
from Base_franka_joint_velocity_controller import FrankaJointVelocityController

# Target initial pose averaged from 32 demonstration trajectories
TARGET_Q = np.array([0.032, -0.083, 0.007, -2.686, 0.002, 2.591, 0.822], dtype=np.float64)

def main():
    rospy.init_node("go_to_demo_pose", anonymous=True)
    arm = FrankaJointVelocityController()
    time.sleep(1.0)

    curr_q = arm.get_joint_positions()
    if curr_q is None:
        print("[ERROR] Cannot read robot joint positions. Is the controller running?")
        return

    print(f"Current Joint Positions: {[round(x, 3) for x in curr_q]}")
    print(f"Target Demo Joint Pose:  {[round(x, 3) for x in TARGET_Q]}")
    diff = TARGET_Q - curr_q
    print(f"Max joint diff: {np.max(np.abs(diff)):.3f} rad ({np.degrees(np.max(np.abs(diff))):.1f} deg)")

    input("Press [ENTER] to smoothly move to target demo pose (Hold E-STOP)...")

    rate = rospy.Rate(100) # 100 Hz velocity command
    MAX_V = 0.15 # safe slow speed 0.15 rad/s

    while not rospy.is_shutdown():
        q_now = arm.get_joint_positions()
        err = TARGET_Q - q_now
        dist = np.linalg.norm(err)
        if dist < 0.015:
            arm.stop()
            print("[SUCCESS] Reached demonstration starting pose!")
            arm.open_gripper(width=0.08, speed=0.35)
            break
        
        # P-controller on joint velocities
        cmd_v = 1.0 * err
        speed = np.max(np.abs(cmd_v))
        if speed > MAX_V:
            cmd_v = cmd_v * (MAX_V / speed)
        arm.set_joint_velocities(cmd_v)
        rate.sleep()

if __name__ == "__main__":
    main()
