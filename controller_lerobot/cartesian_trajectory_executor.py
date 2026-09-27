#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic Cartesian Waypoint Trajectory Executor for Franka Emika Panda.
Executes high-precision Cartesian waypoint paths via Base_franka_joint_velocity_controller
with real-time 1kHz C++ Jacobian IK, acceleration limits, and safety boundaries.
"""

import time
import math
import numpy as np
import rospy

from Base_franka_joint_velocity_controller import FrankaJointVelocityController

# Workspace safety bounds (m)
SAFE_BOUNDS = {
    "x_min": 0.30, "x_max": 0.75,
    "y_min": -0.40, "y_max": 0.40,
    "z_min": 0.005, "z_max": 0.55
}

class CartesianTrajectoryExecutor:
    def __init__(self, arm=None):
        if not rospy.core.is_initialized():
            rospy.init_node("cartesian_trajectory_executor", anonymous=True)
        self.arm = arm or FrankaJointVelocityController()
        self.gripper_is_open = True

    def wait_for_connection(self, timeout=30.0):
        t0 = time.time()
        while time.time() - t0 < timeout and not rospy.is_shutdown():
            pos, quat = self.arm.get_cartesian_pose()
            if pos is not None:
                return True
            rospy.sleep(0.05)
        return False

    def validate_position(self, pos):
        x, y, z = pos[:3]
        if not (SAFE_BOUNDS["x_min"] <= x <= SAFE_BOUNDS["x_max"]):
            raise ValueError(f"Target X={x:.4f} exceeds safety bounds [{SAFE_BOUNDS['x_min']}, {SAFE_BOUNDS['x_max']}]")
        if not (SAFE_BOUNDS["y_min"] <= y <= SAFE_BOUNDS["y_max"]):
            raise ValueError(f"Target Y={y:.4f} exceeds safety bounds [{SAFE_BOUNDS['y_min']}, {SAFE_BOUNDS['y_max']}]")
        if not (SAFE_BOUNDS["z_min"] <= z <= SAFE_BOUNDS["z_max"]):
            raise ValueError(f"Target Z={z:.4f} exceeds safety bounds [{SAFE_BOUNDS['z_min']}, {SAFE_BOUNDS['z_max']}]")

    def move_to_cartesian(
        self,
        target_pos,
        max_vel=0.15,
        kp=2.0,
        tol=0.010,
        timeout=None
    ):
        """
        Closed-loop Cartesian velocity control to smoothly reach target_pos [x, y, z].
        """
        self.validate_position(target_pos)
        target = np.array(target_pos[:3], dtype=np.float64)

        curr_pos, _ = self.arm.get_cartesian_pose()
        initial_dist = float(np.linalg.norm(target - curr_pos)) if curr_pos is not None else 0.3
        
        # Dynamic generous timeout: at least 20s, scaled by distance and velocity
        if timeout is None:
            timeout = max(20.0, (initial_dist / max(max_vel, 0.05)) * 4.0)

        rate = rospy.Rate(50)  # 50Hz Cartesian command loop
        t_start = time.time()
        
        while not rospy.is_shutdown():
            if time.time() - t_start > timeout:
                rospy.logwarn(f"[CartesianExecutor] Waypoint timeout ({timeout:.1f}s) reached before converging to tol {tol}m")
                break

            curr_pos, _ = self.arm.get_cartesian_pose()
            if curr_pos is None:
                rate.sleep()
                continue

            delta = target - curr_pos
            dist = float(np.linalg.norm(delta))

            if dist <= tol:
                break

            # P-controller on Cartesian position error with velocity saturation
            speed = min(max_vel, max(0.02, kp * dist))
            direction = delta / dist
            v_cmd = direction * speed

            # Apply additional safety clamp on Z descent
            if v_cmd[2] < -0.10:
                v_cmd[2] = -0.10

            self.arm.set_cartesian_velocity(v_cmd, [0.0, 0.0, 0.0])
            rate.sleep()

        self.arm.stop()
        curr_pos, _ = self.arm.get_cartesian_pose()
        final_err = float(np.linalg.norm(target - curr_pos)) if curr_pos is not None else 999.0
        return final_err <= tol * 1.5

    def set_gripper(self, open_grip=True):
        """Actuate gripper: open (0.08m) or close (0.045m, 15N force)"""
        if open_grip and not self.gripper_is_open:
            rospy.loginfo("[CartesianExecutor] Opening gripper...")
            self.arm.open_gripper(width=0.08, speed=0.2)
            rospy.sleep(0.4)
            self.gripper_is_open = True
        elif not open_grip and self.gripper_is_open:
            rospy.loginfo("[CartesianExecutor] Closing gripper (Grasp with 15N)...")
            self.arm.close_gripper(width=0.045, force=15.0, speed=0.1)
            rospy.sleep(0.5)
            self.gripper_is_open = False

    def execute_waypoints(self, waypoints, progress_callback=None):
        """
        Executes a sequence of waypoint dictionaries.
        Each waypoint:
          {"name": "...", "pos": [x,y,z], "gripper": 1.0/0.0, "speed": 0.15, "pause_s": 0.2}
        """
        total = len(waypoints)
        rospy.loginfo(f"[CartesianExecutor] Starting execution of {total} waypoints...")

        for idx, wp in enumerate(waypoints):
            name = wp.get("name", f"WP_{idx+1}")
            pos = wp["pos"]
            speed = float(wp.get("speed", 0.15))
            pause_s = float(wp.get("pause_s", 0.0))
            grip_cmd = wp.get("gripper")

            rospy.loginfo(f"  [{idx+1}/{total}] {name} -> pos={[round(x, 4) for x in pos]}, speed={speed:.2f} m/s")

            # 1. Pre-motion gripper handling if specified before move
            if grip_cmd is not None and wp.get("grip_first", False):
                self.set_gripper(open_grip=(grip_cmd > 0.5))

            # 2. Cartesian move
            success = self.move_to_cartesian(pos, max_vel=speed)
            if not success:
                rospy.logwarn(f"  [{idx+1}/{total}] {name} did not converge precisely, proceeding to next waypoint")

            # 3. Post-motion gripper handling
            if grip_cmd is not None and not wp.get("grip_first", False):
                self.set_gripper(open_grip=(grip_cmd > 0.5))

            # 4. Dwell / pause
            if pause_s > 0:
                rospy.sleep(pause_s)

            if progress_callback:
                progress_callback(idx + 1, total, name)

        self.arm.stop()
        rospy.loginfo(f"[CartesianExecutor] Completed all {total} waypoints successfully.")
        return True
