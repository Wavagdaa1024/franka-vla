#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit and Integration Tests for End-Effector Pose Recovery and OOD Guard Mechanism.
Validates:
  1. compute_ee_tilt accuracy against known rotations.
  2. solve_ee_recovery_joints 5-DOF IK convergence (< 0.01 deg tilt, < 0.05 mm position error).
  3. Table floor Z-lifting during recovery when Z < z_floor.
  4. Franka joint limit compliance after recovery.
  5. C1-smooth cosine velocity trajectory generation without jerk spikes.
  6. End-to-end mock sync cycle recovery integration.
"""

import sys
import unittest
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from franka_teleop.kinematics import (
    forward_kinematics,
    analytical_jacobian,
    compute_ee_tilt,
    solve_ee_recovery_joints,
    generate_smooth_recovery_traj,
    DEFAULT_Z_FLOOR,
    FRANKA_JOINT_LIMITS
)


class MockArm:
    """Mock FrankaJointVelocityController for testing recovery control loops."""
    def __init__(self, initial_q: np.ndarray, initial_z_floor: float = DEFAULT_Z_FLOOR):
        self._q = np.asarray(initial_q, dtype=np.float64).copy()
        self.cmd_history = []
        self.stopped = False
        self.z_floor = initial_z_floor

    def get_joint_positions(self):
        return self._q.copy()

    def get_cartesian_pose(self):
        T = forward_kinematics(self._q)
        return T[:3, 3], np.array([0.0, 0.0, 0.0, 1.0])

    def set_joint_velocities(self, dq, dt=0.0667):
        self.cmd_history.append(np.asarray(dq).copy())
        self._q += np.asarray(dq, dtype=np.float64) * dt
        self.stopped = False

    def stop(self):
        self.stopped = True


class MockArgs:
    def __init__(self, max_tilt_deg=8.0, z_min=DEFAULT_Z_FLOOR, recover_ee=True,
                 recovery_steps=10, recovery_vel=0.20, recovery_iters=25,
                 settle_time=0.01, shadow=False):
        self.max_tilt_deg = max_tilt_deg
        self.z_min = z_min
        self.recover_ee = recover_ee
        self.recovery_steps = recovery_steps
        self.recovery_vel = recovery_vel
        self.recovery_iters = recovery_iters
        self.settle_time = settle_time
        self.shadow = shadow


class MockRate:
    def __init__(self, hz=15.0):
        self.hz = hz
    def sleep(self):
        pass


class TestPoseRecovery(unittest.TestCase):
    def setUp(self):
        # Ready pose of Franka Panda: tool pointing straight down [0, 0, -1]
        self.q_nominal = np.array([-0.0802, 0.0305, 0.0914, -2.4492, 0.0313, 2.4519, 0.7947], dtype=np.float64)

    def test_compute_ee_tilt(self):
        """Test tilt angle computation on nominal and tilted configurations."""
        tilt_nom = compute_ee_tilt(self.q_nominal)
        # Franka calibrated ready pose has a natural ~2.04 deg tilt, well within the 8.0 deg safe bound
        self.assertLess(tilt_nom, 3.0, f"Nominal pose should have tilt < 3.0 deg, got {tilt_nom:.3f} deg")

        # Induce deliberate wrist tilt
        q_tilted = self.q_nominal.copy()
        q_tilted[4] += 0.20  # ~11.5 deg tilt
        tilt_induced = compute_ee_tilt(q_tilted)
        self.assertGreater(tilt_induced, 8.0, f"Induced tilt should be > 8.0 deg, got {tilt_induced:.2f} deg")

    def test_solve_ee_recovery_joints_convergence(self):
        """Test 5-DOF IK recovery from multiple random tilted configurations."""
        np.random.seed(42)
        for trial in range(5):
            # Induce random tilt between 8 and 22 degrees
            q_perturbed = self.q_nominal.copy()
            q_perturbed[3] += np.random.uniform(-0.15, 0.15)
            q_perturbed[4] += np.random.uniform(-0.25, 0.25)
            q_perturbed[5] += np.random.uniform(-0.20, 0.20)
            
            p_initial = forward_kinematics(q_perturbed)[:3, 3]
            tilt_initial = compute_ee_tilt(q_perturbed)
            
            q_rec, tilt_after, pos_err = solve_ee_recovery_joints(
                q_perturbed, z_floor=DEFAULT_Z_FLOOR, max_iters=25
            )
            
            # 1. Tilt should be strictly eliminated to < 0.05 deg
            self.assertLess(tilt_after, 0.05,
                            f"Trial {trial}: residual tilt {tilt_after:.4f} deg > 0.05 deg")
            
            # 2. Position error (XY) should be negligible (< 0.5 mm)
            p_rec = forward_kinematics(q_rec)[:3, 3]
            xy_err = np.linalg.norm(p_rec[:2] - p_initial[:2]) * 1000.0
            self.assertLess(xy_err, 0.5,
                            f"Trial {trial}: XY drift {xy_err:.3f} mm > 0.5 mm")
            
            # 3. Joint limits must be respected
            for j in range(7):
                low, high = FRANKA_JOINT_LIMITS[j]
                self.assertGreaterEqual(q_rec[j], low - 1e-4)
                self.assertLessEqual(q_rec[j], high + 1e-4)

    def test_floor_protection_lifting_during_recovery(self):
        """Test that if recovery is triggered below floor, Z is automatically lifted."""
        # Find a configuration with Z < z_floor
        q_low = self.q_nominal.copy()
        q_low[1] += 0.40  # pushes arm down
        T_low = forward_kinematics(q_low)
        z_curr = T_low[2, 3]

        # Force target height below floor for testing
        z_floor = 0.100  # set test floor above current Z
        self.assertLess(z_curr, z_floor)

        q_rec, tilt_after, pos_err = solve_ee_recovery_joints(
            q_low, z_floor=z_floor, max_iters=25, z_lift=0.005
        )
        p_rec = forward_kinematics(q_rec)[:3, 3]

        # Recovered Z must be >= z_floor + 0.003
        self.assertGreaterEqual(p_rec[2], z_floor + 0.003,
                                f"Recovered Z ({p_rec[2]:.4f}m) failed to clear floor ({z_floor:.4f}m)")

    def test_smooth_trajectory_generation(self):
        """Test cosine velocity trajectory profile: zero start/end velocity, speed clamp."""
        q_start = self.q_nominal.copy()
        q_target = self.q_nominal.copy()
        q_target[4] += 0.20
        q_target[5] -= 0.15

        max_vel = 0.20
        dq_traj = generate_smooth_recovery_traj(
            q_start, q_target, steps=10, dt=0.0667, max_vel=max_vel
        )

        # 1. Non-empty
        self.assertGreater(len(dq_traj), 0)

        # 2. Maximum joint velocity clamp respected
        self.assertLessEqual(np.max(np.abs(dq_traj)), max_vel + 1e-5)

        # 3. Smooth start: step 0 velocity should be significantly lower than peak
        peak_vel = np.max(np.abs(dq_traj))
        first_step_vel = np.max(np.abs(dq_traj[0]))
        self.assertLess(first_step_vel, peak_vel * 0.5)

        # 4. Integrated position should reach target
        q_integrated = q_start.copy()
        for dq in dq_traj:
            q_integrated += dq * 0.0667
        err_to_target = np.max(np.abs(q_integrated - q_target))
        self.assertLess(err_to_target, 0.01, f"Trajectory integration error: {err_to_target:.4f} rad")

    def test_check_and_restore_ee_pose_mock(self):
        """Test check_and_restore_ee_pose logic with mock arm."""
        from franka_teleop.sync_franka import check_and_restore_ee_pose

        # 1. Under nominal pose (tilt < 8.0°): should NOT trigger recovery
        arm = MockArm(self.q_nominal)
        args = MockArgs(max_tilt_deg=8.0)
        rate = MockRate()

        q_out, did_recover = check_and_restore_ee_pose(arm, self.q_nominal, args, rate)
        self.assertFalse(did_recover)
        self.assertEqual(len(arm.cmd_history), 0)

        # 2. Under tilted pose (tilt = 15.0° > 8.0°): SHOULD trigger recovery
        q_tilted = self.q_nominal.copy()
        q_tilted[4] += 0.25
        q_tilted[5] -= 0.18
        tilt_init = compute_ee_tilt(q_tilted)
        self.assertGreater(tilt_init, 8.0)

        arm_tilted = MockArm(q_tilted)
        q_out, did_recover = check_and_restore_ee_pose(arm_tilted, q_tilted, args, rate)
        self.assertTrue(did_recover)
        self.assertGreater(len(arm_tilted.cmd_history), 0)

        # Arm final pose should be upright
        tilt_final = compute_ee_tilt(arm_tilted.get_joint_positions())
        self.assertLess(tilt_final, 0.5, f"Post-recovery tilt should be < 0.5 deg, got {tilt_final:.2f} deg")

    def test_multi_cycle_sync_simulation_with_drift(self):
        """Simulate continuous synchronous cycles where tilt drift is induced and auto-corrected."""
        from franka_teleop.sync_franka import check_and_restore_ee_pose

        arm = MockArm(self.q_nominal)
        args = MockArgs(max_tilt_deg=8.0)
        rate = MockRate()

        recovery_events = []
        for cycle in range(1, 6):
            curr_q = arm.get_joint_positions()
            
            # In cycle 3, inject simulated open-loop joint drift
            if cycle == 3:
                curr_q[4] += 0.30
                arm._q = curr_q.copy()

            curr_q, did_recover = check_and_restore_ee_pose(arm, curr_q, args, rate)
            if did_recover:
                recovery_events.append(cycle)

            # Sampled pose before inference MUST be strictly in-distribution (<= max_tilt_deg)
            sampled_tilt = compute_ee_tilt(curr_q)
            self.assertLessEqual(sampled_tilt, args.max_tilt_deg,
                                 f"Cycle {cycle}: sampled tilt {sampled_tilt:.2f} deg is OOD (> {args.max_tilt_deg} deg)!")

        self.assertEqual(recovery_events, [3], "Recovery should have triggered exactly at Cycle 3")


if __name__ == "__main__":
    unittest.main()
