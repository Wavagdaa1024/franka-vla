"""Unit tests for Franka Panda safety guards and action bounds."""
import unittest
import numpy as np
from franka_teleop.live_guards import (
    validate_joint_velocity_chunk,
    validate_joint_position_chunk,
    FRANKA_JOINT_LIMITS,
    action_is_fresh
)

class TestSafetyGuards(unittest.TestCase):
    def test_joint_velocity_clamp(self):
        vels = np.zeros((10, 7))
        grip = np.full(10, 0.5)
        v_out, g_out = validate_joint_velocity_chunk(vels, grip)
        self.assertEqual(v_out.shape, (10, 7))
        self.assertEqual(g_out.shape, (10,))

    def test_joint_position_clamp_limits(self):
        bad_pos = np.array([[10.0] * 7])
        grip = np.array([0.5])
        q_clamped, _ = validate_joint_position_chunk(bad_pos, grip)
        for j in range(7):
            _, high = FRANKA_JOINT_LIMITS[j]
            self.assertAlmostEqual(q_clamped[0, j], high)

    def test_freshness_ttl(self):
        now = 100.0
        self.assertTrue(action_is_fresh(99.8, now=now, ttl_s=0.5))
        self.assertFalse(action_is_fresh(98.0, now=now, ttl_s=0.5))
        self.assertFalse(action_is_fresh(101.0, now=now, ttl_s=0.5))

if __name__ == "__main__":
    unittest.main()
