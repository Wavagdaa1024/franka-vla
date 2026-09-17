# -*- coding: utf-8 -*-
import unittest
import numpy as np

from obstacle_avoidance.safety.franka_capsule_model import FrankaCapsuleModel, CAPSULE_RADII


class TestFrankaCapsuleModel(unittest.TestCase):
    def setUp(self):
        self.model = FrankaCapsuleModel()

    def test_zero_pose_capsules(self):
        q_zero = np.zeros(7)
        capsules = self.model.get_capsules(q_zero)
        self.assertEqual(len(capsules), 8)
        
        # Verify base origin is at [0, 0, 0]
        np.testing.assert_allclose(capsules[0].p1, [0.0, 0.0, 0.0], atol=1e-5)
        # Verify radii match specification
        for i, cap in enumerate(capsules):
            self.assertEqual(cap.radius, CAPSULE_RADII[i])
            self.assertEqual(len(cap.p1), 3)
            self.assertEqual(len(cap.p2), 3)

    def test_ready_pose_kinematics(self):
        # Ready pose commonly used in Franka demos
        q_ready = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        capsules = self.model.get_capsules(q_ready)
        self.assertEqual(len(capsules), 8)
        
        # EE tool tip should be forward and above table
        ee_tip = capsules[7].p2
        self.assertGreater(ee_tip[0], 0.20, "EE X must reach forward")
        self.assertGreater(ee_tip[2], 0.05, "EE Z must be above base")


if __name__ == "__main__":
    unittest.main()
