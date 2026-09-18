# -*- coding: utf-8 -*-
import unittest
import numpy as np

from obstacle_avoidance.perception.envelope_extractor import ObstacleOBB
from obstacle_avoidance.safety.franka_capsule_model import FrankaCapsuleModel
from obstacle_avoidance.safety.distance_engine import evaluate_whole_arm_distance
from obstacle_avoidance.safety.real_arm_safety_filter import RealArmSafetyFilter


class TestDistanceAndSafetyFilter(unittest.TestCase):
    def setUp(self):
        self.model = FrankaCapsuleModel()
        self.filter = RealArmSafetyFilter(warn_distance=0.08, stop_distance=0.02)
        self.q_ready = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

    def test_clear_space_no_intervention(self):
        # Obstacle far away at X=1.5m
        far_obb = ObstacleOBB(
            center=np.array([1.5, 0.0, 0.2]),
            extents=np.array([0.1, 0.1, 0.1]),
            rotation_matrix=np.eye(3),
            yaw_rad=0.0,
            corners=np.zeros((8, 3)),
            num_points=100,
            margin_added=0.03
        )
        capsules = self.model.get_capsules(self.q_ready)
        report = evaluate_whole_arm_distance(capsules, [far_obb])
        self.assertGreater(report.min_distance, 0.5)
        self.assertFalse(report.is_collision)

        safe_q, intervened, info = self.filter.filter_joint_action(
            self.q_ready + 0.01, self.q_ready, [far_obb]
        )
        self.assertFalse(intervened)
        self.assertEqual(info["reason"], "clear")

    def test_collision_hazard_freeze(self):
        # Place obstacle directly at current EE tool tip location
        ee_pos = self.model.get_capsules(self.q_ready)[7].p2
        hazard_obb = ObstacleOBB(
            center=ee_pos,
            extents=np.array([0.10, 0.10, 0.10]),
            rotation_matrix=np.eye(3),
            yaw_rad=0.0,
            corners=np.zeros((8, 3)),
            num_points=100,
            margin_added=0.03
        )
        # Try moving further into obstacle
        safe_q, intervened, info = self.filter.filter_joint_action(
            self.q_ready + 0.02, self.q_ready, [hazard_obb]
        )
        self.assertTrue(intervened)
        self.assertIn("stop", info["reason"])
        # Should freeze motion (safe_q close to current_q)
        np.testing.assert_allclose(safe_q, self.q_ready, atol=1e-5)

    def test_evaluation_latency(self):
        import time
        # Test 100 queries of whole-arm distance against multiple obstacles
        obbs = [
            ObstacleOBB(
                center=np.array([0.4 + 0.1 * i, 0.1 * (i - 1), 0.15]),
                extents=np.array([0.08, 0.08, 0.12]),
                rotation_matrix=np.eye(3),
                yaw_rad=0.0,
                corners=np.zeros((8, 3)),
                num_points=100,
                margin_added=0.03
            ) for i in range(3)
        ]
        capsules = self.model.get_capsules(self.q_ready)
        
        t0 = time.perf_counter()
        N = 100
        for _ in range(N):
            _ = evaluate_whole_arm_distance(capsules, obbs)
        t_total = time.perf_counter() - t0
        avg_ms = (t_total / N) * 1000.0
        print(f"\n[Benchmark] 8-Capsule vs 3-OBB average latency: {avg_ms:.3f} ms per step")
        # Ensure it is well below 5ms (typically 0.1~0.3ms)
        self.assertLess(avg_ms, 5.0)


if __name__ == "__main__":
    unittest.main()
