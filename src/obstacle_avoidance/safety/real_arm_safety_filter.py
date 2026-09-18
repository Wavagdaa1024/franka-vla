# -*- coding: utf-8 -*-
"""
Real-Arm Whole-Arm Safety Filter & Collision Interceptor.
Replaces simulation-only safety filter. Operates on real 7-DOF joint commands,
guaranteeing step delta limits and obstacle clearance.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from obstacle_avoidance.perception.envelope_extractor import ObstacleOBB
from obstacle_avoidance.safety.franka_capsule_model import FrankaCapsuleModel
from obstacle_avoidance.safety.distance_engine import evaluate_whole_arm_distance, DistanceReport


class RealArmSafetyFilter:
    """
    Continuous collision prevention and action intervention filter for Franka Panda.
    """
    def __init__(
        self,
        warn_distance: float = 0.080,    # 8cm warning clearance
        stop_distance: float = 0.025,    # 2.5cm emergency stop threshold
        max_step_delta: float = 0.050,   # Maximum allowed joint delta per step (rad)
        intervention_threshold: float = 1e-3
    ):
        self.warn_distance = float(warn_distance)
        self.stop_distance = float(stop_distance)
        self.max_step_delta = float(max_step_delta)
        self.intervention_threshold = float(intervention_threshold)
        self.robot_model = FrankaCapsuleModel()

    def filter_joint_action(
        self,
        proposed_q: np.ndarray,
        current_q: np.ndarray,
        obstacles: List[ObstacleOBB]
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """
        Filters a proposed 7-DOF joint configuration target.
        
        Args:
            proposed_q: (7,) proposed next joint target from VLA or trajectory planner
            current_q: (7,) current robot joint state
            obstacles: list of active ObstacleOBB envelopes
            
        Returns:
            safe_q: (7,) safe, validated joint configuration
            intervened: bool indicating if modification exceeded threshold
            info: dict containing distance metrics, scale factor, and intervention reason
        """
        proposed_q = np.asarray(proposed_q, dtype=np.float64).flatten()
        current_q = np.asarray(current_q, dtype=np.float64).flatten()

        # 1. Delta clamping
        raw_delta = proposed_q - current_q
        clamped_delta = np.clip(raw_delta, -self.max_step_delta, self.max_step_delta)
        target_q = current_q + clamped_delta

        if not obstacles:
            residual = np.linalg.norm(target_q - proposed_q)
            return target_q, residual > self.intervention_threshold, {
                "min_distance_m": float("inf"),
                "scale_factor": 1.0,
                "reason": "no_obstacles",
                "clamped": not np.allclose(target_q, proposed_q)
            }

        # 2. Evaluate distance at current and proposed configurations
        curr_capsules = self.robot_model.get_capsules(current_q)
        target_capsules = self.robot_model.get_capsules(target_q)

        curr_report = evaluate_whole_arm_distance(curr_capsules, obstacles)
        target_report = evaluate_whole_arm_distance(target_capsules, obstacles)

        # 3. Decision Logic:
        # Zone A: Completely clear (> warn_distance)
        if target_report.min_distance > self.warn_distance:
            residual = np.linalg.norm(target_q - proposed_q)
            return target_q, residual > self.intervention_threshold, {
                "min_distance_m": target_report.min_distance,
                "scale_factor": 1.0,
                "reason": "clear",
                "closest_link": target_report.closest_capsule_name
            }

        # Zone B: Deceleration / Damping zone (stop_distance < d <= warn_distance)
        if target_report.min_distance > self.stop_distance:
            # Linear scaling of motion towards obstacle
            # alpha in (0.0, 1.0)
            alpha = (target_report.min_distance - self.stop_distance) / (self.warn_distance - self.stop_distance)
            alpha = float(np.clip(alpha, 0.1, 1.0))
            
            # If moving away from obstacle, allow full delta
            if target_report.min_distance >= curr_report.min_distance:
                alpha = 1.0

            scaled_delta = clamped_delta * alpha
            safe_q = current_q + scaled_delta
            residual = np.linalg.norm(safe_q - proposed_q)

            return safe_q, residual > self.intervention_threshold, {
                "min_distance_m": target_report.min_distance,
                "scale_factor": alpha,
                "reason": "velocity_scaled_near_obstacle",
                "closest_link": target_report.closest_capsule_name
            }

        # Zone C: Critical Stop Zone (d <= stop_distance)
        # Check if proposed action is moving AWAY from the obstacle:
        if target_report.min_distance > curr_report.min_distance + 1e-4:
            # Evasive motion allowed with small safe velocity
            safe_q = current_q + clamped_delta * 0.3
            return safe_q, True, {
                "min_distance_m": target_report.min_distance,
                "scale_factor": 0.3,
                "reason": "evasion_allowed",
                "closest_link": target_report.closest_capsule_name
            }
        else:
            # Freeze joint motion completely to prevent physical collision
            safe_q = current_q.copy()
            return safe_q, True, {
                "min_distance_m": target_report.min_distance,
                "scale_factor": 0.0,
                "reason": "hazard_freeze_stop",
                "closest_link": target_report.closest_capsule_name
            }
