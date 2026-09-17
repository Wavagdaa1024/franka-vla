"""
Whole-arm Safety Filter with Distance Projection and Velocity Scaling.
Operates on proposed joint actions, guaranteeing step delta limits and clearance from obstacles.
"""

from typing import Tuple, Dict, Any, List
import numpy as np

class SafetyFilter:
    def __init__(
        self,
        max_step_delta: float = 0.05,
        clearance_margin: float = 0.06,
        stop_distance: float = 0.02,
        intervention_threshold: float = 1e-3,
    ):
        self.max_step_delta = max_step_delta
        self.clearance_margin = clearance_margin
        self.stop_distance = stop_distance
        self.intervention_threshold = intervention_threshold
        
        # Key arm bodies to track for obstacle clearance
        self.monitored_bodies = [
            "vx300s_left/gripper_link",
            "vx300s_left/wrist_link",
            "vx300s_left/lower_forearm_link",
            "vx300s_right/gripper_link",
            "vx300s_right/wrist_link",
            "vx300s_right/lower_forearm_link",
        ]

    def filter_action(
        self,
        proposed_action: np.ndarray,
        current_qpos: np.ndarray,
        physics,
        obstacle_positions: List[np.ndarray],
        obstacle_radius: float = 0.035
    ) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """
        Filters proposed 14-dim action.
        Returns:
            safe_action: 14-dim numpy array
            intervened: bool indicating if modification exceeded threshold
            info: dict containing min_distance, scale_factor, residual_norm
        """
        proposed_action = np.asarray(proposed_action, dtype=np.float64)
        current_qpos = np.asarray(current_qpos, dtype=np.float64)
        
        # 1. Delta clamping
        raw_delta = proposed_action - current_qpos
        clamped_delta = np.clip(raw_delta, -self.max_step_delta, self.max_step_delta)
        
        # 2. Obstacle distance check via forward kinematics
        min_dist = float("inf")
        closest_body = None
        
        for body_name in self.monitored_bodies:
            try:
                body_xpos = physics.named.data.xpos[body_name]
                for obs_pos in obstacle_positions:
                    dist = float(np.linalg.norm(body_xpos - obs_pos) - obstacle_radius)
                    if dist < min_dist:
                        min_dist = dist
                        closest_body = body_name
            except Exception:
                pass
                
        # 3. Velocity scaling if inside clearance boundary
        scale_factor = 1.0
        if min_dist < self.clearance_margin:
            if min_dist <= self.stop_distance:
                scale_factor = 0.0
            else:
                # Linear velocity ramp down
                scale_factor = (min_dist - self.stop_distance) / (self.clearance_margin - self.stop_distance)
                scale_factor = float(np.clip(scale_factor, 0.0, 1.0))
                
        # Scale arm motion
        # Aloha 14 dim: [left 7 joints, right 7 joints]
        scaled_delta = clamped_delta * scale_factor
        safe_action = current_qpos + scaled_delta
        
        # Measure modification residual
        residual_norm = float(np.linalg.norm(safe_action - proposed_action))
        intervened = residual_norm > self.intervention_threshold
        
        info = {
            "min_distance": min_dist,
            "closest_body": closest_body,
            "scale_factor": scale_factor,
            "residual_norm": residual_norm,
            "raw_delta_norm": float(np.linalg.norm(raw_delta)),
        }
        
        return safe_action.astype(np.float32), intervened, info
