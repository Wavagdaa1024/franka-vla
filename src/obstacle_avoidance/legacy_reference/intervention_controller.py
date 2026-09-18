"""
Intervention Controller and Action Chunk Manager for M3/M4 benchmark variants.
Supports:
1. 'original_vla': Direct unconstrained execution
2. 'silent_filter': Filter intervenes silently without notifying policy or clearing chunk
3. 'intervene_and_replan': Filter intervention triggers chunk invalidation and immediate replan
4. 'matched_replan': Periodic replanning matching intervention computation budget
"""

import time
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import torch

from src.safety_filter import SafetyFilter
from src.collision_oracle import CollisionOracle

class InterventionController:
    def __init__(
        self,
        policy,
        preproc,
        postproc,
        safety_filter: SafetyFilter,
        collision_oracle: CollisionOracle,
        variant: str = "intervene_and_replan",
        device: str = "cuda",
        matched_replan_period: int = 10,
    ):
        self.policy = policy
        self.preproc = preproc
        self.postproc = postproc
        self.safety_filter = safety_filter
        self.collision_oracle = collision_oracle
        self.variant = variant
        self.device = device
        self.matched_replan_period = matched_replan_period
        
        # Internal state
        self.action_queue: List[np.ndarray] = []
        self.step_in_chunk = 0
        self.total_policy_calls = 0
        self.total_interventions = 0
        self.total_replan_latency = 0.0
        self.cumulative_correction_norm = 0.0
        self.intervention_history: List[bool] = []

    def reset(self):
        self.policy.reset()
        self.action_queue.clear()
        self.step_in_chunk = 0
        self.total_policy_calls = 0
        self.total_interventions = 0
        self.total_replan_latency = 0.0
        self.cumulative_correction_norm = 0.0
        self.intervention_history.clear()

    def _query_policy_chunk(self, obs: dict) -> Tuple[List[np.ndarray], float]:
        """Queries policy and returns newly predicted action chunk and latency."""
        t0 = time.perf_counter()
        
        img = torch.from_numpy(obs["pixels"]["top"]).permute(2, 0, 1).float() / 255.0
        state = torch.from_numpy(obs["agent_pos"]).float()
        
        obs_dict = {
            "observation.images.top": img,
            "observation.state": state,
        }
        
        proc_dict = self.preproc(obs_dict)
        proc_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in proc_dict.items()}
        
        with torch.no_grad():
            # In ACT, select_action manages the internal queue if called step-by-step,
            # or forward() produces chunk.
            # Calling select_action yields the current action.
            # To get explicit chunk control: select_action returns next action.
            raw_action = self.policy.select_action(proc_dict)
            
        action = self.postproc(raw_action)
        if isinstance(action, torch.Tensor):
            action = action.squeeze(0).cpu().numpy()
            
        latency = time.perf_counter() - t0
        self.total_policy_calls += 1
        return [action], latency

    def get_action(
        self,
        obs: dict,
        physics,
        obstacle_positions: List[np.ndarray]
    ) -> Tuple[np.ndarray, bool, bool, Dict[str, Any]]:
        """
        Computes the action to execute according to the variant rules.
        Returns:
            executed_action: 14-dim numpy array
            intervened: bool
            collided: bool (ground truth collision from oracle)
            step_stats: dict
        """
        current_qpos = obs["agent_pos"]
        replan_triggered = False
        replan_latency = 0.0
        
        # 1. Check if policy needs to be queried (chunk empty or matched replan)
        need_query = False
        if len(self.action_queue) == 0:
            need_query = True
        elif self.variant == "matched_replan" and (self.step_in_chunk % self.matched_replan_period == 0):
            need_query = True
            self.action_queue.clear()
            
        if need_query:
            actions, lat = self._query_policy_chunk(obs)
            self.action_queue.extend(actions)
            replan_latency += lat
            
        proposed_action = self.action_queue.pop(0)
        self.step_in_chunk += 1
        
        # 2. Safety filter execution
        if self.variant == "original_vla":
            safe_action = proposed_action
            intervened = False
            filter_info = {"residual_norm": 0.0, "scale_factor": 1.0, "min_distance": float("inf")}
        else:
            safe_action, intervened, filter_info = self.safety_filter.filter_action(
                proposed_action=proposed_action,
                current_qpos=current_qpos,
                physics=physics,
                obstacle_positions=obstacle_positions
            )
            
        if intervened:
            self.total_interventions += 1
            corr_norm = filter_info.get("residual_norm", 0.0)
            self.cumulative_correction_norm += corr_norm
            
            # 3. Intervene-and-replan: flush chunk queue and replan
            if self.variant == "intervene_and_replan":
                replan_triggered = True
                # Clear stale chunk
                self.action_queue.clear()
                self.step_in_chunk = 0
                self.policy.reset()
                
                # Immediately replan from current state
                actions, lat = self._query_policy_chunk(obs)
                self.action_queue.extend(actions)
                replan_latency += lat
                self.total_replan_latency += lat
                
        self.intervention_history.append(intervened)
        
        # 4. Check ground truth collision oracle
        has_collision, col_details = self.collision_oracle.check_collision(physics)
        
        step_stats = {
            "proposed_action": proposed_action,
            "safe_action": safe_action,
            "intervened": intervened,
            "has_collision": has_collision,
            "collision_details": col_details,
            "replan_triggered": replan_triggered,
            "replan_latency": replan_latency,
            "filter_info": filter_info
        }
        
        return safe_action, intervened, has_collision, step_stats
