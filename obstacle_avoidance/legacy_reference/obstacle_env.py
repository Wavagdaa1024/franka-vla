"""
Aloha Environment with Configurable Obstacles (Static Narrow & Low-Speed Dynamic).
Wraps MuJoCo simulation, maintaining 100% state save/restore fidelity and native cameras.
"""

from pathlib import Path
from typing import Optional, List, Tuple
import shutil
import numpy as np
import gymnasium as gym
import gym_aloha.env as ga_env
from dm_control import mujoco as dm_mujoco
from dm_control.rl import control

ENV_ASSETS_DIR = Path("C:/Users/74727/vla_obstacle_experiments/task4_s4_end_to_end/models/envs")

def ensure_assets_copied():
    ENV_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for f in ga_env.ASSETS_DIR.glob("*.*"):
        target = ENV_ASSETS_DIR / f.name
        if not target.exists() or target.stat().st_size != f.stat().st_size:
            shutil.copy(f, target)

def build_obstacle_xml(task_name: str, obstacle_type: str = "none") -> Path:
    ensure_assets_copied()
    if obstacle_type == "none":
        return ga_env.ASSETS_DIR / f"bimanual_viperx_{task_name}.xml"
        
    xml_filename = f"{task_name}_{obstacle_type}_obs.xml"
    out_path = ENV_ASSETS_DIR / xml_filename
    
    base_xml = (ga_env.ASSETS_DIR / f"bimanual_viperx_{task_name}.xml").read_text(encoding="utf-8")
    
    if obstacle_type == "static":
        # Static narrow obstacle placed in the transfer/insertion work zone
        obs_xml = """
        <!-- Static Narrow Obstacle -->
        <body name="obstacle_body" pos="0.0 0.50 0.08">
            <geom name="obstacle_geom" type="cylinder" size="0.025 0.08" rgba="0.9 0.2 0.2 0.9" contype="1" conaffinity="1"/>
        </body>
        """
    elif obstacle_type == "dynamic":
        # Low speed moving obstacle traversing between x = -0.12 and 0.12
        obs_xml = """
        <!-- Dynamic Low-Speed Obstacle -->
        <body name="obstacle_body" pos="-0.10 0.50 0.08">
            <joint name="obstacle_slide_x" type="slide" axis="1 0 0" range="-0.15 0.15" damping="10"/>
            <inertial pos="0 0 0" mass="0.5" diaginertia="0.005 0.005 0.005" />
            <geom name="obstacle_geom_moving" type="cylinder" size="0.025 0.08" rgba="0.2 0.4 0.9 0.9" contype="1" conaffinity="1"/>
        </body>
        """
    else:
        raise ValueError(f"Unknown obstacle_type: {obstacle_type}")

    modified_xml = base_xml.replace("</worldbody>", f"{obs_xml}\n    </worldbody>")
    out_path.write_text(modified_xml, encoding="utf-8")
    return out_path

class AlohaObstacleEnv(ga_env.AlohaEnv):
    def __init__(
        self,
        task_name: str = "transfer_cube",
        obstacle_type: str = "none",
        obs_type: str = "pixels_agent_pos",
        dynamic_velocity: float = 0.02, # m/s for dynamic obstacle
        **kwargs
    ):
        self.task_name_base = task_name
        self.obstacle_type = obstacle_type
        self.dynamic_velocity = dynamic_velocity
        self._obstacle_xml_path = build_obstacle_xml(task_name, obstacle_type)
        super().__init__(task=task_name, obs_type=obs_type, **kwargs)

    def _make_env_task(self, task_name):
        time_limit = float("inf")
        physics = dm_mujoco.Physics.from_xml_path(str(self._obstacle_xml_path))
        
        if task_name == "transfer_cube":
            task = ga_env.TransferCubeTask()
        elif task_name == "insertion":
            task = ga_env.InsertionTask()
        else:
            raise NotImplementedError(task_name)

        return control.Environment(
            physics, task, time_limit, control_timestep=ga_env.DT, n_sub_steps=None, flat_observation=False
        )

    def step(self, action):
        # Update dynamic obstacle motion if dynamic
        if self.obstacle_type == "dynamic":
            try:
                # Move obstacle back and forth smoothly
                t = float(self._env.physics.data.time)
                # Sine oscillation between -0.08 and 0.08 m
                target_pos = 0.08 * np.sin(self.dynamic_velocity * 2.0 * np.pi * t)
                self._env.physics.named.data.qpos["obstacle_slide_x"] = target_pos
            except Exception:
                pass

        return super().step(action)

    def get_obstacle_positions(self) -> List[np.ndarray]:
        """Returns list of 3D center positions of obstacles."""
        if self.obstacle_type == "none":
            return []
        try:
            pos = np.array(self._env.physics.named.data.xpos["obstacle_body"], copy=True)
            return [pos]
        except Exception:
            return []

    def get_physics_state(self) -> np.ndarray:
        return self._env.physics.get_state()

    def set_physics_state(self, state: np.ndarray):
        self._env.physics.set_state(state)
        self._env.physics.forward()
