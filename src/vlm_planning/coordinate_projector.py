import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

def pixel_to_camera(u, v, depth_m, intrinsics):
    """Back-project an aligned depth pixel into the RealSense camera frame."""
    fx, fy, cx, cy = (intrinsics[k] for k in ('fx', 'fy', 'cx', 'cy'))
    z = float(depth_m)
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=float)

def transform_point(point, camera_to_base):
    p = np.r_[np.asarray(point, dtype=float), 1.0]
    return (np.asarray(camera_to_base, dtype=float) @ p)[:3]

def validate_transform(T):
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError('变换矩阵必须为 4x4')
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError('齐次矩阵末行非法')
    if not np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-3):
        raise ValueError('旋转矩阵非法')
    return T


class TablePlaneProjector:
    """
    Planar Affine / Homography Projection mapping front-camera 2D pixels (u, v)
    directly to Franka base Cartesian coordinates (X, Y, Z).
    
    Calibrated against human teleoperation demonstration datasets:
      - u (horizontal pixel 0..640): maps to Franka Y axis (robot left/right)
      - v (vertical pixel 0..480): maps to Franka X axis (robot reach forward/backward)
    """
    DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "table_plane_calibration.json"

    # Default parameters calibrated from teleop demonstration dataset:
    # Basket center: (u=166.5, v=213.5) -> (X=0.530, Y=-0.116, Z=0.165)
    # Red cube (Ep00): (u=374.5, v=194.0) -> (X=0.500, Y=+0.124, Z=0.122)
    DEFAULT_PARAMS = {
        "kx": 0.00105,       # dX / dv (m per pixel)
        "bx": 0.2964,        # X intercept (m)
        "ky": 0.001154,      # dY / du (m per pixel)
        "by": -0.3081,       # Y intercept (m)
        "z_hover": 0.160,    # Safe transit height (m)
        "z_approach": 0.070, # Pre-grasp approach height (m)
        "z_grasp": 0.013,    # Table grasp height (m) - aligned with live physical arm: 0.0128m
        "z_drop": 0.075,     # Basket drop height (m)
        "ready_pose_xyz": [0.4421, -0.0291, 0.200],
        "default_quat_xyzw": [1.0, 0.0, 0.0, 0.0],
        "workspace_bounds": {
            "x_min": 0.32, "x_max": 0.72,
            "y_min": -0.35, "y_max": 0.35,
            "z_min": 0.005, "z_max": 0.50
        }
    }

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        self.params = dict(self.DEFAULT_PARAMS)
        self.load_calibration()

    def load_calibration(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.params.update(data)
            except Exception as e:
                raise ValueError(f"calibration file is invalid: {self.config_path}") from e

    def save_calibration(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.params, f, indent=2, ensure_ascii=False)

    def project_pixel(self, u: float, v: float, z: Optional[float] = None) -> np.ndarray:
        """
        Projects pixel coordinates (u, v) on the table plane to robot base (X, Y, Z).
        Uses calibrated 2D Affine transformation matrix if present, else fallback to 1D linear.
        """
        if "affine_matrix" in self.params:
            M = self.params["affine_matrix"]
            x = float(M["a11"] * u + M["a12"] * v + M["b1"])
            y = float(M["a21"] * u + M["a22"] * v + M["b2"])
        else:
            kx = self.params["kx"]
            bx = self.params["bx"]
            ky = self.params["ky"]
            by = self.params["by"]
            x = float(kx * v + bx)
            y = float(ky * u + by)

        z_out = float(z if z is not None else self.params["z_grasp"])

        self.validate_bounds(x, y, z_out)
        return np.array([round(x, 4), round(y, 4), round(z_out, 4)], dtype=float)

    def validate_bounds(self, x: float, y: float, z: float):
        b = self.params["workspace_bounds"]
        if not all(np.isfinite(float(value)) for value in (x, y, z)):
            raise ValueError("Cartesian target must be finite")
        if not all(np.isfinite(float(b[key])) for key in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")):
            raise ValueError("workspace bounds must be finite")
        if b["x_min"] > b["x_max"] or b["y_min"] > b["y_max"] or b["z_min"] > b["z_max"]:
            raise ValueError("workspace bounds are inverted")
        if not (b["x_min"] <= x <= b["x_max"]):
            raise ValueError(f"Cartesian X={x:.4f} outside safe bounds [{b['x_min']}, {b['x_max']}]")
        if not (b["y_min"] <= y <= b["y_max"]):
            raise ValueError(f"Cartesian Y={y:.4f} outside safe bounds [{b['y_min']}, {b['y_max']}]")
        if not (b["z_min"] <= z <= b["z_max"]):
            raise ValueError(f"Cartesian Z={z:.4f} outside safe bounds [{b['z_min']}, {b['z_max']}]")

    def generate_pick_and_place_waypoints(
        self,
        cube_uv: Tuple[float, float],
        basket_uv: Tuple[float, float],
        z_grasp: Optional[float] = None,
        z_drop: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Generates deterministic Cartesian waypoint sequence for picking target cube
        and placing into basket.
        
        Gripper convention: 1.0 = OPEN, 0.0 = CLOSED
        """
        z_g = float(z_grasp if z_grasp is not None else self.params["z_grasp"])
        z_d = float(z_drop if z_drop is not None else self.params["z_drop"])

        p_ready = np.array(self.params["ready_pose_xyz"], dtype=float)
        p_cube_hover = self.project_pixel(cube_uv[0], cube_uv[1], z=self.params["z_hover"])
        p_cube_approach = self.project_pixel(cube_uv[0], cube_uv[1], z=self.params["z_approach"])
        p_cube_grasp = self.project_pixel(cube_uv[0], cube_uv[1], z=z_g)
        
        p_basket_hover = self.project_pixel(basket_uv[0], basket_uv[1], z=self.params["z_hover"])
        p_basket_drop = self.project_pixel(basket_uv[0], basket_uv[1], z=z_d)

        quat = self.params["default_quat_xyzw"]

        waypoints = [
            # 1. Start from ready pose, open gripper
            {
                "step": 1,
                "name": "PREPARE_READY",
                "pos": p_ready.tolist(),
                "quat": quat,
                "gripper": 1.0,
                "speed": 0.15,
                "pause_s": 0.2
            },
            # 2. Hover horizontally over target cube
            {
                "step": 2,
                "name": "HOVER_OVER_CUBE",
                "pos": p_cube_hover.tolist(),
                "quat": quat,
                "gripper": 1.0,
                "speed": 0.15,
                "pause_s": 0.1
            },
            # 3. Approach cube to pre-grasp height
            {
                "step": 3,
                "name": "APPROACH_CUBE",
                "pos": p_cube_approach.tolist(),
                "quat": quat,
                "gripper": 1.0,
                "speed": 0.10,
                "pause_s": 0.05
            },
            # 4. Descend to table grasp height
            {
                "step": 4,
                "name": "DESCEND_TO_GRASP",
                "pos": p_cube_grasp.tolist(),
                "quat": quat,
                "gripper": 1.0,
                "speed": 0.06,
                "pause_s": 0.2
            },
            # 5. Grip cube securely
            {
                "step": 5,
                "name": "CLOSE_GRIPPER",
                "pos": p_cube_grasp.tolist(),
                "quat": quat,
                "gripper": 0.0,
                "speed": 0.05,
                "pause_s": 0.6
            },
            # 6. Lift cube vertically back to hover height
            {
                "step": 6,
                "name": "LIFT_OBJECT",
                "pos": p_cube_hover.tolist(),
                "quat": quat,
                "gripper": 0.0,
                "speed": 0.10,
                "pause_s": 0.1
            },
            # 7. Translate horizontally to hover over basket
            {
                "step": 7,
                "name": "HOVER_OVER_BASKET",
                "pos": p_basket_hover.tolist(),
                "quat": quat,
                "gripper": 0.0,
                "speed": 0.15,
                "pause_s": 0.1
            },
            # 8. Descend slightly into basket
            {
                "step": 8,
                "name": "DESCEND_INTO_BASKET",
                "pos": p_basket_drop.tolist(),
                "quat": quat,
                "gripper": 0.0,
                "speed": 0.08,
                "pause_s": 0.1
            },
            # 9. Release cube into basket
            {
                "step": 9,
                "name": "OPEN_GRIPPER_RELEASE",
                "pos": p_basket_drop.tolist(),
                "quat": quat,
                "gripper": 1.0,
                "speed": 0.05,
                "pause_s": 0.6
            },
            # 10. Ascend vertically out of basket
            {
                "step": 10,
                "name": "ASCEND_FROM_BASKET",
                "pos": p_basket_hover.tolist(),
                "quat": quat,
                "gripper": 1.0,
                "speed": 0.12,
                "pause_s": 0.1
            },
            # 11. Return smoothly to tabletop ready pose
            {
                "step": 11,
                "name": "RETURN_TO_READY",
                "pos": p_ready.tolist(),
                "quat": quat,
                "gripper": 1.0,
                "speed": 0.15,
                "pause_s": 0.2
            }
        ]
        return waypoints
