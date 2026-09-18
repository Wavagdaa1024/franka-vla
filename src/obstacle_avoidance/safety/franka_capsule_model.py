# -*- coding: utf-8 -*-
"""
Kinematic Capsule Model for Franka Emika Panda.
Models the robot arm links and end-effector as 8 line-segment swept spheres (Capsules).
Computes analytical forward kinematics from joint configuration q (7,) to 8 space capsules
in < 0.1ms.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Capsule:
    """
    Geometric representation of a robot link capsule.
    A capsule is a line segment [p1, p2] swept by a sphere of radius r.
    """
    link_id: int
    link_name: str
    p1: np.ndarray      # (3,) Start point of the capsule axis in base frame
    p2: np.ndarray      # (3,) End point of the capsule axis in base frame
    radius: float       # Capsule cylinder/sphere radius in meters


# Modified DH Parameters (Craig's convention) for Franka Emika Panda:
# [a_{i-1}, d_i, alpha_{i-1}]
FRANKA_MDH = [
    [0.0,      0.333,   0.0],
    [0.0,      0.0,    -np.pi / 2],
    [0.0,      0.316,   np.pi / 2],
    [0.0825,   0.0,     np.pi / 2],
    [-0.0825,  0.384,  -np.pi / 2],
    [0.0,      0.0,     np.pi / 2],
    [0.088,    0.0,     np.pi / 2],
    [0.0,      0.107,   0.0]
]

# Standard Franka hand flange-to-fingertip transformation
DEFAULT_F_T_EE = np.array([
    [1.0,  0.0,  0.0,  0.0],
    [0.0,  1.0,  0.0,  0.0],
    [0.0,  0.0,  1.0,  0.1034],
    [0.0,  0.0,  0.0,  1.0]
], dtype=np.float64)

# Optimized physical capsule radii for Panda arm links (in meters)
CAPSULE_RADII = [
    0.080,   # Link 1: Base shoulder
    0.080,   # Link 2: Shoulder swivel
    0.075,   # Link 3: Upper arm
    0.075,   # Link 4: Elbow
    0.070,   # Link 5: Lower arm
    0.070,   # Link 6: Wrist
    0.065,   # Link 7: Flange
    0.060    # Link 8: Gripper fingers
]

LINK_NAMES = [
    "link1_base",
    "link2_shoulder",
    "link3_upperarm",
    "link4_elbow",
    "link5_lowerarm",
    "link6_wrist",
    "link7_flange",
    "link8_gripper"
]


class FrankaCapsuleModel:
    """
    Franka Panda 8-Capsule whole-arm kinematic collision model.
    """
    def __init__(self, f_t_ee: Optional[np.ndarray] = None):
        self.f_t_ee = np.asarray(f_t_ee if f_t_ee is not None else DEFAULT_F_T_EE, dtype=np.float64)

    def compute_link_frames(self, q: np.ndarray) -> List[np.ndarray]:
        """
        Computes forward kinematics for all intermediate joint origins.
        Returns 9 4x4 transform matrices: T_0, T_1, ..., T_7, T_ee
        """
        q = np.asarray(q, dtype=np.float64).flatten()
        assert len(q) == 7, f"q must have 7 elements, got {len(q)}"

        frames = []
        T_curr = np.eye(4, dtype=np.float64)
        frames.append(T_curr.copy())  # Base frame T_0

        for i in range(7):
            a, d, alpha = FRANKA_MDH[i]
            theta = q[i]
            ca, sa = np.cos(alpha), np.sin(alpha)
            ct, st = np.cos(theta), np.sin(theta)

            A_i = np.array([
                [ct,      -st,      0.0,   a],
                [st * ca,  ct * ca, -sa,  -sa * d],
                [st * sa,  ct * sa,  ca,   ca * d],
                [0.0,      0.0,      0.0,  1.0]
            ], dtype=np.float64)

            T_curr = T_curr @ A_i
            frames.append(T_curr.copy())  # T_1 to T_7

        # Frame 8 / Tool Flange
        a8, d8, alpha8 = FRANKA_MDH[7]
        ca8, sa8 = np.cos(alpha8), np.sin(alpha8)
        A_8 = np.array([
            [1.0, 0.0, 0.0, a8],
            [0.0, ca8, -sa8, -sa8 * d8],
            [0.0, sa8, ca8, ca8 * d8],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        T_flange = T_curr @ A_8

        # Frame EE / Gripper fingertip center
        T_ee = T_flange @ self.f_t_ee
        frames.append(T_ee.copy())

        return frames

    def get_capsules(self, q: np.ndarray) -> List[Capsule]:
        """
        Given 7-DOF joint configuration q, returns the list of 8 spatial capsules.
        """
        frames = self.compute_link_frames(q)
        # Origins of joints in robot base frame:
        # p[0]: Base origin [0, 0, 0]
        # p[1]..p[7]: Link 1..7 origins
        # p[8]: End-Effector tool tip
        origins = [F[:3, 3] for F in frames]

        capsules = []
        # Segment 0: Base to Joint 1
        capsules.append(Capsule(0, LINK_NAMES[0], origins[0], origins[1], CAPSULE_RADII[0]))
        # Segment 1: Joint 1 to Joint 2
        capsules.append(Capsule(1, LINK_NAMES[1], origins[1], origins[2], CAPSULE_RADII[1]))
        # Segment 2: Joint 2 to Joint 3
        capsules.append(Capsule(2, LINK_NAMES[2], origins[2], origins[3], CAPSULE_RADII[2]))
        # Segment 3: Joint 3 to Joint 4
        capsules.append(Capsule(3, LINK_NAMES[3], origins[3], origins[4], CAPSULE_RADII[3]))
        # Segment 4: Joint 4 to Joint 5
        capsules.append(Capsule(4, LINK_NAMES[4], origins[4], origins[5], CAPSULE_RADII[4]))
        # Segment 5: Joint 5 to Joint 6
        capsules.append(Capsule(5, LINK_NAMES[5], origins[5], origins[6], CAPSULE_RADII[5]))
        # Segment 6: Joint 6 to Joint 7
        capsules.append(Capsule(6, LINK_NAMES[6], origins[6], origins[7], CAPSULE_RADII[6]))
        # Segment 7: Joint 7 to EE Tool tip
        capsules.append(Capsule(7, LINK_NAMES[7], origins[7], origins[8], CAPSULE_RADII[7]))

        return capsules
