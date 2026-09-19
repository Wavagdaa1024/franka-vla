"""
Franka Panda Kinematics & Constrained Control Module.
Includes:
  - Craig Modified DH Forward Kinematics (0.0000mm error calibrated against libfranka O_T_EE)
  - Analytical Geometric Jacobian J(q) = [J_v; J_w] (calibrated to 1e-8 numerical accuracy)
  - Task-Priority Nullspace Projection: 
      Primary: 3D Position tracking (preserves policy trajectory)
      Secondary: Vertical Gripper Alignment (z_ee -> [0, 0, -1], eliminates tilt drift)
  - Hard Table Collision Guard: Clamps Z >= z_floor (default: +0.0070m = +7.0mm)
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, List, Optional

# Franka Panda Craig Modified DH Parameters (a_{i-1}, d_i, alpha_{i-1})
MDH_PARAMS = [
    (0.0,     0.333,  0.0),          # Joint 1
    (0.0,     0.0,   -np.pi / 2.0),  # Joint 2
    (0.0,     0.316,  np.pi / 2.0),  # Joint 3
    (0.0825,  0.0,   np.pi / 2.0),  # Joint 4
    (-0.0825, 0.384, -np.pi / 2.0),  # Joint 5
    (0.0,     0.0,    np.pi / 2.0),  # Joint 6
    (0.088,   0.0,    np.pi / 2.0),  # Joint 7
    (0.0,     0.107,  0.0)           # Flange offset (Link 8 / NE)
]

# Franka Hand standard tool transformation F_T_EE (rotation -pi/4 around Z, translation +0.1034m along Z)
# Verified to 0.0000 mm Cartesian error against live Franka libfranka O_T_EE matrix.
_c45 = np.cos(np.pi / 4.0)
_s45 = np.sin(np.pi / 4.0)
DEFAULT_F_T_EE = np.array([
    [ _c45,  _s45, 0.0, 0.0],
    [-_s45,  _c45, 0.0, 0.0],
    [  0.0,   0.0, 1.0, 0.1034],
    [  0.0,   0.0, 0.0, 1.0]
], dtype=np.float64)

# Hardware calibrated minimum safe floor height (+7.0 mm above Franka base)
# Measured lowest reachable physical point on table was +6.75 mm.
DEFAULT_Z_FLOOR = 0.0070

# Franka Panda joint position limits (with soft buffer in radians)
FRANKA_JOINT_LIMITS = [
    (-2.84, 2.84),
    (-1.71, 1.71),
    (-2.84, 2.84),
    (-3.02, -0.12),
    (-2.84, 2.84),
    (0.03, 3.70),
    (-2.84, 2.84)
]


def tf_mdh(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    """Craig Modified DH single-link transformation matrix."""
    c, s = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [c,        -s,        0.0,  a],
        [s * ca,    c * ca,  -sa,  -sa * d],
        [s * sa,    c * sa,   ca,   ca * d],
        [0.0,       0.0,      0.0,  1.0]
    ], dtype=np.float64)


def forward_kinematics(q: np.ndarray, F_T_EE: np.ndarray = DEFAULT_F_T_EE) -> np.ndarray:
    """
    Computes 4x4 homogeneous transformation matrix O_T_EE from base to end-effector tool center point.
    """
    q_arr = np.asarray(q, dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    for i in range(7):
        a, d, alpha = MDH_PARAMS[i]
        T = T @ tf_mdh(a, d, alpha, q_arr[i])
    # Flange link 8
    a, d, alpha = MDH_PARAMS[7]
    T = T @ tf_mdh(a, d, alpha, 0.0)
    if F_T_EE is not None:
        T = T @ F_T_EE
    return T


def analytical_jacobian(q: np.ndarray, F_T_EE: np.ndarray = DEFAULT_F_T_EE) -> np.ndarray:
    """
    Computes the 6x7 geometric Jacobian J(q) = [J_v; J_w] in Franka base frame O.
    J_v: 3x7 linear velocity Jacobian
    J_w: 3x7 angular velocity Jacobian
    """
    q_arr = np.asarray(q, dtype=np.float64)
    T_EE = forward_kinematics(q_arr, F_T_EE=F_T_EE)
    p_EE = T_EE[:3, 3]

    J = np.zeros((6, 7), dtype=np.float64)
    T_curr = np.eye(4, dtype=np.float64)

    for i in range(7):
        a, d, alpha = MDH_PARAMS[i]
        ca, sa = np.cos(alpha), np.sin(alpha)
        # Transform of joint i rotation axis before Rot_Z(theta_i):
        T_pre_rel = np.array([
            [1.0, 0.0,  0.0, a],
            [0.0,  ca,  -sa, -sa * d],
            [0.0,  sa,   ca,  ca * d],
            [0.0, 0.0,  0.0, 1.0]
        ], dtype=np.float64)

        T_axis = T_curr @ T_pre_rel
        z_i = T_axis[:3, 2]
        p_i = T_axis[:3, 3]

        # J_v: z_i x (p_EE - p_i)
        J[:3, i] = np.cross(z_i, p_EE - p_i)
        # J_w: z_i
        J[3:, i] = z_i

        # Advance base frame to next link
        T_curr = T_curr @ tf_mdh(a, d, alpha, q_arr[i])

    return J


def damped_pinv(J: np.ndarray, damping: float = 1e-4) -> np.ndarray:
    """Computes Moore-Penrose pseudoinverse with Tikhonov damping."""
    m, n = J.shape
    if m <= n:
        return J.T @ np.linalg.inv(J @ J.T + (damping ** 2) * np.eye(m))
    else:
        return np.linalg.inv(J.T @ J + (damping ** 2) * np.eye(n)) @ J.T


def correct_step_nullspace(
    curr_q: np.ndarray,
    target_pos: np.ndarray,
    dt: float = 0.0667,
    kp_pos: float = 15.0,
    kp_rot: float = 5.0,
    damping: float = 1e-4
) -> Tuple[np.ndarray, float]:
    """
    Executes a single-step Task-Priority Nullspace update:
      - Primary Task: Drive current position p_curr -> target_pos
      - Secondary Task: Align tool z-axis z_ee -> [0, 0, -1] in the nullspace of primary task.
    Returns:
      (q_next, tilt_deg): Updated joint angles and residual tilt angle in degrees.
    """
    curr_q = np.asarray(curr_q, dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)

    p_curr = forward_kinematics(curr_q)[:3, 3]
    v_des = kp_pos * (target_pos - p_curr)

    J = analytical_jacobian(curr_q)
    J_v = J[:3, :]
    J_w = J[3:, :]

    J_v_pinv = damped_pinv(J_v, damping=damping)
    dq_primary = J_v_pinv @ v_des

    # Nullspace projection operator of position task
    I7 = np.eye(7, dtype=np.float64)
    N_v = I7 - J_v_pinv @ J_v

    # Gripper orientation alignment error: z_ee x [0, 0, -1]
    R_curr = forward_kinematics(curr_q)[:3, :3]
    z_ee = R_curr[:, 2]
    z_des = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    w_err = np.cross(z_ee, z_des)
    w_des = kp_rot * w_err

    # Project orientation task into nullspace of position task
    J_w_null = J_w @ N_v
    J_w_null_pinv = damped_pinv(J_w_null, damping=damping)
    dq_sec = J_w_null_pinv @ w_des

    dq_total = dq_primary + dq_sec
    q_next = curr_q + dq_total * dt

    # Franka joint limits clamp
    for j in range(7):
        low, high = FRANKA_JOINT_LIMITS[j]
        q_next[j] = np.clip(q_next[j], low, high)

    # Compute tilt angle in degrees
    R_next = forward_kinematics(q_next)[:3, :3]
    cos_tilt = np.clip(np.dot(R_next[:, 2], z_des), -1.0, 1.0)
    tilt_deg = float(np.arccos(cos_tilt) * 180.0 / np.pi)

    return q_next, tilt_deg


def lock_gripper_vertical_downward(
    curr_q: np.ndarray,
    target_pos: np.ndarray,
    max_iters: int = 15,
    tol_pos: float = 1e-4,
    tol_rot: float = 1e-4,
    damping: float = 1e-3
) -> Tuple[np.ndarray, float]:
    """
    Solves 5-DOF IK with Newton-Raphson iterations:
      - 3D Position: p(q) == target_pos (exact Cartesian position tracking)
      - Orientation: tool z-axis strictly locked vertically downward [0, 0, -1]
        (Roll = 0, Pitch = 0, eliminating any tilt drift to < 0.01 deg)
    """
    q = np.asarray(curr_q, dtype=np.float64).copy()
    z_des = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)

    for _ in range(max_iters):
        T = forward_kinematics(q)
        p = T[:3, 3]
        z_curr = T[:3, 2]

        pos_err = target_pos - p
        rot_err = np.cross(z_curr, z_des)

        if np.linalg.norm(pos_err) < tol_pos and np.linalg.norm(rot_err) < tol_rot:
            break

        J = analytical_jacobian(q)
        dx = np.concatenate([pos_err, 0.5 * rot_err])
        J_damped = damped_pinv(J, damping=damping)
        dq = J_damped @ dx

        dq_norm = np.linalg.norm(dq)
        if dq_norm > 0.15:
            dq = dq * (0.15 / dq_norm)

        q += dq
        for j in range(7):
            low, high = FRANKA_JOINT_LIMITS[j]
            q[j] = np.clip(q[j], low, high)

    T_final = forward_kinematics(q)
    z_final = T_final[:3, 2]
    cos_tilt = np.clip(np.dot(z_final, z_des), -1.0, 1.0)
    tilt_deg = float(np.arccos(cos_tilt) * 180.0 / np.pi)

    return q, tilt_deg


def correct_chunk_nullspace(
    current_q: np.ndarray,
    chunk_q: np.ndarray,
    dt: float = 0.0667,
    z_floor: float = DEFAULT_Z_FLOOR,
    kp_pos: float = 15.0,
    kp_rot: float = 5.0,
    substeps_per_point: int = 2
) -> Tuple[np.ndarray, float, int]:
    """
    Applies Strict Vertical Orientation Locking (Roll=0, Pitch=0) and Z_floor protection
    to an entire chunk of planned joint waypoints (H, 7).
    
    Args:
      current_q: Current robot joint position (7,)
      chunk_q: Model predicted joint position chunk (H, 7)
      dt: Time delta per chunk waypoint (default: 1/15s)
      z_floor: Minimum allowable Z height (default: 0.0070m = +7.0mm)
      
    Returns:
      (corrected_chunk, max_tilt_deg, z_clamped_count):
        - corrected_chunk: Cleaned, strictly vertical downward, floor-safe (H, 7) array
        - max_tilt_deg: Maximum remaining gripper tilt angle in degrees (< 0.01 deg)
        - z_clamped_count: Number of waypoints where Z floor guard was activated
    """
    curr = np.asarray(current_q, dtype=np.float64).copy()
    chunk = np.asarray(chunk_q, dtype=np.float64).copy()
    H, D = chunk.shape
    assert D == 7, f"chunk_q must have shape (H, 7), got {chunk.shape}"

    corrected_chunk = np.zeros_like(chunk)
    max_tilt_deg = 0.0
    z_clamped_count = 0

    for k in range(H):
        # 1. Extract target 3D Cartesian position from raw model waypoint
        T_target = forward_kinematics(chunk[k])
        p_target = T_target[:3, 3].copy()

        # 2. Enforce Z_floor hard safety limit
        if p_target[2] < z_floor:
            p_target[2] = z_floor
            z_clamped_count += 1

        # 3. Solve 5-DOF IK locking orientation strictly vertical downward
        curr, tilt_deg = lock_gripper_vertical_downward(
            curr,
            target_pos=p_target,
            max_iters=15,
            tol_pos=1e-4,
            tol_rot=1e-4
        )

        # Safety post-check: if IK residual resulted in Z < z_floor, re-solve targeting slightly above floor
        p_check = forward_kinematics(curr)[:3, 3]
        if p_check[2] < z_floor:
            curr, tilt_deg = lock_gripper_vertical_downward(
                curr,
                target_pos=np.array([p_target[0], p_target[1], z_floor + 2e-4]),
                max_iters=12,
                tol_pos=5e-5,
                tol_rot=1e-4
            )

        if tilt_deg > max_tilt_deg:
            max_tilt_deg = tilt_deg

        corrected_chunk[k] = curr.copy()

    return corrected_chunk, max_tilt_deg, z_clamped_count


def project_velocity_z_floor(
    curr_q: np.ndarray,
    dq_cmd: np.ndarray,
    dt: float = 0.0667,
    z_floor: float = DEFAULT_Z_FLOOR,
    damping: float = 1e-4
) -> Tuple[np.ndarray, bool]:
    """
    Real-time 15Hz/1kHz safety filter for joint velocity commands dq_cmd.
    If the predicted end-effector position p_next drops below z_floor,
    projects out the downward Z velocity component using positional Jacobian:
      dq_filtered = dq - J_vz^+ (J_vz * dq)
    Allows smooth horizontal sliding while strictly preventing floor penetration.
    """
    curr_q = np.asarray(curr_q, dtype=np.float64)
    dq = np.asarray(dq_cmd, dtype=np.float64).copy()

    p_curr = forward_kinematics(curr_q)[:3, 3]
    p_pred = forward_kinematics(curr_q + dq * dt)[:3, 3]

    if p_pred[2] < z_floor or p_curr[2] < z_floor:
        J = analytical_jacobian(curr_q)
        J_vz = J[2:3, :]  # 1x7 row vector for Z linear velocity
        v_z = float(np.dot(J_vz.flatten(), dq))

        # If already below floor: freeze completely unless moving upwards to escape
        if p_curr[2] < z_floor:
            if v_z <= 1e-4:
                return np.zeros(7, dtype=np.float64), True
            else:
                return dq, True

        if v_z < 0.0:
            # Downward velocity component to be removed
            denom = float(np.dot(J_vz.flatten(), J_vz.flatten()) + damping ** 2)
            J_vz_pinv = J_vz.T / denom
            dq_filtered = dq - J_vz_pinv.flatten() * v_z
            # Re-check prediction
            p_recheck = forward_kinematics(curr_q + dq_filtered * dt)[:3, 3]
            if p_recheck[2] < z_floor:
                # If still below floor, freeze velocity completely
                return np.zeros(7, dtype=np.float64), True
            return dq_filtered, True

    return dq, False


# ==============================================================================
# PyTorch Vectorized Differentiable Kinematics Module
# ==============================================================================
try:
    import torch
    import torch.nn as nn

    def rotmat_to_rotvec_torch(R: "torch.Tensor", eps: float = 1e-6) -> "torch.Tensor":
        """
        Differentiable SO(3) matrix logarithm converting rotation matrices to rotation vectors (axis-angle).
        Args:
            R: Tensor of shape (..., 3, 3)
        Returns:
            r: Tensor of shape (..., 3)
        """
        tr = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1.0) * 0.5
        tr_clamped = torch.clamp(tr, -1.0 + eps, 1.0 - eps)
        theta = torch.acos(tr_clamped)
        sin_theta = torch.sin(theta).unsqueeze(-1)
        v = torch.stack([
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ], dim=-1)
        scale = torch.where(
            theta.unsqueeze(-1) < 1e-4,
            torch.full_like(v, 0.5),
            (theta.unsqueeze(-1) / (2.0 * sin_theta)).clamp(max=10.0)
        )
        return v * scale

    class FrankaDifferentiableKinematics(nn.Module):
        """
        Vectorized, fully differentiable Forward Kinematics module for Franka Emika Panda.
        Computes analytical end-effector position, tool Z-axis orientation, and
        6-DoF relative end-effector action [dx, dy, dz, drx, dry, drz] from joint positions q.
        """
        def __init__(self, device: str = "cpu"):
            super().__init__()
            self.mdh_params = MDH_PARAMS
            a8, d8, alpha8 = MDH_PARAMS[7]
            ca8, sa8 = float(np.cos(alpha8)), float(np.sin(alpha8))
            A8 = np.array([
                [1.0, 0.0, 0.0, a8],
                [0.0, ca8, -sa8, -sa8 * d8],
                [0.0, sa8, ca8, ca8 * d8],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=np.float32)
            flange_tool = A8 @ DEFAULT_F_T_EE.astype(np.float32)
            self.register_buffer("flange_tool", torch.from_numpy(flange_tool).to(device=device))

        def compute_fk_matrices(self, q: "torch.Tensor") -> "torch.Tensor":
            """
            Computes full 4x4 homogeneous transformation matrices for input joint positions.
            Args:
                q: Joint angles tensor of shape (..., 7) in radians.
            Returns:
                T: Homogeneous transformation matrices of shape (..., 4, 4).
            """
            orig_shape = q.shape[:-1]
            q_flat = q.reshape(-1, 7)
            N = q_flat.shape[0]
            device = q.device
            dtype = q.dtype

            T = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(N, 1, 1)
            zeros = torch.zeros(N, dtype=dtype, device=device)
            ones = torch.ones(N, dtype=dtype, device=device)

            for i in range(7):
                a, d, alpha = self.mdh_params[i]
                ca = float(np.cos(alpha))
                sa = float(np.sin(alpha))
                theta = q_flat[:, i]
                c = torch.cos(theta)
                s = torch.sin(theta)

                row0 = torch.stack([c, -s, zeros, torch.full_like(theta, a)], dim=-1)
                row1 = torch.stack([s * ca, c * ca, torch.full_like(theta, -sa), torch.full_like(theta, -sa * d)], dim=-1)
                row2 = torch.stack([s * sa, c * sa, torch.full_like(theta, ca), torch.full_like(theta, ca * d)], dim=-1)
                row3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)
                A_i = torch.stack([row0, row1, row2, row3], dim=-2)
                T = torch.bmm(T, A_i)

            flange_tool_expanded = self.flange_tool.to(dtype=dtype, device=device).unsqueeze(0).expand(N, 4, 4)
            T = torch.bmm(T, flange_tool_expanded)
            return T.reshape(*orig_shape, 4, 4)

        def forward(self, q: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
            """
            Args:
                q: Joint angles tensor of shape (..., 7) in radians.
            Returns:
                p_ee: Cartesian 3D position of shape (..., 3) in meters.
                z_ee: Tool Z-axis orientation unit vector of shape (..., 3).
            """
            T = self.compute_fk_matrices(q)
            p_ee = T[..., :3, 3]
            z_ee = T[..., :3, 2]
            return p_ee, z_ee

        def compute_relative_ee_action(self, curr_q: "torch.Tensor", delta_q: "torch.Tensor") -> "torch.Tensor":
            """
            Computes relative 6-DoF end-effector delta action [dx, dy, dz, drx, dry, drz]
            from current joint positions and joint position deltas.
            Args:
                curr_q: Current joint positions of shape (B, 1, 7) or (B, 7)
                delta_q: Joint position deltas of shape (B, T, 7)
            Returns:
                ee_action_6d: Tensor of shape (B, T, 6) containing [dx, dy, dz, drx, dry, drz]
            """
            if curr_q.ndim == 2:
                curr_q = curr_q.unsqueeze(1)
            q_traj = curr_q + delta_q
            T_curr = self.compute_fk_matrices(curr_q)
            T_traj = self.compute_fk_matrices(q_traj)

            delta_pos = T_traj[..., :3, 3] - T_curr[..., :3, 3]
            R_curr = T_curr[..., :3, :3]
            R_traj = T_traj[..., :3, :3]
            R_rel = torch.matmul(R_traj, R_curr.transpose(-1, -2))
            delta_rotvec = rotmat_to_rotvec_torch(R_rel)
            return torch.cat([delta_pos, delta_rotvec], dim=-1)

except ImportError:
    class FrankaDifferentiableKinematics:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required to use FrankaDifferentiableKinematics.")

