"""Pure validation shared by experimental live policy and controller code."""
from __future__ import annotations

from typing import Iterable, Optional
import numpy as np

from franka_teleop.kinematics import (
    FRANKA_JOINT_LIMITS,
    DEFAULT_Z_FLOOR,
    forward_kinematics,
    analytical_jacobian
)


def validate_joint_velocity_chunk(velocities: Iterable[Iterable[float]], gripper: Iterable[float], *, max_steps: int = 64):
    v = np.asarray(velocities, dtype=np.float64)
    g = np.asarray(gripper, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 7 or not 1 <= v.shape[0] <= max_steps:
        raise ValueError("joint velocity chunk must have shape (1..max_steps, 7)")
    if g.shape != (v.shape[0],) or not np.isfinite(v).all() or not np.isfinite(g).all():
        raise ValueError("live action chunk contains invalid values or mismatched gripper length")
    if np.any(g < -0.05) or np.any(g > 1.05):
        raise ValueError("gripper position must be in [0,1]")
    g = np.clip(g, 0.0, 1.0)
    return v, g


def validate_joint_position_chunk(
    positions: Iterable[Iterable[float]],
    gripper: Iterable[float],
    current_q: Iterable[float] | None = None,
    *,
    max_steps: int = 64,
    max_step_delta: float = 0.25,
    z_floor: Optional[float] = DEFAULT_Z_FLOOR
):
    """
    Validates and clamps an absolute joint position chunk.
    Enforces:
      1. Franka Panda soft joint limits
      2. Step-to-step jerk limits (anti-jump protection)
      3. Z_floor table collision guard (EE z >= z_floor)
      4. Gripper bounds [0, 1]
    """
    q = np.asarray(positions, dtype=np.float64).copy()
    g = np.asarray(gripper, dtype=np.float64).copy()

    if q.ndim != 2 or q.shape[1] != 7 or not 1 <= q.shape[0] <= max_steps:
        raise ValueError("joint position chunk must have shape (1..max_steps, 7)")
    if g.shape != (q.shape[0],) or not np.isfinite(q).all() or not np.isfinite(g).all():
        raise ValueError("live action chunk contains non-finite values or mismatched gripper length")

    # 1. Franka Panda soft joint limits clamp
    for j in range(7):
        low, high = FRANKA_JOINT_LIMITS[j]
        q[:, j] = np.clip(q[:, j], low, high)

    # 2. Step-to-step jerk clamp (anti-jump protection)
    ref_q = np.asarray(current_q, dtype=np.float64) if current_q is not None else q[0]
    for step in range(q.shape[0]):
        prev = ref_q if step == 0 else q[step - 1]
        delta = q[step] - prev
        clamped_delta = np.clip(delta, -max_step_delta, max_step_delta)
        q[step] = prev + clamped_delta

    # 3. Real Z-floor table collision guard (lifting penetrated waypoints)
    if z_floor is not None:
        for step in range(q.shape[0]):
            p_ee = forward_kinematics(q[step])[:3, 3]
            if p_ee[2] < z_floor:
                dz = float(z_floor - p_ee[2])
                J = analytical_jacobian(q[step])
                J_z = J[2, :]
                dq_lift = J_z * (dz / (np.dot(J_z, J_z) + 1e-4))
                q[step] += dq_lift
                for j in range(7):
                    q[step, j] = np.clip(q[step, j], FRANKA_JOINT_LIMITS[j][0], FRANKA_JOINT_LIMITS[j][1])

    # 4. Gripper bounds
    g = np.clip(g, 0.0, 1.0)
    return q, g


def action_is_fresh(received_monotonic: float, *, now: float, ttl_s: float) -> bool:
    if not np.isfinite(received_monotonic) or not np.isfinite(now) or not np.isfinite(ttl_s) or ttl_s <= 0:
        return False
    return 0 <= now - received_monotonic <= ttl_s
