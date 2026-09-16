"""Pure validation shared by experimental live policy and controller code."""
from __future__ import annotations

from typing import Iterable
import numpy as np


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


FRANKA_JOINT_LIMITS = [
    (-2.84, 2.84),
    (-1.71, 1.71),
    (-2.84, 2.84),
    (-3.02, -0.12),
    (-2.84, 2.84),
    (0.03, 3.70),
    (-2.84, 2.84)
]


def validate_joint_position_chunk(
    positions: Iterable[Iterable[float]],
    gripper: Iterable[float],
    current_q: Iterable[float] | None = None,
    *,
    max_steps: int = 64,
    max_step_delta: float = 0.25
):
    """
    Validates and clamps an absolute joint position chunk.
    Enforces Franka soft joint limits and step-to-step jerk limits.
    """
    q = np.asarray(positions, dtype=np.float64)
    g = np.asarray(gripper, dtype=np.float64)

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

    # 3. Gripper bounds
    g = np.clip(g, 0.0, 1.0)
    return q, g


def action_is_fresh(received_monotonic: float, *, now: float, ttl_s: float) -> bool:
    if not np.isfinite(received_monotonic) or not np.isfinite(now) or not np.isfinite(ttl_s) or ttl_s <= 0:
        return False
    return 0 <= now - received_monotonic <= ttl_s

