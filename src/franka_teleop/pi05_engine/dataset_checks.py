"""Fail-closed checks for native pi05-DROID training arrays."""
from __future__ import annotations

from typing import Iterable
import numpy as np


def validate_droid_training_actions(actions: Iterable[Iterable[float]], *, max_abs_joint_velocity: float = 2.0) -> np.ndarray:
    """Validate ``(N, 8)`` actions before they reach a GPU training loop.

    The threshold is a data-quality gate, not a controller limit. It catches
    timestamp/frame corruption while leaving the final motion limiter to the
    hardware-side controller.
    """
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 8 or values.shape[0] < 1:
        raise ValueError("DROID training actions must have shape (N>=1, 8)")
    if not np.isfinite(values).all():
        raise ValueError("DROID training actions contain NaN/Inf")
    if not np.isfinite(max_abs_joint_velocity) or max_abs_joint_velocity <= 0:
        raise ValueError("max_abs_joint_velocity must be finite and positive")
    observed = float(np.max(np.abs(values[:, :7])))
    if observed > max_abs_joint_velocity:
        raise ValueError(f"joint-velocity outlier {observed:.6g} exceeds data gate {max_abs_joint_velocity:.6g} rad/s")
    if np.any(values[:, 7] < 0.0) or np.any(values[:, 7] > 1.0):
        raise ValueError("gripper positions must be in [0, 1]")
    return values.astype(np.float32, copy=False)


def validate_paired_lengths(*, state_count: int, action_count: int,
                            front_frame_count: int, wrist_frame_count: int) -> None:
    counts = (state_count, action_count, front_frame_count, wrist_frame_count)
    if any(count < 1 for count in counts):
        raise ValueError(f"paired streams must be non-empty: {counts}")
    if len(set(counts)) != 1:
        raise ValueError(f"state/action/front/wrist lengths are not aligned: {counts}")
