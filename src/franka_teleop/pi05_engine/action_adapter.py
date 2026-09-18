"""Explicit pi05-DROID action semantics; no robot I/O.

The pretrained DROID policy emits seven joint velocities (rad/s) followed by
one gripper-position value. This module deliberately returns validated values
for a downstream controller; it never turns them into Cartesian poses or
publishes commands.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


ACTION_DIM = 8
ARM_DIM = 7


@dataclass(frozen=True)
class DroidActionChunk:
    joint_velocity_rad_s: np.ndarray
    gripper_position: np.ndarray
    dt_s: float
    semantics: str = "droid_joint_velocity_gripper_position"

    @property
    def shape(self) -> tuple[int, int]:
        return (self.joint_velocity_rad_s.shape[0], ACTION_DIM)

    def as_array(self) -> np.ndarray:
        return np.concatenate((self.joint_velocity_rad_s,
                               self.gripper_position[:, None]), axis=1)


def validate_droid_action_chunk(actions: Iterable[Iterable[float]]) -> np.ndarray:
    """Return a finite ``(T, 8)`` float32 chunk or raise a clear error."""
    array = np.asarray(actions, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != ACTION_DIM or array.shape[0] < 1:
        raise ValueError("DROID action chunk must have shape (T>=1, 8)")
    if not np.isfinite(array).all():
        raise ValueError("DROID action chunk contains NaN/Inf")
    if np.any(array[:, ARM_DIM] < 0.0) or np.any(array[:, ARM_DIM] > 1.0):
        raise ValueError("gripper position must be in [0, 1]")
    return array.astype(np.float32, copy=False)


def split_droid_action_chunk(actions: Iterable[Iterable[float]], *, dt_s: float) -> DroidActionChunk:
    """Split a policy chunk without changing units or silently clipping values."""
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be finite and positive")
    array = validate_droid_action_chunk(actions)
    return DroidActionChunk(array[:, :ARM_DIM].copy(), array[:, ARM_DIM].copy(), float(dt_s))


def integrate_for_offline_preview(current_q: Iterable[float], chunk: DroidActionChunk,
                                  *, joint_limits: Optional[np.ndarray] = None) -> np.ndarray:
    """Integrate velocities for plotting/replay only; never use as a pose command.

    ``joint_limits`` may be an explicit ``(7, 2)`` array. Omitting it avoids
    inventing robot limits; values outside limits are rejected by the caller's
    safety layer rather than clipped here.
    """
    q = np.asarray(current_q, dtype=np.float64)
    if q.shape != (ARM_DIM,) or not np.isfinite(q).all():
        raise ValueError("current_q must be finite with shape (7,)")
    if chunk.joint_velocity_rad_s.ndim != 2 or chunk.joint_velocity_rad_s.shape[1] != ARM_DIM:
        raise ValueError("chunk arm action must have shape (T, 7)")
    preview = q[None, :] + np.cumsum(chunk.joint_velocity_rad_s * chunk.dt_s, axis=0)
    if not np.isfinite(preview).all():
        raise ValueError("integrated preview is non-finite")
    if joint_limits is not None:
        limits = np.asarray(joint_limits, dtype=np.float64)
        if limits.shape != (ARM_DIM, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] > limits[:, 1]):
            raise ValueError("joint_limits must be finite with shape (7, 2)")
        if np.any(preview < limits[:, 0]) or np.any(preview > limits[:, 1]):
            raise ValueError("integrated preview exceeds explicit joint limits")
    return preview.astype(np.float32)
