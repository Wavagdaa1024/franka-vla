"""Offline and streaming conversion from teleop samples to native pi05-DROID actions.

Supported action representations:
- droid_joint_delta: delta_q = q[t+1] - q[t] (7 rad) + gripper (0.0=OPEN, 1.0=CLOSED).
  Native format for pi05_droid_jointpos SFT.
- droid_joint_velocity: dq/dt (7 rad/s) + gripper (0.0=OPEN, 1.0=CLOSED).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def _vector(sample: dict[str, Any], key: str, size: int) -> np.ndarray:
    value = np.asarray(sample.get(key), dtype=np.float64)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"{key} must be finite with shape ({size},)")
    return value


def droid_joint_velocity_action(current: dict[str, Any], following: dict[str, Any], *,
                                max_dt_s: float = 0.5) -> np.ndarray:
    """Return ``[dq/dt (7 rad/s), gripper_position]`` from two robot samples."""
    q0 = _vector(current, "q", 7)
    q1 = _vector(following, "q", 7)
    t0 = int(current.get("_gpu_receive_monotonic_ns", -1))
    t1 = int(following.get("_gpu_receive_monotonic_ns", -1))
    dt = (t1 - t0) / 1e9
    if not math.isfinite(dt) or dt <= 0 or dt > max_dt_s:
        raise ValueError("consecutive samples require a positive, bounded receive-time delta")
    grip = float(following.get("target_gripper", math.nan))
    if not math.isfinite(grip) or not 0.0 <= grip <= 1.0:
        raise ValueError("target_gripper must be a finite position in [0,1]")
    result = np.concatenate(((q1 - q0) / dt, [grip])).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("converted DROID action is non-finite")
    return result


def droid_joint_delta_action(current: dict[str, Any], following: dict[str, Any], *,
                             max_dt_s: float = 0.5) -> np.ndarray:
    """Return ``[delta_q (7 rad), gripper_position]`` from two robot samples.

    delta_q = q[t+1] - q[t] (joint angle difference in radians, unscaled by dt).
    Gripper semantics: 0.0 = OPEN, 1.0 = CLOSED.
    Target format for pi05_droid_jointpos SFT.
    """
    q0 = _vector(current, "q", 7)
    q1 = _vector(following, "q", 7)
    t0 = int(current.get("_gpu_receive_monotonic_ns", -1))
    t1 = int(following.get("_gpu_receive_monotonic_ns", -1))
    if t0 > 0 and t1 > 0:
        dt = (t1 - t0) / 1e9
        if not math.isfinite(dt) or dt <= 0 or dt > max_dt_s:
            raise ValueError(f"consecutive samples time delta invalid: dt={dt:.4f}s")
    grip = float(following.get("target_gripper", math.nan))
    if not math.isfinite(grip) or not 0.0 <= grip <= 1.0:
        raise ValueError("target_gripper must be a finite position in [0,1]")
    result = np.concatenate((q1 - q0, [grip])).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("converted DROID joint delta action is non-finite")
    return result
