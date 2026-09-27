"""CPU-testable motion timing, proportional limits and observation validation."""
import time
import numpy as np


def hold_period(started, duration, shutdown=lambda: False, clock=None, sleep=None):
    """Never catch up skipped slots after inference or an overrun."""
    clock, sleep = clock or time.monotonic, sleep or time.sleep
    deadline = started + duration
    while not shutdown():
        remaining = deadline - clock()
        if remaining <= 0:
            return
        sleep(min(remaining, 0.01))


def proportional_limit(desired, previous, max_velocity, max_acceleration, dt):
    """Keep desired joint ratios when acceleration permits; otherwise brake as a vector.

    Braking is explicit, not a tracking command. Caller must discard the rest of
    this chunk and reobserve after stopping; never close the gripper while braking.
    """
    desired = np.asarray(desired, dtype=np.float64)
    previous = np.asarray(previous, dtype=np.float64)
    if desired.shape != (7,) or previous.shape != (7,) or not np.isfinite([desired, previous]).all():
        raise ValueError("Expected finite 7D velocities")
    if not all(np.isfinite(x) and x > 0 for x in (max_velocity, max_acceleration, dt)):
        raise ValueError("Positive finite limits required")
    if np.max(np.abs(previous)) > max_velocity + 1e-9:
        raise ValueError("Previous command exceeds velocity bound")
    speed_scale = min(1.0, max_velocity / max(float(np.max(np.abs(desired))), 1e-15))
    target = desired * speed_scale
    delta = max_acceleration * dt
    lo, hi = 0.0, 1.0
    for value, old in zip(target, previous):
        if abs(value) < 1e-15:
            if abs(old) > delta + 1e-12:
                hi = -1.0
                break
        else:
            a, b = (old - delta) / value, (old + delta) / value
            lo, hi = max(lo, min(a,b)), min(hi, max(a,b))
    if hi >= lo and hi >= 0 and (hi > 1e-12 or np.max(np.abs(target)) < 1e-12):
        return target * hi, dict(braking=False, speed_scale=speed_scale, acceleration_scale=hi)
    scale = max(0.0, 1.0 - delta / max(float(np.max(np.abs(previous))), 1e-15))
    return previous * scale, dict(braking=True, speed_scale=speed_scale, acceleration_scale=None)


def checked_snapshot(arm, now=None, max_age=0.1):
    snap = arm.get_state_snapshot()
    now = time.monotonic() if now is None else now
    if snap is None:
        raise RuntimeError("No timestamped robot state")
    q, dq = np.asarray(snap["q"]), np.asarray(snap["dq"])
    age = now - snap["received_monotonic"]
    if q.shape != (7,) or dq.shape != (7,) or not np.isfinite([q,dq]).all():
        raise RuntimeError("Invalid robot feedback")
    if not np.isfinite(age) or not 0 <= age <= max_age:
        raise RuntimeError("Stale robot feedback")
    if not np.isfinite(snap["source_stamp"]) or snap["source_stamp"] <= 0:
        raise RuntimeError("Missing ROS source timestamp")
    return snap


def wait_until_settled(arm, minimum=0.1, timeout=2.0, max_age=0.1,
                       velocity_tol=0.01, position_tol=0.002,
                       shutdown=lambda: False, clock=None, sleep=None):
    clock, sleep = clock or time.monotonic, sleep or time.sleep
    started = clock()
    stable_since = None
    anchor = None
    last_seq = None
    last_stamp = None
    while not shutdown() and clock() - started < timeout:
        arm.stop()
        snap = checked_snapshot(arm, clock(), max_age)
        if snap["sequence"] != last_seq:
            stamp = snap["source_stamp"]
            if last_stamp is not None and stamp <= last_stamp:
                raise RuntimeError("Robot source timestamp did not advance")
            last_seq, last_stamp = snap["sequence"], stamp
            q, dq = np.asarray(snap["q"]), np.asarray(snap["dq"])
            if np.max(np.abs(dq)) > velocity_tol:
                stable_since, anchor = None, None
            elif anchor is None or np.max(np.abs(q-anchor)) > position_tol:
                stable_since, anchor = clock(), q.copy()
            elif clock() - stable_since >= minimum:
                return snap
        sleep(0.01)
    raise RuntimeError("Robot did not settle with fresh feedback before timeout")
