"""Pure frame validation. Caller holds the streamer's frame lock."""
import math

def validated_frame_pair(frames, timestamps, now, max_age_s=0.5, max_sync_diff_s=0.08):
    if max_age_s <= 0 or max_sync_diff_s < 0:
        raise ValueError("Invalid frame timing bounds")
    for role in ("front", "wrist"):
        stamp = timestamps.get(role)
        if frames.get(role) is None or stamp is None or not math.isfinite(stamp):
            return None, None
        age = now - stamp
        if not math.isfinite(age) or age < 0 or age > max_age_s:
            return None, None
    if abs(timestamps["front"] - timestamps["wrist"]) > max_sync_diff_s:
        return None, None
    return frames["front"].copy(), frames["wrist"].copy()
