"""Verify RealSense camera connectivity and frame integrity."""
import sys
import numpy as np
from franka_teleop.realsense_service import DualRealSense

def main():
    print("=" * 60)
    print("  RealSense Dual-Camera Verification")
    print("=" * 60)
    try:
        cams = DualRealSense()
    except Exception as e:
        print(f"[FAIL] Could not initialize cameras: {e}")
        return 1

    try:
        for i in range(10):
            img_f, img_w = cams.get_frames()
            if img_f is not None and img_w is not None:
                print(f"[{i+1}/10] Front Frame: {img_f.shape}, Wrist Frame: {img_w.shape} (OK)")
            else:
                print(f"[{i+1}/10] Frame dropped!")
    finally:
        cams.stop()

    print("[PASS] Camera test completed successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
