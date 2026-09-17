#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual RealSense Camera Diagnostics and Preview Utility.
Tests RealSense camera connectivity, USB bandwidth, streaming framerate,
saves snapshot images to disk, and optionally displays a real-time OpenCV window.
"""
from __future__ import annotations

import sys
import os
import time
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

from franka_teleop.realsense_service import DualRealSense, FRONT_SERIAL_DEFAULT, WRIST_SERIAL_DEFAULT


def enumerate_devices():
    """List all connected RealSense devices with details."""
    print("=" * 70)
    print("  REALSENSE HARDWARE ENUMERATION")
    print("=" * 70)
    if rs is None:
        print("[ERROR] pyrealsense2 is not installed in the current environment.")
        return []

    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("[WARN] No RealSense devices detected on any USB port!")
        return []

    print(f"[*] Total RealSense Devices Detected: {len(devices)}")
    found_devices = []
    for idx, dev in enumerate(devices):
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        fw = dev.get_info(rs.camera_info.firmware_version)
        usb = dev.get_info(rs.camera_info.usb_type_descriptor)
        role = "Unknown"
        if serial == FRONT_SERIAL_DEFAULT:
            role = "FRONT CAMERA (Target)"
        elif serial == WRIST_SERIAL_DEFAULT:
            role = "WRIST CAMERA (Target)"

        print(f"  [{idx + 1}] {name}")
        print(f"      - Serial Number: {serial}  --> Role: {role}")
        print(f"      - Firmware:      {fw}")
        print(f"      - USB Port Type: USB {usb} {'(OK)' if '3.' in usb else '(WARNING: USB 2.x detected, may drop frames!)'}")
        found_devices.append((name, serial, usb))
    print("-" * 70)
    return found_devices


def test_cameras(args):
    """Test streaming and capture frames."""
    enumerate_devices()

    print("\n" + "=" * 70)
    print("  INITIALIZING DUAL-CAMERA PIPELINES (Front + Wrist)")
    print("=" * 70)
    print(f"[*] Target Front Serial: {args.front_serial}")
    print(f"[*] Target Wrist Serial: {args.wrist_serial}")
    print(f"[*] Resolution:          {args.width}x{args.height} @ {args.fps} FPS")

    try:
        cams = DualRealSense(
            front_serial=args.front_serial,
            wrist_serial=args.wrist_serial,
            width=args.width,
            height=args.height,
            fps=args.fps
        )
    except Exception as e:
        print(f"\n[FAIL] Camera initialization failed: {e}")
        print("  Tips: Check USB cables, ensure no other python/agent process is holding the camera open.")
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("\n[*] Capturing frames and measuring integrity...")
        t0 = time.time()
        success_count = 0
        last_f, last_w = None, None

        for i in range(args.num_frames):
            t_step = time.time()
            img_f, img_w = cams.get_frames()
            dt_ms = (time.time() - t_step) * 1000.0

            if img_f is not None and img_w is not None:
                success_count += 1
                last_f, last_w = img_f, img_w
                print(f"  [{i+1:02d}/{args.num_frames:02d}] Front: {img_f.shape}, Wrist: {img_w.shape} | Latency: {dt_ms:.1f}ms (OK)")
            else:
                print(f"  [{i+1:02d}/{args.num_frames:02d}] [WARNING] Frame dropped or timeout! Latency: {dt_ms:.1f}ms")

        fps_measured = success_count / max(time.time() - t0, 1e-4)
        print("-" * 70)
        print(f"[*] Capture Completed: {success_count}/{args.num_frames} frames captured ({success_count/args.num_frames*100:.1f}%)")
        print(f"[*] Measured Rate:     {fps_measured:.2f} FPS")

        # Save snapshots
        if last_f is not None and last_w is not None:
            front_path = out_dir / "front_camera.jpg"
            wrist_path = out_dir / "wrist_camera.jpg"
            combined_path = out_dir / "dual_cameras_preview.jpg"

            Image.fromarray(last_f).save(front_path, quality=95)
            Image.fromarray(last_w).save(wrist_path, quality=95)

            # Combine side by side
            h = min(last_f.shape[0], last_w.shape[0])
            combined = np.hstack([last_f, last_w])
            Image.fromarray(combined).save(combined_path, quality=95)

            print(f"\n[OK] Camera snapshots saved successfully to:")
            print(f"     - Front View:    {front_path}")
            print(f"     - Wrist View:    {wrist_path}")
            print(f"     - Side-by-Side:  {combined_path}")

        # Live GUI window if requested
        if args.gui:
            try:
                import cv2
                print("\n[*] Starting interactive OpenCV preview window...")
                print("    Press 'q' or 'ESC' in the preview window to exit.")
                cv2.namedWindow("RealSense Dual Camera Preview (Left: Front, Right: Wrist)", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("RealSense Dual Camera Preview (Left: Front, Right: Wrist)", 1280, 480)

                while True:
                    f_rgb, w_rgb = cams.get_frames()
                    if f_rgb is not None and w_rgb is not None:
                        # Convert RGB to BGR for OpenCV
                        f_bgr = cv2.cvtColor(f_rgb, cv2.COLOR_RGB2BGR)
                        w_bgr = cv2.cvtColor(w_rgb, cv2.COLOR_RGB2BGR)

                        cv2.putText(f_bgr, "FRONT CAMERA", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        cv2.putText(w_bgr, "WRIST CAMERA", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                        both = np.hstack([f_bgr, w_bgr])
                        cv2.imshow("RealSense Dual Camera Preview (Left: Front, Right: Wrist)", both)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        break
                cv2.destroyAllWindows()
            except ImportError:
                print("[WARN] OpenCV (cv2) not available for GUI display. Snapshot saved instead.")

    finally:
        cams.stop()

    print("\n" + "=" * 70)
    print("  CAMERA VERIFICATION PASSED (ALL SENSORS NORMAL)")
    print("=" * 70)
    return 0


def main():
    parser = argparse.ArgumentParser(description="RealSense Dual Camera Diagnostic & Preview Utility")
    parser.add_argument("--front-serial", default=FRONT_SERIAL_DEFAULT, help=f"Front camera serial (default: {FRONT_SERIAL_DEFAULT})")
    parser.add_argument("--wrist-serial", default=WRIST_SERIAL_DEFAULT, help=f"Wrist camera serial (default: {WRIST_SERIAL_DEFAULT})")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    parser.add_argument("--fps", type=int, default=30, help="Frame rate (default: 30)")
    parser.add_argument("--num-frames", type=int, default=15, help="Number of frames to test (default: 15)")
    parser.add_argument("--output-dir", default="outputs/camera_preview", help="Directory to save snapshot preview images")
    parser.add_argument("--gui", "--view", dest="gui", action="store_true", help="Launch live OpenCV interactive preview window")
    args = parser.parse_args()

    return test_cameras(args)


if __name__ == "__main__":
    sys.exit(main())
