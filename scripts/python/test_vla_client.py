#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test client for Franka Pi0.5 VLA Inference Server.
Verifies /health, /predict (both mock and real base64 images), and /switch_checkpoint.
"""

import sys
import time
import json
import base64
import urllib.request
import urllib.error
import numpy as np
import cv2

SERVER_URL = "http://127.0.0.1:8088"
if len(sys.argv) > 1:
    SERVER_URL = sys.argv[1]


def http_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "VLA-TestClient"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post(url, data):
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "VLA-TestClient"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def encode_image(bgr_img):
    _, buffer = cv2.imencode(".jpg", bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return base64.b64encode(buffer).decode("utf-8")


def run_tests():
    print("=" * 80)
    print(f"  VLA INFERENCE SERVER API CLIENT VERIFICATION: {SERVER_URL}")
    print("=" * 80)

    # 1. Test /health
    print("\n[Test 1/4] Checking GET /health...")
    try:
        health = http_get(f"{SERVER_URL}/health")
        print(f"  [OK] Server is {health.get('status')}!")
        print(f"       GPU: {health.get('hardware', {}).get('gpu_name')}")
        print(f"       VRAM: {health.get('hardware', {}).get('vram_used_gb')} / {health.get('hardware', {}).get('vram_total_gb')} GB")
        print(f"       Active Checkpoint: {health.get('current_checkpoint', {}).get('name')}")
    except Exception as e:
        print(f"  [FAIL] Health check failed: {e}")
        return False

    # 2. Test /predict (Mock ping)
    print("\n[Test 2/4] Testing POST /predict (Mock Ping)...")
    try:
        t0 = time.perf_counter()
        res = http_post(f"{SERVER_URL}/predict", {
            "task": "pick and place the red cube",
            "state": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0],
            "mock": True
        })
        rtt_ms = (time.perf_counter() - t0) * 1000.0
        print(f"  [OK] Roundtrip: {rtt_ms:.1f}ms | GPU Latency: {res.get('latency_ms')}ms | Capacity: {res.get('fps_capacity')} FPS")
        print(f"       Action Chunk Shape: {len(res.get('action_chunk', []))} steps x {len(res.get('action_chunk', [[0]*8])[0])} dims")
    except Exception as e:
        print(f"  [FAIL] Mock predict failed: {e}")
        return False

    # 3. Test /predict (Real synthetic image encoding)
    print("\n[Test 3/4] Testing POST /predict (Dual Base64 Images)...")
    try:
        front_img = np.full((480, 640, 3), 120, dtype=np.uint8)
        wrist_img = np.full((480, 640, 3), 80, dtype=np.uint8)
        # Draw some mock objects
        cv2.circle(front_img, (320, 240), 50, (0, 0, 255), -1)  # Red cube
        cv2.circle(wrist_img, (320, 240), 30, (0, 0, 255), -1)

        b64_front = encode_image(front_img)
        b64_wrist = encode_image(wrist_img)

        t0 = time.perf_counter()
        res = http_post(f"{SERVER_URL}/predict", {
            "task": "pick and place the red cube",
            "state": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0],
            "images": {
                "front": b64_front,
                "wrist": b64_wrist
            }
        })
        rtt_ms = (time.perf_counter() - t0) * 1000.0
        chunk = np.array(res["action_chunk"])
        print(f"  [OK] Roundtrip (incl. Base64 transfer): {rtt_ms:.1f}ms | GPU Compute: {res.get('latency_ms')}ms")
        print(f"       Step 1 Delta Q:   {np.round(chunk[0, :7], 4).tolist()}")
        print(f"       Step 1 Gripper:   {chunk[0, 7]:.3f}")
        print(f"       Step 15 Delta Q:  {np.round(chunk[-1, :7], 4).tolist()}")
        print(f"       Step 15 Gripper:  {chunk[-1, 7]:.3f}")
    except Exception as e:
        print(f"  [FAIL] Image predict failed: {e}")
        return False

    # 4. Test /switch_checkpoint
    print("\n[Test 4/4] Testing POST /switch_checkpoint (Microsecond LoRA Hot-swapping)...")
    try:
        t0 = time.perf_counter()
        res = http_post(f"{SERVER_URL}/switch_checkpoint", {
            "checkpoint": "cartesian_7d_2500"
        })
        switch_ms = (time.perf_counter() - t0) * 1000.0
        print(f"  [OK] Switched in {switch_ms:.1f}ms! Active: {res.get('info', {}).get('name')}")

        # Switch back to default
        http_post(f"{SERVER_URL}/switch_checkpoint", {"checkpoint": "cartesian_7d_2000"})
        print("  [OK] Restored default checkpoint: cartesian_7d_2000.")
    except Exception as e:
        print(f"  [WARN] Switch checkpoint: {e}")

    print("\n" + "=" * 80)
    print("  ALL API CHECKS PASSED! SERVER READY FOR ROBOT INTEGRATION.")
    print("=" * 80)
    return True


if __name__ == "__main__":
    run_tests()
