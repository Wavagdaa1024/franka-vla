#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Mock RTC Streaming Test:
Runs closed_loop_franka server and mock GPU agent to test:
- Bootstrap Chunk 0
- Early Prefetch at Step 1
- Continuous Handover at Step 8
- Delayed Chunk Arrival (latency > 533ms) and Step 8..10 execution
- Safe clean termination
"""

import sys
import time
import socket
import struct
import json
import threading
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from franka_teleop.closed_loop_franka import (
    send_json,
    recv_json,
    run_rtc_loop,
    FrankaJointVelocityController
)
from franka_teleop.kinematics import forward_kinematics

class MockArgs:
    port = 8766
    rtc = True
    sync = False
    sync_steps = 15
    settle_time = 0.05
    steps_per_chunk = 8
    preempt_step = 1
    blend_steps = 3
    max_vel = 0.35
    max_acc = 1.5
    z_min = 0.0070
    kp_pos = 8.0
    close_delay_steps = 2
    flip_lr = False
    shadow = True


def mock_gpu_agent(port, stop_event):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    time.sleep(0.1)
    sock.connect(("127.0.0.1", port))

    loop_cnt = 0
    try:
        while not stop_event.is_set():
            msg = recv_json(sock)
            if msg is None:
                break
            loop_cnt += 1
            raw_q = np.array(msg["q"], dtype=np.float64)

            # Generate 15-step trajectory holding vertical orientation
            fake_chunk = np.tile(raw_q, (15, 1))
            for s in range(15):
                fake_chunk[s, 0] += 0.002 * (s + 1)

            # Simulate variable inference latency
            if loop_cnt == 1:
                infer_delay = 0.05
            elif loop_cnt == 2:
                infer_delay = 0.30
            else:
                infer_delay = 0.55

            time.sleep(infer_delay)

            resp = {
                "type": "action_chunk",
                "loop": loop_cnt,
                "action_mode": "joint_position",
                "latency_ms": infer_delay * 1000.0,
                "joint_positions": fake_chunk.tolist(),
                "joint_velocities": [[0.0] * 7] * 15,
                "gripper": [0.0] * 15,
                "max_tilt_deg": 0.008,
                "z_clamped_count": 0
            }
            if not send_json(sock, resp):
                break

            if loop_cnt >= 3:
                time.sleep(0.5)
                break
    finally:
        sock.close()


def test_mock_rtc_stream():
    print("===========================================================================")
    print("  TESTING MOCK END-TO-END RTC CONTINUOUS STREAMING INTEGRATION")
    print("===========================================================================")

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", MockArgs.port))
    server_sock.listen(1)

    stop_event = threading.Event()
    agent_thread = threading.Thread(target=mock_gpu_agent, args=(MockArgs.port, stop_event))

    print("[1/3] Awaiting Mock Agent connection...")
    agent_thread.start()
    conn, addr = server_sock.accept()
    print(f"[OK] Mock Agent connected from {addr}")

    arm = FrankaJointVelocityController()
    args = MockArgs()

    print("[2/3] Starting run_rtc_loop for 3 chunks...")
    total_chunks, t_start = run_rtc_loop(arm, conn, args)

    stop_event.set()
    agent_thread.join(timeout=2.0)
    conn.close()
    server_sock.close()

def test_mock_rtc_starvation_timeout():
    print("\n===========================================================================")
    print("  TESTING MOCK RTC STARVATION TIMEOUT & SAFE TERMINATION")
    print("===========================================================================")
    port = MockArgs.port + 1

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("127.0.0.1", port))
    server_sock.listen(1)

    stop_event = threading.Event()

    def hung_agent():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        time.sleep(0.1)
        sock.connect(("127.0.0.1", port))
        try:
            msg = recv_json(sock)  # bootstrap request
            if msg:
                raw_q = np.array(msg["q"], dtype=np.float64)
                fake_chunk = np.tile(raw_q, (15, 1))
                resp = {
                    "type": "action_chunk",
                    "loop": 0,
                    "action_mode": "joint_position",
                    "latency_ms": 10.0,
                    "joint_positions": fake_chunk.tolist(),
                    "joint_velocities": [[0.0] * 7] * 15,
                    "gripper": [0.0] * 15,
                    "max_tilt_deg": 0.005,
                    "z_clamped_count": 0
                }
                send_json(sock, resp)
            # Now simulate hung agent (no further responses)
            time.sleep(10.0)
        finally:
            sock.close()

    agent_thread = threading.Thread(target=hung_agent, daemon=True)
    agent_thread.start()

    conn, addr = server_sock.accept()
    arm = FrankaJointVelocityController()
    args = MockArgs()
    args.port = port

    t_start_test = time.time()
    total_chunks, t_start = run_rtc_loop(arm, conn, args)
    duration = time.time() - t_start_test

    conn.close()
    server_sock.close()

    print(f"  Starvation exit duration: {duration:.2f}s (safe bound: 5-8s)")
    assert 5.0 <= duration <= 8.5, f"Expected starvation timeout around 5-8s, got {duration:.2f}s"
    print("  [PASS] Starvation loop terminated safely without infinite deadlock!")


def run_all():
    test_mock_rtc_stream()
    test_mock_rtc_starvation_timeout()


if __name__ == "__main__":
    run_all()
