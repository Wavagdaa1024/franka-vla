#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Franka Hybrid Mode Execution Server.
Runs on Franka Linux Control PC (10.197.16.43:8765).
Accepts structured Cartesian waypoint missions from the Windows VLM orchestrator
and executes them deterministically with safety guards and status telemetry.
"""

import sys
import time
import json
import struct
import socket
import argparse
import traceback
import rospy

from cartesian_trajectory_executor import CartesianTrajectoryExecutor

DEFAULT_PORT = 8765

def send_json(conn, payload):
    data = json.dumps(payload).encode("utf-8")
    header = struct.pack("!I", len(data))
    conn.sendall(header + data)

def recv_json(conn):
    header = conn.recv(4)
    if not header or len(header) < 4:
        return None
    size = struct.unpack("!I", header)[0]
    data = bytearray()
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return json.loads(data.decode("utf-8"))

def main():
    parser = argparse.ArgumentParser(description="Franka Hybrid Cartesian Waypoint Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to listen on")
    args = parser.parse_args()

    print("=" * 72)
    print("  FRANKA HYBRID MODE CARTESIAN SERVER (10.197.16.43)")
    print("=" * 72)
    # Bind socket immediately so Windows client never gets Connection Refused
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    server_sock.bind(("0.0.0.0", args.port))
    server_sock.listen(5)
    print(f"[Server Ready] Listening for VLM orchestrator on port {args.port}...")

    if not rospy.core.is_initialized():
        rospy.init_node("hybrid_franka_server", anonymous=True)

    executor = CartesianTrajectoryExecutor()
    print("[*] Connecting to Franka hardware controller...")
    if executor.wait_for_connection(timeout=30.0):
        pos, _ = executor.arm.get_cartesian_pose()
        print(f"[OK] Franka connected. Current EE pos: {[round(x, 4) for x in pos]} m")
    else:
        print("[WARNING] Could not immediately read Franka state. Will connect upon request...")

    while not rospy.is_shutdown():
        try:
            conn, addr = server_sock.accept()
            print(f"\n[Client Connected] Connection from {addr[0]}:{addr[1]}")

            while not rospy.is_shutdown():
                req = recv_json(conn)
                if req is None:
                    print(f"[Client Disconnected] {addr[0]}:{addr[1]}")
                    break

                cmd = req.get("cmd", "").upper()

                if cmd == "PING":
                    pos, _ = executor.arm.get_cartesian_pose()
                    send_json(conn, {
                        "status": "ok",
                        "type": "pong",
                        "robot_ready": (pos is not None)
                    })

                elif cmd == "GET_STATE":
                    pos, quat = executor.arm.get_cartesian_pose()
                    q = executor.arm.get_joint_positions()
                    w = executor.arm.get_gripper_width()
                    send_json(conn, {
                        "status": "ok",
                        "pos": pos.tolist() if pos is not None else None,
                        "quat": quat.tolist() if quat is not None else None,
                        "joints": q.tolist() if q is not None else None,
                        "gripper_width": w
                    })

                elif cmd == "STOP":
                    executor.arm.stop()
                    send_json(conn, {"status": "ok", "message": "Robot stopped"})

                elif cmd == "EXECUTE_WAYPOINTS":
                    waypoints = req.get("waypoints", [])
                    print(f"[Execution] Received mission with {len(waypoints)} waypoints.")

                    def on_progress(current, total, name):
                        try:
                            send_json(conn, {
                                "status": "progress",
                                "current": current,
                                "total": total,
                                "step_name": name
                            })
                        except Exception:
                            pass

                    try:
                        success = executor.execute_waypoints(waypoints, progress_callback=on_progress)
                        send_json(conn, {
                            "status": "success" if success else "failed",
                            "message": "Mission execution completed"
                        })
                    except Exception as ex:
                        traceback.print_exc()
                        executor.arm.stop()
                        send_json(conn, {
                            "status": "error",
                            "error": str(ex)
                        })

                else:
                    send_json(conn, {"status": "error", "message": f"Unknown command: {cmd}"})

            conn.close()

        except KeyboardInterrupt:
            print("\n[Server Shutdown] Exiting...")
            break
        except Exception as e:
            print(f"[Server Error] {e}")
            time.sleep(0.5)

    server_sock.close()

if __name__ == "__main__":
    main()
