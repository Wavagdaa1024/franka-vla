#!/usr/bin/env python3
"""Run the existing Touch teleop and stream episode events plus Franka samples to the GPU."""

from __future__ import annotations

import argparse
import math
import select
import socket
import sys
import termios
import threading
import time
import tty
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import rospy

from Teleop_controller import TouchFrankaTeleopController


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "bridge"))
from protocol import PROTOCOL_VERSION, send_message  # noqa: E402


class Keyboard:
    def __enter__(self) -> "Keyboard":
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def read(self) -> str | None:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def __exit__(self, *_: Any) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


class OutboundStreamServer:
    """Keep teleoperation non-blocking while reliably preserving episode events."""

    def __init__(self, bind: str, port: int) -> None:
        self.bind = bind
        self.port = port
        self._condition = threading.Condition()
        self._events: deque[dict[str, Any]] = deque()
        self._sample: dict[str, Any] | None = None
        self._sample_seq = -1
        self._connected = False
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="lerobot-stream-server", daemon=True)

    @property
    def connected(self) -> bool:
        with self._condition:
            return self._connected

    def start(self) -> None:
        self._thread.start()

    def send_event(self, message: dict[str, Any]) -> None:
        with self._condition:
            self._events.append(message)
            self._condition.notify_all()

    def update_sample(self, sample: dict[str, Any]) -> None:
        with self._condition:
            self._sample = sample
            self._sample_seq = int(sample["seq"])
            self._condition.notify_all()

    def flush_events(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._events and self._connected and time.monotonic() < deadline:
                self._condition.wait(timeout=max(0.0, deadline - time.monotonic()))

    def stop(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.bind, self.port))
            server.listen(1)
            server.settimeout(1.0)
            rospy.loginfo("LeRobot recording stream listening on %s:%d", self.bind, self.port)
            while not rospy.is_shutdown():
                with self._condition:
                    if self._stop:
                        return
                try:
                    connection, address = server.accept()
                except socket.timeout:
                    continue
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self._condition:
                    self._connected = True
                    self._condition.notify_all()
                rospy.loginfo("GPU LeRobot recorder connected from %s:%d", *address)
                try:
                    self._serve(connection)
                except (BrokenPipeError, ConnectionError, OSError) as exc:
                    rospy.logwarn("GPU recorder disconnected: %s", exc)
                finally:
                    connection.close()
                    with self._condition:
                        self._connected = False
                        self._events.clear()
                        self._condition.notify_all()

    def _serve(self, connection: socket.socket) -> None:
        sent_sample_seq = -1
        while not rospy.is_shutdown():
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stop or self._events or self._sample_seq != sent_sample_seq,
                    timeout=1.0,
                )
                if self._stop:
                    return
                if self._events:
                    message = self._events.popleft()
                    self._condition.notify_all()
                elif self._sample is not None and self._sample_seq != sent_sample_seq:
                    message = dict(self._sample)
                    sent_sample_seq = self._sample_seq
                else:
                    continue
            send_message(connection, message)


class TeleopRecordingStreamer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.teleop = TouchFrankaTeleopController(control_hz=args.control_hz, controller_type=getattr(args, 'controller_type', 'velocity'))
        self.server = OutboundStreamServer(args.bind, args.port)
        self.sample_seq = 0
        self.episode_id = 0
        self.recording = False

    def _event(self, event_type: str, *, save: bool | None = None) -> dict[str, Any]:
        event = {
            "type": event_type,
            "version": PROTOCOL_VERSION,
            "episode_id": self.episode_id,
            "source_time_ns": time.time_ns(),
        }
        if save is not None:
            event["save"] = save
        return event

    def _sample(self) -> dict[str, Any] | None:
        arm = self.teleop.arm
        q = arm.get_joint_positions()
        position, quaternion = arm.get_cartesian_pose()
        gripper_width = arm.get_gripper_width()
        if q is None or position is None or quaternion is None or gripper_width is None:
            return None

        if self.teleop._teleop_enabled:
            target_position = self.teleop._target_pos.copy()
            target_quaternion = self.teleop._target_quat.copy()
        else:
            target_position = np.asarray(position, dtype=np.float64)
            target_quaternion = np.asarray(quaternion, dtype=np.float64)
        vectors = [q, position, quaternion, target_position, target_quaternion]
        if not all(np.all(np.isfinite(vector)) for vector in vectors) or not math.isfinite(float(gripper_width)):
            rospy.logwarn_throttle(1.0, "Skipping non-finite Franka sample")
            return None

        sample = {
            "type": "teleop_sample",
            "version": PROTOCOL_VERSION,
            "seq": self.sample_seq,
            "episode_id": self.episode_id,
            "source_time_ns": time.time_ns(),
            "q": np.asarray(q, dtype=np.float64).tolist(),
            "eef_position": np.asarray(position, dtype=np.float64).tolist(),
            "eef_quaternion": np.asarray(quaternion, dtype=np.float64).tolist(),
            "gripper_width_m": float(gripper_width),
            "target_position": np.asarray(target_position, dtype=np.float64).tolist(),
            "target_quaternion": np.asarray(target_quaternion, dtype=np.float64).tolist(),
            "target_gripper": 1.0 if self.teleop._gripper_closed else 0.0,  # DROID: 0.0=OPEN, 1.0=CLOSED
            "teleop_enabled": bool(self.teleop._teleop_enabled),
            "recording": self.recording,
        }
        self.sample_seq += 1
        return sample

    def _handle_key(self, key: str | None) -> bool:
        if key == "s" and not self.recording:
            if not self.server.connected:
                rospy.logwarn("Cannot start: GPU recorder is not connected")
            elif not self.teleop._teleop_enabled:
                rospy.logwarn("Cannot start: enable Touch teleoperation first")
            else:
                self.recording = True
                self.server.send_event(self._event("episode_start"))
                sample = self._sample()
                if sample is not None:
                    self.server.update_sample(sample)
                rospy.loginfo("Episode %d recording started", self.episode_id)
        elif key in ("e", "d") and self.recording:
            save = key == "e"
            self.server.send_event(self._event("episode_end", save=save))
            rospy.loginfo("Episode %d %s", self.episode_id, "save requested" if save else "discarded")
            self.recording = False
            self.episode_id += 1
        elif key == "p":
            rospy.loginfo(
                "gpu_connected=%s teleop_enabled=%s recording=%s episode=%d sample_seq=%d",
                self.server.connected,
                self.teleop._teleop_enabled,
                self.recording,
                self.episode_id,
                self.sample_seq,
            )
        elif key == "q":
            if self.recording:
                self.server.send_event(self._event("episode_end", save=True))
                self.recording = False
            self.server.flush_events(timeout_s=1.0)
            return False
        return True

    def run(self) -> int:
        if not self.teleop.initialize(timeout=5.0):
            return 1
        width = self.teleop.arm.get_gripper_width()
        if width is not None:
            self.teleop._gripper_closed = float(width) < 0.04

        self.server.start()
        control_rate = rospy.Rate(self.args.control_hz)
        sample_period = 1.0 / self.args.sample_hz
        next_sample = time.monotonic()
        rospy.loginfo("TeleopRecordingStreamer: control_hz=%.1f, sample_hz=%.1f (DROID 15Hz Standard)", self.args.control_hz, self.args.sample_hz)
        rospy.loginfo("Touch UP: teleop on/off; DOWN: gripper. Keyboard s/e/d/p/q: record control")

        try:
            with Keyboard() as keyboard:
                running = True
                while not rospy.is_shutdown() and running:
                    edges = self.teleop.touch.get_button_edges()
                    if edges["up_pressed"]:
                        self.teleop.toggle_teleop()
                    if edges["down_pressed"]:
                        self.teleop.toggle_gripper()
                    running = self._handle_key(keyboard.read())

                    if self.recording and not self.server.connected:
                        self.recording = False
                        self.episode_id += 1
                        rospy.logerr("GPU disconnected: current episode aborted")

                    self.teleop.step()
                    now = time.monotonic()
                    if now >= next_sample:
                        sample = self._sample()
                        if sample is not None:
                            self.server.update_sample(sample)
                        next_sample = now + sample_period
                    control_rate.sleep()
        except rospy.ROSInterruptException:
            pass
        finally:
            self.teleop.hold_current_pose()
            self.server.stop()
            self.teleop.touch.close()
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--control-hz", type=float, default=100.0)
    parser.add_argument("--sample-hz", type=float, default=15.0)
    parser.add_argument("--controller-type", choices=("velocity", "impedance", "auto"), default="velocity", help="Underlying Franka controller (default: velocity)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.port <= 65535 or args.control_hz <= 0 or args.sample_hz <= 0:
        raise SystemExit("invalid port or frequency")
    if args.sample_hz > args.control_hz:
        raise SystemExit("sample-hz cannot exceed control-hz")
    return TeleopRecordingStreamer(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
