#!/usr/bin/env python3
"""Pair Franka teleop samples with two local RealSense streams and write LeRobotDataset."""

from __future__ import annotations

import argparse
import math
import queue
import shutil
import socket
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver

# Global state for web streaming
_latest_preview_jpeg = None
_preview_lock = threading.Lock()
_web_stream_active = False


class _TeleopWebHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress request logging

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()

    def do_GET(self):
        global _latest_preview_jpeg, _web_stream_active
        if self.path in ("/", "/index.html"):
            html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Franka Teleop Recording Live Stream</title>
    <style>
        body {
            margin: 0;
            background: #0d1117;
            color: #c9d1d9;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
        }
        .header {
            margin: 12px 0 8px 0;
            font-size: 20px;
            font-weight: 600;
            color: #58a6ff;
            letter-spacing: 0.5px;
        }
        .stream-box {
            position: relative;
            max-width: 98vw;
            border: 2px solid #30363d;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 8px 24px rgba(0,0,0,0.6);
            background: #161b22;
        }
        img {
            display: block;
            width: 100%;
            height: auto;
            max-height: 82vh;
        }
        .instructions {
            margin-top: 10px;
            font-size: 13px;
            color: #8b949e;
        }
        .badge {
            background: #21262d;
            border: 1px solid #30363d;
            padding: 2px 8px;
            border-radius: 4px;
            color: #f0883e;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <div class="header">Franka Panda &bull; LeRobot Teleop Live Monitor</div>
    <div class="stream-box">
        <img src="/stream.mjpg" alt="Franka Teleop Camera Stream">
    </div>
    <div class="instructions">
        Franka Keybinds: <span class="badge">'s'=Start REC</span> <span class="badge">'e'=Save</span> <span class="badge">'d'=Discard</span> <span class="badge">'q'=Quit</span>
    </div>
</body>
</html>"""
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html.encode("utf-8"))))
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            while _web_stream_active:
                with _preview_lock:
                    frame_bytes = _latest_preview_jpeg
                if frame_bytes is not None:
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(frame_bytes)))
                        self.end_headers()
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        break
                time.sleep(0.04)


class _ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _start_teleop_web_server(port: int = 8080) -> tuple[Any, int]:
    global _web_stream_active
    _web_stream_active = True
    for p in [port, port + 1, 8088, 8090]:
        try:
            server = _ThreadedHTTPServer(("0.0.0.0", p), _TeleopWebHandler)
            t = threading.Thread(target=server.serve_forever, name="teleop-web-stream", daemon=True)
            t.start()
            return server, p
        except OSError:
            continue
    return None, 0


def _draw_recording_dashboard(
    front_bgr: np.ndarray,
    wrist_bgr: np.ndarray,
    *,
    recording: bool,
    active_episode_id: int | None,
    episode_frames: int,
    fps: float,
    target_fps: int,
    task: str,
    total_episodes: int,
    last_status: tuple[str, str, float] | None,
    teleop_enabled: bool,
    gripper_open: bool | None,
    front_serial: str,
    wrist_serial: str,
) -> np.ndarray:
    import cv2
    h, w = front_bgr.shape[:2]
    now = time.time()

    # 1. Front Camera View
    f = front_bgr.copy()
    cv2.rectangle(f, (0, 0), (w, 32), (15, 15, 15), -1)
    cv2.putText(f, f"FRONT CAMERA ({front_serial})", (10, 22), cv2.FONT_HERSHEY_DUPLEX, 0.52, (50, 205, 50), 1, cv2.LINE_AA)
    cv2.putText(f, f"{fps:.1f} FPS", (w - 85, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
    cx, cy = w // 2, h // 2
    cv2.line(f, (cx - 15, cy), (cx + 15, cy), (0, 255, 255), 1)
    cv2.line(f, (cx, cy - 15), (cx, cy + 15), (0, 255, 255), 1)

    # 2. Wrist Camera View
    wr = wrist_bgr.copy()
    cv2.rectangle(wr, (0, 0), (w, 32), (15, 15, 15), -1)
    cv2.putText(wr, f"WRIST CAMERA ({wrist_serial})", (10, 22), cv2.FONT_HERSHEY_DUPLEX, 0.52, (255, 191, 0), 1, cv2.LINE_AA)
    grip_str = "GRIPPER: OPEN" if gripper_open else ("GRIPPER: CLOSED" if gripper_open is False else "GRIPPER: --")
    grip_color = (0, 255, 0) if gripper_open else (0, 165, 255)
    cv2.putText(wr, grip_str, (w - 170, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, grip_color, 1, cv2.LINE_AA)
    cv2.line(wr, (cx - 15, cy), (cx + 15, cy), (0, 255, 255), 1)
    cv2.line(wr, (cx, cy - 15), (cx, cy + 15), (0, 255, 255), 1)

    combined = np.hstack([f, wr])
    cw_total = 2 * w

    # 3. Mega Top Status Banner (height: 42px)
    top_banner = np.zeros((42, cw_total, 3), dtype=np.uint8)
    if recording:
        blink = int(now * 2) % 2 == 0
        bg_color = (0, 0, 180) if blink else (0, 0, 130)
        top_banner[:] = bg_color
        rec_text = f" [REC] Episode #{active_episode_id}  |  Frames: {episode_frames} ({episode_frames / max(target_fps, 1):.1f}s)  |  Task: \"{task}\""
        cv2.putText(top_banner, rec_text, (15, 28), cv2.FONT_HERSHEY_DUPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(top_banner, "RECORDING ACTIVE", (cw_total - 180, 28), cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
    elif last_status is not None and (now - last_status[2]) < 3.5:
        text, kind, _ = last_status
        if kind == "save":
            top_banner[:] = (34, 139, 34)  # Forest green
            cv2.putText(top_banner, f"[SAVED] {text}  |  Total Episodes Saved: {total_episodes}", (15, 28), cv2.FONT_HERSHEY_DUPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            top_banner[:] = (0, 100, 200)  # Orange for discard
            cv2.putText(top_banner, f"[DISCARDED] {text}", (15, 28), cv2.FONT_HERSHEY_DUPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        top_banner[:] = (45, 35, 25)
        teleop_txt = "TELEOP: ACTIVE" if teleop_enabled else "TELEOP: STANDBY"
        cv2.putText(top_banner, f"[IDLE / READY] Press 's' on Franka terminal to start REC  |  Next Ep: #{total_episodes}  |  Task: \"{task}\"", (15, 28), cv2.FONT_HERSHEY_DUPLEX, 0.52, (100, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(top_banner, teleop_txt, (cw_total - 160, 28), cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 255, 0) if teleop_enabled else (160, 160, 160), 1, cv2.LINE_AA)

    # 4. Bottom Footer Bar (28px high)
    footer = np.zeros((28, cw_total, 3), dtype=np.uint8)
    footer[:] = (20, 20, 20)
    cv2.putText(footer, f"Dataset: Total Episodes = {total_episodes}", (15, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(footer, "Franka Terminal Controls:  's'=Start REC    'e'=Save    'd'=Discard    'q'=Quit", (cw_total - 580, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 180, 50), 1, cv2.LINE_AA)

    return np.vstack([top_banner, combined, footer])



PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "bridge"))
from franka_teleop.protocol import PROTOCOL_VERSION, receive_message  # noqa: E402


FRONT_SERIAL = "254322072252"
WRIST_SERIAL = "348122070854"
# Dataset storage must be chosen explicitly; never reuse a Linux path on Windows.
GRIPPER_MAX_WIDTH_M = 0.08
GRIPPER_WIDTH_TOLERANCE_M = 0.006  # Franka Hand open can read ~0.081m due to mechanical pads/calibration


def _finite_vector(message: dict[str, Any], key: str, length: int) -> list[float]:
    value = message.get(key)
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{key} must contain {length} values")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{key} contains a non-finite value")
    return result


def validate_stream_message(message: dict[str, Any]) -> dict[str, Any]:
    if message.get("version") != PROTOCOL_VERSION:
        raise ValueError("unsupported stream version")
    message_type = message.get("type")
    if message_type in ("episode_start", "episode_end"):
        if not isinstance(message.get("episode_id"), int) or message["episode_id"] < 0:
            raise ValueError("invalid episode_id")
        if message_type == "episode_end" and not isinstance(message.get("save"), bool):
            raise ValueError("episode_end.save must be bool")
        return message
    if message_type != "teleop_sample":
        raise ValueError(f"unsupported stream message: {message_type}")
    if not isinstance(message.get("seq"), int) or message["seq"] < 0:
        raise ValueError("invalid sample seq")
    if not isinstance(message.get("episode_id"), int) or message["episode_id"] < 0:
        raise ValueError("invalid sample episode_id")
    _finite_vector(message, "q", 7)
    _finite_vector(message, "eef_position", 3)
    _finite_vector(message, "eef_quaternion", 4)
    _finite_vector(message, "target_position", 3)
    quaternion = _finite_vector(message, "target_quaternion", 4)
    if math.sqrt(sum(value * value for value in quaternion)) < 1e-9:
        raise ValueError("target_quaternion has zero norm")
    width = float(message.get("gripper_width_m", math.nan))
    gripper = float(message.get("target_gripper", math.nan))
    if (
        not math.isfinite(width)
        or width < -GRIPPER_WIDTH_TOLERANCE_M
        or width > GRIPPER_MAX_WIDTH_M + GRIPPER_WIDTH_TOLERANCE_M
    ):
        raise ValueError(f"gripper_width_m outside sensor tolerance: {width:.9g} m")
    message["gripper_width_m"] = min(GRIPPER_MAX_WIDTH_M, max(0.0, width))
    if not math.isfinite(gripper) or not 0.0 <= gripper <= 1.0:
        raise ValueError("target_gripper must be within [0,1]")
    if not isinstance(message.get("teleop_enabled"), bool) or not isinstance(message.get("recording"), bool):
        raise ValueError("teleop_enabled and recording must be bool")
    return message


class RobotStreamClient:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.messages: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=256)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="franka-record-stream", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.ident is not None:
            self._thread.join(timeout=6.0)

    def get(self, timeout_s: float) -> dict[str, Any] | None:
        try:
            return self.messages.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def _put(self, message: dict[str, Any]) -> None:
        if message.get("type") == "teleop_sample" and not message.get("recording"):
            return
        try:
            self.messages.put(message, timeout=0.5)
        except queue.Full as exc:
            raise ConnectionError("local recorder queue overflow") from exc

    def _run(self) -> None:
        while not self._stop.is_set():
            connected = False
            try:
                with socket.create_connection((self.host, self.port), timeout=3.0) as connection:
                    connection.settimeout(2.0)
                    connected = True
                    print(f"Connected to Franka recording stream at {self.host}:{self.port}")
                    while not self._stop.is_set():
                        message = receive_message(connection)
                        if message is None:
                            raise ConnectionError("Franka stream closed")
                        validate_stream_message(message)
                        message["_gpu_receive_monotonic_ns"] = time.perf_counter_ns()
                        self._put(message)
            except (ConnectionError, OSError, TypeError, ValueError) as exc:
                if connected:
                    try:
                        self._put({"type": "connection_lost", "reason": str(exc)})
                    except ConnectionError:
                        pass
                if not self._stop.is_set():
                    print(f"Franka stream unavailable: {exc}; retrying...")
                    self._stop.wait(1.0)


@dataclass
class CameraFrame:
    role: str
    frame_number: int
    host_time_ns: int
    image_rgb: np.ndarray


def _bgr8_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert a RealSense BGR8 frame to contiguous RGB HWC uint8."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError("RealSense color frame must be uint8 HWC with 3 channels")
    return np.ascontiguousarray(image_bgr[:, :, ::-1])


class DualRealSenseBuffer:
    def __init__(self, *, width: int, height: int, fps: int, warmup_frames: int,
                 front_serial: str = FRONT_SERIAL, wrist_serial: str = WRIST_SERIAL) -> None:
        if not front_serial or not wrist_serial or front_serial == wrist_serial:
            raise ValueError("front and wrist must have distinct nonempty serials")
        self.serials = {"front": front_serial, "wrist": wrist_serial}
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup_frames = warmup_frames
        self._condition = threading.Condition()
        self._buffers = {"front": deque(maxlen=12), "wrist": deque(maxlen=12)}
        self._errors: dict[str, BaseException] = {}
        self._ready: set[str] = set()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        for role, serial in self.serials.items():
            thread = threading.Thread(target=self._capture, args=(role, serial), name=f"capture-{role}", daemon=True)
            self._threads.append(thread)
            thread.start()
        deadline = time.monotonic() + 20.0
        with self._condition:
            while len(self._ready) < 2 and not self._errors:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("RealSense startup timed out")
                self._condition.wait(timeout=remaining)
            if self._errors:
                role, error = next(iter(self._errors.items()))
                raise RuntimeError(f"{role} camera failed: {error}")

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=3.0)

    def latest_images(self) -> dict[str, np.ndarray] | None:
        with self._condition:
            if self._errors:
                role, error = next(iter(self._errors.items()))
                raise RuntimeError(f"{role} camera failed: {error}")
            if not all(self._buffers[role] for role in ("front", "wrist")):
                return None
            return {role: self._buffers[role][-1].image_rgb.copy() for role in ("front", "wrist")}

    def match(
        self,
        target_time_ns: int,
        last_frame_numbers: dict[str, int],
        *,
        timeout_s: float,
        max_skew_s: float | None = None,
    ) -> dict[str, CameraFrame]:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            if self._errors:
                role, error = next(iter(self._errors.items()))
                raise RuntimeError(f"{role} camera failed: {error}")
            # Wait for a frame newer than the last frame consumed by the
            # previous sample/episode, not merely for any buffered frame.
            while not all(
                any(frame.frame_number > last_frame_numbers[role] for frame in self._buffers[role])
                and self._buffers[role][-1].host_time_ns >= target_time_ns
                for role in ("front", "wrist")
            ):
                if self._errors:
                    role, error = next(iter(self._errors.items()))
                    raise RuntimeError(f"{role} camera failed: {error}")
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(timeout=remaining)

            result = {}
            for role in ("front", "wrist"):
                candidates = [frame for frame in self._buffers[role] if frame.frame_number > last_frame_numbers[role]]
                if not candidates:
                    raise ValueError(f"no unused {role} frame")
                frame = min(candidates, key=lambda item: abs(item.host_time_ns - target_time_ns))
                result[role] = frame
            limit_ns = (max_skew_s if max_skew_s is not None else 1.0 / self.fps) * 1e9
            timestamps = [target_time_ns, *(frame.host_time_ns for frame in result.values())]
            if max(timestamps) - min(timestamps) > limit_ns:
                raise ValueError("camera/telemetry arrival skew exceeds limit")
            return result

    def _capture(self, role: str, serial: str) -> None:
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        # BGR8 is the stream format verified on both attached D435i cameras.
        # Convert explicitly below because LeRobot image features expect RGB HWC.
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        started = False
        try:
            pipeline.start(config)
            started = True
            received = 0
            while received < self.warmup_frames:
                if pipeline.wait_for_frames(timeout_ms=2000).get_color_frame():
                    received += 1
            with self._condition:
                self._ready.add(role)
                self._condition.notify_all()
            while not self._stop.is_set():
                color = pipeline.wait_for_frames(timeout_ms=2000).get_color_frame()
                if not color:
                    continue
                frame = CameraFrame(
                    role=role,
                    frame_number=int(color.get_frame_number()),
                    host_time_ns=time.perf_counter_ns(),
                    image_rgb=_bgr8_to_rgb(np.asanyarray(color.get_data())),
                )
                with self._condition:
                    self._buffers[role].append(frame)
                    self._condition.notify_all()
        except BaseException as exc:  # Hardware boundary: preserve exact RealSense error.
            with self._condition:
                self._errors[role] = exc
                self._condition.notify_all()
        finally:
            if started:
                pipeline.stop()


def _normalize_quaternion(quaternion: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in quaternion))
    if magnitude < 1e-9:
        raise ValueError("zero quaternion")
    return [value / magnitude for value in quaternion]


def _quaternion_multiply(left: list[float], right: list[float]) -> list[float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return [
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ]


def target_delta_action(current: dict[str, Any], following: dict[str, Any]) -> np.ndarray:
    current_position = _finite_vector(current, "target_position", 3)
    following_position = _finite_vector(following, "target_position", 3)
    translation = [following_position[i] - current_position[i] for i in range(3)]
    current_quaternion = _normalize_quaternion(_finite_vector(current, "target_quaternion", 4))
    following_quaternion = _normalize_quaternion(_finite_vector(following, "target_quaternion", 4))
    inverse_current = [-current_quaternion[0], -current_quaternion[1], -current_quaternion[2], current_quaternion[3]]
    delta = _normalize_quaternion(_quaternion_multiply(following_quaternion, inverse_current))
    if delta[3] < 0.0:
        delta = [-value for value in delta]
    vector_norm = math.sqrt(sum(value * value for value in delta[:3]))
    if vector_norm < 1e-9:
        rotvec = [0.0, 0.0, 0.0]
    else:
        angle = 2.0 * math.atan2(vector_norm, max(delta[3], 0.0))
        rotvec = [value * angle / vector_norm for value in delta[:3]]
    return np.asarray(translation + rotvec + [float(following["target_gripper"])], dtype=np.float32)


def dataset_features(height: int, width: int, *, action_space: str = "droid_joint_delta") -> dict[str, dict[str, Any]]:
    if action_space == "cartesian_delta":
        action_shape = (7,)
        action_names = ["delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz", "gripper"]
    elif action_space == "droid_joint_velocity":
        action_shape = (8,)
        action_names = [*[f"panda_joint{i}.velocity" for i in range(1, 8)], "gripper.position"]
    elif action_space == "droid_joint_delta":
        action_shape = (8,)
        action_names = [*[f"panda_joint{i}.pos_delta" for i in range(1, 8)], "gripper.position"]
    else:
        raise ValueError("action_space must be cartesian_delta, droid_joint_velocity, or droid_joint_delta")
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": [*[f"panda_joint{i}.pos" for i in range(1, 8)], "gripper.position"],
        },
        "action": {
            "dtype": "float32",
            "shape": action_shape,
            "names": action_names,
        },
        "observation.images.front": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
    }


def _state(sample: dict[str, Any]) -> np.ndarray:
    # DROID/OpenPI standard: 0.0 = OPEN, 1.0 = CLOSED
    # RealSense/Franka width: 0.08m when OPEN, 0.00m when CLOSED
    gripper_closed_ratio = 1.0 - min(1.0, max(0.0, float(sample["gripper_width_m"]) / GRIPPER_MAX_WIDTH_M))
    return np.asarray(_finite_vector(sample, "q", 7) + [gripper_closed_ratio], dtype=np.float32)


def _check_resume(dataset: Any, expected_features: dict[str, dict[str, Any]], fps: int) -> None:
    if dataset.fps != fps:
        raise ValueError(f"resume fps mismatch: dataset={dataset.fps}, requested={fps}")
    for key, expected in expected_features.items():
        actual = dataset.features.get(key)
        if actual is None or actual["dtype"] != expected["dtype"] or tuple(actual["shape"]) != tuple(expected["shape"]):
            raise ValueError(f"resume feature mismatch: {key}")


def _self_check() -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    test_bgr = np.asarray([[[1, 2, 3], [10, 20, 30]]], dtype=np.uint8)
    test_rgb = _bgr8_to_rgb(test_bgr)
    assert test_rgb.tolist() == [[[3, 2, 1], [30, 20, 10]]]
    assert test_rgb.flags.c_contiguous

    # Gripper semantics check: 0.0 = OPEN, 1.0 = CLOSED
    sample_open = {"q": [0.0] * 7, "gripper_width_m": GRIPPER_MAX_WIDTH_M}
    sample_closed = {"q": [0.0] * 7, "gripper_width_m": 0.0}
    state_open = _state(sample_open)
    state_closed = _state(sample_closed)
    assert np.isclose(state_open[7], 0.0), f"open gripper state must be 0.0, got {state_open[7]}"
    assert np.isclose(state_closed[7], 1.0), f"closed gripper state must be 1.0, got {state_closed[7]}"

    # Action semantics check for droid_joint_delta
    from action_semantics import droid_joint_delta_action
    s0 = {"q": [0.0] * 7, "_gpu_receive_monotonic_ns": 1_000_000_000}
    s1 = {"q": [0.01] * 7, "target_gripper": 1.0, "_gpu_receive_monotonic_ns": 1_066_666_667}
    act_delta = droid_joint_delta_action(s0, s1)
    assert np.allclose(act_delta[:7], 0.01), f"expected delta 0.01, got {act_delta[:7]}"
    assert np.isclose(act_delta[7], 1.0), f"expected gripper 1.0, got {act_delta[7]}"

    identity_sample = {
        "target_position": [0.0, 0.0, 0.0],
        "target_quaternion": [0.0, 0.0, 0.0, 1.0],
        "target_gripper": 1.0,
    }
    moved_sample = {**identity_sample, "target_position": [0.001, 0.0, 0.0]}
    action = target_delta_action(identity_sample, moved_sample)
    assert np.allclose(action, np.asarray([0.001, 0, 0, 0, 0, 0, 1], dtype=np.float32))

    temporary_parent = Path(tempfile.mkdtemp(prefix="pi05_lerobot_selfcheck_"))
    root = temporary_parent / "dataset"
    try:
        features = dataset_features(48, 64, action_space="droid_joint_delta")
        dataset = LeRobotDataset.create(
            repo_id="local/pi05_lerobot_selfcheck",
            fps=15,
            root=root,
            robot_type="franka_panda_touch",
            features=features,
            use_videos=True,
            image_writer_threads=2,
        )
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        for _ in range(3):
            dataset.add_frame(
                {
                    "observation.state": np.zeros(8, dtype=np.float32),
                    "action": np.zeros(8, dtype=np.float32),
                    "observation.images.front": image,
                    "observation.images.wrist": image,
                    "task": "self check",
                }
            )
        dataset.save_episode()
        dataset.finalize()
        assert dataset.num_episodes == 1 and dataset.num_frames == 3
    finally:
        shutil.rmtree(temporary_parent, ignore_errors=True)
    print("LeRobot teleop recorder self-check passed (DROID 15Hz & joint delta)")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", help="Franka control PC address (required for recording)")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--repo-id", default="local/midterm_franka_teleop")
    parser.add_argument("--root", type=Path, help="Explicit absolute dataset path on the Windows server")
    parser.add_argument("--front-serial", default=FRONT_SERIAL)
    parser.add_argument("--wrist-serial", default=WRIST_SERIAL)
    parser.add_argument("--task")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--action-space", choices=("cartesian_delta", "droid_joint_velocity", "droid_joint_delta"),
                        default="droid_joint_delta",
                        help="Action representation: droid_joint_delta (7 delta_q rad + gripper 0=open/1=closed for pi05_droid_jointpos), droid_joint_velocity, or cartesian_delta")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--max-skew-ms", type=float, default=100.0, help="maximum camera/telemetry arrival skew")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--streaming-encoding", action="store_true")
    preview_grp = parser.add_mutually_exclusive_group()
    preview_grp.add_argument("--preview", dest="preview", action="store_true", help="Enable live camera & recording dashboard (default: True)")
    preview_grp.add_argument("--no-preview", dest="preview", action="store_false", help="Disable live preview (headless mode)")
    parser.set_defaults(preview=True)
    parser.add_argument("--no-gui", action="store_true", help="Disable desktop OpenCV window (web stream only)")
    parser.add_argument("--web-port", type=int, default=8080, help="HTTP Web streaming port (default: 8080, 0 to disable)")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_check:
        return _self_check()
    if not args.task:
        raise SystemExit("--task is required for recording")
    if not args.host or args.root is None or not args.root.is_absolute():
        raise SystemExit("--host and an absolute --root are required for recording")
    if not args.front_serial or not args.wrist_serial or args.front_serial == args.wrist_serial:
        raise SystemExit("camera serials must be distinct and nonempty")
    if args.resume and not (args.root / "meta" / "info.json").is_file():
        raise SystemExit("--resume requires an existing local dataset; Hub download is disabled")
    if not 1 <= args.port <= 65535 or args.fps <= 0 or args.width <= 0 or args.height <= 0:
        raise SystemExit("invalid port, fps or image size")
    if args.warmup_frames < 0 or args.max_skew_ms <= 0:
        raise SystemExit("invalid warmup frame count")

    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = dataset_features(args.height, args.width, action_space=args.action_space)
    droid_action = None
    if args.action_space == "droid_joint_velocity":
        from action_semantics import droid_joint_velocity_action
        droid_action = droid_joint_velocity_action
    elif args.action_space == "droid_joint_delta":
        from action_semantics import droid_joint_delta_action
        droid_action = droid_joint_delta_action
    rgb_encoder = RGBEncoderConfig(
        vcodec="h264",
        pix_fmt="yuv420p",
        crf=23,
        preset="veryfast",
        fast_decode=1,
    )
    writer_options = {
        "root": args.root,
        "video_backend": "pyav",
        "streaming_encoding": args.streaming_encoding,
        "encoder_threads": 2,
        "image_writer_threads": 4,
        "rgb_encoder": rgb_encoder,
    }
    if args.resume:
        dataset = LeRobotDataset.resume(args.repo_id, **writer_options)
        try:
            _check_resume(dataset, features, args.fps)
        except Exception:
            dataset.finalize()
            raise
    else:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            features=features,
            robot_type="franka_panda_touch",
            use_videos=True,
            **writer_options,
        )

    client = RobotStreamClient(args.host, args.port)
    cameras = DualRealSenseBuffer(
        width=args.width,
        height=args.height,
        fps=args.fps,
        warmup_frames=args.warmup_frames,
        front_serial=args.front_serial,
        wrist_serial=args.wrist_serial,
    )

    recording = False
    episode_valid = False
    active_episode_id: int | None = None
    episode_frames = 0
    pending: tuple[dict[str, Any], dict[str, CameraFrame]] | None = None
    previous_seq: int | None = None
    last_frame_numbers = {"front": -1, "wrist": -1}

    def abort_episode(reason: str) -> None:
        nonlocal episode_valid, pending
        if recording and episode_valid:
            dataset.clear_episode_buffer()
            print(f"Episode {active_episode_id} invalidated: {reason}")
        episode_valid = False
        pending = None

    actual_web_port = 0
    if args.preview and args.web_port > 0:
        _, actual_web_port = _start_teleop_web_server(args.web_port)

    print("\n" + "=" * 80)
    print("  FRANKA LEROBOT TELEOP RECORDER: LIVE CAMERA & TELEMETRY MONITOR")
    print("=" * 80)
    if args.preview:
        if not args.no_gui:
            print("[*] Desktop Window: Active (Press 'q' in window or on Franka to quit)")
        else:
            print("[*] Desktop Window: Disabled (--no-gui)")
        if actual_web_port > 0:
            print(f"[*] Web Dashboard:  http://localhost:{actual_web_port}  (LAN: http://10.70.242.38:{actual_web_port})")
    else:
        print("[*] Visual Stream:  Disabled (--no-preview)")
    print(f"[*] Dataset Root:   {args.root}")
    print(f"[*] Task:           \"{args.task}\"")
    print(f"[*] Target Rate:    {args.fps} Hz (Action: {args.action_space})")
    print("[*] Status:         Waiting for Franka episode events. Control with s/e/d/p/q on Franka.")
    print("=" * 80 + "\n")

    last_status_event: tuple[str, str, float] | None = None
    teleop_active = False
    last_gripper_open: bool | None = None
    calc_fps = float(args.fps)
    last_fps_time = time.perf_counter()
    render_frame_count = 0

    try:
        cameras.start()
        client.start()
        if args.preview:
            import cv2

        while True:
            message = client.get(timeout_s=0.03)

            if args.preview:
                images = cameras.latest_images()
                if images is not None:
                    now_t = time.perf_counter()
                    render_frame_count += 1
                    if now_t - last_fps_time >= 1.0:
                        calc_fps = render_frame_count / (now_t - last_fps_time)
                        render_frame_count = 0
                        last_fps_time = now_t

                    f_bgr = cv2.cvtColor(images["front"], cv2.COLOR_RGB2BGR)
                    w_bgr = cv2.cvtColor(images["wrist"], cv2.COLOR_RGB2BGR)

                    dashboard = _draw_recording_dashboard(
                        f_bgr,
                        w_bgr,
                        recording=recording,
                        active_episode_id=active_episode_id,
                        episode_frames=episode_frames,
                        fps=calc_fps,
                        target_fps=args.fps,
                        task=args.task,
                        total_episodes=dataset.num_episodes,
                        last_status=last_status_event,
                        teleop_enabled=teleop_active,
                        gripper_open=last_gripper_open,
                        front_serial=args.front_serial,
                        wrist_serial=args.wrist_serial,
                    )

                    if actual_web_port > 0:
                        ret, jpeg = cv2.imencode(".jpg", dashboard, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        if ret:
                            with _preview_lock:
                                _latest_preview_jpeg = jpeg.tobytes()

                    if not args.no_gui:
                        cv2.imshow("Franka LeRobot Teleop Recorder [Live Stream & HUD]", dashboard)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

            if message is None:
                continue

            message_type = message["type"]
            if message_type == "connection_lost":
                abort_episode(message["reason"])
                recording = False
                active_episode_id = None
                continue
            if message_type == "episode_start":
                if recording:
                    abort_episode("new episode started before previous episode ended")
                recording = True
                episode_valid = True
                active_episode_id = int(message["episode_id"])
                episode_frames = 0
                pending = None
                previous_seq = None
                last_status_event = None
                print(f"Episode {active_episode_id} started: task={args.task!r}")
                continue
            if message_type == "episode_end":
                if recording and int(message["episode_id"]) != active_episode_id:
                    abort_episode("episode_end id mismatch")
                if recording and int(message["episode_id"]) == active_episode_id:
                    if message["save"] and episode_valid and episode_frames > 0:
                        dataset.save_episode()
                        last_status_event = (f"Episode #{active_episode_id} Saved ({episode_frames} frames)", "save", time.time())
                        print(f"Episode {active_episode_id} saved: {episode_frames} frames")
                    else:
                        dataset.clear_episode_buffer()
                        last_status_event = (f"Episode #{active_episode_id} Discarded", "discard", time.time())
                        print(f"Episode {active_episode_id} discarded")
                recording = False
                episode_valid = False
                active_episode_id = None
                pending = None
                continue
            if message_type != "teleop_sample" or not recording or not episode_valid:
                continue
            if int(message["episode_id"]) != active_episode_id or not message["recording"]:
                abort_episode("episode id/recording flag mismatch")
                continue
            teleop_active = bool(message.get("teleop_enabled", True))
            target_grip_val = message.get("target_gripper")
            if target_grip_val is not None:
                last_gripper_open = bool(float(target_grip_val) < 0.5)

            if not message["teleop_enabled"]:
                abort_episode("teleoperation was disabled during recording")
                continue
            seq = int(message["seq"])
            if previous_seq is not None and seq != previous_seq + 1:
                abort_episode(f"telemetry sequence gap: {previous_seq}->{seq}")
                continue

            try:
                frames = cameras.match(
                    int(message["_gpu_receive_monotonic_ns"]),
                    last_frame_numbers,
                    timeout_s=1.0 / args.fps,
                    max_skew_s=args.max_skew_ms / 1000.0,
                )
            except (RuntimeError, ValueError) as exc:
                abort_episode(str(exc))
                continue

            if pending is not None:
                current_sample, current_frames = pending
                if droid_action is None:
                    action = target_delta_action(current_sample, message)
                else:
                    action = droid_action(current_sample, message)
                dataset.add_frame(
                    {
                        "observation.state": _state(current_sample),
                        "action": action,
                        "observation.images.front": current_frames["front"].image_rgb,
                        "observation.images.wrist": current_frames["wrist"].image_rgb,
                        "task": args.task,
                    }
                )
                episode_frames += 1
            pending = (message, frames)
            previous_seq = seq
            for role, frame in frames.items():
                last_frame_numbers[role] = frame.frame_number
    except KeyboardInterrupt:
        print("Stopping GPU recorder")
    finally:
        if recording:
            dataset.clear_episode_buffer()
            print("Pending episode discarded because GPU recorder stopped before episode_end")
        try:
            cameras.stop()
        finally:
            try:
                client.stop()
            finally:
                dataset.finalize()
        global _web_stream_active
        _web_stream_active = False
        if args.preview and not args.no_gui:
            import cv2
            cv2.destroyAllWindows()
    print(f"Dataset finalized at {args.root}; episodes={dataset.num_episodes}, frames={dataset.num_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
