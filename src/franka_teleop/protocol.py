"""Small validated TCP protocol for PI05 state/action shadow traffic."""

from __future__ import annotations

import json
import math
import socket
import struct
from typing import Any


PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 1_000_000
ACTION_DIM = 7


def _vector(message: dict[str, Any], key: str, length: int) -> list[float]:
    value = message.get(key)
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{key} must contain {length} numbers")
    converted = [float(item) for item in value]
    if not all(math.isfinite(item) for item in converted):
        raise ValueError(f"{key} contains a non-finite value")
    return converted


def validate_state(message: dict[str, Any]) -> dict[str, Any]:
    if message.get("type") != "state" or message.get("version") != PROTOCOL_VERSION:
        raise ValueError("unsupported state message")
    if not isinstance(message.get("seq"), int) or message["seq"] < 0:
        raise ValueError("state seq must be a non-negative integer")
    if not isinstance(message.get("source_time_ns"), int) or message["source_time_ns"] < 0:
        raise ValueError("source_time_ns must be a non-negative integer")
    if not isinstance(message.get("robot_mode"), int) or not 0 <= message["robot_mode"] <= 6:
        raise ValueError("robot_mode must be an integer within [0, 6]")
    _vector(message, "q", 7)
    _vector(message, "dq", 7)
    _vector(message, "O_T_EE", 16)
    width = message.get("gripper_width_m")
    if width is not None and (not math.isfinite(float(width)) or not -0.006 <= float(width) <= 0.086):
        raise ValueError("gripper_width_m must be null or within [-0.006, 0.086]")
    return message


def validate_action_chunk(
    message: dict[str, Any],
    *,
    max_chunk_size: int = 50,
    max_translation_m: float = 0.01,
    max_rotation_rad: float = 0.10,
    max_chunk_duration_s: float = 4.0,
) -> dict[str, Any]:
    if message.get("type") != "action_chunk" or message.get("version") != PROTOCOL_VERSION:
        raise ValueError("unsupported action message")
    if not isinstance(message.get("seq"), int) or message["seq"] < 0:
        raise ValueError("action seq must be a non-negative integer")
    if not isinstance(message.get("source_state_seq"), int) or message["source_state_seq"] < 0:
        raise ValueError("source_state_seq must be a non-negative integer")
    dt_s = float(message.get("dt_s", 0.0))
    if not math.isfinite(dt_s) or not 0.0 < dt_s <= 1.0:
        raise ValueError("dt_s must be within (0, 1]")
    actions = message.get("actions")
    if not isinstance(actions, list) or not 1 <= len(actions) <= max_chunk_size:
        raise ValueError(f"actions must contain 1..{max_chunk_size} steps")
    if len(actions) * dt_s > max_chunk_duration_s:
        raise ValueError(f"action chunk duration exceeds {max_chunk_duration_s} s")
    for index, action in enumerate(actions):
        if not isinstance(action, list) or len(action) != ACTION_DIM:
            raise ValueError(f"action {index} must contain {ACTION_DIM} numbers")
        values = [float(item) for item in action]
        if not all(math.isfinite(item) for item in values):
            raise ValueError(f"action {index} contains a non-finite value")
        if math.sqrt(sum(value * value for value in values[:3])) > max_translation_m:
            raise ValueError(f"action {index} translation exceeds {max_translation_m} m")
        if math.sqrt(sum(value * value for value in values[3:6])) > max_rotation_rad:
            raise ValueError(f"action {index} rotation exceeds {max_rotation_rad} rad")
        if not 0.0 <= values[6] <= 1.0:
            raise ValueError(f"action {index} gripper must be within [0, 1]")
    return message


def send_message(connection: socket.socket, message: dict[str, Any]) -> None:
    payload = json.dumps(message, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("message is too large")
    connection.sendall(struct.pack("!I", len(payload)) + payload)


def _receive_exact(connection: socket.socket, size: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            return None
        chunks.extend(chunk)
    return bytes(chunks)


def receive_message(connection: socket.socket) -> dict[str, Any] | None:
    header = _receive_exact(connection, 4)
    if header is None:
        return None
    size = struct.unpack("!I", header)[0]
    if size == 0 or size > MAX_MESSAGE_BYTES:
        raise ValueError(f"invalid message size: {size}")
    payload = _receive_exact(connection, size)
    if payload is None:
        raise ConnectionError("connection closed mid-message")
    message = json.loads(payload)
    if not isinstance(message, dict):
        raise ValueError("message must be a JSON object")
    return message


def _self_check() -> None:
    state = {
        "type": "state",
        "version": 1,
        "seq": 3,
        "source_time_ns": 1,
        "q": [0.0] * 7,
        "dq": [0.0] * 7,
        "O_T_EE": [0.0] * 16,
        "gripper_width_m": 0.04,
        "robot_mode": 2,
    }
    validate_state(state)
    actions = {
        "type": "action_chunk",
        "version": 1,
        "seq": 4,
        "source_state_seq": 3,
        "dt_s": 1 / 15,
        "actions": [[0.001, 0, 0, 0, 0, 0.01, 0.5]],
    }
    validate_action_chunk(actions)
    left, right = socket.socketpair()
    try:
        send_message(left, state)
        assert receive_message(right) == state
    finally:
        left.close()
        right.close()
    print("protocol self-check passed")


if __name__ == "__main__":
    _self_check()
