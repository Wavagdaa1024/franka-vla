"""RealSense dual-camera (front/wrist) service with auto-reconnect and threading buffer."""
from __future__ import annotations

import time
import threading
from typing import Optional, Tuple
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

FRONT_SERIAL_DEFAULT = "254322072252"
WRIST_SERIAL_DEFAULT = "348122070854"


class DualRealSense:
    """Synchronized dual RealSense camera capture for front and wrist views."""

    def __init__(
        self,
        front_serial: str = FRONT_SERIAL_DEFAULT,
        wrist_serial: str = WRIST_SERIAL_DEFAULT,
        width: int = 640,
        height: int = 480,
        fps: int = 30
    ):
        self.front_serial = front_serial
        self.wrist_serial = wrist_serial
        self.width = width
        self.height = height
        self.fps = fps

        self.front_pipe = None
        self.wrist_pipe = None
        self.lock = threading.Lock()

        self._start_cameras()

    def _start_cameras(self):
        if rs is None:
            raise RuntimeError("pyrealsense2 is not installed!")

        print(f"[RealSense] Initializing Front Camera (Serial: {self.front_serial})...")
        f_cfg = rs.config()
        f_cfg.enable_device(self.front_serial)
        f_cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        self.front_pipe = rs.pipeline()
        self.front_pipe.start(f_cfg)

        print(f"[RealSense] Initializing Wrist Camera (Serial: {self.wrist_serial})...")
        w_cfg = rs.config()
        w_cfg.enable_device(self.wrist_serial)
        w_cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        self.wrist_pipe = rs.pipeline()
        self.wrist_pipe.start(w_cfg)

        # Warmup
        for _ in range(15):
            self.front_pipe.wait_for_frames()
            self.wrist_pipe.wait_for_frames()
        print("[RealSense] Both cameras warmed up successfully.")

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (front_rgb, wrist_rgb) as uint8 numpy arrays [H, W, 3]."""
        with self.lock:
            try:
                f_fs = self.front_pipe.wait_for_frames(timeout_ms=1000)
                w_fs = self.wrist_pipe.wait_for_frames(timeout_ms=1000)

                f_color = f_fs.get_color_frame()
                w_color = w_fs.get_color_frame()

                if not f_color or not w_color:
                    return None, None

                img_f = np.asanyarray(f_color.get_data())
                img_w = np.asanyarray(w_color.get_data())
                return img_f, img_w
            except Exception as e:
                print(f"[RealSense Warning] Frame drop/timeout: {e}")
                return None, None

    def stop(self):
        with self.lock:
            if self.front_pipe:
                try:
                    self.front_pipe.stop()
                except Exception:
                    pass
                self.front_pipe = None
            if self.wrist_pipe:
                try:
                    self.wrist_pipe.stop()
                except Exception:
                    pass
                self.wrist_pipe = None
            print("[RealSense] Cameras stopped.")
