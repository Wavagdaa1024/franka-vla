#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Franka Third-Person Camera Fast Alignment & Verification Tool (Dual GUI + Web Edition).
Designed for Franka Imitation Learning (ACT / Diffusion Policy / VLA) and Hand-Eye Calibration.

Features:
1. RealSense hardware auto-detection & native color intrinsics querying.
2. 60mm ChArUco 4x4 board sub-pixel pose estimation (solvePnP).
3. "Golden Baseline" saving and persistent storage (baseline_pose.json).
4. Real-time delta display: dX, dY, dZ (mm), dRoll, dPitch, dYaw (deg).
5. Actionable physical adjustment hints (e.g., 向左推 5mm / 俯角下压 1.2度).
6. Semi-transparent Ghost Overlay (press G to blend baseline reference image).
7. Big Green "ALIGNED / 已精准复位" status banner when within tolerance.
8. Dual Display: Native Windows Desktop Window (cv2.imshow) + Web Dashboard (http://10.70.242.38:8088).

Hotkeys (Desktop & Web Buttons):
  [S] / Button: Save current pose & image as Golden Baseline
  [G] / Button: Toggle Semi-transparent Ghost Overlay mode
  [R] / Button: Reset / clear existing baseline
  [Q] / [ESC] : Exit program
"""

import os
import sys
import json
import time
import math
import threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

# Default Front Camera Serial Number
FRONT_SERIAL_DEFAULT = "254322072252"

# ChArUco Board Specifications
SQUARES_X = 4
SQUARES_Y = 4
SQUARE_LENGTH_M = 0.015   # 15.0 mm square length
MARKER_LENGTH_M = 0.011   # 11.0 mm marker length
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50

# Tolerances for "Aligned" state
TOLERANCE_POS_MM = 3.0       # <= 3mm translation error
TOLERANCE_ANGLE_DEG = 1.0    # <= 1.0 degree orientation error

# Global state for web streaming
latest_jpeg_frame = None
frame_lock = threading.Lock()
stream_running = True
web_command_queue = []


def rvec_to_euler_angles(rvec):
    """Convert Rodrigues rotation vector to Euler angles (Roll, Pitch, Yaw in degrees)."""
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([math.degrees(x), math.degrees(y), math.degrees(z)])


class AlignmentWebHandler(BaseHTTPRequestHandler):
    """Serves Web Dashboard and MJPEG live stream."""

    def log_message(self, format, *args):
        pass  # Suppress logging spam

    def do_GET(self):
        global latest_jpeg_frame, stream_running, web_command_queue
        if self.path == "/" or self.path == "/index.html":
            html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Franka Camera Fast Alignment Dashboard</title>
    <style>
        body {
            background-color: #121212;
            color: #E0E0E0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 15px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .container {
            background: #1E1E1E;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.6);
            max-width: 900px;
            width: 100%;
            text-align: center;
        }
        h1 {
            font-size: 20px;
            color: #4CAF50;
            margin-top: 5px;
            margin-bottom: 15px;
            letter-spacing: 0.5px;
        }
        img.stream {
            width: 100%;
            height: auto;
            border-radius: 8px;
            border: 2px solid #333;
            display: block;
        }
        .btn-group {
            display: flex;
            justify-content: center;
            gap: 12px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        button {
            padding: 10px 20px;
            font-size: 15px;
            font-weight: 600;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            transition: 0.2s;
        }
        .btn-save { background: #2E7D32; color: #FFF; }
        .btn-save:hover { background: #388E3C; }
        .btn-ghost { background: #0277BD; color: #FFF; }
        .btn-ghost:hover { background: #0288D1; }
        .btn-reset { background: #C62828; color: #FFF; }
        .btn-reset:hover { background: #D32F2F; }
        .toast {
            margin-top: 12px;
            font-size: 14px;
            color: #81C784;
            min-height: 20px;
        }
        .tips {
            margin-top: 15px;
            text-align: left;
            background: #252525;
            padding: 10px 15px;
            border-radius: 6px;
            font-size: 13px;
            color: #AAA;
            line-height: 1.6;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>FRANKA FRONT CAMERA FAST ALIGNMENT</h1>
        <img class="stream" src="/stream.mjpg" alt="Camera Feed">
        <div class="btn-group">
            <button class="btn-save" onclick="sendCmd('save')">&#10004; [S] Save Golden Baseline</button>
            <button class="btn-ghost" onclick="sendCmd('ghost')">&#128065; [G] Toggle Ghost Overlay</button>
            <button class="btn-reset" onclick="sendCmd('reset')">&#8634; [R] Reset Baseline</button>
        </div>
        <div id="toast" class="toast"></div>
        <div class="tips">
            <b>操作指南：</b><br>
            1. 首次使用将相机摆好后，点击 <b>[Save Golden Baseline]</b>（或按键盘 <code>S</code> 键）保存黄金位姿；<br>
            2. 日常复位时微调三脚架云台，画面上<b>青色实时框</b>对准<b>黄色基准框</b>；<br>
            3. 点击 <b>[Toggle Ghost Overlay]</b>（或按键盘 <code>G</code> 键）开启 50% 半透明鬼影叠图；<br>
            4. 误差处于 <code>&le; 3mm / &le; 1.0&deg;</code> 内时，顶部 Banner 自动变绿显示 <b>[ALIGNED]</b>。
        </div>
    </div>
    <script>
        function sendCmd(cmd) {
            fetch('/api/' + cmd)
                .then(r => r.text())
                .then(msg => {
                    const t = document.getElementById('toast');
                    t.innerText = msg;
                    setTimeout(() => { t.innerText = ''; }, 3000);
                });
        }
    </script>
</body>
</html>
""".encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()

            try:
                while stream_running:
                    with frame_lock:
                        if latest_jpeg_frame is None:
                            time.sleep(0.01)
                            continue
                        frame_bytes = latest_jpeg_frame

                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame_bytes)))
                    self.end_headers()
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.033)  # ~30 FPS
            except (ConnectionResetError, BrokenPipeError):
                pass

        elif self.path == "/api/save":
            web_command_queue.append("save")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write("✔ Save Baseline command sent!".encode("utf-8"))

        elif self.path == "/api/ghost":
            web_command_queue.append("ghost")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write("✔ Ghost Mode toggled!".encode("utf-8"))

        elif self.path == "/api/reset":
            web_command_queue.append("reset")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write("✔ Baseline reset!".encode("utf-8"))

        else:
            self.send_error(404)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


class CameraAligner:
    def __init__(self, serial=FRONT_SERIAL_DEFAULT, width=640, height=480, fps=30, port=8088):
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.port = port
        self.pipeline = None
        self.K = None
        self.D = None

        # Data directory
        self.data_dir = Path(__file__).resolve().parent
        self.baseline_json = self.data_dir / "baseline_pose.json"
        self.baseline_img_path = self.data_dir / "baseline_ref.png"

        # Baseline cache
        self.baseline = None
        self.baseline_img = None
        self.ghost_mode = False
        self.load_baseline()

        # Setup ChArUco Board & Detector
        self.dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
        self.board = cv2.aruco.CharucoBoard(
            (SQUARES_X, SQUARES_Y), SQUARE_LENGTH_M, MARKER_LENGTH_M, self.dictionary
        )
        self.detector = cv2.aruco.CharucoDetector(self.board)

        # 3D points of 4 outer board corners for projecting bounding box
        board_w_m = SQUARES_X * SQUARE_LENGTH_M
        board_h_m = SQUARES_Y * SQUARE_LENGTH_M
        self.outer_corners_3d = np.array([
            [0.0, 0.0, 0.0],
            [board_w_m, 0.0, 0.0],
            [board_w_m, board_h_m, 0.0],
            [0.0, board_h_m, 0.0]
        ], dtype=np.float32)

    def load_baseline(self):
        """Load stored baseline pose and reference image if available."""
        if self.baseline_json.exists():
            try:
                with open(self.baseline_json, "r", encoding="utf-8") as f:
                    self.baseline = json.load(f)
                print(f"[INFO] Loaded baseline from {self.baseline_json}")
            except Exception as e:
                print(f"[WARN] Failed to load baseline JSON: {e}")
                self.baseline = None

        if self.baseline_img_path.exists():
            try:
                self.baseline_img = cv2.imread(str(self.baseline_img_path))
                print(f"[INFO] Loaded baseline reference image from {self.baseline_img_path}")
            except Exception as e:
                print(f"[WARN] Failed to load baseline image: {e}")
                self.baseline_img = None

    def save_baseline(self, tvec, rvec, euler, corners_2d, frame_bgr):
        """Save current detection as Golden Baseline."""
        data = {
            "tvec_m": tvec.flatten().tolist(),
            "rvec": rvec.flatten().tolist(),
            "euler_deg": euler.tolist(),
            "corners_2d": corners_2d.reshape(-1, 2).tolist(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pattern": "CharucoBoard",
            "squares": [SQUARES_X, SQUARES_Y],
            "square_length_m": SQUARE_LENGTH_M,
            "marker_length_m": MARKER_LENGTH_M
        }
        with open(self.baseline_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        cv2.imwrite(str(self.baseline_img_path), frame_bgr)
        self.baseline = data
        self.baseline_img = frame_bgr.copy()
        print(f"[SUCCESS] Baseline saved successfully to {self.baseline_json}")

    def clear_baseline(self):
        """Clear existing baseline."""
        if self.baseline_json.exists():
            self.baseline_json.unlink()
        if self.baseline_img_path.exists():
            self.baseline_img_path.unlink()
        self.baseline = None
        self.baseline_img = None
        print("[INFO] Baseline cleared.")

    def start_camera(self):
        """Initialize RealSense camera stream and retrieve factory intrinsics."""
        if rs is None:
            raise RuntimeError("pyrealsense2 is not installed!")

        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No RealSense devices found! Please check USB cable.")

        found_serial = None
        for dev in devices:
            sn = dev.get_info(rs.camera_info.serial_number)
            if sn == self.serial:
                found_serial = sn
                break

        if found_serial is None:
            found_serial = devices[0].get_info(rs.camera_info.serial_number)
            print(f"[WARN] Target front serial {self.serial} not found. Using connected device: {found_serial}")
            self.serial = found_serial

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)

        profile = self.pipeline.start(config)
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()

        # Build OpenCV Camera Matrix K and Distortion Coefficients D
        self.K = np.array([
            [intr.fx, 0.0,     intr.ppx],
            [0.0,     intr.fy, intr.ppy],
            [0.0,     0.0,     1.0]
        ], dtype=np.float64)
        self.D = np.array(intr.coeffs, dtype=np.float64)

        print("=" * 70)
        print("  REALSENSE CAMERA CONNECTED SUCCESSFULLY")
        print("=" * 70)
        print(f"  Device S/N:  {self.serial}")
        print(f"  Resolution:  {self.width}x{self.height} @ {self.fps} FPS")
        print(f"  Intrinsics:  fx={intr.fx:.1f}, fy={intr.fy:.1f}, cx={intr.ppx:.1f}, cy={intr.ppy:.1f}")
        print(f"  Web Server:  http://localhost:{self.port}  (or http://10.70.242.38:{self.port})")
        print("=" * 70)

    def run(self):
        global latest_jpeg_frame, stream_running, web_command_queue
        self.start_camera()

        # Start Web Server in Background Thread
        httpd = ThreadedHTTPServer(("0.0.0.0", self.port), AlignmentWebHandler)
        web_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        web_thread.start()

        # Try to initialize native OpenCV desktop window
        win_title = "Franka Camera Alignment Tool (ChArUco)"
        has_gui = False
        try:
            cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_title, 1100, 750)
            has_gui = True
            print("[INFO] Native desktop GUI window started.")
        except Exception as e:
            print(f"[INFO] Running in Web-only mode ({e}). Open http://10.70.242.38:{self.port} in browser.")

        flash_message = ""
        flash_time = 0

        try:
            while stream_running:
                frames = self.pipeline.wait_for_frames(5000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                frame_rgb = np.asanyarray(color_frame.get_data())
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                display = frame_bgr.copy()

                # Ghost Overlay Mode
                if self.ghost_mode and self.baseline_img is not None:
                    if self.baseline_img.shape == display.shape:
                        display = cv2.addWeighted(display, 0.55, self.baseline_img, 0.45, 0)
                        cv2.putText(display, "[GHOST OVERLAY: ON]", (display.shape[1] - 280, 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

                # ChArUco Board Detection with robust error handling
                target_detected = False
                curr_tvec = None
                curr_rvec = None
                curr_euler = None
                curr_corners_2d = None

                try:
                    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                    charucoCorners, charucoIds, markerCorners, markerIds = self.detector.detectBoard(gray)

                    if charucoCorners is not None and charucoIds is not None and len(charucoCorners) >= 4:
                        objPoints, imgPoints = self.board.matchImagePoints(charucoCorners, charucoIds)
                        if len(objPoints) >= 4:
                            success, rvec, tvec = cv2.solvePnP(
                                objPoints, imgPoints, self.K, self.D, flags=cv2.SOLVEPNP_ITERATIVE
                            )
                            if success:
                                target_detected = True
                                curr_tvec = tvec
                                curr_rvec = rvec
                                curr_euler = rvec_to_euler_angles(rvec)

                                pts_2d, _ = cv2.projectPoints(self.outer_corners_3d, rvec, tvec, self.K, self.D)
                                curr_corners_2d = pts_2d.reshape(-1, 2)

                                # Robust subpixel corner drawing (completely avoids OpenCV 5.0 C++ assertion bug)
                                for pt in charucoCorners.reshape(-1, 2):
                                    px, py = int(round(pt[0])), int(round(pt[1]))
                                    cv2.circle(display, (px, py), 3, (0, 255, 0), -1)
                                    cv2.circle(display, (px, py), 5, (0, 200, 0), 1)

                                cv2.polylines(display, [np.int32(curr_corners_2d)], isClosed=True, color=(255, 200, 0), thickness=2)
                                cv2.drawFrameAxes(display, self.K, self.D, rvec, tvec, 0.040, 2)
                except Exception as detect_err:
                    pass  # Keep streaming smoothly even if detection glitched

                # Draw Baseline Target Box if baseline exists
                if self.baseline is not None:
                    base_corners = np.array(self.baseline["corners_2d"], dtype=np.int32)
                    cv2.polylines(display, [base_corners], isClosed=True, color=(0, 230, 255), thickness=2, lineType=cv2.LINE_AA)
                    cv2.putText(display, "TARGET BASELINE", (base_corners[0][0], max(base_corners[0][1] - 8, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 2)

                # UI Dashboard & Guidance
                if target_detected:
                    t_mm = curr_tvec.flatten() * 1000.0
                    eu = curr_euler

                    if self.baseline is not None:
                        base_t_mm = np.array(self.baseline["tvec_m"]) * 1000.0
                        base_eu = np.array(self.baseline["euler_deg"])

                        dt_mm = t_mm - base_t_mm
                        deu = eu - base_eu
                        deu = (deu + 180.0) % 360.0 - 180.0

                        dx, dy, dz = dt_mm[0], dt_mm[1], dt_mm[2]
                        droll, dpitch, dyaw = deu[0], deu[1], deu[2]
                        total_pos_err = math.sqrt(dx*dx + dy*dy + dz*dz)
                        total_ang_err = max(abs(droll), abs(dpitch), abs(dyaw))

                        is_aligned = (total_pos_err <= TOLERANCE_POS_MM) and (total_ang_err <= TOLERANCE_ANGLE_DEG)

                        # Top Status Banner
                        banner_h = 70
                        overlay = display.copy()
                        if is_aligned:
                            cv2.rectangle(overlay, (0, 0), (display.shape[1], banner_h), (0, 140, 0), -1)
                        else:
                            cv2.rectangle(overlay, (0, 0), (display.shape[1], banner_h), (20, 20, 180), -1)
                        display = cv2.addWeighted(display, 0.25, overlay, 0.75, 0)

                        if is_aligned:
                            cv2.putText(display, "[ALIGNED] CAMERA RESTORED TO TARGET POSE", (20, 32),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                            cv2.putText(display, f"Pos Error: {total_pos_err:.1f} mm <= {TOLERANCE_POS_MM} mm | Angle: {total_ang_err:.1f} deg <= {TOLERANCE_ANGLE_DEG} deg (PASS)",
                                        (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
                        else:
                            cv2.putText(display, "[ADJUSTING] CAMERA POSE MISALIGNED", (20, 32),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                            cv2.putText(display, f"Pos Error: {total_pos_err:.1f} mm (Tol: {TOLERANCE_POS_MM}mm) | Angle: {total_ang_err:.1f} deg (Tol: {TOLERANCE_ANGLE_DEG}deg)",
                                        (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 255), 1)

                        # Side Guidance Panel
                        p_x, p_y = 15, 95
                        cv2.rectangle(display, (p_x - 5, p_y - 18), (p_x + 360, p_y + 195), (15, 15, 15), -1)
                        cv2.rectangle(display, (p_x - 5, p_y - 18), (p_x + 360, p_y + 195), (90, 90, 90), 1)

                        cv2.putText(display, "--- 6-DoF DELTA & HINTS ---", (p_x, p_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                        hints = []
                        hints.append((f"dX:    {dx:+5.1f} mm", "Shift LEFT" if dx > 1.5 else ("Shift RIGHT" if dx < -1.5 else "OK")))
                        hints.append((f"dY:    {dy:+5.1f} mm", "Raise UP" if dy > 1.5 else ("Lower DOWN" if dy < -1.5 else "OK")))
                        hints.append((f"dZ:    {dz:+5.1f} mm", "Move FORWARD" if dz > 2.0 else ("Move BACK" if dz < -2.0 else "OK")))
                        hints.append((f"Pitch: {dpitch:+5.1f} deg", "Tilt UP" if dpitch > 0.8 else ("Tilt DOWN" if dpitch < -0.8 else "OK")))
                        hints.append((f"Yaw:   {dyaw:+5.1f} deg", "Pan LEFT" if dyaw > 0.8 else ("Pan RIGHT" if dyaw < -0.8 else "OK")))
                        hints.append((f"Roll:  {droll:+5.1f} deg", "Roll Level" if abs(droll) > 0.8 else "OK"))

                        for idx, (val_str, act_str) in enumerate(hints):
                            col = (0, 255, 0) if act_str == "OK" else (0, 165, 255)
                            cv2.putText(display, f"{val_str} -> {act_str}", (p_x, p_y + 25 + idx * 26),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                    else:
                        cv2.rectangle(display, (15, 15), (460, 85), (35, 35, 35), -1)
                        cv2.putText(display, "ChArUco Detected! No Baseline Saved.", (25, 42),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        cv2.putText(display, "Position camera, then press [S] to Save Baseline.",
                                    (25, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                else:
                    cv2.rectangle(display, (15, 15), (460, 75), (0, 0, 150), -1)
                    cv2.putText(display, "[WARN] 60mm ChArUco Board NOT FOUND!", (25, 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    cv2.putText(display, "Ensure board is in view and well-illuminated.",
                                (25, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

                # Bottom Hotkey Bar
                h_disp, w_disp = display.shape[:2]
                cv2.rectangle(display, (0, h_disp - 32), (w_disp, h_disp), (15, 15, 15), -1)
                help_text = f"[S] Save Baseline | [G] Ghost Overlay | [R] Reset | Web: http://10.70.242.38:{self.port}"
                cv2.putText(display, help_text, (15, h_disp - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

                # Toast Message
                if flash_message and (time.time() - flash_time < 3.0):
                    cv2.rectangle(display, (w_disp // 2 - 180, h_disp // 2 - 30),
                                  (w_disp // 2 + 180, h_disp // 2 + 25), (0, 160, 0), -1)
                    cv2.putText(display, flash_message, (w_disp // 2 - 160, h_disp // 2 + 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

                # Update Web JPEG frame buffer
                ret, jpeg = cv2.imencode(".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ret:
                    with frame_lock:
                        latest_jpeg_frame = jpeg.tobytes()

                # Process Web API commands
                while len(web_command_queue) > 0:
                    cmd = web_command_queue.pop(0)
                    if cmd == "save":
                        if target_detected:
                            self.save_baseline(curr_tvec, curr_rvec, curr_euler, curr_corners_2d, frame_bgr)
                            flash_message = "Baseline Saved Successfully!"
                            flash_time = time.time()
                        else:
                            flash_message = "Cannot Save: Board Not Detected!"
                            flash_time = time.time()
                    elif cmd == "ghost":
                        self.ghost_mode = not self.ghost_mode
                    elif cmd == "reset":
                        self.clear_baseline()
                        flash_message = "Baseline Cleared."
                        flash_time = time.time()

                # Desktop GUI event loop
                if has_gui:
                    cv2.imshow(win_title, display)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord('q'), ord('Q'), 27]:
                        break
                    elif key in [ord('s'), ord('S')]:
                        if target_detected:
                            self.save_baseline(curr_tvec, curr_rvec, curr_euler, curr_corners_2d, frame_bgr)
                            flash_message = "Baseline Saved Successfully!"
                            flash_time = time.time()
                        else:
                            flash_message = "Cannot Save: Board Not Detected!"
                            flash_time = time.time()
                    elif key in [ord('g'), ord('G')]:
                        self.ghost_mode = not self.ghost_mode
                    elif key in [ord('r'), ord('R')]:
                        self.clear_baseline()
                        flash_message = "Baseline Cleared."
                        flash_time = time.time()
                else:
                    time.sleep(0.01)

        finally:
            stream_running = False
            if self.pipeline:
                self.pipeline.stop()
            if has_gui:
                cv2.destroyAllWindows()
            try:
                httpd.shutdown()
            except:
                pass
            print("[INFO] Alignment tool closed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Franka Camera Fast Alignment Tool")
    parser.add_argument("--serial", default=FRONT_SERIAL_DEFAULT, help="Front camera serial number")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS")
    parser.add_argument("--port", type=int, default=8088, help="Web dashboard port")
    args = parser.parse_args()

    aligner = CameraAligner(serial=args.serial, width=args.width, height=args.height, fps=args.fps, port=args.port)
    aligner.run()
