#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Continuous RealSense Dual-Camera Live Viewer & Web Streamer.
Supports simultaneous:
1. Native Windows Desktop GUI window (OpenCV)
2. Web Browser Live Streaming via HTTP MJPEG (http://localhost:8080 or http://10.70.242.38:8080)
"""
from __future__ import annotations

import sys
import os
import time
import threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import argparse
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from franka_teleop.realsense_service import DualRealSense, FRONT_SERIAL_DEFAULT, WRIST_SERIAL_DEFAULT

try:
    import cv2
except ImportError:
    cv2 = None

# Global shared state for HTTP streamer
latest_jpeg_frame = None
frame_lock = threading.Lock()
stream_running = True


class StreamingHandler(BaseHTTPRequestHandler):
    """Serves MJPEG live stream and HTML web dashboard."""

    def log_message(self, format, *args):
        pass  # Suppress request spam

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()

    def do_GET(self):
        global latest_jpeg_frame, stream_running
        if self.path == "/" or self.path == "/index.html":
            content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Franka Dual RealSense Live Stream</title>
    <style>
        body {
            background-color: #121212;
            color: #E0E0E0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        h1 {
            margin-top: 0;
            font-size: 24px;
            color: #4CAF50;
            letter-spacing: 1px;
        }
        .container {
            background: #1E1E1E;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
            max-width: 95vw;
        }
        img.stream {
            width: 100%;
            max-width: 1280px;
            height: auto;
            border-radius: 8px;
            display: block;
            border: 2px solid #333;
        }
        .info-bar {
            display: flex;
            justify-content: space-between;
            margin-top: 12px;
            font-size: 14px;
            color: #AAA;
        }
        .tag {
            background: #2C2C2C;
            padding: 4px 10px;
            border-radius: 4px;
            border: 1px solid #444;
        }
        .tag-green { color: #81C784; }
        .tag-blue { color: #64B5F6; }
    </style>
</head>
<body>
    <div class="container">
        <h1>FRANKA DUAL REALSENSE LIVE STREAM</h1>
        <img class="stream" src="/stream.mjpg" alt="Live Camera Stream">
        <div class="info-bar">
            <span class="tag tag-green">&#9679; LEFT: Front Camera (Serial: 254322072252)</span>
            <span class="tag tag-blue">&#9679; RIGHT: Wrist Camera (Serial: 348122070854)</span>
            <span class="tag">Native 30 FPS / 640x480</span>
        </div>
    </div>
</body>
</html>
""".encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

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
                        frame = latest_jpeg_frame
                    if frame is not None:
                        self.wfile.write(b"--FRAME\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(frame)))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.033)
            except Exception:
                pass
        else:
            self.send_error(404)
            self.end_headers()


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_web_server(port=8080):
    server = ThreadedHTTPServer(("0.0.0.0", port), StreamingHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def draw_hud(frame_bgr: np.ndarray, title: str, color_bgr: tuple, fps: float, is_wrist: bool = False) -> np.ndarray:
    """Draw professional HUD labels, crosshair, and timestamps."""
    h, w = frame_bgr.shape[:2]
    # Header bar
    cv2.rectangle(frame_bgr, (0, 0), (w, 32), (20, 20, 20), -1)
    cv2.line(frame_bgr, (0, 32), (w, 32), color_bgr, 2)
    cv2.putText(frame_bgr, title, (10, 22), cv2.FONT_HERSHEY_DUPLEX, 0.6, color_bgr, 1, cv2.LINE_AA)

    # Status info
    status_text = f"640x480 | {fps:.1f} FPS"
    text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    cv2.putText(frame_bgr, status_text, (w - text_size[0] - 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # If wrist camera, draw center targeting crosshair
    if is_wrist:
        cx, cy = w // 2, h // 2
        arm_len = 20
        # Center dot
        cv2.circle(frame_bgr, (cx, cy), 3, (0, 255, 255), -1)
        # Crosshair lines
        cv2.line(frame_bgr, (cx - arm_len, cy), (cx - 6, cy), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(frame_bgr, (cx + 6, cy), (cx + arm_len, cy), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(frame_bgr, (cx, cy - arm_len), (cx, cy - 6), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(frame_bgr, (cx, cy + 6), (cx, cy + arm_len), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame_bgr, "GRIPPER TCP CENTER", (cx - 60, cy + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

    return frame_bgr


def main():
    parser = argparse.ArgumentParser(description="Continuous Dual RealSense Live Viewer")
    parser.add_argument("--front-serial", default=FRONT_SERIAL_DEFAULT, help="Front camera serial")
    parser.add_argument("--wrist-serial", default=WRIST_SERIAL_DEFAULT, help="Wrist camera serial")
    parser.add_argument("--port", type=int, default=8080, help="Web streaming HTTP port (default: 8080)")
    parser.add_argument("--no-gui", action="store_true", help="Disable OpenCV desktop window (web stream only)")
    args = parser.parse_args()

    global latest_jpeg_frame, stream_running

    print("=" * 75)
    print("  FRANKA REALSENSE CONTINUOUS DUAL-CAMERA STREAMER")
    print("=" * 75)
    print(f"[*] Front Serial: {args.front_serial}")
    print(f"[*] Wrist Serial: {args.wrist_serial}")
    print(f"[*] Web Server:   http://localhost:{args.port}  (or http://10.70.242.38:{args.port})")
    print("=" * 75)

    # 1. Start Web Server
    server = start_web_server(args.port)
    print(f"[OK] Web Stream Server running at: http://localhost:{args.port}")
    print(f"     -> You can open this URL in any browser to watch the continuous stream!")

    # 2. Start Cameras
    print("[*] Connecting to Dual RealSense hardware...")
    try:
        cams = DualRealSense(
            front_serial=args.front_serial,
            wrist_serial=args.wrist_serial,
            width=640,
            height=480,
            fps=30
        )
    except Exception as e:
        print(f"[ERROR] Failed to open RealSense cameras: {e}")
        return 1

    gui_available = (not args.no_gui) and (cv2 is not None)
    window_name = "Franka RealSense Dual View (Left: Front, Right: Wrist) - Press Q to Exit"

    if gui_available:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 480)
            print("[OK] Native OpenCV desktop window initialized.")
            print("     -> Press 'q' or 'ESC' on the video window to stop.")
        except Exception as e:
            print(f"[INFO] GUI desktop window not available in current session ({e}). Falling back to Web Stream only.")
            gui_available = False

    print("\n" + "-" * 75)
    print("  [LIVE] STREAMING ACTIVE. PRESS CTRL+C IN TERMINAL (OR 'Q' IN GUI) TO STOP.")
    print("-" * 75)

    fps_counter = 0
    fps_start = time.time()
    current_fps = 30.0

    try:
        while True:
            f_rgb, w_rgb = cams.get_frames()
            if f_rgb is None or w_rgb is None:
                continue

            fps_counter += 1
            if fps_counter >= 15:
                now = time.time()
                current_fps = fps_counter / max(now - fps_start, 1e-4)
                fps_counter = 0
                fps_start = now

            # Convert RGB to BGR
            if cv2 is not None:
                f_bgr = cv2.cvtColor(f_rgb, cv2.COLOR_RGB2BGR)
                w_bgr = cv2.cvtColor(w_rgb, cv2.COLOR_RGB2BGR)
            else:
                f_bgr = f_rgb[..., ::-1].copy()
                w_bgr = w_rgb[..., ::-1].copy()

            f_hud = draw_hud(f_bgr, "FRONT CAMERA (Global)", (50, 205, 50), current_fps, is_wrist=False)
            w_hud = draw_hud(w_bgr, "WRIST CAMERA (Gripper)", (255, 191, 0), current_fps, is_wrist=True)

            combined_bgr = np.hstack([f_hud, w_hud])

            # Update HTTP JPEG frame
            if cv2 is not None:
                ret, jpeg = cv2.imencode(".jpg", combined_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ret:
                    with frame_lock:
                        latest_jpeg_frame = jpeg.tobytes()

            # Display on native GUI if active
            if gui_available:
                try:
                    cv2.imshow(window_name, combined_bgr)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        print("\n[*] 'q' pressed. Stopping live stream...")
                        break
                except Exception:
                    gui_available = False
                    print("[INFO] GUI window closed or lost focus. Continuing in Web mode.")

    except KeyboardInterrupt:
        print("\n[*] Interrupted by user (Ctrl+C). Exiting...")
    finally:
        stream_running = False
        if gui_available and cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        cams.stop()
        print("[OK] Cameras released cleanly.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
