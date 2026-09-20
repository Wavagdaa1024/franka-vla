#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pi0.5 Franka VLA High-Performance Inference Server.
Provides a clean, unified HTTP REST API for real-time robot policy inference.

Features:
  - Strict physical GPU isolation (defaults to GPU 1, RTX 5090 32GB).
  - Fast JSON REST API (/predict, /health, /switch_checkpoint).
  - Base64 image decoding (Front + Wrist RealSense cameras).
  - Validated LoRA checkpoint hot-swapping without reloading base 4.14B model.
  - Zero-dependency (pure Python standard library http.server + torch + cv2/PIL).
  - Built-in Web UI dashboard with live VRAM meter and interactive testing.
"""

import os
import sys
import json
import time
import base64
import argparse
import threading
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver

# GPU Isolation (defaults to GPU 1 if not explicitly set)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
if "CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"].strip()
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import cv2
import torch

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from franka_teleop.pi05_engine.runtime import PI05Inference
from franka_teleop.pi05_engine.contracts import read_checkpoint, checkpoint_state, resolve_crop, lora_spec
from franka_teleop.pi05_engine.lora import inject_pi05_lora, load_lora_state_dict

# Default Paths & Checkpoint Aliases
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "pi05_droid_jointpos"
STATS_PATH = CHECKPOINT_DIR / "auxiliary" / "openpi_droid_jointpos_norm_stats.json"
TOKENIZER_PATH = CHECKPOINT_DIR / "auxiliary" / "paligemma_tokenizer.model"

CKPT_ALIASES = {
    # Canonical 50k Pure Flow Checkpoints
    "pure_flow": PROJECT_ROOT / "outputs" / "checkpoints" / "pi05_lora_pure_flow_50k" / "latest.pt",
    "pure_flow_latest": PROJECT_ROOT / "outputs" / "checkpoints" / "pi05_lora_pure_flow_50k" / "latest.pt",
    "pure_flow_50k": PROJECT_ROOT / "outputs" / "checkpoints" / "pi05_lora_pure_flow_50k" / "latest.pt",
    "pure_flow_50000": PROJECT_ROOT / "outputs" / "checkpoints" / "pi05_lora_pure_flow_50k" / "step_50000.pt",
    "pure_flow_2500": PROJECT_ROOT / "outputs" / "checkpoints" / "pi05_lora_pure_flow_50k" / "step_02500.pt",
    # Archived / Ablation Aliases
    "cartesian_7d": PROJECT_ROOT / "outputs" / "checkpoints" / "pi05_lora_cartesian_7d" / "pi05_lora_multitask_step_2000.pt",
    "cartesian_7d_2000": PROJECT_ROOT / "outputs" / "checkpoints" / "pi05_lora_cartesian_7d" / "pi05_lora_multitask_step_2000.pt",
}

DEFAULT_CKPT = CKPT_ALIASES["pure_flow"]


class VLAEngine:
    """Manages model lifecycle, warm-up, thread-safe inference, and checkpoint hot-swapping."""
    def __init__(self, default_ckpt_key_or_path: str = "pure_flow", device: str = "cuda:0", image_crop: str = "auto"):
        self.device = device
        self.image_crop = image_crop
        self._lora_spec = None
        self.lock = threading.Lock()
        self.current_ckpt_info = {"name": "None", "path": "", "step": 0, "loss": 0.0, "type": "base"}

        print(f"[VLA Engine] Initializing Pi0.5 base model on {self.device}...")
        t0 = time.perf_counter()
        self.inference = PI05Inference.from_checkpoint(
            CHECKPOINT_DIR,
            stats_path=STATS_PATH,
            tokenizer_path=TOKENIZER_PATH,
            device=self.device,
            profile="droid_jointpos",
            trainable_fp32=True
        )
        print(f"[VLA Engine] Base model loaded in {time.perf_counter()-t0:.2f}s.")

        # Validate and load before warming up or accepting requests.
        self.load_checkpoint(default_ckpt_key_or_path)
        self._warmup()

    def _warmup(self):
        """Runs single dummy inference pass to compile CUDA graphs and cache kernels."""
        print("[VLA Engine] Warming up inference engine...")
        dummy_front = torch.zeros(3, 480, 640, dtype=torch.uint8)
        dummy_wrist = torch.zeros(3, 480, 640, dtype=torch.uint8)
        dummy_state = np.zeros(8, dtype=np.float32)
        obs = {
            "observation.images.base_0_rgb": dummy_front,
            "observation.images.left_wrist_0_rgb": dummy_wrist,
            "observation.state": dummy_state,
        }
        with torch.no_grad():
            _ = self.inference.predict_action_chunk(obs, "pick and place the red cube")
        torch.cuda.synchronize()
        print("[VLA Engine] Warm-up completed.")

    def resolve_checkpoint_path(self, ckpt_key_or_path: str) -> Path:
        if ckpt_key_or_path in CKPT_ALIASES:
            return CKPT_ALIASES[ckpt_key_or_path]
        p = Path(ckpt_key_or_path)
        if p.exists():
            return p
        # Try finding in outputs/checkpoints
        candidate = PROJECT_ROOT / "outputs" / "checkpoints" / ckpt_key_or_path
        if candidate.exists():
            return candidate
        return p

    def load_checkpoint(self, ckpt_key_or_path: str) -> dict:
        """Loads or hot-swaps a LoRA checkpoint into the active model."""
        with self.lock:
            ckpt_path = self.resolve_checkpoint_path(ckpt_key_or_path)
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            ckpt = read_checkpoint(ckpt_path)
            state_dict = checkpoint_state(ckpt)
            crop = resolve_crop(ckpt, ckpt_path, self.image_crop)
            if not any("lora_" in key for key in state_dict):
                raise ValueError("HTTP hot-swap supports LoRA checkpoints only; use a dedicated process for full weights")
            spec = lora_spec(ckpt)
            if self._lora_spec is not None and spec != self._lora_spec:
                raise ValueError("LoRA architecture differs from the running model; restart with the selected checkpoint")
            if self._lora_spec is None:
                inject_pi05_lora(self.inference.network, **spec)
            load_lora_state_dict(self.inference.network, state_dict, strict=True)
            self._lora_spec = spec
            self.inference.network.eval()
            self.inference.processor.crop_mode = crop
            self.inference.reset()
            self.current_ckpt_info = {
                "name": ckpt_path.stem, "path": str(ckpt_path),
                "step": ckpt.get("step", 0),
                "loss": ckpt.get("flow_loss", ckpt.get("loss", 0.0)),
                "type": "LoRA", "image_crop": crop,
                "projection_precision": "float32",
            }
            return {"status": "success", "info": dict(self.current_ckpt_info)}

    def predict(self, front_rgb: np.ndarray, wrist_rgb: np.ndarray, state_8d: np.ndarray, task: str) -> dict:
        """Executes forward VLA policy inference."""
        t0 = time.perf_counter()
        front_chw = torch.from_numpy(np.transpose(front_rgb, (2, 0, 1)))
        wrist_chw = torch.from_numpy(np.transpose(wrist_rgb, (2, 0, 1)))

        obs = {
            "observation.images.base_0_rgb": front_chw,
            "observation.images.left_wrist_0_rgb": wrist_chw,
            "observation.state": state_8d,
        }

        with self.lock:
            with torch.no_grad():
                pred_chunk = self.inference.predict_action_chunk(obs, task)
                pred_chunk_np = pred_chunk.cpu().numpy()[0]  # (15, 8)
            checkpoint_name = self.current_ckpt_info["name"]

        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Unpack delta joint angles, absolute joint positions, and gripper
        curr_q = state_8d[:7]
        pred_delta_q = pred_chunk_np[:, :7]
        pred_gripper = pred_chunk_np[:, 7].tolist()
        pred_abs_q = (curr_q[None, :] + pred_delta_q).tolist()

        return {
            "status": "success",
            "latency_ms": round(latency_ms, 1),
            "fps_capacity": round(1000.0 / max(latency_ms, 1e-3), 1),
            "action_chunk": pred_chunk_np.tolist(),
            "joint_positions": pred_abs_q,
            "joint_deltas": pred_delta_q.tolist(),
            "gripper": pred_gripper,
            "checkpoint": checkpoint_name,
        }


def decode_base64_image(b64_str: str) -> np.ndarray:
    """Decodes a base64 encoded image string to an RGB numpy array (480, 640, 3)."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    raw_bytes = base64.b64decode(b64_str)
    np_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode image from base64 string")
    if bgr.shape[:2] != (480, 640):
        bgr = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


def make_vla_handler(engine: VLAEngine):
    class VLAHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Suppress noisy standard request logs unless error
            if " 5" in args[1] or " 4" in args[1]:
                super().log_message(format, *args)

        def _send_json(self, data: dict, status: int = 200):
            body = json.dumps(data, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/health", "/status"):
                peak_vram_gb = torch.cuda.max_memory_allocated("cuda:0") / (1024 ** 3)
                total_vram_gb = torch.cuda.get_device_properties("cuda:0").total_memory / (1024 ** 3)
                gpu_name = torch.cuda.get_device_name("cuda:0")
                self._send_json({
                    "status": "healthy",
                    "hardware": {
                        "gpu_name": gpu_name,
                        "vram_used_gb": round(peak_vram_gb, 2),
                        "vram_total_gb": round(total_vram_gb, 2),
                        "cuda_device": os.environ.get("CUDA_VISIBLE_DEVICES", "1")
                    },
                    "current_checkpoint": engine.current_ckpt_info,
                    "available_aliases": list(CKPT_ALIASES.keys())
                })
            elif path in ("/", "/ui"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Franka Pi0.5 VLA Inference Server</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 40px; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 24px; max-width: 800px; margin: 0 auto; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
        h1 {{ margin-top: 0; color: #38bdf8; font-size: 24px; }}
        .badge {{ display: inline-block; padding: 4px 10px; border-radius: 9999px; background: #059669; color: white; font-weight: bold; font-size: 13px; }}
        .metric {{ background: #334155; padding: 12px; border-radius: 8px; margin: 8px 0; }}
        pre {{ background: #090d16; padding: 12px; border-radius: 8px; overflow-x: auto; color: #a5f3fc; }}
        button {{ background: #2563eb; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: bold; }}
        button:hover {{ background: #1d4ed8; }}
    </style>
</head>
<body>
    <div class="card">
        <h1>🤖 Franka Pi0.5 VLA High-Performance Inference Server</h1>
        <p><span class="badge">● ONLINE</span> Running on <strong>{torch.cuda.get_device_name('cuda:0')}</strong> (GPU {os.environ.get('CUDA_VISIBLE_DEVICES', '1')})</p>
        <div class="metric">
            <strong>Active Checkpoint:</strong> {engine.current_ckpt_info['name']} ({engine.current_ckpt_info['type']})<br>
            <strong>Step:</strong> {engine.current_ckpt_info['step']} | <strong>Loss:</strong> {engine.current_ckpt_info['loss']}
        </div>
        <h3>📡 API Endpoints</h3>
        <ul>
            <li><code>POST /predict</code> - Send images (front/wrist base64) + state (8D) + task string -> Returns 15-step action chunk</li>
            <li><code>GET /health</code> - Returns JSON health and VRAM usage metrics</li>
            <li><code>POST /switch_checkpoint</code> - Hot-swaps active LoRA adapter without restarting</li>
        </ul>
        <h3>⚡ Live API Self-Test</h3>
        <button onclick="runTest()">Run Dummy /predict Test</button>
        <div id="testResult" style="margin-top: 15px;"></div>
    </div>
    <script>
        async function runTest() {{
            const div = document.getElementById('testResult');
            div.innerHTML = 'Executing test...';
            const t0 = performance.now();
            const res = await fetch('/predict', {{
                method: 'POST',
                headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{
                    task: 'pick and place the red cube',
                    state: [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0],
                    mock: true
                }})
            }});
            const data = await res.json();
            const totalMs = (performance.now() - t0).toFixed(1);
            div.innerHTML = `<pre>Roundtrip: ${{totalMs}}ms (GPU Compute: ${{data.latency_ms}}ms)\n` + JSON.stringify(data, null, 2) + `</pre>`;
        }}
    </script>
</body>
</html>"""
                self.wfile.write(html.encode("utf-8"))
            else:
                self._send_json({"error": "Not Found", "path": path}, status=404)

        def do_POST(self):
            path = self.path.split("?")[0]
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)

            try:
                payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
            except Exception as e:
                self._send_json({"status": "error", "message": f"Invalid JSON body: {e}"}, status=400)
                return

            if path == "/predict":
                task = payload.get("task", "pick and place the red cube")
                state = payload.get("state", [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.0])
                state_8d = np.array(state, dtype=np.float32)

                is_mock = payload.get("mock", False)
                images = payload.get("images", {})

                if is_mock or ("front" not in images and "wrist" not in images):
                    # Mock zero images for fast network/server benchmarking
                    front_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
                    wrist_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    try:
                        front_rgb = decode_base64_image(images["front"])
                        wrist_rgb = decode_base64_image(images["wrist"])
                    except Exception as e:
                        self._send_json({"status": "error", "message": f"Image decode error: {e}"}, status=400)
                        return

                try:
                    result = engine.predict(front_rgb, wrist_rgb, state_8d, task)
                    self._send_json(result)
                except Exception as e:
                    self._send_json({"status": "error", "message": f"Inference error: {e}"}, status=500)

            elif path == "/switch_checkpoint":
                ckpt = payload.get("checkpoint")
                if not ckpt:
                    self._send_json({"status": "error", "message": "Missing 'checkpoint' parameter"}, status=400)
                    return
                try:
                    res = engine.load_checkpoint(ckpt)
                    self._send_json(res)
                except (ValueError, KeyError, FileNotFoundError) as exc:
                    self._send_json({"status": "error", "message": str(exc)}, status=400)

            else:
                self._send_json({"error": "Endpoint not found", "path": path}, status=404)

    return VLAHandler


def main():
    parser = argparse.ArgumentParser(description="Franka Pi0.5 VLA High-Performance Inference Server.")
    parser.add_argument("--port", type=int, default=8088, help="Server port (default: 8088)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--checkpoint", type=str, default="pure_flow", help="Initial checkpoint alias or path")
    parser.add_argument("--image-crop", choices=("auto", "none", "16_9"), default="auto")
    args = parser.parse_args()

    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "1")
    print("=" * 80)
    print("  FRANKA Pi0.5 VLA HIGH-PERFORMANCE INFERENCE SERVER")
    print("=" * 80)
    print(f"[*] Port:              {args.port}")
    print(f"[*] Bind Host:         {args.host}")
    print(f"[*] Initial Ckpt:      {args.checkpoint}")
    print(f"[*] Target GPU:        Physical GPU {gpu_id} (RTX 5090 32GB, CUDA_VISIBLE_DEVICES={gpu_id})")
    print("-" * 80)

    engine = VLAEngine(default_ckpt_key_or_path=args.checkpoint, device="cuda:0", image_crop=args.image_crop)
    handler_cls = make_vla_handler(engine)
    server = ThreadedHTTPServer((args.host, args.port), handler_cls)

    print("\n" + "=" * 80)
    print(f"  [SERVER READY] Listening on http://0.0.0.0:{args.port}")
    print(f"  * Web Dashboard:       http://localhost:{args.port}/ui")
    print(f"  * Predict Endpoint:    POST http://localhost:{args.port}/predict")
    print(f"  * Health Endpoint:     GET  http://localhost:{args.port}/health")
    print(f"  * Hot-swap Endpoint:   POST http://localhost:{args.port}/switch_checkpoint")
    print("=" * 80)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
        server.server_close()
        print("[Server] Gracefully stopped.")


if __name__ == "__main__":
    main()
