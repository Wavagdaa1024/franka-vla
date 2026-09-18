#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Path 2: Hybrid VLM Grounding + Deterministic Cartesian Controller (Interactive Session).
Preloads RoboBrain2.5-8B-NV onto Physical GPU 1 (RTX 5090) ONCE.
Accepts continuous natural language pick-and-place targets with zero model reload delay.
Dispatches waypoints to Franka Linux Control PC (10.197.16.43:8765).
"""

import os
import sys

# Strictly isolate GPU 1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import time
import json
import socket
import struct
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MYCODE_SRC = PROJECT_ROOT / "src"
if str(MYCODE_SRC) not in sys.path:
    sys.path.insert(0, str(MYCODE_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm_planning.perception.coordinate_transform import TablePlaneProjector
from vlm_planning.perception.visual_grounding_agent import VisualGroundingAgent

DEFAULT_HOST = "10.197.16.43"
DEFAULT_PORT = 8765
FRONT_CAMERA_SERIAL = "254322072252"
OUTPUTS_DIR = SCRIPT_DIR / "outputs"

def send_json(sock, payload):
    data = json.dumps(payload).encode("utf-8")
    header = struct.pack("!I", len(data))
    sock.sendall(header + data)

def recv_json(sock):
    header = sock.recv(4)
    if not header or len(header) < 4:
        return None
    size = struct.unpack("!I", header)[0]
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return json.loads(data.decode("utf-8"))

def capture_realsense_frame(serial_number=FRONT_CAMERA_SERIAL):
    """Captures a single 640x480 RGB frame from RealSense."""
    import pyrealsense2 as rs
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    print(f"[Camera] Connecting to RealSense (S/N: {serial_number})...")
    profile = pipeline.start(config)
    try:
        # Warmup Auto-Exposure
        for _ in range(15):
            pipeline.wait_for_frames(2000)
        frames = pipeline.wait_for_frames(3000)
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Failed to obtain color frame from RealSense")
        data = np.asanyarray(color_frame.get_data())
        return Image.fromarray(data).convert("RGB")
    finally:
        pipeline.stop()

def run_single_mission(
    target_query: str,
    container_query: str,
    vlm: VisualGroundingAgent,
    projector: TablePlaneProjector,
    sock: socket.socket,
    dry_run: bool = False,
    camera_serial: str = FRONT_CAMERA_SERIAL,
    z_grasp: float = None,
    z_drop: float = None,
    offline_image: str = None
):
    print("\n" + "-" * 72)
    print(f"[*] 执行任务: 抓取目标='{target_query}' -> 目标容器='{container_query}'")
    print(f"[*] 演练模式 (Dry-Run): {dry_run}")
    print("-" * 72)

    # 1. 拍照
    t0 = time.time()
    if offline_image:
        print(f"[1/3] 读取离线图像: {offline_image}")
        raw_image = Image.open(offline_image).convert("RGB")
    else:
        print(f"[1/3] 正在从 RealSense ({camera_serial}) 实时拍照...")
        raw_image = capture_realsense_frame(camera_serial)

    snapshot_path = OUTPUTS_DIR / "hybrid_live_snapshot.jpg"
    raw_image.save(snapshot_path, quality=95)
    print(f"[OK] 图像采集成功 ({raw_image.size[0]}x{raw_image.size[1]})，耗时 {time.time()-t0:.2f}s")

    # 2. VLM 目标检测 (模型已常驻显存，极速推理)
    print(f"[2/3] 运行大模型视觉定位 (GPU 1, 零加载延迟)...")
    t_vlm = time.time()
    queries = [
        {"id": "target_obj", "query": target_query},
        {"id": "container_obj", "query": container_query}
    ]
    detections = vlm.detect_multi(raw_image, queries)

    if "target_obj" not in detections:
        print(f"[ERROR] 画面中未识别到目标: '{target_query}'，请检查名称或物体摆放！")
        return False
    if "container_obj" not in detections:
        print(f"[ERROR] 画面中未识别到容器: '{container_query}'！")
        return False

    target_uv = detections["target_obj"]["center_px"]
    container_uv = detections["container_obj"]["center_px"]
    print(f"[OK] 目标中心像素: {target_uv} | 容器中心像素: {container_uv} (耗时 {time.time()-t_vlm:.2f}s)")

    # 保存可视化图片
    annotated_path = OUTPUTS_DIR / "hybrid_grounding_result.jpg"
    VisualGroundingAgent.annotate_image(raw_image, detections, output_path=annotated_path)
    print(f"[OK] 标注规划图已更新: {annotated_path}")

    # 3. 仿射空间投影与笛卡尔路点规划
    p_target = projector.project_pixel(target_uv[0], target_uv[1], z=z_grasp)
    p_container = projector.project_pixel(container_uv[0], container_uv[1], z=z_drop)
    print(f"[*] 空间落点预测: Target=({p_target[0]:.4f}, {p_target[1]:.4f}, {p_target[2]:.4f})m")
    print(f"[*] 容器落点预测: Container=({p_container[0]:.4f}, {p_container[1]:.4f}, {p_container[2]:.4f})m")

    waypoints = projector.generate_pick_and_place_waypoints(
        target_uv, container_uv, z_grasp=z_grasp, z_drop=z_drop
    )

    # 4. 执行路点
    if dry_run or sock is None:
        print(f"[DRY-RUN] 11 步空间路点生成成功，未驱动实体机械臂。")
        return True

    print(f"[3/3] 正在向 Franka 发送 11 步无碰撞轨迹指令...")
    send_json(sock, {"cmd": "EXECUTE_WAYPOINTS", "waypoints": waypoints})

    while True:
        resp = recv_json(sock)
        if resp is None:
            print("[ERROR] 与 Franka 服务端的网络连接断开！")
            return False

        status = resp.get("status")
        if status == "progress":
            cur = resp.get("current")
            tot = resp.get("total")
            name = resp.get("step_name")
            print(f"  [机械臂动作] 步骤 {cur:02d}/{tot:02d}: {name}")
        elif status == "success":
            print(f"\n[SUCCESS] 机械臂抓取与放置任务圆满完成！已自动回位。")
            return True
        elif status == "error":
            print(f"\n[ERROR] 机械臂执行出错: {resp.get('error')}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Path 2: Hybrid VLM Grounding + Deterministic Controller (Interactive Session)")
    parser.add_argument("--target", type=str, default=None, help="Initial target object (e.g. 'green chili', 'orange block')")
    parser.add_argument("--container", type=str, default="basket", help="Target container description (default: 'basket')")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Franka Server IP")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Franka Server Port")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="Explicitly allow sending waypoints to the controller")
    mode.add_argument("--dry-run", action="store_true", help="Dry run mode without moving robot (default)")
    parser.add_argument("--camera-serial", type=str, default=FRONT_CAMERA_SERIAL, help="RealSense serial number")
    parser.add_argument("--z-grasp", type=float, default=None, help="Override table grasp height in meters (default from calib: ~0.018m)")
    parser.add_argument("--z-drop", type=float, default=None, help="Override basket drop height in meters")
    parser.add_argument("--image", type=str, default=None, help="Offline image for test")
    parser.add_argument("--once", action="store_true", help="Exit after first mission instead of entering interactive loop")
    args = parser.parse_args()
    args.dry_run = not args.live
    if args.dry_run and not args.image:
        parser.error("dry-run requires --image; use --live only after field safety approval")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("  FRANKA HYBRID MODE: VLM PERSISTENT INTERACTIVE SESSION")
    print("  Physical GPU 1 (RTX 5090) | Zero Model-Reloading Overhead")
    print("=" * 75)

    # 1. 预加载大模型到 GPU 1 显存（一次性加载，常驻显存）
    print("\n[系统初始化 1/3] 正在加载 RoboBrain2.5-8B-NV 大模型到 GPU 1 显存...")
    t_start = time.time()
    vlm = VisualGroundingAgent(device_id=0)
    print(f"[OK] 大模型加载完毕，耗时 {time.time()-t_start:.2f}s！后续所有指令无需重复加载。")

    # 2. 初始化空间几何投影器
    print("\n[系统初始化 2/3] 加载全仿射空间标定矩阵...")
    projector = TablePlaneProjector()

    # 3. 建立与 Franka 控制端长连接
    sock = None
    if args.live:
        print(f"\n[系统初始化 3/3] 连接 Franka 控制机 ({args.host}:{args.port})...")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            sock.settimeout(60.0)
            sock.connect((args.host, args.port))
            send_json(sock, {"cmd": "PING"})
            pong = recv_json(sock)
            print(f"[OK] Franka 控制机已连接！当前状态: {pong}")
        except Exception as e:
            print(f"[警告] 无法连接到 Franka 服务端 ({args.host}:{args.port}): {e}")
            print("若要驱动真机，请确保 Linux 端运行 'python3 hybrid_franka_server.py'。")
            print("当前将自动以演练模式 (Dry-Run) 运行。\n")
            sock = None
            raise SystemExit("Live controller connection failed; refusing automatic dry-run fallback")

    # 4. 如果命令行指定了目标，先执行该目标
    if args.target:
        run_single_mission(
            target_query=args.target,
            container_query=args.container,
            vlm=vlm,
            projector=projector,
            sock=sock,
            dry_run=args.dry_run,
            camera_serial=args.camera_serial,
            z_grasp=args.z_grasp,
            z_drop=args.z_drop,
            offline_image=args.image
        )
        if args.once:
            if sock:
                sock.close()
            return

    # 5. 进入常驻交互式循环（用户输入任意物体直接抓，零加载延迟）
    print("\n" + "=" * 75)
    print("  [交互式会话已就绪] 大模型常驻显存，随时接收目标！")
    print("  使用说明:")
    print("    - 直接输入物体名称抓取 (例如: green chili, orange block, carrot, yellow pepper)")
    print("    - 输入 'ready' : 机械臂复位到预备姿态")
    print("    - 输入 'q' 或 'exit' : 退出程序")
    print("=" * 75)

    try:
        while True:
            prompt_str = f"\n[VLM就绪] 请输入目标物体名称 > "
            user_input = input(prompt_str).strip()
            if not user_input:
                continue
            if user_input.lower() in ("q", "quit", "exit"):
                print("[退出] 正在结束交互会话...")
                break

            if user_input.lower() == "ready":
                if sock:
                    print("[复位] 正在让机械臂回位...")
                    # Generate ready waypoints
                    wp_ready = [{
                        "name": "RETURN_TO_READY",
                        "pos": projector.params["ready_pose_xyz"],
                        "quat": projector.params["default_quat_xyzw"],
                        "gripper": 1.0,
                        "speed": 0.15,
                        "pause_s": 0.1
                    }]
                    send_json(sock, {"cmd": "EXECUTE_WAYPOINTS", "waypoints": wp_ready})
                    resp = recv_json(sock)
                    while resp and resp.get("status") == "progress":
                        resp = recv_json(sock)
                    print("[OK] 机械臂已回到 Ready 姿态。")
                else:
                    print("[提示] 当前未连接 Franka。")
                continue

            # 执行抓取
            run_single_mission(
                target_query=user_input,
                container_query=args.container,
                vlm=vlm,
                projector=projector,
                sock=sock,
                dry_run=args.dry_run,
                camera_serial=args.camera_serial,
                z_grasp=args.z_grasp,
                z_drop=args.z_drop
            )

    except (KeyboardInterrupt, EOFError):
        print("\n[中断] 退出中...")
    finally:
        if sock:
            try:
                sock.close()
            except Exception:
                pass
        print("[关闭] 会话结束。")

if __name__ == "__main__":
    main()
