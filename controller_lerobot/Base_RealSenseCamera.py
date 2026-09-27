#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import cv2
import numpy as np
import pyrealsense2 as rs


class RealSenseCamera:
    def __init__(self, serial="254322072252", width=640, height=480, fps=30, warmup_frames=10):
        """
        默认初始化第三视角相机（序列号 254322072252）
        """
        self.pipeline = rs.pipeline()
        config = rs.config()

        if serial is not None:
            config.enable_device(serial)

        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.running = True

        for _ in range(warmup_frames):
            self.pipeline.wait_for_frames()

    # 获取图像数据
    def get_frames(self):
        frames = self.align.process(self.pipeline.wait_for_frames())

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        color = np.asanyarray(color_frame.get_data()) if color_frame else None
        depth = np.asanyarray(depth_frame.get_data()) if depth_frame else None
        color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB) if color is not None else None

        return color, color_rgb, depth

    # 显示图像数据
    def show_frames(self, color=None, depth=None, alpha=0.03):
        if color is not None:
            cv2.imshow("Color", color)

        if depth is not None:
            depth_color = cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=alpha),
                cv2.COLORMAP_JET
            )
            cv2.imshow("Depth", depth_color)

        cv2.waitKey(1)

    # 停止
    def stop(self):
        if self.running:
            self.pipeline.stop()
            self.running = False
        cv2.destroyAllWindows()


class MultiCameraManager:
    def __init__(self, configs=None):
        """
        默认初始化两台相机：
        - 编号0 Franka上面 D435I 348122070854
        - 编号1 第三视角 D435I 254322072252
        """
        if configs is None:
            configs = [
                {"serial": "348122070854", "width": 640, "height": 480, "fps": 30},
                {"serial": "254322072252", "width": 640, "height": 480, "fps": 30}
            ]
        self.cameras = [RealSenseCamera(**cfg) for cfg in configs]

    def get_frames_all(self):
        return [cam.get_frames() for cam in self.cameras]

    def show_frames_all(self, alpha=0.03):
        for i, cam in enumerate(self.cameras):
            color, _, depth = cam.get_frames()
            if color is not None:
                cv2.imshow(f"Camera{i}_Color", color)
            if depth is not None:
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth, alpha=alpha),
                    cv2.COLORMAP_JET
                )
                cv2.imshow(f"Camera{i}_Depth", depth_colormap)
        cv2.waitKey(1)

    def stop_all(self):
        for cam in self.cameras:
            cam.stop()






class RealSenseCamera_Tester:
    def __init__(self, multi=False):
        self.multi = multi
        self.cam = MultiCameraManager() if multi else RealSenseCamera()

    # 单相机流式显示测试
    def test_stream(self):
        print("Start RealSense stream test, press Ctrl+C to stop.")
        try:
            while True:
                if self.multi:
                    self.cam.show_frames_all()
                else:
                    color, _, depth = self.cam.get_frames()
                    self.cam.show_frames(color, depth)
        except KeyboardInterrupt:
            print("Stream test stopped.")

    # 单相机定时采集测试
    def test_capture_interval(self, interval=0.5):
        count = 0
        print("Start RealSense capture interval test, press Ctrl+C to stop.")
        try:
            while True:
                if self.multi:
                    self.cam.show_frames_all()
                else:
                    color, _, depth = self.cam.get_frames()
                    self.cam.show_frames(color, depth)

                count += 1
                print("Captured frames:", count)
                time.sleep(interval)
        except KeyboardInterrupt:
            print("Capture interval test stopped.")

    def stop(self):
        if self.multi:
            self.cam.stop_all()
        else:
            self.cam.stop()



def get_all_serials():

    ctx = rs.context()
    devices = ctx.query_devices()

    if len(devices) == 0:
        print("没有检测到 RealSense 相机")
    else:
        for i, dev in enumerate(devices):
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)

            print("相机编号:", i)
            print("名称:", name)
            print("序列号:", serial)
            print("-" * 30)

# 相机编号: 0
# 名称: Intel RealSense D435I[franka上面那个]
# 序列号: 348122070854

# 相机编号: 1
# 名称: Intel RealSense D435I【第三视角】
# 序列号: 254322072252


if __name__ == "__main__":
    tester = RealSenseCamera_Tester(multi=True)  # 默认单相机，第三视角
    # tester = RealSenseCamera_Tester(multi=True)  # 双相机

    try:
        tester.test_stream()
        # tester.test_capture_interval(interval=0.5)
    finally:
        tester.stop()