#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
from dataclasses import dataclass, field
from pyOpenHaptics.hd_device import HapticDevice
from pyOpenHaptics.hd_callback import hd_callback
import pyOpenHaptics.hd as hd

#============ 数据结构 ================
@dataclass
class TouchState:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    button_down: bool = False
    button_up: bool = False
    timestamp: float = 0.0


_ACTIVE_CONTROLLER = None  # 当前激活的 TouchController 实例

#============ 回调函数 ================
@hd_callback
def _touch_state_callback():
    global _ACTIVE_CONTROLLER
    controller = _ACTIVE_CONTROLLER
    if controller is None:
        return

    transform = hd.get_transform()
    button_state = hd.get_buttons()
    position = np.array([transform[3][0], transform[3][1], transform[3][2]], dtype=np.float64)

    controller._state.position = position
    controller._state.button_down = bool(button_state & 1)
    controller._state.button_up = bool(button_state & 2)
    controller._state.timestamp = time.time()
    controller._has_state = True


#============ 控制器类 ================
class TouchController:
    def __init__(self, device_name="Default Device", auto_start=True):
        self.device_name = device_name
        self._state = TouchState()
        self._has_state = False
        self._zero_position = np.zeros(3, dtype=np.float64)
        self._prev_button_down = False
        self._prev_button_up = False
        self._device = None
        if auto_start:
            self.start()

    # 启动 Touch 设备
    def start(self):
        global _ACTIVE_CONTROLLER
        if self._device is not None:
            return
        _ACTIVE_CONTROLLER = self
        self._device = HapticDevice(device_name=self.device_name, callback=_touch_state_callback)

    # 关闭 Touch 设备
    def close(self):
        global _ACTIVE_CONTROLLER
        if self._device is not None:
            self._device.close()
            self._device = None
        if _ACTIVE_CONTROLLER is self:
            _ACTIVE_CONTROLLER = None

    # 等待设备第一次回调状态
    def wait_until_ready(self, timeout=2.0):
        start = time.time()
        while time.time() - start < timeout:
            if self._has_state:
                return True
            time.sleep(0.01)
        return False

    # 获取当前状态快照
    def get_state(self):
        return TouchState(
            position=self._state.position.copy(),
            button_down=self._state.button_down,
            button_up=self._state.button_up,
            timestamp=self._state.timestamp
        )

    # 获取当前三维位置
    def get_position(self):
        return self.get_state().position

    # 获取相对于零点的位移
    def get_relative_position(self):
        return self.get_position() - self._zero_position

    # 设置零点
    def zero(self):
        self._zero_position = self.get_position().copy()
        self._prev_button_down = self._state.button_down
        self._prev_button_up = self._state.button_up
        return self._zero_position.copy()

    # 映射到 Franka 坐标系
    def get_mapped_delta(self):
        delta = self.get_relative_position()
        return np.array([-delta[2], -delta[0], delta[1]], dtype=np.float64)

    # 获取按钮边沿事件
    def get_button_edges(self):
        state = self.get_state()
        down_pressed = state.button_down and not self._prev_button_down
        down_released = not state.button_down and self._prev_button_down
        up_pressed = state.button_up and not self._prev_button_up
        up_released = not state.button_up and self._prev_button_up
        self._prev_button_down = state.button_down
        self._prev_button_up = state.button_up
        return {
            "down_pressed": down_pressed,
            "down_released": down_released,
            "up_pressed": up_pressed,
            "up_released": up_released
        }

    # 让记录程序能知道 Touch 设备是否有有效状态数据
    def has_state(self):
        return self._has_state

#============ 测试类 ================
class TouchControllerTester:
    def __init__(self, touch: TouchController, hz=20):
        self.touch = touch
        self.dt = 1.0 / hz

    def test_print_state(self):
        print("Start test_print_state, press Ctrl+C to stop.")
        try:
            while True:
                s = self.touch.get_state()
                print(f"pos = [{s.position[0]:.3f}, {s.position[1]:.3f}, {s.position[2]:.3f}] | "
                      f"down = {s.button_down} | up = {s.button_up} | t = {s.timestamp:.3f}")
                time.sleep(self.dt)
        except KeyboardInterrupt:
            print("Stream test stopped.")

    def test_print_mapped_delta(self):
        if not self.touch._has_state:
            print("Touch state not ready.")
            return
        self.touch.zero()
        print("Zero position recorded. Start test_print_mapped_delta, press Ctrl+C to stop.")
        try:
            while True:
                raw = self.touch.get_position()
                rel = self.touch.get_relative_position()
                mapped = self.touch.get_mapped_delta()
                edges = self.touch.get_button_edges()
                print(f"raw = [{raw[0]:.3f}, {raw[1]:.3f}, {raw[2]:.3f}] | "
                      f"rel = [{rel[0]:.3f}, {rel[1]:.3f}, {rel[2]:.3f}] | "
                      f"mapped = [{mapped[0]:.3f}, {mapped[1]:.3f}, {mapped[2]:.3f}] | "
                      f"up_pressed = {edges['up_pressed']} | down_pressed = {edges['down_pressed']}")
                time.sleep(self.dt)
        except KeyboardInterrupt:
            print("Mapped delta test stopped.")


#============ 示例 ================
if __name__ == "__main__":
    touch = TouchController()
    tester = TouchControllerTester(touch, hz=10)
    try:
        if touch.wait_until_ready(timeout=2.0):
            tester.test_print_mapped_delta()
        else:
            print("Touch device not ready.")
    finally:
        touch.close()