#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import rospy

from Base_touch_controller import TouchController

class TouchFrankaTeleopController:
    #============ 初始化主从控制器 ================
    def __init__(self,
                 touch=None,
                 arm=None,
                 control_hz=100,
                 touch_scale=0.003,
                 touch_deadband=0.5,
                 down_quaternion=None,
                 p_gain=5.0,
                 pos_deadband=0.002,   # 2mm 闭环死区
                 max_cart_vel=0.25,
                 controller_type='velocity'):
        self.touch = touch if touch is not None else TouchController()
        
        if arm is not None:
            self.arm = arm
        else:
            choice = controller_type or 'velocity'
            if choice == 'auto':
                try:
                    topics = [t[0] for t in rospy.get_published_topics()]
                    if any('joint_velocity' in t for t in topics):
                        choice = 'velocity'
                    elif any('cartesian_impedance' in t for t in topics):
                        choice = 'impedance'
                    else:
                        choice = 'velocity'
                except Exception:
                    choice = 'velocity'

            if choice == 'impedance':
                from Base_franka_cartesian_impedance_controller import FrankaCartesianImpedanceController
                self.arm = FrankaCartesianImpedanceController()
                rospy.loginfo('TouchFrankaTeleopController: 使用 FrankaCartesianImpedanceController (笛卡尔阻抗控制器)')
            else:
                from Base_franka_joint_velocity_controller import FrankaJointVelocityController
                self.arm = FrankaJointVelocityController()
                rospy.loginfo('TouchFrankaTeleopController: 使用 FrankaJointVelocityController (带 1kHz 平滑保护的关节速度控制器)')

        self.is_velocity_controller = hasattr(self.arm, 'set_cartesian_velocity')

        self.control_hz = control_hz
        self.touch_scale = touch_scale
        self.p_gain = p_gain
        self.pos_deadband = pos_deadband
        self.max_cart_vel = max_cart_vel
        self.touch_deadband = touch_deadband

        self._v_filtered = np.zeros(3, dtype=np.float64)

        self._down_quat = None
        if down_quaternion is not None:
            quaternion = np.asarray(down_quaternion, dtype=np.float64)
            norm = np.linalg.norm(quaternion)
            if quaternion.shape != (4,) or not np.isfinite(norm) or norm < 1e-9:
                raise ValueError("down_quaternion must contain a valid quaternion")
            self._down_quat = quaternion / norm

        self._running = False
        self._teleop_enabled = False
        self._gripper_closed = False

        self._franka_anchor = np.zeros(3, dtype=np.float64)
        self._target_pos = np.zeros(3, dtype=np.float64)
        self._target_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    #============ 初始化和等待设备就绪 ================
    def initialize(self, timeout=3.0):
        if not self.touch.wait_until_ready(timeout=timeout):
            rospy.logerr("Touch device not ready.")
            return False

        start_time = time.time()
        while time.time() - start_time < timeout:
            pos, quat = self.arm.get_cartesian_pose()
            if pos is not None and quat is not None:
                rospy.loginfo("Touch and Franka are ready.")
                return True
            rospy.sleep(0.01)

        rospy.logerr("Franka cartesian state not ready.")
        return False

    #============ 标定零点锚点 ================
    def reset_anchor(self):
        current_pos, current_quat = self.arm.get_cartesian_pose()
        if current_pos is None or current_quat is None:
            rospy.logwarn("Franka pose not ready, reset anchor failed.")
            return False

        self.touch.zero()
        self._v_filtered.fill(0.0)
        if self.is_velocity_controller:
            self.arm.stop()

        self._franka_anchor = current_pos.copy()
        self._target_pos = current_pos.copy()
        if self._down_quat is None:
            quaternion = np.asarray(current_quat, dtype=np.float64)
            norm = np.linalg.norm(quaternion)
            if not np.isfinite(norm) or norm < 1e-9:
                rospy.logwarn("Invalid current orientation, reset anchor failed.")
                return False
            quaternion /= norm
            down_quaternion = np.array([quaternion[0], quaternion[1], 0.0, 0.0], dtype=np.float64)
            down_norm = np.linalg.norm(down_quaternion)
            if down_norm < 1e-9:
                down_quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            else:
                down_quaternion /= down_norm
            self._down_quat = down_quaternion
            rospy.loginfo("Strict tool-down orientation calibrated: %s", self._down_quat)
        self._target_quat = self._down_quat.copy()

        if not self.is_velocity_controller:
            self.arm.set_cartesian_pose(self._target_pos, self._target_quat)

        rospy.loginfo("Teleop anchor reset.")
        rospy.loginfo("Franka anchor: x={:.4f}, y={:.4f}, z={:.4f}".format(
            self._franka_anchor[0], self._franka_anchor[1], self._franka_anchor[2]
        ))
        return True

    #============ Touch 映射和滤波 ================
    def _apply_touch_deadband(self, delta):
        filtered = delta.copy()
        for i in range(3):
            if abs(filtered[i]) < self.touch_deadband:
                filtered[i] = 0.0
        return filtered

    # 计算目标末端位置 (手跟手 1:1 绝对位移映射)
    def compute_target_position(self):
        mapped_delta = self.touch.get_mapped_delta()
        mapped_delta = self._apply_touch_deadband(mapped_delta)

        offset = self.touch_scale * mapped_delta
        self._target_pos = self._franka_anchor + offset
        return self._target_pos.copy()

    #============ 停止或保持当前姿态 ================
    def hold_current_pose(self):
        self._v_filtered.fill(0.0)
        if self.is_velocity_controller:
            self.arm.stop()
        else:
            pos, quat = self.arm.get_cartesian_pose()
            if pos is not None and quat is not None:
                hold_quaternion = self._down_quat if self._down_quat is not None else quat
                self.arm.set_cartesian_pose(pos, hold_quaternion)

    #============ 控制主循环步进 ================
    def step(self):
        if not self._teleop_enabled:
            self.hold_current_pose()
            return

        target_pos = self.compute_target_position()

        if self.is_velocity_controller:
            current_pos, current_quat = self.arm.get_cartesian_pose()
            if current_pos is not None:
                pos_err = target_pos - current_pos
                dist = float(np.linalg.norm(pos_err))
                
                # 2mm 死区：消除人手微颤引发的低速粘滑摩擦
                if dist < self.pos_deadband:
                    v_target = np.zeros(3, dtype=np.float64)
                else:
                    direction = pos_err / dist
                    effective_dist = dist - self.pos_deadband
                    v_mag = np.clip(self.p_gain * effective_dist, 0.0, self.max_cart_vel)
                    v_target = direction * v_mag

                # 轻量平滑 (80% 直通，极度干脆跟手)
                alpha = 0.8
                self._v_filtered = alpha * v_target + (1.0 - alpha) * self._v_filtered
                self.arm.set_cartesian_velocity(self._v_filtered, [0.0, 0.0, 0.0])
                self._target_quat = self._down_quat if self._down_quat is not None else current_quat.copy()
        else:
            self.arm.set_cartesian_pose(target_pos, self._target_quat)

    #============ 模式切换 ================
    def start_teleop(self):
        if self.reset_anchor():
            self._teleop_enabled = True
            rospy.loginfo("Teleoperation started.")

    def stop_teleop(self):
        self._teleop_enabled = False
        self.hold_current_pose()
        rospy.loginfo("Teleoperation stopped.")

    def toggle_teleop(self):
        if self._teleop_enabled:
            self.stop_teleop()
        else:
            self.start_teleop()

    def toggle_gripper(self):
        try:
            if self._gripper_closed:
                self.arm.open_gripper()
                self._gripper_closed = False
                rospy.loginfo("Gripper opened.")
            else:
                self.arm.close_gripper()
                self._gripper_closed = True
                rospy.loginfo("Gripper closed.")
        except AttributeError:
            rospy.logwarn("Current arm controller does not provide gripper control interface.")

    #============ 主循环 ================
    def run(self):
        if not self.initialize():
            return

        rate = rospy.Rate(self.control_hz)
        self._running = True

        rospy.loginfo("Touch teleop loop started. Press Touch UP to toggle teleop, DOWN to toggle gripper.")

        try:
            while not rospy.is_shutdown() and self._running:
                edges = self.touch.get_button_edges()

                if edges["up_pressed"]:
                    self.toggle_teleop()

                if edges["down_pressed"]:
                    self.toggle_gripper()

                self.step()
                rate.sleep()

        except rospy.ROSInterruptException:
            pass
        finally:
            self.hold_current_pose()
            self.touch.close()

    def stop(self):
        self._running = False
        self.stop_teleop()


if __name__ == "__main__":
    teleop = TouchFrankaTeleopController()
    teleop.run()