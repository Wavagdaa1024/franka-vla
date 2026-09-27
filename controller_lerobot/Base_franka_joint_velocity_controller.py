import os
import sys

CATKIN_WS2_PYTHON = "/home/ssui/franka_ros_ws/catkin_ws2/devel/lib/python3/dist-packages"
if os.path.exists(CATKIN_WS2_PYTHON) and CATKIN_WS2_PYTHON not in sys.path:
    sys.path.insert(0, CATKIN_WS2_PYTHON)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import rospy
from geometry_msgs.msg import TwistStamped, PoseStamped
from std_msgs.msg import Float64MultiArray
from franka_msgs.msg import FrankaState
from sensor_msgs.msg import JointState
from franka_gripper.msg import MoveActionGoal, GraspActionGoal, GraspGoal
from tf.transformations import quaternion_from_matrix


class FrankaJointVelocityController:
    """
    Franka 关节速度控制器上层 Python 适配器。
    适配底层 C++ 1kHz 带有 franka::limitRate 平滑保护的 JointVelocityExampleController。
    支持：
      1. 7 维关节角速度直接下发 (set_joint_velocities)
      2. 6 维末端笛卡尔速度下发 (set_cartesian_velocity，由底层 C++ 雅可比解算)
      3. 高刚度安全位置保持 (stop)
      4. 夹爪开闭接口 (open_gripper, close_gripper)
      5. 状态读取 (get_cartesian_pose, get_joint_positions, get_joint_velocities, get_gripper_width)
    """

    def __init__(self,
                 state_topic='/franka_state_controller/franka_states',
                 joint_vel_topic='/joint_velocity_example_controller/joint_velocities',
                 cart_vel_topic='/joint_velocity_example_controller/cartesian_velocity'):
        if not rospy.core.is_initialized():
            rospy.init_node('franka_joint_velocity_controller_client', anonymous=True)

        # 状态订阅
        self.state_sub = rospy.Subscriber(state_topic, FrankaState, self._state_callback, queue_size=1)
        self.gripper_state_sub = rospy.Subscriber('/franka_gripper/joint_states', JointState, self._gripper_state_callback, queue_size=1)

        # 速度指令发布器
        self.joint_vel_pub = rospy.Publisher(joint_vel_topic, Float64MultiArray, queue_size=1)
        self.cart_vel_pub = rospy.Publisher(cart_vel_topic, TwistStamped, queue_size=1)

        # 夹爪控制发布器
        self.gripper_move_pub = rospy.Publisher('/franka_gripper/move/goal', MoveActionGoal, queue_size=10)
        self.gripper_grasp_pub = rospy.Publisher('/franka_gripper/grasp/goal', GraspActionGoal, queue_size=10)

        # 内部状态缓存
        self.current_pose_matrix = None
        self.current_pos = None
        self.current_quat = None
        self.current_joint_pos = None
        self.current_joint_vel = None
        self.gripper_positions = []
        self._last_state_time = 0.0

        rospy.loginfo("Franka Joint Velocity Controller Client 初始化完成，等待状态更新...")

    # ================= 状态回调 =================
    def _state_callback(self, msg):
        matrix = np.array(msg.O_T_EE, dtype=np.float64).reshape(4, 4, order='F')
        self.current_pose_matrix = matrix
        self.current_pos = matrix[:3, 3]
        self.current_quat = quaternion_from_matrix(matrix)
        self.current_joint_pos = np.array(msg.q, dtype=np.float64)
        self.current_joint_vel = np.array(msg.dq, dtype=np.float64)
        self._last_state_time = time.time()

    def _gripper_state_callback(self, msg):
        if msg.position is not None:
            self.gripper_positions = list(msg.position)

    # ================= 状态查询接口 =================
    def get_cartesian_pose(self):
        """获取当前末端笛卡尔位姿 (pos: 3, quat: 4 [x, y, z, w])"""
        if self.current_pos is None or self.current_quat is None:
            return None, None
        return self.current_pos.copy(), self.current_quat.copy()

    def get_joint_positions(self):
        """获取当前 7 维关节位置 (q: 7)"""
        if self.current_joint_pos is None:
            return None
        return self.current_joint_pos.copy()

    def get_joint_velocities(self):
        """获取当前 7 维关节速度 (dq: 7)"""
        if self.current_joint_vel is None:
            return None
        return self.current_joint_vel.copy()

    def get_gripper_width(self):
        """获取当前夹爪张开宽度 (m)"""
        if not self.gripper_positions:
            return None
        return float(sum(self.gripper_positions))

    # ================= 控制指令接口 =================
    def set_joint_velocities(self, dq):
        """
        向底层控制器发送 7 维关节角速度 (rad/s)。
        参数:
            dq: 7 维数组/列表
        """
        dq = np.asarray(dq, dtype=np.float64)
        if dq.shape != (7,):
            raise ValueError(f"Expected 7-dim joint velocities, got shape {dq.shape}")

        msg = Float64MultiArray()
        msg.data = dq.tolist()
        self.joint_vel_pub.publish(msg)

    def set_cartesian_velocity(self, v_linear, v_angular=None, frame_id='panda_link0'):
        """
        向底层控制器发送末端笛卡尔线速度 (m/s) 与角速度 (rad/s)。
        由底层 C++ 实时执行雅可比解算并进行 C2 连续平滑。
        """
        v_linear = np.asarray(v_linear, dtype=np.float64)
        if v_angular is None:
            v_angular = np.zeros(3, dtype=np.float64)
        else:
            v_angular = np.asarray(v_angular, dtype=np.float64)

        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id
        msg.twist.linear.x = float(v_linear[0])
        msg.twist.linear.y = float(v_linear[1])
        msg.twist.linear.z = float(v_linear[2])
        msg.twist.angular.x = float(v_angular[0])
        msg.twist.angular.y = float(v_angular[1])
        msg.twist.angular.z = float(v_angular[2])
        self.cart_vel_pub.publish(msg)

    def stop(self):
        """发送 0 速度，依靠速度闭环刚性锁死当前位置，绝不下垂漂移"""
        msg = Float64MultiArray()
        msg.data = [0.0] * 7
        self.joint_vel_pub.publish(msg)

    # 兼容阻抗控制器的接口签名，避免外部代码报错
    def set_cartesian_pose(self, position, quaternion, frame_id='panda_link0'):
        """
        闭环位置跟踪：计算当前位姿到目标位姿的误差，换算成速度下发。
        """
        pos, quat = self.get_cartesian_pose()
        if pos is None:
            return
        pos_err = np.asarray(position) - pos
        # P 环增益 (1/s)
        k_p = 2.0
        v_linear = np.clip(k_p * pos_err, -0.15, 0.15)
        self.set_cartesian_velocity(v_linear, [0.0, 0.0, 0.0], frame_id=frame_id)

    # ================= 夹爪控制接口 =================
    def open_gripper(self, width=0.08, speed=0.5):
        move_goal = MoveActionGoal()
        move_goal.goal.width = width
        move_goal.goal.speed = speed
        self.gripper_move_pub.publish(move_goal)

    def close_gripper(self, width=0.055, force=5.0, speed=0.1, inner_epsilon=0.005, outer_epsilon=0.005):
        grasp_goal = GraspGoal()
        grasp_goal.width = width
        grasp_goal.speed = speed
        grasp_goal.force = force
        grasp_goal.epsilon.inner = inner_epsilon
        grasp_goal.epsilon.outer = outer_epsilon

        action_goal = GraspActionGoal()
        action_goal.goal = grasp_goal
        self.gripper_grasp_pub.publish(action_goal)