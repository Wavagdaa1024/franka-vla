
import rospy
import numpy as np
from franka_msgs.msg import FrankaState
from tf.transformations import quaternion_from_matrix
from sensor_msgs.msg import JointState
from franka_gripper.msg import MoveActionGoal, GraspActionGoal, GraspGoal
from geometry_msgs.msg import PoseStamped

class FrankaCartesianImpedanceController:
    def __init__(self, state_topic='/franka_state_controller/franka_states'):
        """
        初始化 Franka 末端笛卡尔阻抗控制器适配器。
        主要用于读取状态（位姿和关节角），未来可扩展发送力矩指令。
        """
        if not rospy.core.is_initialized():
            rospy.init_node('franka_cartesian_impedance_controller', anonymous=True)

        # 订阅 FrankaState，获取机械臂当前状态
        self.state_sub = rospy.Subscriber(state_topic, FrankaState, self._state_callback)

        # 夹爪控制发布器
        self.gripper_move_pub = rospy.Publisher(
            '/franka_gripper/move/goal',
            MoveActionGoal,
            queue_size=10
        )

        self.gripper_grasp_pub = rospy.Publisher(
            '/franka_gripper/grasp/goal',
            GraspActionGoal,
            queue_size=10
        )

        # 发布 equilibrium_pose
        self.equilibrium_pose_pub = rospy.Publisher(
            '/cartesian_impedance_example_controller/equilibrium_pose',
            PoseStamped,
            queue_size=1
)

        # 夹爪状态订阅器
        self.gripper_state_sub = rospy.Subscriber(
            '/franka_gripper/joint_states',
            JointState,
            self._gripper_state_callback,
            queue_size=1
        )

        # 夹爪状态缓存
        self.gripper_positions = []


        # 内部状态缓存
        self.current_pose_matrix = None          # 4x4 齐次矩阵
        self.current_pos = np.zeros(3)           # 末端位置
        self.current_quat = np.array([0.0, 0.0, 0.0, 1.0])  # 四元数
        self.current_joint_pos = np.zeros(7)     # 当前 7 维关节角

        rospy.loginfo("Franka Cartesian Impedance Controller 初始化完成，等待状态更新...")



    # 信息获取接口————————————————————————————————————————————————————————————————————
    # 获取当前末端笛卡尔位姿
    def get_cartesian_pose(self):
        """
        获取当前末端笛卡尔位姿。
        返回:
            position: np.array(3,) - 末端位置
            quaternion: np.array(4,) - 末端四元数姿态
        """
        if self.current_pose_matrix is None:
            rospy.logwarn_once("尚未接收到 Franka 末端笛卡尔状态数据。")
            return None, None
        return self.current_pos.copy(), self.current_quat.copy()

    # 获取当前 7 维关节角
    def get_joint_positions(self):
        """
        获取当前 7 维关节角。
        返回:
            joint_positions: np.array(7,)
        """
        if self.current_joint_pos is None:
            rospy.logwarn_once("尚未接收到 Franka 关节状态数据。")
            return None
        return self.current_joint_pos.copy()

    # 获取当前夹爪宽度
    def get_gripper_width(self):
        """
        获取当前夹爪张开宽度。

        返回:
            width: float，单位 m
        """
        if not self.gripper_positions:
            rospy.logwarn_once("尚未接收到 Franka 夹爪状态数据。")
            return None

        return float(sum(self.gripper_positions))



    # 操控接口————————————————————————————————————————————————————————————————————
    # 打开夹爪
    def open_gripper(self, width=0.08, speed=0.5):
        """
        打开夹爪。

        参数:
            width: 目标张开宽度，单位 m
            speed: 夹爪运动速度，单位 m/s
        """
        while self.gripper_move_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.sleep(0.05)

        move_goal = MoveActionGoal()
        move_goal.goal.width = width
        move_goal.goal.speed = speed

        self.gripper_move_pub.publish(move_goal)

    # 闭合夹爪
    def close_gripper(self, width=0.055, force=5.0, speed=0.1,
                    inner_epsilon=0.005, outer_epsilon=0.005):
        """
        闭合夹爪。

        参数:
            width: 目标闭合宽度，单位 m
            force: 抓取力，单位 N
            speed: 夹爪闭合速度，单位 m/s
            inner_epsilon: 内侧容差
            outer_epsilon: 外侧容差
        """
        while self.gripper_grasp_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            rospy.sleep(0.05)

        grasp_goal = GraspGoal()
        grasp_goal.width = width
        grasp_goal.speed = speed
        grasp_goal.force = force
        grasp_goal.epsilon.inner = inner_epsilon
        grasp_goal.epsilon.outer = outer_epsilon

        action_goal = GraspActionGoal()
        action_goal.goal = grasp_goal

        self.gripper_grasp_pub.publish(action_goal)

    # 发送末端目标位姿
    def set_cartesian_pose(self, position, quaternion, frame_id='panda_link0'):
        """
        发送末端目标位姿给笛卡尔阻抗控制器。

        参数:
            position: [x, y, z]
            quaternion: [x, y, z, w]
            frame_id: 参考坐标系，默认 panda_link0
        """
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id

        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])

        msg.pose.orientation.x = float(quaternion[0])
        msg.pose.orientation.y = float(quaternion[1])
        msg.pose.orientation.z = float(quaternion[2])
        msg.pose.orientation.w = float(quaternion[3])

        self.equilibrium_pose_pub.publish(msg)



    # 回调函数————————————————————————————————————————————————————————————————————
    # 解析 FrankaState 消息，更新末端位姿和关节角
    def _state_callback(self, msg):
        """
        解析 FrankaState 消息，更新末端位姿和关节角。
        """
        matrix = np.array(msg.O_T_EE, dtype=np.float64).reshape(4, 4, order='F')
        self.current_pose_matrix = matrix
        self.current_pos = matrix[:3, 3]
        self.current_quat = quaternion_from_matrix(matrix)
        self.current_joint_pos = np.array(msg.q, dtype=np.float64)

    # 更新当前夹爪关节状态
    def _gripper_state_callback(self, msg):
        
        """
        内部回调函数。

        读取 /franka_gripper/joint_states 中的夹爪关节位置。
        两个手指的位置相加就是当前夹爪宽度。
        """
        if msg.position is not None:
            self.gripper_positions = list(msg.position)





class FrankaCartesianImpedanceController_Tester:
    def __init__(self):
        if not rospy.core.is_initialized():
            rospy.init_node('franka_tester', anonymous=True)
        self.arm = FrankaCartesianImpedanceController()
        self.rate = rospy.Rate(20)  # 默认 20 Hz

    # 测试 1：获取末端位姿和关节角
    def test_cartesian(self):
        while not rospy.is_shutdown():
            pos, quat = self.arm.get_cartesian_pose()
            q = self.arm.get_joint_positions()
            if pos is not None and q is not None:
                print("pos =", np.round(pos, 6))
                print("quat =", np.round(quat, 6))
                print("q =", np.round(q, 6))
                print("-" * 40)
            self.rate.sleep()

    # 测试 2：夹爪开闭
    def test_gripper(self):
        rospy.sleep(1.0)
        print("当前夹爪宽度 =", self.arm.get_gripper_width())

        print("打开夹爪")
        self.arm.open_gripper()
        rospy.sleep(2.0)
        print("打开后夹爪宽度 =", self.arm.get_gripper_width())

        print("关闭夹爪")
        self.arm.close_gripper()
        rospy.sleep(2.0)
        print("关闭后夹爪宽度 =", self.arm.get_gripper_width())

    # 测试 3：沿 z 方向发送末端目标位姿
    def test_send_pose(self, dz=0.005, z_range=0.2):
        while not rospy.is_shutdown():
            pos, quat = self.arm.get_cartesian_pose()
            if pos is not None:
                break
            print("等待获取末端位姿数据...")
            self.rate.sleep()

        target_pos = pos.copy()
        target_quat = quat.copy()
        z_center = pos[2]
        z_min, z_max = z_center - z_range, z_center + z_range
        direction = 1

        while not rospy.is_shutdown():
            target_pos[2] += direction * dz
            if target_pos[2] >= z_max:
                target_pos[2] = z_max
                direction = -1
            elif target_pos[2] <= z_min:
                target_pos[2] = z_min
                direction = 1

            print("发送目标位姿 pos =", target_pos, "quat =", target_quat)
            pos1, quat1 = self.arm.get_cartesian_pose()
            print("当前位姿 pos =", pos1, "quat =", quat1)
            self.arm.set_cartesian_pose(target_pos, target_quat)
            self.rate.sleep()


if __name__ == '__main__':
    tester = FrankaCartesianImpedanceController_Tester()

    # 按需选择运行的测试
    # tester.test_cartesian()
    # tester.test_gripper()
    # tester.test_send_pose()







        
