#!/usr/bin/env python3
import rospy
from Base_franka_joint_velocity_controller import FrankaJointVelocityController

def main():
    rospy.init_node('read_current_pose', anonymous=True)
    arm = FrankaJointVelocityController()
    rospy.sleep(0.5)
    pos, quat = arm.get_cartesian_pose()
    if pos is not None:
        print('=' * 50)
        print(f'当前机械臂末端位置 (Base 坐标系):')
        print(f'  X: {pos[0]:.4f} m')
        print(f'  Y: {pos[1]:.4f} m')
        print(f'  Z: {pos[2]:.4f} m  <--- (这就是真实抓取高度)')
        print(f'  姿态四元数: {[round(q, 4) for q in quat]}')
        print('=' * 50)
    else:
        print('未接收到机械臂位姿状态，请确认 ROS 控制器节点已启动。')

if __name__ == '__main__':
    main()
