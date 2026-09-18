# Obstacle Avoidance & Constraint Perception Module

## 1. 模块定位
本模块负责具身机械臂操作中的**物理约束感知、场景语义解耦、障碍物包络面提取与整臂避障干预**，对应申请书研究内容 (1)「深度视觉下的物理约束感知研究」与方向 A「VLA + 深度视觉安全约束」。

## 2. 目录结构
- legacy_reference/: 历史避障与约束代码全量汇总（包含 MuJoCo 仿真 Safety Filter、Aloha 接触流形碰撞检测、单点距离门禁等 15 个脚本）
- perception/: 场景语义理解、桌面点云滤波 (RANSAC)、前景物体解耦、3D 包络面 (OBB / Convex Hull) 提取
- safety/: Franka 连杆胶囊体 (Kinematic Capsules) 建模、微秒级 Capsule-to-Box / Capsule-to-Hull 距离计算、CBF / 动作残差截断
- 	ests/: 单元测试与真机/离线点云测试数据

## 3. 开发规划
1. **阶段 1 (感知闭环)**: RealSense RGB-D 深度图对齐 -> RANSAC 台面滤除 -> 欧氏聚类 -> 提取 3D OBB/凸包包络面
2. **阶段 2 (机械臂本体建模)**: Franka 7 轴 + 夹爪的 8 胶囊体运动学几何模型构建与极速距离场
3. **阶段 3 (控制干预接入)**: 改造 safety_filter.py 接入实时代理 live_vla_agent.py 或笛卡尔控制器
