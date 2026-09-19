# 05. 第三视角相机快速复位与手眼标定指引 (05_camera_alignment_and_calibration.md)

在具身模仿学习（VLA / ACT / Diffusion Policy）中，第三视角相机（RealSense D435I, S/N: `254322072252`）固定在外部三脚架上，极易因意外碰撞或震动发生微小偏位，导致测试图像与训练数据分布发生漂移（OOD）。

本模块采用 **Franka 手腕顶面 60mm ChArUco 标定板 + 机械臂一键就位 + 6-DoF 毫米级姿态闭环 + 半透明鬼影叠图**，提供 10 秒级快速复位能力。

---

## 核心架构与入口（相机与机械臂彻底解耦）

> [!IMPORTANT]
> **相机调试与机械臂运动是 100% 物理完全解耦的两个独立模块**：
> - **相机复位监控**（`scripts\align_camera.bat`）：在 Windows GPU 服务器上运行，为纯视觉、只读的轻量感知工具，内部绝无任何机器人控制代码，调试相机时机械臂绝不会误动。
> - **机械臂就位**（ROS Launch：`move_to_camera_calib.launch`）：在 Franka 控制电脑上直接通过 ROS Launch 运行，仅在需要机械臂归位到标定姿态时单独调用一次。

- **机械臂就位**：`roslaunch franka_example_controllers move_to_camera_calib.launch robot_ip:=172.16.0.2`（Franka 控制端独立调用）
- **相机复位监控**：`scripts\align_camera.bat`（Windows 端纯视觉只读）
- **基准存储文件**：`tests\camera_alignment\baseline_pose.json`
- **局域网手机/平板 Web 看板**：`http://10.70.242.38:8088`

---

## 快速复位操作三步法

### 第一步：让机械臂运行到标定位置
在对准相机前，在 Franka 控制电脑（`franka-control`）终端运行标准 ROS Launch 节点，机械臂将平滑运动至标准标定位姿后自动退出：

```bash
source /opt/ros/noetic/setup.bash
source /home/ssui/franka_ros_ws/catkin_ws2/devel/setup.bash
roslaunch franka_example_controllers move_to_camera_calib.launch robot_ip:=172.16.0.2
```

---

### 第二步：启动对齐与监测程序
在 Windows 服务器（`win-74727`）上：
- 双击运行：`scripts\align_camera.bat`；
- 桌面将弹出高帧率对准窗口，同时局域网网页服务已同步在 `http://10.70.242.38:8088` 运行（手机浏览器打开即可站在三脚架旁实时查看）。

---

### 第三步：微调三脚架至吸合对齐
1. 观察画面：
   - 标定板上会自动锁定 9 个亚像素绿色角点，外框为青色；
   - 之前保存的黄金基准框为**黄色虚线框**；
2. 参考左上角极简半透明 HUD 的偏差指示（微调三脚架）：
   - `dX`: 水平偏差（提示向左/向右微推）；
   - `dY`: 垂直高度偏差（提示升高/降低）；
   - `dZ`: 前后进深偏差（提示推前/拉后）；
   - `Pitch / Yaw / Roll`: 角度微调提示；
3. **关键快捷键（亦可在手机网页上点击）**：
   - **`H` 键**：**一键隐藏/显示所有文字 HUD**（纯净视野，彻底不遮挡机械臂和物料框）；
   - **`G` 键**：开启/关闭 50% 半透明“鬼影”叠图，利用操作台背景边缘肉眼验证；
   - **`S` 键**：如果重新确立了新的理想位置，按 `S` 重新覆盖黄金基准；
   - **`Q` / `ESC`**：退出程序。
4. 当所有偏差均处于阈值内（误差 $\le 3\text{mm}, \le 1.0^\circ$）时，顶部状态条瞬间变绿：
   ```text
   [✔ ALIGNED] Err: 1.2mm / 0.4deg (PASS)
   ```
   此时拧紧三脚架旋钮，按 `Q` 退出，相机即已完美复位！

---

## 附：已记录的 Franka 黄金标定姿态

系统在 `baseline_pose.json` 中已自动绑定记录了当前姿态：
- **关节角 ($q_{\text{rad}}$)**：
  `[0.0284, 0.7187, 0.0210, -1.7251, -0.0087, 2.4741, 0.8302]`
- **末端笛卡尔坐标 ($O\_T\_EE$)**：
  `X: 673.7mm, Y: 32.9mm, Z: 67.8mm`
