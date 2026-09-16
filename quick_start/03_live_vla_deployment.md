# 教程 03：实机闭环 VLA 部署四步法 (03_live_vla_deployment.md)

本教程指导如何将训练好的 VLA 模型（如微调好的 Pi0.5 / DROID jointpos 模型）接入实机 Franka 机械臂，进行闭环自主推理与抓取控制。

---

## 第 1 步：在 Franka Linux 终端启动底层关节速度控制器

> [!CAUTION]
> 必须在系统原生 ROS 环境运行，**严禁在 Conda 环境中启动**！

打开 Franka 控制机终端 1，运行：
```bash
source /opt/ros/noetic/setup.bash
source /home/ssui/franka_ros_ws/catkin_ws2/devel/setup.bash

# 启动底层 1kHz 关节速度控制器（带硬件限速与平滑保护）
roslaunch franka_example_controllers joint_velocity_example_controller.launch robot_ip:=172.16.0.2
```

---

## 第 2 步：在 Franka 新终端启动闭环执行服务

新开一个 Franka 终端 2，运行：
```bash
source /opt/ros/noetic/setup.bash
cd /home/ssui/project/embodied_midterm/controller_lerobot

# 启动执行服务（监听 8765 端口，开启 Y 轴镜像以匹配前置相机朝向）
python3 closed_loop_franka_server.py --port 8765 --invert-y
```
*(终端将打印机械臂当前关节就绪状态，并等待 Windows 端 Agent 建立 TCP 连接)*

---

## 第 3 步：在 Windows GPU 服务器启动 VLA 推理 Agent

在 Windows 端打开 CMD 或 PowerShell，运行：

### 方式 A（推荐，一键批处理脚本）：
```cmd
cd /d C:\Users\74727\Desktop\project\VLA_franka
scripts\launch_live_agent.bat --live --task "pick up the blue block and place it in the brown basket"
```

### 方式 B（完整命令行方式）：
```cmd
cd /d C:\Users\74727\Desktop\project\VLA_franka

set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=1

C:\Users\74727\miniconda3\envs\lerobot\python.exe franka_teleop\closed_loop_franka.py ^
  --profile jointpos ^
  --checkpoint outputs\checkpoints\action_expert_final.pt ^
  --task "pick and place the red cube" ^
  --live
```

### 技巧：如何切换模型与任务对比测试？
- **测试不同微调 checkpoint（如 Step 500 模型）**：
  ```cmd
  scripts\launch_live_agent.bat --live --checkpoint outputs\checkpoints\action_expert_step500.pt --task "pick up the blue block and place it in the brown basket"
  ```
- **切换抓取目标**：只需修改 `--task` 参数，例如：`--task "pick and place the red cube"`。

---

## 第 4 步：确认急停，回车启动闭环

两端网络握手成功后，Franka 终端 2 会打印：
```text
[READY] Hold physical E-STOP. Press [ENTER] to start live closed-loop control...
```

1. **现场操作人员手持物理急停按钮**，目视核验机械臂周围无人与危险杂物；
2. 在 Franka 终端按下 **[回车 (ENTER)]**，正式下发动作，机械臂开始闭环执行抓取！
3. **紧急中止**：
   - 调试暂停：在终端按下 `Ctrl + C`；
   - 突发碰撞风险：**立即拍下手中的物理急停开关**！
