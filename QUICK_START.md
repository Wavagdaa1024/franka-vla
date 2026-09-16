# VLA_franka 快速使用与实机运行指南 (QUICK_START.md)

本文档提供从离线自检、多终端遥操作数据采集，到实机闭环 VLA 部署的端到端标准操作手册。

---

## 零、前置自检与环境校验 (Windows GPU 服务器)

在连接机械臂前，建议先在 Windows 终端执行自检：

```bash
# 1. 验证模型权重与资产完整性
python tests\check_assets.py

# 2. 运行安全门禁单元测试（软限位、限幅与 TTL）
python tests\check_safety_guards.py

# 3. 运行 GPU 1 单卡隔离离线影子运行 (验证推理延迟与 0 NaN/Inf)
scripts\launch_shadow_run.bat
```

---

## 一、数据采集流程 (LeRobot 格式)

### 架构数据流

```text
Franka ROS / 笛卡尔阻抗控制器
        ↓
TouchFrankaTeleopController
        ↓  TCP 8766
Windows 双相机 + LeRobot 录制器
        ↓
保存 episode 数据集
```

### 终端 1：Franka 控制电脑，启动 ROS 和底层控制器

#### 1. 阻抗控制底层 (推荐用于遥操作采集)
```bash
source /opt/ros/noetic/setup.bash
source /home/ssui/franka_ros_ws/catkin_ws2/devel/setup.bash

roslaunch franka_example_controllers \
  cartesian_impedance_example_controller.launch \
  robot_ip:=172.16.0.2
```

#### 2. 速度控制底层 (备用)
```bash
source /opt/ros/noetic/setup.bash
source /home/ssui/franka_ros_ws/catkin_ws2/devel/setup.bash

# 启动关节速度控制器（带 1kHz 平滑保护）
roslaunch franka_example_controllers \
  joint_velocity_example_controller.launch \
  robot_ip:=172.16.0.2
```

---

### 终端 2：Franka 控制电脑，启动 Franka 控制器和夹爪

```bash
source /opt/ros/noetic/setup.bash
source /home/ssui/franka_ros_ws/catkin_ws2/devel/setup.bash

cd /home/ssui/project/embodied_midterm/controller_lerobot
python3 Teleop_dataset_recorder.py --sample-hz 15
```

---

### 终端 3：Windows 服务器，启动录制端

#### (1) 开始新的数据集目录 (不要加 `--resume`)
```powershell
cd C:\Users\74727\Desktop\project\VLA_franka

C:\Users\74727\miniconda3\envs\lerobot\python.exe franka_teleop\record_teleop.py `
  --host 10.197.16.43 `
  --port 8766 `
  --action-space droid_joint_delta `
  --task "pick and place the red cube" `
  --repo-id local/franka_red_cube `
  --root "C:\Users\74727\Desktop\project\VLA_franka\dataset\teleop_pick_cube_15hz_001" `
  --front-serial 254322072252 `
  --wrist-serial 348122070854 `
  --fps 15 `
  --no-preview
```

*(或使用一键批处理：`scripts\launch_record.bat --repo-id local/franka_red_cube --root "C:\Users\74727\Desktop\project\VLA_franka\dataset\teleop_pick_cube_15hz_001" --fps 15`)*

#### (2) 后续向已有目录继续追加录制 (必须加 `--resume`)
```powershell
cd C:\Users\74727\Desktop\project\VLA_franka

C:\Users\74727\miniconda3\envs\lerobot\python.exe franka_teleop\record_teleop.py `
  --host 10.197.16.43 `
  --port 8766 `
  --action-space droid_joint_delta `
  --task "pick and place the orange cube" `
  --repo-id local/franka_red_cube `
  --root "C:\Users\74727\Desktop\project\VLA_franka\dataset\teleop_pick_cube_15hz_001" `
  --front-serial 254322072252 `
  --wrist-serial 348122070854 `
  --fps 15 `
  --resume `
  --no-preview
```

---

### 录制控制与操作顺序

所有键盘操作均在 **Franka 的终端 2** 进行：

| 按键 / 操作 | 功能说明 |
|---|---|
| **Touch UP** | 开启 / 关闭遥操作 |
| **Touch DOWN** | 开启 / 关闭夹爪 |
| **`s`** | 开始当前 episode 录制 |
| **`e`** | 保存当前 episode |
| **`d`** | 丢弃当前 episode（异常或碰撞时按此键） |
| **`p`** | 打印当前连接与录制状态 |
| **`q`** | 退出程序 |

#### 实际标准操作顺序：
1. 启动 ROS (终端 1)
2. 启动 Franka 笛卡尔阻抗控制器 (终端 1)
3. 启动 `TouchFrankaTeleopController` (终端 2)
4. 启动 Windows 录制器 (终端 3)
5. 确认 Windows 终端显示已连接
6. Touch 手柄按 **UP** 开启遥操作跟随
7. 键盘按 **`s`** 开始录制本条轨迹
8. 控制机械臂完成抓取与放置
9. 键盘按 **`e`** 保存本条轨迹

> **重要验证**：GPU 端必须看到打印：
> ```text
> Episode 0 started
> Episode 0 saved: ... frames
> ```
> 看到 `Episode X saved` 后，这段数据才真正安全写入磁盘。

---

## 二、实机闭环 VLA 部署流程

### 第 1 步：在 Franka Linux 终端启动底层关节速度控制器（不可在 conda 环境）

打开 Franka 控制机终端，运行：
```bash
source /opt/ros/noetic/setup.bash
source /home/ssui/franka_ros_ws/catkin_ws2/devel/setup.bash

# 启动底层 1kHz 关节速度控制器
roslaunch franka_example_controllers joint_velocity_example_controller.launch robot_ip:=172.16.0.2
```

---

### 第 2 步：在 Franka 新终端启动闭环执行服务

新开一个 Franka 终端，运行：
```bash
source /opt/ros/noetic/setup.bash
cd /home/ssui/project/embodied_midterm/controller_lerobot

# 启动执行服务（监听 8765 端口，开启 Y 轴镜像以匹配相机朝向）
python3 closed_loop_franka_server.py --port 8765 --invert-y
```
*(终端将打印机械臂就绪状态，并等待 Windows 端 Agent 建立 TCP 连接)*

---

### 第 3 步：在 Windows GPU 服务器启动 VLA 推理 Agent

在 Windows 端打开 CMD 或 PowerShell，运行：

#### 方式 A（推荐，一键启动批处理脚本）：
```cmd
cd /d C:\Users\74727\Desktop\project\VLA_franka
scripts\launch_live_agent.bat --live --task "pick up the blue block and place it in the brown basket"
```

#### 方式 B（完整命令行方式）：
```cmd
cd /d C:\Users\74727\Desktop\project\VLA_franka

set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=1

C:\Users\74727\miniconda3\envs\lerobot\python.exe franka_teleop\closed_loop_franka.py ^
  --profile jointpos ^
  --checkpoint checkpoints\pi05_droid\action_expert_final.pt ^
  --task "pick and place the red cube" ^
  --live
```

#### 提示：如何切换模型对比测试？
- **默认模型**：直接运行上述命令，脚本会自动加载默认审计模型。
- **指定 Step 500 检查点**：
  ```cmd
  scripts\launch_live_agent.bat --live --checkpoint checkpoints\pi05_droid\action_expert_step500.pt --task "pick up the blue block and place it in the brown basket"
  ```
- **抓取红方块**：只需修改 `--task "pick and place the red cube"`。

---

### 第 4 步：确认急停，回车启动闭环

两端网络握手成功后，Franka 终端会提示：
```text
[READY] Hold physical E-STOP. Press [ENTER] to start live closed-loop control...
```

1. **现场操作人员手持物理急停按钮**，目视确认机械臂周围无障碍物。
2. 在 Franka 终端按下 **[回车 (ENTER)]**，即可正式启动闭环实时推理与抓取！
3. **安全中止**：过程中若需暂停或复位，可随时在终端按 `Ctrl + C`，或直接拍下物理急停开关。
