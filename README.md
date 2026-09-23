# franka-vla: Franka Panda 具身智能 VLA 部署工程

本项目面向 **Franka Panda** 机械臂具身操作任务，基于 Hugging Face **LeRobot** 规范实现端到端数据采集、Pi0.5 策略微调（LoRA）、三维工作空间安全防护与实时闭环推理部署。

---

## 🏗️ 整体系统架构与工作流

系统采用“**GPU 算力工作站 + 实时机械臂控制电脑**”双机协同架构：

```text
               ┌─────────────────────────────────────────────────────┐
               │              Windows GPU 推理工作站                  │
               │   (RTX 5090 32GB / Python / PyTorch / Pi0.5 LoRA)   │
               └───────────────▲─────────────────────▲───────────────┘
                               │                     │
                    (双目图像 Front+Wrist)       (15Hz 动作分块 Chunk)
                               │                     │
               ┌───────────────┴─────────────────────┴───────────────┐
               │               Linux 机械臂控制主机                   │
               │         (Ubuntu 20.04 / ROS Noetic / 1kHz)          │
               └─────────────────────────▲───────────────────────────┘
                                         │
                                  (libfranka 驱动)
                                         │
                               ┌─────────┴─────────┐
                               │ Franka Panda 机械臂│
                               └───────────────────┘
```

---

## 🚀 核心代码入口 (Core Entry Points)

仓库核心功能已高度收敛封装，日常运行只需关注以下 4 个最关键入口：

| 任务场景 | 核心入口脚本 / 命令 | 底层实现文件 | 说明与关键特性 |
| :--- | :--- | :--- | :--- |
| **1. 实机自主推理 (首选)** | `scripts\run_sync_agent.bat` | `scripts/python/sync_vla_agent.py` | **Stop-and-Go 同步执行**：执行后停稳50ms无模糊采图，高精抓取首选 |
| **1.1 实机推理 (RTC 流动)** | `scripts\run_async_agent.bat` | `scripts/python/async_rtc_vla_agent.py` | **异步 RTC 模式**：滚动时域衔接，适合大范围平滑连续移动 |
| **1.2 独立推理服务** | `scripts\launch_server.bat` | `scripts/python/vla_inference_server.py` | GPU 1 常驻模型推理服务，供客户端通过网络远程调用 |
| **2. 模型微调训练** | `scripts\train_pure_flow_50k_gpu1.bat` | `scripts/python/train_pi05_lora.py` | 50,000 步纯净流匹配微调，带 LoRA Dropout 与 WandB 监控 |
| **3. 遥操作数据采集** | `scripts\launch_record.bat` | `src/franka_teleop/record_teleop.py` | 15Hz 同步录制机械臂关节与双摄画面，自动输出 LeRobot 格式 |
| **4. 控制端底层服务** | `python3 sync_franka.py --z-min 0.0070` | `src/franka_teleop/sync_franka.py` | Linux 端 ROS 服务，内置 $Z \ge 7\text{mm}$ 防撞保护与垂直姿态锁定 |

---

## ⚡ 三步上手流程 (Quickstart)

### 第一步：在 Linux 控制主机启动机械臂服务

```bash
# 激活 ROS 环境并进入控制目录 (位于 10.197.16.43 控制电脑)
source /opt/ros/noetic/setup.bash
cd /home/ssui/project/embodied_midterm/controller_lerobot

# 启动底层关节速度控制器与安全同步服务
roslaunch franka_example_controllers joint_velocity_example_controller.launch robot_ip:=172.16.0.2
python3 sync_franka.py --z-min 0.0070
```

### 第二步（可选）：数据录制与模型训练

* **录制新任务数据**（连接双目 RealSense 与示教夹爪）：
  ```cmd
  scripts\launch_record.bat
  ```
* **一键启动策略微调**（在 GPU 1 上训练 50k 步）：
  ```cmd
  scripts\train_pure_flow_50k_gpu1.bat
  ```

### 第三步：在 Windows 工作站启动实机闭环推理

```cmd
:: 运行默认同步推理（自动加载 50k 步最新检查点并连接控制端）
scripts\run_sync_agent.bat

:: 或者统一调度总入口：
scripts\run_agent.bat
```

---

## 📁 目录架构导航

```text
franka-vla/
├── src/                                  # 核心系统源码
│   ├── franka_teleop/                    # 机械臂驱动、RTC 闭环通信、Pi0.5 引擎与正逆运动学
│   ├── obstacle_avoidance/               # 视觉包络提取、机械臂胶囊模型与实时避障滤波
│   └── vlm_planning/                     # 大模型（Qwen / RoboBrain）空间规划与目标映射
├── scripts/                              # 用户操作一键批处理层
│   ├── run_sync_agent.bat                # 【实机推理】同步执行首选入口
│   ├── run_async_agent.bat               # 【实机推理】异步连续流入口
│   ├── launch_server.bat                 # 【服务部署】独立推理服务
│   ├── train_pure_flow_50k_gpu1.bat      # 【微调训练】50k 步纯净流训练入口
│   ├── train_crop169_20k_gpu1.bat        # 【微调训练】20k 步视场裁剪微调入口
│   ├── launch_record.bat                 # 【数据采集】遥操作数据同步录制
│   └── python/                           # 底层 Python 实现脚本库
├── quick_start/                          # 详细步骤文档 (01环境检查 ~ 04微调实操)
├── tests/                                # 算法单测与相机 ChArUco 手眼标定工具
├── dataset/                              # 录制轨迹数据集 (LeRobot / Parquet 规范)
└── outputs/                              # 模型 Checkpoints 权重与推理评测日志
```

---

## 📋 运行依赖与环境

* **Windows GPU 推理端**：
  * Python 3.10+ (推荐 Conda `lerobot` 环境)
  * PyTorch 2.2+, CUDA 12.x
  * NVIDIA RTX 4090 / 5090 (建议显存 $\ge$ 24GB)
  * RealSense SDK (`pyrealsense2`), OpenCV, Accelerate
* **Linux 机械臂控制端**：
  * Ubuntu 20.04 LTS
  * ROS Noetic (`rospy`, `sensor_msgs`, `franka_ros`)
  * `libfranka` 0.8.0+
