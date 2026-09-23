# franka-vla: Franka Panda 具身智能 VLA 部署工程

本项目面向 **Franka Panda** 机械臂具身操作任务，基于 Hugging Face **LeRobot** 规范实现端到端数据采集、Pi0.5 策略微调与真机闭环推理。

系统采用“**GPU 算力工作站 (Windows / RTX 5090) + 机械臂控制主机 (Linux / ROS Noetic)**”双机架构。

---

## 🚀 核心代码入口

日常运行只需使用以下 4 个核心入口：

### 1. 实机闭环抓取 (Inference)
* **同步推理 (Stop-and-Go，默认推荐)**
  ```cmd
  scripts\run_sync_agent.bat
  ```
  > 执行完分块后平滑停稳 50ms 采图，消除运动模糊与时间戳竞态，抓取定位精度最高。
* **异步 RTC 推理 (连续流动)**
  ```cmd
  scripts\run_async_agent.bat
  ```
  > 滚动时域平滑衔接，动作连续不卡顿。

### 2. 策略微调训练 (Training)
* **50k 步纯净流匹配训练 (GPU 1)**
  ```cmd
  scripts\train_pure_flow_50k_gpu1.bat
  ```
  > 原生 8D 关节流匹配微调，带 LoRA Dropout 与 WandB 监控。

### 3. 遥操作数据采集 (Data Collection)
* **真机数据录制**
  ```cmd
  scripts\launch_record.bat
  ```
  > 15Hz 同步录制机械臂关节与双目相机（前视 + 腕部）画面，直接输出标准 LeRobot 格式。

### 4. 机械臂底层控制服务 (Robot Server)
* **Linux 控制端启动** (控制电脑 `10.197.16.43`)
  ```bash
  source /opt/ros/noetic/setup.bash
  cd /home/ssui/project/embodied_midterm/controller_lerobot
  python3 sync_franka.py --z-min 0.0070
  ```
  > 内置桌面防撞保护（$Z \ge 7\text{mm}$）、末端姿态垂直对齐与速度平滑限幅。

---

## 📁 核心目录简览

```text
franka-vla/
├── src/franka_teleop/     # 机械臂驱动、RTC 闭环通信、运动学与安全滤波
├── scripts/               # 一键运行入口 (.bat / .sh) 与核心 Python 脚本
├── quick_start/           # 详细部署与调试指南 (01环境检查 ~ 04模型微调)
├── tests/                 # 单元测试与相机标定工具
├── dataset/               # 轨迹数据集 (LeRobot / Parquet 规范)
└── outputs/               # 训练权重与日志
```

---

## 📋 环境要求

* **GPU 推理端 (Windows)**: Python 3.10+, PyTorch 2.2+, CUDA 12.x, RTX 4090/5090
* **控制端 (Linux)**: Ubuntu 20.04, ROS Noetic, libfranka 0.8.0+
