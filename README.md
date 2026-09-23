# franka-vla: Franka Panda 具身智能 VLA 部署工程

本项目面向 Franka Panda 机械臂具身操作任务，基于 Hugging Face **LeRobot** 规范实现端到端数据采集、Pi0.5 具身策略微调、三维视觉避障安全滤波与实时 RTC 闭环部署。

---

## 🚀 核心代码入口 (Core Entry Points)

日常所有核心任务（实机闭环、模型训练、遥操作录制、控制服务）均已封装为标准入口脚本，请按需调用：

### 1. 实机闭环推理 (Inference & Autonomous Control)

> **核心实现**：`scripts/python/sync_vla_agent.py`（同步模式） / `scripts/python/async_rtc_vla_agent.py`（异步模式）

* **同步推理 (Stop-and-Go，推荐首选)**
  ```cmd
  scripts\run_sync_agent.bat
  :: 或使用统一总入口：
  scripts\run_agent.bat
  ```
  * **特点**：动作分块执行完毕后平滑停稳 50ms 采图，彻底消除运动模糊与时间戳竞态，抓取定位精度最高。
  * **硬件绑定**：GPU 1 (RTX 5090 32GB)
  * **默认权重**：`outputs\checkpoints\pi05_lora_pure_flow_50k\latest.pt`

* **异步 RTC 推理 (Real-Time Chunking)**
  ```cmd
  scripts\run_async_agent.bat
  ```
  * **特点**：滚动时域平滑衔接（Receding Horizon），连续流动控制，适合大范围平滑移动。

* **独立推理服务 (High-Performance Server)**
  ```cmd
  scripts\launch_server.bat 8088 pure_flow
  ```
  * **底层脚本**：`scripts/python/vla_inference_server.py`
  * **特点**：将 Pi0.5 模型作为独立常驻服务部署于 GPU 1，支持客户端通过网络高并发调用。

---

### 2. 具身策略微调与训练 (Policy Training)

> **核心实现**：`scripts/python/train_pi05_lora.py`

* **50,000 步纯净流匹配微调 (Canonical 50k Pure Flow)**
  ```cmd
  scripts\train_pure_flow_50k_gpu1.bat
  ```
  * **特点**：100% 原生纯净关节流匹配（Loss Mode: `pure_flow`），带 LoRA Dropout 与 WandB 云端实时指标监控。
  * **硬件资源**：物理 GPU 1 (RTX 5090 32GB)

* **20,000 步视野裁剪微调 (Crop 16:9)**
  ```cmd
  scripts\train_crop169_20k_gpu1.bat
  ```

* **自定义训练命令行**
  ```cmd
  python scripts/python/train_pi05_lora.py ^
      --dataset dataset\teleop_pick_cube_15hz_004 ^
      --output-dir outputs\checkpoints\my_pi05_lora ^
      --steps 20000 --loss-mode pure_flow
  ```

---

### 3. 真机遥操作与数据录制 (Teleoperation & Data Collection)

> **核心实现**：`src/franka_teleop/record_teleop.py`

* **实机数据同步录制**
  ```cmd
  scripts\launch_record.bat
  ```
  * **特点**：15Hz 同步采集 Franka 机械臂状态与双目 RealSense 相机画面（Front 视角 + Wrist 腕部视角），自动分切并保存为标准 LeRobot / Parquet 数据集格式。

---

### 4. 机械臂控制与安全服务 (Robot Controller & Safety Guard)

> **核心实现**：`src/franka_teleop/sync_franka.py` / `src/franka_teleop/closed_loop_franka.py`

* **Linux 控制端启动同步服务端**
  ```bash
  # 必须在原生 ROS 环境中执行 (Linux 控制主机 10.197.16.43)
  source /opt/ros/noetic/setup.bash
  cd /home/ssui/project/embodied_midterm/controller_lerobot
  python3 sync_franka.py --z-min 0.0070
  ```
  * **安全特性**：硬编码桌面安全底面（$Z \ge +7.0\text{ mm}$ 防撞保护）、末端姿态垂直对齐、关节速度限幅与急停刹车平滑过滤。

---

## 📁 项目关键目录架构

```text
franka-vla/
├── src/                                  # 核心源码
│   ├── franka_teleop/                    # 机械臂驱动、RTC闭环服务、Pi0.5引擎、正逆运动学
│   ├── obstacle_avoidance/               # 3D包络感知、Franka胶囊碰撞体、实时安全滤波
│   └── vlm_planning/                     # 大模型任务规划与目标空间映射
├── scripts/                              # 用户操作统一入口层
│   ├── run_sync_agent.bat                # 【实机推理】同步模式首选入口
│   ├── run_async_agent.bat               # 【实机推理】异步RTC流动入口
│   ├── launch_server.bat                 # 【推理服务】常驻后台推理服务
│   ├── train_pure_flow_50k_gpu1.bat      # 【模型训练】50k步Pure Flow训练入口
│   ├── train_crop169_20k_gpu1.bat        # 【模型训练】20k步裁剪微调入口
│   ├── launch_record.bat                 # 【数据采集】遥操作数据同步录制
│   └── python/                           # 底层 Python 实现脚本
├── tests/                                # 单元测试与标定工具
├── outputs/                              # 训练权重与日志 (已加入 .gitignore)
├── dataset/                              # 轨迹数据集 (已加入 .gitignore)
└── quick_start/                          # 分步部署详细教程
```
