# VLA_franka: Franka Panda 具身智能 VLA 部署工程

本项目采用与 Hugging Face 官方 **LeRobot** 并列的模块化架构，实现 Franka Panda 机械臂的高效数据采集、安全自检、VLM 规划与 VLA 闭环部署。

## 目录结构

- **`lerobot/`**：官方原生核心库，提供 Policy（PI0, ACT, Diffusion, SmolVLA 等）与标准数据集。
- **`franka_teleop/`**：机械臂驱动、双相机流与 LeRobot 数据采集。
- **`vlm_planning/`**：RoboBrain / Qwen 任务规划与空间坐标映射。
- **`scripts/`**：快捷命令入口（绑定 GPU 1 与 Conda 环境）。
- **`tests/`**：安全门禁、相机检测与 GPU 1 影子运行自检。
- **`docs/`**：详尽接口文档（`docs/SCRIPTS_AND_INTERFACES.md`）。
- **`checkpoints/`**：预训练模型权重。
- **`dataset/`**：LeRobot 格式轨迹数据集。

## 快速使用与实机运行

详细的多终端（Franka 控制机 + Windows GPU 录制端/推理端）端到端实机操作手册请参阅：  
👉 **[快速使用与实机运行指南 (QUICK_START.md)](QUICK_START.md)**

- **数据采集**：涵盖 ROS 阻抗控制、Touch 遥操作手柄控制以及 LeRobot 双相机流同步录制规范。
- **实机部署**：涵盖 1kHz 关节速度底层、闭环执行服务与 Windows VLA 推理 Agent。
- **自检测试**：资产检查与 GPU 1 影子运行自检。

各自研模块、类接口与 Python API 说明请参阅：  
👉 **[自研脚本与接口规格说明书 (docs/SCRIPTS_AND_INTERFACES.md)](docs/SCRIPTS_AND_INTERFACES.md)**
