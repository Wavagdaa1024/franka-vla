# VLA_franka: Franka Panda 具身智能 VLA 部署工程

本项目采用与 Hugging Face 官方 **LeRobot** 并列的模块化架构，实现 Franka Panda 机械臂的高效数据采集、安全自检、VLM 规划与 VLA 闭环部署。

## 目录结构

- **`lerobot/`**：官方原生核心库，提供 Policy（PI0, ACT, Diffusion, SmolVLA 等）与标准数据集。
- **`franka_teleop/`**：机械臂驱动、双相机流与 LeRobot 数据采集。
- **`vlm_planning/`**：RoboBrain / Qwen 任务规划与空间坐标映射。
- **`scripts/`**：快捷命令入口（绑定 GPU 1 与 Conda 环境）。
- **`tests/`**：安全门禁、相机检测与 GPU 1 影子运行自检。
- **`quick_start/`**：分步教程手册（自检、数据录制、实机部署、模型训练）。
- **`docs/`**：详尽接口文档（`docs/SCRIPTS_AND_INTERFACES.md`）。
- **`checkpoints/`**：预训练模型权重。
- **`dataset/`**：LeRobot 格式轨迹数据集。

## 快速使用分步教程

详细的分步教程请参阅 **[quick_start/ 教程目录](quick_start/)**：

1. **[01. 离线环境与安全自检 (quick_start/01_offline_sanity_checks.md)](quick_start/01_offline_sanity_checks.md)**：模型权重自检、安全门禁测试、GPU 1 离线影子推理。
2. **[02. 遥操作数据采集全流程 (quick_start/02_teleop_data_collection.md)](quick_start/02_teleop_data_collection.md)**：Franka ROS + Touch 手柄 + Windows LeRobot 数据集同步录制。
3. **[03. 实机闭环 VLA 部署四步法 (quick_start/03_live_vla_deployment.md)](quick_start/03_live_vla_deployment.md)**：1kHz 关节速度底层、闭环服务端、Windows Agent 推理与急停安全启动。
4. **[04. LeRobot 官方模型训练与微调 (quick_start/04_model_training.md)](quick_start/04_model_training.md)**：调用官方 `train.py` 进行 Policy 微调。

各自研模块、类接口与 Python API 说明请参阅：  
👉 **[自研脚本与接口规格说明书 (docs/SCRIPTS_AND_INTERFACES.md)](docs/SCRIPTS_AND_INTERFACES.md)**
