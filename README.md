# VLA_franka: Franka Panda 具身智能 VLA 部署工程

本项目采用与 Hugging Face 官方 **LeRobot** 并列的模块化架构，实现 Franka Panda 机械臂的高效数据采集、安全自检、VLM 规划与 VLA 闭环部署。

## 目录结构

- **`lerobot/`**：官方原生核心库，提供 Policy（PI0, ACT, Diffusion, SmolVLA 等）与标准数据集。
- **`franka_teleop/`**：机械臂驱动、双相机流与 LeRobot 数据采集。
- **`sanity_checks/`**：安全门禁、相机检测与 GPU 1 影子运行自检。
- **`vlm_planning/`**：RoboBrain / Qwen 任务规划与空间坐标映射。
- **`scripts/`**：快捷命令入口（绑定 GPU 1 与 Conda 环境）。
- **`docs/`**：详尽接口文档（`docs/SCRIPTS_AND_INTERFACES.md`）。
- **`checkpoints/`**：预训练模型权重。
- **`dataset/`**：LeRobot 格式轨迹数据集。

## 快速使用

### 1. 验证模型资产与安全门禁
```bash
python sanity_checks/check_assets.py
python sanity_checks/check_safety_guards.py
```

### 2. 离线影子运行 (GPU 1)
```bash
scripts\\launch_shadow_run.bat
```

### 3. 数据采集 (LeRobot 格式)
```bash
scripts\\launch_record.bat --repo_id teleop_pick_cube_15hz_001
```

详见 [docs/SCRIPTS_AND_INTERFACES.md](docs/SCRIPTS_AND_INTERFACES.md)。
