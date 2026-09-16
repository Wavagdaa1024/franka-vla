# VLA_franka: Franka Panda 具身智能 VLA 部署工程

本项目采用与 Hugging Face 官方 **LeRobot** 并列的模块化架构，实现 Franka Panda 机械臂的高效数据采集、安全自检、VLM 规划与 VLA 闭环部署。

## 目录结构

VLA_franka/
├── lerobot/          # 【官方核心库】直接 Clone / Submodule
│
├── franka_teleop/    # 【自己的一级业务】Franka 机械臂驱动、双相机流、遥操作录制与闭环
├── vlm_planning/     # 【自己的一级业务】RoboBrain / Qwen 任务规划与空间坐标标定
│
├── scripts/          # 【日常运行入口】一键录制、一键训练、一键实机启动
├── tests/            # 【自检与测试】移到这里（原 sanity_checks：相机检查、限位测试、影子运行）
├── docs/             # 【接口与架构文档】
│
├── dataset/          # 【数据集】
└── checkpoints/      # 【模型权重】

## 快速使用

### 1. 验证模型资产与安全门禁
```bash
python tests/check_assets.py
python tests/check_safety_guards.py
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
