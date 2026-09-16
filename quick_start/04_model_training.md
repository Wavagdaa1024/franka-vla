# 教程 04：LeRobot 官方模型训练与微调 (04_model_training.md)

本教程指导如何使用官方原生 `lerobot` 训练脚本，基于本地录制的数据集对 Policy 进行微调训练。

---

## 1. 训练核心原则

1. **全面对齐官方生态**：直接调用 `lerobot.scripts.train`，使用标准 Hydra / Draccus 配置，不手写训练循环。
2. **GPU 1 物理隔离**：严格使用物理卡 1，避免干扰服务器 GPU 0。
3. **数据集对齐**：输入数据集必须由 `franka_teleop/record_teleop.py` 写入，符合 LeRobotDataset 规范。

---

## 2. 启动训练

在 Windows GPU 服务器上，运行：

### 方式 A（一键批处理）：
```cmd
cd /d C:\Users\74727\Desktop\project\VLA_franka
scripts\run_lerobot_train.bat dataset_repo_id=local/franka_red_cube policy=act
```

### 方式 B（完整命令行）：
```cmd
cd /d C:\Users\74727\Desktop\project\VLA_franka\lerobot

set CUDA_DEVICE_ORDER=PCI_BUS_ID
set CUDA_VISIBLE_DEVICES=1

C:\Users\74727\miniconda3\envs\lerobot\python.exe -m lerobot.scripts.train ^
  --dataset.repo_id="C:\Users\74727\Desktop\project\VLA_franka\dataset\teleop_pick_cube_15hz_001" ^
  --policy.type=pi0 ^
  --output_dir="outputs\train_pi0_run1" ^
  --batch_size=8 ^
  --steps=5000
```

---

## 3. 常见参数说明

| 参数项 | 说明 | 示例 |
|---|---|---|
| `--dataset.repo_id` | 本地数据集路径或 HuggingFace Hub repo id | `dataset/teleop_pick_cube_15hz_001` |
| `--policy.type` | 策略架构类型（`pi0`, `smolvla`, `act`, `diffusion`） | `pi0` |
| `--batch_size` | 训练批次大小（RTX 5090 推荐 8~16） | `8` |
| `--steps` | 总训练迭代步数 | `5000` |
| `--output_dir` | 权重检查点保存位置 | `outputs/checkpoints/...` |
| `--save_freq` | 检查点保存频率（步） | `500` |
| `--wandb.enable` | 是否开启 W&B 实验在线追踪 | `true` |
