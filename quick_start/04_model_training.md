# 教程 04：模型训练与微调 (04_model_training.md)

本教程指导如何使用工程内的微调训练套件，基于录制的 LeRobot 格式轨迹对 Pi0.5 VLA 进行高效 LoRA 微调。

---

## 1. 训练核心原则

1. **GPU 0 物理隔离**：训练脚本通过 `CUDA_VISIBLE_DEVICES=0` 绑定物理卡 0 独立运行，不占用 GPU 1（保留给实机实时推理使用）。
2. **DFK 笛卡尔损失 + 倾角约束**：通过可微正向运动学（DFK），在训练阶段直接对末端执行器 3D 位姿和垂直向下方向施加物理几何损失，消除末端滑移与偏斜。
3. **数据集对齐**：使用 `scripts\launch_record.bat` 录制的标准 LeRobotDataset 轨迹数据集（如 `dataset\teleop_pick_cube_15hz_001`、`002`）。

---

## 2. 启动训练

在 Windows GPU 服务器上进入工程根目录：

```cmd
cd /d C:\Users\74727\Desktop\project\VLA_franka
```

### 推荐首选：DFK 笛卡尔空间损失微调 (带垂直向下约束)
训练出的模型末端姿态极其稳定（倾角误差 $< 0.5^\circ$）：
```cmd
scripts\train_cartesian_lora_gpu0.bat
```

### 备用方式：纯关节空间损失微调
```cmd
scripts\train_red_cube_lora_gpu0.bat
```

---

## 3. 自定义参数训练 (底层 Python 引擎)

底层 Python 脚本位于 `scripts\python\train_pi05_lora.py`，支持高度灵活的自定义传参：

```powershell
C:\Users\74727\miniconda3\envs\lerobot\python.exe scripts\python\train_pi05_lora.py `
  --dataset dataset\teleop_pick_cube_15hz_001 dataset\teleop_pick_cube_15hz_002 `
  --task-filter "red cube" `
  --output-dir outputs\checkpoints\my_custom_lora `
  --cartesian-loss-weight 5.0 `
  --vertical-loss-weight 2.0 `
  --steps 3000 `
  --lr 1e-4 `
  --save-freq 500 `
  --eval-freq 250
```

### 常用核心参数说明：

| 参数项 | 默认值 | 作用说明 |
| :--- | :---: | :--- |
| `--dataset` | - | 训练数据集路径列表（可同时传入多个数据集目录） |
| `--output-dir` | - | 权重检查点保存目录 |
| `--cartesian-loss-weight` | `5.0` | DFK 末端执行器 3D 坐标 MSE 损失权重 |
| `--vertical-loss-weight` | `2.0` | 末端执行器工具 Z 轴垂直向下 `[0, 0, -1]` 约束权重 |
| `--steps` | `2000` | 训练总迭代步数 |
| `--save-freq` | `500` | Checkpoint 检查点保存频率（步） |
| `--eval-freq` | `250` | 离线验证评估频率（步） |
| `--lang-rank` | `16` | PaliGemma-2B 语言视觉主干 LoRA Rank |
| `--expert-rank` | `32` | Gemma-300M Action Expert LoRA Rank |

---

## 4. 查询已保存的权重清单

训练完成后，使用一键工具查询所有产出权重文件的步数、Loss 与参数配置：

```cmd
scripts\list_ckpts.bat
```
