# 教程 04：模型训练与微调 (04_model_training.md)

本教程指导如何使用工程内的微调训练套件，基于录制的真实遥操作数据集对 Pi0.5 4.14B VLA 机械臂模型进行 100% 原生纯净流匹配（Pure Flow Matching）LoRA 微调。

---

## 1. 训练核心原则与规范

1. **GPU 1 物理专卡运行**：
   - 训练脚本强制通过 `CUDA_VISIBLE_DEVICES=1` 绑定物理卡 1（RTX 5090 32GB）运行。
   - **严格禁止触碰 GPU 0**（物理卡 0 专供其他 PINN / 仿真任务运行）。
2. **100% 原生纯净关节流匹配（Pure Flow Matching）**：
   - 严格遵循 OpenPI 核心动力学损失，直接监督 15 步未来相对关节增量 $a[k] = q[t+k+1] - q[t]$ 与夹爪状态 $g \in [0, 1]$。
   - **摒弃人为状态加噪**（State Noise 设为 0.0 rad），防止破坏 256-bin 分词器与造成运动学真值冲突。
   - **摒弃伪逆解笛卡尔损失**，由神经网络端到端学习平滑、连续的 7 轴机械臂本体动力学分布。
3. **严格 Episode 级隔离切分（Zero Leakage）**：
   - 按整条轨迹（Episode）严格隔离训练集与验证集（默认 85% 训练 / 15% 验证），绝不在时间维度截断造成数据泄露。
4. **W&B 全生命周期云端同步**：
   - 训练 Loss、学习率衰退、每步时延（~885 ms/step）及验证集指标自动在线实时同步至 Weights & Biases 看板。

---

## 2. 一键启动 50,000 步正式训练

在 Windows GPU 服务器上进入工程根目录：

```cmd
cd /d C:\Users\74727\Desktop\project\VLA_franka
```

### 推荐首选：一键批处理启动
```cmd
.\scripts\train_pure_flow_50k_gpu1.bat
```
- **显存与硬件占用**：峰值显存约 `23.2 GB / 32.6 GB`（占用率 ~72.9%），GPU 算力利用率 90%~95%。
- **预计耗时**：全量 50k 步约 11.5 小时（吞吐率约 1.13 steps/s）。
- **快照保存**：每 5,000 步自动保存一次完整检查点（包含 LoRA 矩阵、动作投影层与优化器状态）至 `outputs\checkpoints\pi05_lora_pure_flow_50k\`。
- **自动验证**：每 2,500 步在 5 个未见测试轨迹上自动执行真实的 10 步 Euler ODE 动力学积分评测。

### 快捷预检模式（Preflight Benchmark）
在正式训练前，若只需快速验证模型加载、断点权重对接与 5 步试跑工况：
```cmd
.\scripts\train_pure_flow_50k_gpu1.bat --preflight-only
```

---

## 3. 自定义参数训练 (底层 Python 引擎)

底层核心位于 `scripts\python\train_pi05_lora.py`，支持高度灵活的自定义调整：

```powershell
C:\Users\74727\miniconda3\envs\lerobot\python.exe scripts\python\train_pi05_lora.py `
  --dataset dataset\teleop_pick_cube_15hz_002 `
  --output-dir outputs\checkpoints\my_custom_pure_flow `
  --steps 50000 `
  --batch-size 4 `
  --grad-accum 2 `
  --lr 1e-4 `
  --warmup-steps 200 `
  --lora-dropout 0.05 `
  --save-freq 5000 `
  --eval-freq 2500 `
  --val-ratio 0.15 `
  --wandb `
  --wandb-project pi05-franka-vla `
  --wandb-name my_custom_pure_flow_run
```

### 常用核心参数说明：

| 参数项 | 默认值 | 作用说明 |
| :--- | :---: | :--- |
| `--dataset` | - | 训练数据集路径（可传入一个或多个目录） |
| `--output-dir` | - | 权重检查点保存目录 |
| `--resume` | - | 从现有 Checkpoint (.pt) 恢复断点续训（自动延续优化器与学习率曲线） |
| `--steps` | `50000` | 训练总迭代步数 |
| `--batch-size` | `4` | 单步 Batch 大小（配合 grad-accum=2 等效 Batch=8） |
| `--lr` | `1e-4` | 学习率（余弦衰退调度至 1e-6） |
| `--lora-dropout` | `0.05` | LoRA 矩阵 Dropout 概率（提升小样本泛化性） |
| `--save-freq` | `5000` | Checkpoint 检查点落盘保存周期（步） |
| `--eval-freq` | `2500` | 独立测试集 10-step Euler ODE 泛化评测周期（步） |
| `--lang-rank` | `16` | PaliGemma-2B 视觉语言骨干 LoRA Rank |
| `--expert-rank` | `32` | Gemma-300M 动作专员 LoRA Rank |
| `--wandb` | - | 开启 Weights & Biases 实时云端看板追踪 |

---

## 4. 查询已保存的权重清单

训练过程中或完成后，运行一键查询工具即可列出所有 Checkpoint 的步数、参数量与文件大小：

```cmd
.\scripts\list_ckpts.bat
```
- `latest.pt` 始终自动指向最新保存的最佳/终态检查点。
