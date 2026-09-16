# 教程 01：前置自检与离线测试 (01_offline_sanity_checks.md)

本教程指导在正式连接机械臂前，如何在 Windows GPU 服务器上完成资产自检、安全门禁单元测试、数据集完整性校验以及离线影子运行。

---

## 1. 验证模型资产完整性

检查预训练模型权重、分词器与统计量文件是否在 `checkpoints/` 目录下就绪：

```cmd
cd /d C:\Users\74727\Desktop\project\VLA_franka
C:\Users\74727\miniconda3\envs\lerobot\python.exe tests\check_assets.py
```

- **预期输出**：
  ```text
  [PASS] pi05_droid_jointpos   : Found (10 files)
  [PASS] Qwen3.5-9B            : Found (17 files)
  [PASS] RoboBrain2.5-8B-NV    : Found (16 files)
  [ALL PASS] All required model checkpoints and assets are present.
  ```

---

## 2. 运行安全门禁单元测试

验证 Franka 7 轴软限位、防急跳跳变限幅以及动作生命周期（TTL）超时拦截逻辑：

```cmd
cd /d C:\Users\74727\Desktop\project\VLA_franka
C:\Users\74727\miniconda3\envs\lerobot\python.exe tests\check_safety_guards.py
```

- **预期输出**：
  ```text
  Ran 3 tests in 0.001s
  OK
  ```

---

## 3. 验证本地数据集 LeRobot 官方兼容性

自动校验 `dataset/` 目录下所有数据集的分块视频文件、Parquet 元数据及 `meta/info.json`：

```cmd
cd /d C:\Users\74727\Desktop\project\VLA_franka
C:\Users\74727\miniconda3\envs\lerobot\python.exe tests\check_datasets.py
```

- **预期输出**：
  ```text
  ============================================================
    LeRobot Dataset Compliance Verification
  ============================================================
  --> Validating teleop_pick_cube_15hz_001...
      Episodes : 28
      Frames   : 4301 (FPS: 15)
      Features : ['observation.state', 'action', 'observation.images.front', 'observation.images.wrist', ...]
  [PASS] teleop_pick_cube_15hz_001: 100% LeRobotDataset compliant.

  --> Validating teleop_pick_vegetables_15hz_001...
      Episodes : 27
      Frames   : 4471 (FPS: 15)
      Features : ['observation.state', 'action', 'observation.images.front', 'observation.images.wrist', ...]
  [PASS] teleop_pick_vegetables_15hz_001: 100% LeRobotDataset compliant.
  ------------------------------------------------------------
  [ALL PASS] All checked datasets are 100% valid and ready for training.
  ```

---

## 4. 运行 GPU 1 单卡隔离影子推理自检 (Shadow Run)

使用本地保存的真实双相机测试帧与 Franka 初始位姿，执行离线推理测试。  
**核心特性**：严格限制在物理 GPU 1，物理 GPU 0 绝对零占用，仅推理不下发实机动作。

```cmd
cd /d C:\Users\74727\Desktop\project\VLA_franka
scripts\launch_shadow_run.bat
```

- **关键指标核验**：
  - `CUDA_VISIBLE_DEVICES = 1`
  - 4 项基准任务推理通过（`pick up the purple onion`, `place into the brown basket` 等）
  - 平均推理时延 $\approx 310$ ms（15 步 action chunk 耗时，远小于物理动作周期 500ms）
  - 实时因子 RTF $< 1.0$（实测 $\approx 0.62$）
  - 数值安全：**100% 有限值，0 NaN / 0 Inf**
  - 结果自动输出至 `outputs\m3_shadow_run.json`
