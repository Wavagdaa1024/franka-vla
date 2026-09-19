# 教程 01：前置自检与离线测试 (01_offline_sanity_checks.md)

本教程指导在正式连接机械臂前，如何在 Windows GPU 服务器上完成模型资产自检、安全门禁单测、核心运动学单测以及离线影子运行。

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
  Ran 3 tests in 0.000s
  OK
  ```

---

## 3. 运行运动学、零空间自稳与 RTC 闭环全套单测 (10 项核心验证)

验证解析正向运动学、解析雅可比矩阵、零空间姿态纠偏、桌面硬地面防碰撞、15Hz 速度投影滤波与实时分块时间戳等：

```cmd
cd /d C:\Users\74727\Desktop\project\VLA_franka
C:\Users\74727\miniconda3\envs\lerobot\python.exe tests\test_kinematics_and_rtc.py
```

- **预期输出**：
  ```text
  [Test 1/6] FK Accuracy vs Franka Hardware Truth: Error < 0.0001 mm [PASS]
  [Test 2/6] Analytical Jacobian vs Numerical Differentiation [PASS]
  [Test 3/6] Task-Priority Nullspace Gripper Orientation Stabilization [PASS]
  [Test 4/6] Table Floor Collision Prevention (Z >= +7.0 mm) [PASS]
  [Test 5/6] 15Hz/1kHz Real-Time Velocity Projection Floor Guard [PASS]
  [Test 6/6] RTC Receding Horizon Blending Continuity [PASS]
  [Test 7/9] Strict Vertical Downward 5-DOF IK Lock [PASS]
  [Test 8/9] TCP Packet Framing & Fragmentation Resilience [PASS]
  [Test 9/9] RTC Timing, Handover, and Delayed Chunk Resilience [PASS]
  [Test 10/10] Real-Time Closed-Loop Nullspace Orientation Feedback [PASS]
  ALL 10 TESTS PASSED 100% PERFECTLY!
  ```

---

## 4. 运行可微正向运动学 (DFK) 梯度检查

验证 DFK 梯度求导与反向传播是否在 GPU 1 上正确执行（用于支持笛卡尔空间位姿损失训练）：

```cmd
cd /d C:\Users\74727\Desktop\project\VLA_franka
C:\Users\74727\miniconda3\envs\lerobot\python.exe tests\test_differentiable_fk.py
```

- **预期输出**：
  ```text
  [*] GPU CUDA Forward & Backward Pass verified on NVIDIA GeForce RTX 5090
  [SUCCESS] Franka DFK passed all consistency and gradient checks 100%!
  ```

---

## 5. 双摄硬件快速体检与实时查看

- **快速体检（采样 15 帧核验掉帧与序列号）**：
  ```cmd
  cd /d C:\Users\74727\Desktop\project\VLA_franka
  scripts\check_cameras.bat
  ```
  *(如需弹出原生 OpenCV 窗口查看：`scripts\check_cameras.bat --gui`)*

- **持续实时监控（桌面 30 FPS 窗口 + 局域网 Web 推流）**：
  ```cmd
  scripts\live_camera.bat
  ```
  *(浏览器打开 `http://localhost:8080` 即可实时查看头部与腕部双摄)*

---

## 6. 运行 GPU 1 单卡隔离影子推理自检 (Shadow Run)

使用本地保存的双相机基准图与 Franka 初始位姿，执行全流程影子推理测试（仅模型推理，不下发物理动作）：

```cmd
cd /d C:\Users\74727\Desktop\project\VLA_franka
C:\Users\74727\miniconda3\envs\lerobot\python.exe tests\shadow_run_pi05.py
```

- **关键指标核验**：
  - `CUDA_VISIBLE_DEVICES = 1`（物理 GPU 0 绝对零占用）
  - 4 项基准任务推理全部通过（`pick up the purple onion`, `place into the brown basket` 等）
  - 平均推理时延 $\approx 290$ ms（15 步 action chunk 耗时，远小于物理动作周期 500ms）
  - 实时因子 RTF $< 1.0$（实测 $\approx 0.58$）
  - 数值安全：**100% 有限值，0 NaN / 0 Inf**
  - 结果自动保存至 `outputs\m3_shadow_run.json`

---

## 7. 运行 Agent 管道离线 Mock 自检

通过推理主入口验证权重加载与多任务提示词解析（不连接物理机械臂与摄像头）：

```cmd
cd /d C:\Users\74727\Desktop\project\VLA_franka
scripts\run_agent.bat --mock --checkpoint cartesian_1000
```
