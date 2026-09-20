# 教程 01：前置自检与离线测试 (01_offline_sanity_checks.md)

本教程指导在正式连接机械臂前，如何在 Windows GPU 服务器上完成模型资产自检、安全门禁单测、核心运动学单测、模型离线轨迹对比以及影子运行。

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

验证 DFK 梯度求导与反向传播是否在 GPU 1 上正确执行：

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

## 5. 双摄实时查看与硬件体检 (Live Camera & Check)

### 5.1 双摄持续实时监控 (`.\scripts\live_camera.bat`)

在工程根目录下运行：
```powershell
cd C:\Users\74727\Desktop\project\VLA_franka
.\scripts\live_camera.bat
```
- **Windows 桌面窗口**：左右并排显示 Front 头部相机与 Wrist 腕部相机，30 FPS 刷新；
- **局域网 / 浏览器实时流**：在浏览器打开 [http://localhost:8080](http://localhost:8080) 或 `http://<本机IP>:8080`。

### 5.2 硬件快速抽样体检 (`.\scripts\check_cameras.bat`)

快速捕获 15 帧排查连通性与丢包率：
```powershell
.\scripts\check_cameras.bat
```

---

## 6. 运行 GPU 1 离线真实轨迹对比评测 (Ground Truth vs Pred)

直接与人工真实遥操作数据集中的真值动作分块（15 步未来关节增量与夹爪）进行逐帧对比，输出关节 MAE、RMSE、动作幅值比与夹爪开合准确率：

```cmd
cd /d C:\Users\74727\Desktop\project\VLA_franka
C:\Users\74727\miniconda3\envs\lerobot\python.exe tests\eval_offline_jointpos.py --checkpoint pure_flow
```

- **核心评估指标**：
  - **Joint MAE**：$< 0.020\text{ rad} \ (\approx 1.1^\circ)$，跟踪性能优秀。
  - **Gripper Match**：$\ge 95\%$，开合意图完全对齐。
  - **Scale Ratio**：$0.5\text{x} \sim 2.0\text{x}$，无预测塌缩或过度放大。

---

## 7. 运行 Agent 管道离线 Mock 自检

在完全不接物理机械臂与相机的情况下，一键验证 Agent 主脚本的模型加载、LoRA 挂载、前向推理及算力时延：

```cmd
cd /d C:\Users\74727\Desktop\project\VLA_franka
.\scripts\run_agent.bat --mock
```

- **预期输出**：
  ```text
  [Model OK] LoRA Multi-Task Adapters loaded! (Step: 50000)
  [Mock Pass 1/5] Latency: 748.2ms | Step 1 dq[0..2]: [...]
  [Mock Pass 2/5] Latency: 353.1ms | Step 1 dq[0..2]: [...]
  [MOCK TEST PASSED] Steady-state Latency: ~353ms (~2.8 FPS)
  * Model is fully operational and ready for live robot deployment!
  ```
