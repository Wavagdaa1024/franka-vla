# VLA_franka 脚本与接口规格说明书 (SCRIPTS_AND_INTERFACES.md)

本文档详细记录所有自研脚本、模块接口、命令行参数及与官方 LeRobot 架构的集成关系。

---

## 一、架构总览与 LeRobot 配合模式

本工程遵循**与 LeRobot 官方库解耦并列**的架构原则：
1. **`lerobot/`**：官方克隆源码，负责 Policy 模型定义（PI0, ACT, Diffusion, SmolVLA 等）、LeRobotDataset 标准格式、官方训练循环与评测器。
2. **`franka_teleop/`**：机器人与传感器交互层，负责将真实世界（RealSense 相机、Franka 机械臂遥测）与 LeRobotDataset 或 Policy.select_action 桥接。
3. **`tests/`**：安全防护与自检套件，确保实机动作前资产完备、相机畅通、离线数值合规（强制单卡 GPU 1 隔离）。
4. **`vlm_planning/`**：任务规划层，负责高层任务拆解与物体定位，为底层 VLA 提供任务文本和目标路点。

---

## 二、自研核心模块与接口

### 1. `franka_teleop` 模块

#### (1) `action_semantics.py`
- **功能**：将机械臂现场采集的关节角度及夹爪位置，转换为 DROID 8 维标准动作格式。
- **核心接口**：
  - `droid_joint_velocity_action(current: dict, following: dict, max_dt_s=0.5) -> np.ndarray`:
    - 输入：前后两个连续帧采样样本字典（含 `q`, `target_gripper`, `_gpu_receive_monotonic_ns`）。
    - 输出：`(8,)` float32 数组，前 7 维为关节角速度 $\Delta q / \Delta t$（rad/s），第 8 维为夹爪位置（0.0 开，1.0 闭）。
  - `droid_joint_delta_action(current: dict, following: dict, max_dt_s=0.5) -> np.ndarray`:
    - 输入：前后两个连续帧采样样本字典。
    - 输出：`(8,)` float32 数组，前 7 维为关节角度位移 $\Delta q = q_{t+1} - q_t$（rad），第 8 维为夹爪目标位置。

#### (2) `live_guards.py`
- **功能**：实机动作安全门禁与软限位保护。
- **核心接口**：
  - `validate_joint_velocity_chunk(velocities, gripper, max_steps=64) -> (v, g)`:
    - 校验动作块 shape `(N, 7)`，拦截 NaN/Inf，并将夹爪值截断限制在 `[0, 1]`。
  - `validate_joint_position_chunk(positions, gripper, current_q=None, max_step_delta=0.25) -> (q, g)`:
    - 针对 Franka 7 个自由度施加绝对软限位（`FRANKA_JOINT_LIMITS`），并防止步间急跳（单步 delta 超过 0.25 rad 自动限幅）。
  - `action_is_fresh(received_monotonic, now, ttl_s) -> bool`:
    - 动作存活生命周期校验（TTL），防范通信卡顿导致执行过期动作。

#### (3) `realsense_service.py`
- **功能**：前视（base）与腕部（wrist）双目 RealSense 相机硬件服务与缓冲器。
- **核心类**：
  - `DualRealSense(front_serial, wrist_serial, width=640, height=480, fps=30)`:
    - `get_frames() -> (front_img, wrist_img)`: 返回当前同步的 RGB 图像（numpy uint8 `[H, W, 3]`）。
    - `stop()`: 安全释放硬件管线。

#### (4) `record_teleop.py`
- **功能**：接收遥操作控制指令与双相机画面，写入官方 `LeRobotDataset v2` 格式数据集。
- **CLI 参数**：
  - `--repo_id`: 数据集标识名（保存在 `dataset/<repo_id>`）。
  - `--fps`: 采样频率（默认 15 或 30）。
  - `--host`, `--port`: 连接遥操作推流端的 IP 和端口。
  - `--resume`: 增量续录已有数据集。

#### (5) `closed_loop_franka.py`
- **功能**：实机闭环执行服务端，通过 TCP Socket 与 Franka 机械臂控制器长连接通信。
- **CLI 参数**：
  - `--port`: 监听端口（默认 8765）。
  - `--sync` / `--async-mode`: 同步走停模式（Stop-and-Go）或异步双缓冲连续执行模式。
  - `--max-vel`: 关节角速度上限（默认 0.35 rad/s）。
  - `--z-min`: 桌面碰撞保护软地面高度（默认 0.055m）。
  - `--shadow`: 影子模式（只打印校验指令，不下发物理速度）。

---

### 2. `tests` 模块

#### (1) `check_assets.py`
- **功能**：自检所有预训练模型、分词器、统计量资产是否就绪。
- **运行**：`python tests/check_assets.py`。

#### (2) `check_safety_guards.py`
- **功能**：安全门禁单元测试，验证限位、限幅和超时拦截逻辑。
- **运行**：`python tests/check_safety_guards.py`。

#### (3) `check_cameras.py`
- **功能**：检测 RealSense 双相机连通性、帧率与图像数据形状。
- **运行**：`python tests/check_cameras.py`。

#### (4) `shadow_run_pi05.py`
- **功能**：单卡物理 GPU 1 隔离离线推理影子运行，校验策略推理时延与动作输出。
- **关键约束**：强制设置 `CUDA_VISIBLE_DEVICES=1`，物理 GPU 0 完全不占用。

---

### 3. `vlm_planning` 模块

#### (1) `vlm_planner.py`
- **功能**：基于 RoboBrain2.5 或 Qwen3.5 的高层任务规划器，输出规范结构化 JSON。
- **核心接口**：
  - `plan_with_vlm(image, instruction, model_name, gpu_id=1, ...) -> (plan_dict, raw_text, is_valid)`:
    - 图像与自然语言指令输入，输出包含 `steps: [{"action": "pick", "object": "..."}, ...]` 的合法方案。

#### (2) `grounding_detector.py`
- **功能**：视觉定位与目标边界框/中心像素检测。

#### (3) `coordinate_projector.py`
- **功能**：像素坐标经仿射变换映射为 Franka 机械臂基座坐标系下的 Cartesian 路点。

---

## 三、快捷入口脚本与 LeRobot 训练

所有批处理位于 `scripts/` 目录下：
1. **`launch_shadow_run.bat`**：启动离线推理影子自检。
2. **`launch_record.bat`**：启动遥操作数据采集。
3. **`launch_live_agent.bat`**：启动闭环实机代理。
4. **`run_lerobot_train.bat`**：直接调用官方 `lerobot.scripts.train` 启动模型微调。
   - 示例：`scripts\run_lerobot_train.bat policy=pi0 dataset_repo_id=teleop_pick_cube_15hz_001`
