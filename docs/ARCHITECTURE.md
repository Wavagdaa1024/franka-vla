# VLA_franka 架构说明 (ARCHITECTURE.md)

## 整体架构蓝图

```text
+-------------------------------------------------------------------------+
|                              VLA_franka                                 |
+-------------------------------------------------------------------------+
        |                                                 |
        v                                                 v
+-------------------+                           +-------------------------+
|     lerobot/      | <--- (Direct Clone)       |      自研业务代码       |
| (Official Main)   |                           |                         |
|                   |                           | +---------------------+ |
| - policies/       |                           | |    franka_teleop    | |
|   * pi0, pi05     |                           | |  - closed_loop_arm  | |
|   * smolvla       |                           | |  - record_teleop    | |
|   * act, diff     |                           | |  - live_guards      | |
| - datasets/       | <=======================  | |  - realsense_stream | |
|   * LeRobotDataset| (Standard Data Contract)  | +---------------------+ |
| - scripts/        |                           | |    tests    | |
|   * train.py      |                           | |  - shadow_run_gpu1  | |
|   * eval.py       |                           | |  - check_cameras    | |
+-------------------+                           | +---------------------+ |
                                                | |    vlm_planning     | |
                                                | |  - qwen / robobrain | |
                                                | |  - grounding / proj | |
                                                | +---------------------+ |
                                                +-------------------------+
```

## 设计准则
1. **与生态对齐**：使用官方 Policy、标准 LeRobotDataset，不维护私有训练循环。
2. **硬件解耦**：所有相机驱动、Socket 协议和安全保护集中在 `franka_teleop/`。
3. **安全第一**：物理 GPU 0 绝对隔离；实机运动必须经过 `live_guards` 限幅。
