# 教程 03：实机闭环 VLA 部署四步法 (03_live_vla_deployment.md)

本教程指导如何将训练好的 VLA 模型（50,000 步纯净流匹配 Pi0.5 LoRA 模型）接入实机 Franka 机械臂，进行闭环自主推理与抓取控制。

为满足不同实验场景需求，系统已将**同步推理 (Stop-and-Go)** 与 **异步推理 (RTC 连续流)** 彻底解耦为两个独立程序，分别拥有独立的运行入口与参数配置。

---

## 模式选择指南

| 特性维度 | 方案 A：同步推理 (Stop-and-Go) 【推荐首选】 | 方案 B：异步推理 (RTC 连续流) |
| :--- | :--- | :--- |
| **运行脚本 (Windows)** | `scripts\run_sync_agent.bat` | `scripts\run_async_agent.bat` |
| **服务程序 (Linux)** | `src/franka_teleop/sync_franka.py` | `src/franka_teleop/closed_loop_franka.py --rtc` |
| **动作衔接** | 执行完 15 步后平滑停稳 50ms 再采样 | 边走边预取下一动作分块 |
| **控制权冲突** | **零冲突**（起步位置与当前真实位置 100% 吻合） | 若时延预估偏差易产生反向拉扯 |
| **视觉清晰度** | **零运动模糊**（静止采样，抓取定位极准） | 高速运动时可能存在局部模糊 |
| **推荐适用场景** | **精细抓取、防打架调试、稳定可靠部署** | **大范围自由流动探索、高帧率连续推流** |

---

## 第 1 步：在 Franka Linux 终端启动底层关节速度控制器

> [!CAUTION]
> 必须在系统原生 ROS 环境运行，**严禁在 Conda 环境中启动**！

打开 Franka 控制机终端 1，运行：
```bash
source /opt/ros/noetic/setup.bash
source /home/ssui/franka_ros_ws/catkin_ws2/devel/setup.bash

# 启动底层 1kHz 关节速度控制器（带硬件限速与平滑保护）
roslaunch franka_example_controllers joint_velocity_example_controller.launch robot_ip:=172.16.0.2
```

---

## 第 2 步：在 Franka 新终端启动执行服务（二选一）

新开一个 Franka 终端 2，进入代码目录：
```bash
source /opt/ros/noetic/setup.bash
cd /home/ssui/project/embodied_midterm/controller_lerobot
```

### 【推荐首选】方案 A：启动同步执行服务 (Stop-and-Go)
彻底消除“抢控制权”与拉扯抖动，停稳 50ms 静态采样相机画面，纯净执行 50k 模型的原生 7-DOF 轨迹：
```bash
python3 src/franka_teleop/sync_franka.py --z-min 0.0070
```
- `--z-min 0.0070`：硬编码桌面安全底面（$Z \ge +7.0\text{ mm}$）。
- `--settle-time 0.05`：每个分块执行完毕后停稳 50ms 供相机无模糊采图（默认已启用）。
- 姿态控制：默认不开启任何人工竖直强加，完全由 50k 模型自主掌握位姿。

### 方案 B：启动异步连续流服务 (RTC)
适用于高速连续流水线：
```bash
python3 src/franka_teleop/closed_loop_franka.py --rtc --z-min 0.0070
```

---

## 第 3 步：在 Windows GPU 服务器启动 VLA 推理

在 Windows 端打开 CMD 或 PowerShell，进入工程根目录：
```cmd
cd C:\Users\74727\Desktop\project\VLA_franka
```

### 【推荐首选】方案 A：运行同步推理 Agent（`run_sync_agent.bat`）
与 Linux 端 `sync_franka.py` 建立 1:1 请求响应式连接，杜绝时钟漂移与位置差拉扯：
```cmd
.\scripts\run_sync_agent.bat --task "pick and place the red cube"
```
- **默认权重**：自动加载 `outputs\checkpoints\pi05_lora_pure_flow_50k\latest.pt`（50,000 步纯净流匹配最优权重）。
- **默认姿态**：原生 7-DOF 自由位姿，不强制竖直，杜绝人工控制器与模型角力。
- **离线 Mock 快速自检**：
  ```cmd
  .\scripts\run_sync_agent.bat --mock
  ```
  *(验证模型、显卡 GPU 1 与推理速度，正常耗时 ~350ms)*

---

### 方案 B：运行异步 RTC 推理 Agent（`run_async_agent.bat`）
与 Linux 端 `closed_loop_franka.py --rtc` 建立异步流水线连接：
```cmd
.\scripts\run_async_agent.bat --task "pick and place the red cube"
```

---

### 方案 C：高性能 HTTP REST 推理微服务（`launch_server.bat`）
提供轻量零依赖高吞吐 HTTP 服务，适用于多客户端接入或浏览器 Web 控制台压测：
```cmd
.\scripts\launch_server.bat 8088 pure_flow
```
- **Web 可视化控制台**：浏览器直接访问 [http://localhost:8088/ui](http://localhost:8088/ui)。
- **客户端一键回归验证**：
  ```cmd
  C:\Users\74727\miniconda3\envs\lerobot\python.exe scripts\python\test_vla_client.py http://127.0.0.1:8088
  ```

---

### 💡 常用检查点 (Checkpoints) 别名速查

`--checkpoint` 参数支持以下快捷别名：

| 别名 | 对应检查点文件 | 说明 |
| :--- | :--- | :--- |
| **`pure_flow`** / **`pure_flow_latest`** | `outputs/.../pi05_lora_pure_flow_50k/latest.pt` | **【官方推荐】50,000 步纯净流匹配完整训练模型** |
| `pure_flow_50k` | `outputs/.../pi05_lora_pure_flow_50k/step_50000.pt` | 50,000 步终态检查点 |
| `pure_flow_2500` | `outputs/.../pi05_lora_pure_flow_50k/step_02500.pt` | 2,500 步中期检查点 |
| `cartesian_7d` | `outputs/.../pi05_lora_cartesian_7d/pi05_lora_multitask_step_2000.pt` | 历史消融试验模型 |

---

## 第 4 步：确认急停，回车启动闭环

两端网络握手成功后，Franka 终端 2 会打印：
```text
[READY] Hold physical E-STOP. Press [ENTER] to start live closed-loop control...
```

1. **现场操作人员手持物理急停按钮**，目视核验机械臂周围无人与危险杂物；
2. 在 Franka 终端按下 **[回车 (ENTER)]**，正式下发动作，机械臂开始闭环执行抓取！
3. **紧急中止**：
   - 调试暂停：在终端按下 `Ctrl + C`；
   - 突发碰撞风险：**立即拍下手中的物理急停开关**！
