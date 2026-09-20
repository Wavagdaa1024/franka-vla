# 教程 03：实机闭环 VLA 部署四步法 (03_live_vla_deployment.md)

本教程指导如何将训练好的 VLA 模型（50,000 步纯净流匹配 Pi0.5 LoRA 模型）接入实机 Franka 机械臂，进行闭环自主推理与抓取控制。

为满足不同实验场景需求，系统已将**同步推理 (Stop-and-Go)** 与 **异步推理 (RTC 连续流)** 彻底解耦为两个独立程序套件，分别拥有独立的运行脚本、底层服务与参数配置。

---

## 模式选择指南

| 特性维度 | 方案 A：同步推理 (Stop-and-Go) 【强烈推荐首选】 | 方案 B：异步推理 (RTC 连续流) |
| :--- | :--- | :--- |
| **运行脚本 (Windows)** | `scripts\run_sync_agent.bat` | `scripts\run_async_agent.bat` |
| **服务程序 (Linux)** | `sync_franka.py` (或 `sync_franka_server.py`) | `closed_loop_franka_server.py --rtc` |
| **动作衔接协议** | 执行完 15 步后平滑停稳 50ms 再采图 | 边走边预取下一动作分块 |
| **控制权冲突** | **零冲突**（起步位置与当前真实位置 100% 吻合） | 若时延预估偏差易产生反向拉扯 |
| **视觉清晰度** | **零运动模糊**（静止采样，抓取定位极准） | 高速运动时可能存在局部模糊 |
| **推荐适用场景** | **精细抓取、防打架调试、稳定可靠部署** | **大范围自由流动探索、高帧率连续推流** |

---

## 方案 A：同步推理模式 (Stop-and-Go) —— 【推荐首选】

> **设计优势**：机械臂执行完上一个分块后，平滑刹车停稳 50ms（`arm.stop()` + `settle_time=0.05`），此时机械臂静止，双摄相机采到完全无运动模糊的清晰画面，且当前关节角 $q_{curr}$ 毫无时间差；GPU 算好新分块后，机械臂从静止状态平滑起步，**绝对零拉扯、零打架、零卡顿**！

### 终端 1（Linux 控制电脑）：启动底层关节速度控制器

> [!CAUTION]
> 必须在系统原生 ROS 环境运行，**严禁在 Conda 环境中启动**！

```bash
# 激活 ROS 环境变量
source /opt/ros/noetic/setup.bash
source /home/ssui/franka_ros_ws/catkin_ws2/devel/setup.bash

# 启动 1kHz 关节速度控制器（带硬件限速与平滑保护）
roslaunch franka_example_controllers joint_velocity_example_controller.launch robot_ip:=172.16.0.2
```

---

### 终端 2（Linux 控制电脑）：启动同步执行服务端

新开一个 Linux 终端，进入控制目录并启动同步服务：

```bash
# 激活 ROS 环境
source /opt/ros/noetic/setup.bash

# 进入项目目录
cd /home/ssui/project/embodied_midterm/controller_lerobot

# 启动同步执行服务 (桌面硬限高 Z >= 7.0mm)
python3 sync_franka.py --z-min 0.0070
```

> **重要参数说明**：
> - `--z-min 0.0070`：硬编码桌面安全底面（$Z \ge +7.0\text{ mm}$），防止撞击桌面。
> - `--settle-time 0.05`：每个分块执行完毕后停稳 50ms 供相机无模糊采图（默认已启用）。
> - 姿态控制：已默认关闭任何人工强制竖直，完全由 50k 模型自主掌握位姿。
> - 终端打印 `[OK] Listening on port 8765. Awaiting GPU Agent...` 表示就绪。
> - *(注：`python3 sync_franka.py` 与 `python3 src/franka_teleop/sync_franka.py` 均支持)*

---

### 终端 3（Windows GPU 服务器）：启动同步 VLA 推理 Agent

在 Windows 端打开 CMD 或 PowerShell，进入工程根目录：

```cmd
cd C:\Users\74727\Desktop\project\VLA_franka

# 启动同步推理 Agent (自动隔离 GPU 1: RTX 5090 32GB)
.\scripts\run_sync_agent.bat --task "pick and place the red cube"
```

> **参数说明**：
> - `--task "..."`：任务语言提示词（默认："pick and place the red cube"）。
> - **默认模型**：自动加载 `outputs\checkpoints\pi05_lora_pure_flow_50k\latest.pt`（50,000 步纯净流匹配权重）。
> - **安全隔离**：严格绑定 `CUDA_VISIBLE_DEVICES=1`，绝不干扰 GPU 0 任务。
>
> *(可选离线测试命令，不连相机与机械臂快速自测)*：
> ```cmd
> .\scripts\run_sync_agent.bat --mock
> ```

---

### 确认急停并启动

当两端握手成功后，**Linux 终端 2** 会打印：
```text
[READY] Hold physical E-STOP. Press [ENTER] to start closed-loop control...
```
1. 操作人员**手持物理急停手柄**；
2. 在 **Linux 终端 2 按下 [Enter] 回车键**，机械臂立即开始闭环自主抓取！

---

## 方案 B：异步推理模式 (RTC 连续流)

> **设计优势**：流水线预取，边运动边生成下一分块，动作连续不顿挫。
> **注意**：对网络传输与推理耗时波动较为敏感。

### 终端 1（Linux 控制电脑）：底层控制器
```bash
source /opt/ros/noetic/setup.bash
source /home/ssui/franka_ros_ws/catkin_ws2/devel/setup.bash
roslaunch franka_example_controllers joint_velocity_example_controller.launch robot_ip:=172.16.0.2
```

### 终端 2（Linux 控制电脑）：启动 RTC 连续流服务端
```bash
source /opt/ros/noetic/setup.bash
cd /home/ssui/project/embodied_midterm/controller_lerobot
python3 closed_loop_franka_server.py --rtc --z-min 0.0070
```

### 终端 3（Windows GPU 服务器）：启动异步 RTC 推理 Agent
```cmd
cd C:\Users\74727\Desktop\project\VLA_franka
.\scripts\run_async_agent.bat --task "pick and place the red cube"
```

---

## 方案 C：高性能 HTTP REST 推理微服务（跨机 / 网页调试）

适用于跨网络调试、网页可视化看盘或第三方程序调用：

```cmd
cd C:\Users\74727\Desktop\project\VLA_franka

:: 启动 HTTP 微服务 (监听 8088 端口，加载 50k 纯净流匹配权重)
.\scripts\launch_server.bat 8088 pure_flow
```
- **Web 可视化控制台**：浏览器直接访问 [http://localhost:8088/ui](http://localhost:8088/ui)
- **客户端一键连通性测试**：
  ```cmd
  C:\Users\74727\miniconda3\envs\lerobot\python.exe scripts\python\test_vla_client.py http://127.0.0.1:8088
  ```

---

## 💡 常用检查点 (Checkpoints) 别名速查

`--checkpoint` 参数支持以下快捷别名（同步和异步脚本均通用）：

| 别名 | 对应检查点文件 | 说明 |
| :--- | :--- | :--- |
| **`pure_flow`** / **`pure_flow_latest`** | `outputs/.../pi05_lora_pure_flow_50k/latest.pt` | **【官方推荐】50,000 步纯净流匹配完整训练模型** |
| `pure_flow_50k` | `outputs/.../pi05_lora_pure_flow_50k/step_50000.pt` | 50,000 步终态检查点 |
| `pure_flow_2500` | `outputs/.../pi05_lora_pure_flow_50k/step_02500.pt` | 2,500 步中期检查点 |
| `cartesian_7d` | `outputs/.../pi05_lora_cartesian_7d/pi05_lora_multitask_step_2000.pt` | 历史消融试验模型 |

---

## 常见问题与应急排查

1. **`python3: can't open file: No such file or directory`？**
   - 在 Linux 终端已进入 `/home/ssui/project/embodied_midterm/controller_lerobot` 时，直接运行：
     `python3 sync_franka.py --z-min 0.0070`
     *(系统现已建立别名软链，`python3 sync_franka.py` 与 `python3 src/franka_teleop/sync_franka.py` 均支持)*

2. **机械臂抖动、拉扯或顿挫？**
   - **排查 1**：请切换至 **方案 A（同步推理模式）**，运行 `run_sync_agent.bat` + `sync_franka.py`。
   - **排查 2**：确认 Linux 机器后台没有残留其他节点正在发布 `/joint_velocity_example_controller/joint_velocity`。运行 `rostopic info /joint_velocity_example_controller/joint_velocity` 确认仅有 1 个 Publisher。
   - **排查 3**：确认未加 `--enable-nullspace`（默认已禁用），避免人工外力与模型对抗。

3. **急停与安全退出**：
   - 调试暂停：在各终端按下 `Ctrl + C`，控制器会自动平滑制动停臂。
   - 碰撞风险：**立即拍下物理急停开关**！
