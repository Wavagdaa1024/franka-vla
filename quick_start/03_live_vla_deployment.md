# 教程 03：实机闭环 VLA 部署四步法 (03_live_vla_deployment.md)

本教程指导如何将训练好的 VLA 模型（50,000 步纯净流匹配 Pi0.5 LoRA 模型）接入实机 Franka 机械臂，进行闭环自主推理与抓取控制。

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

## 第 2 步：在 Franka 新终端启动闭环执行服务

新开一个 Franka 终端 2，运行：
```bash
source /opt/ros/noetic/setup.bash
cd /home/ssui/project/embodied_midterm/controller_lerobot

# 启动 RTC (Real-Time Chunking) 连续流式闭环服务 (带桌面 Z>=7.0mm 安全门禁)
python3 closed_loop_franka_server.py --rtc --z-min 0.0070
```
*(如果是排查测试，亦可使用传统步进模式：`python3 closed_loop_franka_server.py --sync --sync-steps 15`)*

*(终端将打印机械臂当前关节就绪状态，并等待 Windows 端 Agent 建立 TCP 连接)*

---

## 第 3 步：在 Windows GPU 服务器启动 VLA 推理

在 Windows 端打开 CMD 或 PowerShell，进入工程根目录：

```cmd
cd /d C:\Users\74727\Desktop\project\VLA_franka
```

### 方式 1：推荐首选 —— 闭环实时推理 Agent（`run_agent.bat`）
直接与 Franka 控制机建立 TCP 流式握手，双 RealSense 相机 15Hz 实时推流，执行 50k 纯净流匹配模型预测的原生 7-DOF 轨迹：
```cmd
.\scripts\run_agent.bat --task "pick and place the red cube"
```
- **默认权重**：自动加载 `outputs\checkpoints\pi05_lora_pure_flow_50k\latest.pt`（50,000 步权重）。
- **默认姿态控制**：原生 7-DOF 轨迹执行，零生硬后处理，完美释放机械臂 6-DoF 空间位姿灵活性。
- **安全保障**：实时硬地面碰撞保护（$Z \ge +7.0\text{ mm}$）。

#### 可选姿态调节参数：
- `--enable-nullspace`：开启任务优先级零空间姿态自稳（在不改变末端 XYZ 轨迹的前提下纠偏倾角）。
- `--lock-vertical`：强制 5-DOF 竖直向下几何锁定（消融对比用）。
- `--checkpoint <别名/路径>`：切换不同迭代步数快照（如 `--checkpoint pure_flow_2500`）。

---

### 方式 2：高性能 HTTP REST 推理微服务（`launch_server.bat`）
提供轻量零依赖高吞吐 HTTP 服务，适用于多客户端接入、跨机器远程推理或离线压测：
```cmd
.\scripts\launch_server.bat 8088 pure_flow
```
- **Web 可视化控制台**：浏览器直接访问 [http://localhost:8088/ui](http://localhost:8088/ui)，支持实时显存监测与单步触发测试。
- **核心 API 接口**：
  - `POST /predict`：接收 Base64 双摄图 + 8D 关节状态，返回 15 步动作分块（实测纯 GPU 推理 ~350ms）。
  - `GET /health`：返回显存占用率、健康指标与当前加载的权重信息。
  - `POST /switch_checkpoint`：微秒级无缝热重载 LoRA 权重，无需重启 4.14B 基座模型。
- **客户端一键回归验证**：
  ```cmd
  C:\Users\74727\miniconda3\envs\lerobot\python.exe scripts\python\test_vla_client.py http://127.0.0.1:8088
  ```

---

### 方式 3：离线 Mock 自检模式（不连相机与机械臂）
在不连真实相机和机械臂的情况下，一键验证模型加载、LoRA 挂载、前向推理及算力时延：
```cmd
.\scripts\run_agent.bat --mock
```
- 预期输出 `[MOCK TEST PASSED] Steady-state Latency: ~353ms (~2.8 FPS)`，确认无误后即可连线实机。

---

### 💡 常用检查点 (Checkpoints) 别名速查

`--checkpoint` 参数既支持直接传文件绝对路径，也支持传入以下内置快捷别名：

| 别名 | 对应检查点文件 | 说明 |
| :--- | :--- | :--- |
| **`pure_flow`** / **`pure_flow_latest`** | `outputs/.../pi05_lora_pure_flow_50k/latest.pt` | **【官方推荐】50,000 步纯净流匹配完整训练模型** |
| `pure_flow_50k` | `outputs/.../pi05_lora_pure_flow_50k/step_50000.pt` | 50,000 步终态检查点 |
| `pure_flow_2500` | `outputs/.../pi05_lora_pure_flow_50k/step_02500.pt` | 2,500 步中期检查点 |
| `cartesian_7d` | `outputs/.../pi05_lora_cartesian_7d/pi05_lora_multitask_step_2000.pt` | 历史消融试验模型 |

> **提示**：随时运行 `.\scripts\list_ckpts.bat`，可查看磁盘上所有检查点大小、步数与 LoRA 配置。

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
