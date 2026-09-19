# 教程 03：实机闭环 VLA 部署四步法 (03_live_vla_deployment.md)

本教程指导如何将训练好的 VLA 模型（如微调好的 Pi0.5 / DROID jointpos 模型）接入实机 Franka 机械臂，进行闭环自主推理与抓取控制。

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

# 推荐：启动 RTC (Real-Time Chunking) 连续流式闭环服务 (带桌面 Z>=7.0mm 安全门禁)
python3 closed_loop_franka_server.py --rtc --z-min 0.0070
```
*(如果是排查测试，亦可使用传统步进模式：`python3 closed_loop_franka_server.py --sync --sync-steps 15`)*

*(终端将打印机械臂当前关节就绪状态，并等待 Windows 端 Agent 建立 TCP 连接)*

---

## 第 3 步：在 Windows GPU 服务器启动 VLA 推理 Agent

在 Windows 端打开 CMD 或 PowerShell，进入工程根目录：

```cmd
cd /d C:\Users\74727\Desktop\project\VLA_franka
```

### 方式 1：推荐首选 —— DFK 端到端直接推理（姿态垂直锁死，Tilt < 0.5°）
配合 DFK 笛卡尔空间微调模型，无需额外生硬后处理，端到端自主闭环抓取：
```cmd
scripts\run_agent.bat --raw --checkpoint cartesian_1000 --task "pick and place the red cube"
```

### 方式 2：开启实时零空间自稳与速度滤波（RTC 模式）
开启零空间姿态保护与前置速度投影滤波：
```cmd
scripts\run_agent.bat --rtc --checkpoint cartesian_1000 --task "pick and place the red cube"
```

### 方式 3：离线 Mock 自检（不连机械臂与相机，纯测试模型加载与推理通道）
```cmd
scripts\run_agent.bat --mock --checkpoint cartesian_1000
```

---

### 💡 常用检查点 (Checkpoints) 别名速查

`--checkpoint` 参数既支持直接传文件路径，也支持传入内置快捷别名：

| 别名 | 对应检查点文件 | 说明 |
| :--- | :--- | :--- |
| **`cartesian_1000`** | `outputs/checkpoints/pi05_lora_cartesian_dfk/pi05_lora_multitask_step_1000.pt` | **【推荐】DFK 笛卡尔位姿 LoRA，倾角严控** |
| `cartesian_1500` | `outputs/checkpoints/pi05_lora_cartesian_dfk/pi05_lora_multitask_step_1500.pt` | DFK 笛卡尔位姿 LoRA Step 1500 |
| `cartesian_2000` | `outputs/checkpoints/pi05_lora_cartesian_dfk/pi05_lora_multitask_step_2000.pt` | DFK 笛卡尔位姿 LoRA Step 2000 |
| `red_cube_1500` | `outputs/checkpoints/pi05_lora_red_cube/pi05_lora_multitask_step_1500.pt` | 纯关节空间 LoRA Step 1500 |
| `multitask_5000` | `outputs/checkpoints/pi05_lora_multitask/pi05_lora_multitask_step_5000.pt` | 多任务基准微调权重 |

> **提示**：可直接运行 `scripts\list_ckpts.bat` 查询当前机器上所有保存的 Checkpoints 清单与指标。

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
