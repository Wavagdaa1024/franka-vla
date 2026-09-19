# 教程 02：遥操作数据采集全流程 (02_teleop_data_collection.md)

本教程指导如何协同 Franka 控制电脑与 Windows GPU 服务器，完成基于 Touch 手柄与 RealSense 双相机的 LeRobot 格式轨迹录制。

---

## 1. 架构数据流图

```text
Franka ROS / 笛卡尔阻抗控制器
        ↓
TouchFrankaTeleopController
        ↓  TCP 8766
Windows 双相机 + LeRobot 录制器 (scripts\launch_record.bat)
        ↓
保存 episode 数据集 (dataset\...)
```

---

## 2. 终端 1：Franka 控制电脑，启动 ROS 和底层控制器

打开控制机终端 1，运行：

### 2.1 阻抗控制底层 (推荐遥操作采集使用)
```bash
source /opt/ros/noetic/setup.bash
source /home/ssui/franka_ros_ws/catkin_ws2/devel/setup.bash

roslaunch franka_example_controllers \
  cartesian_impedance_example_controller.launch \
  robot_ip:=172.16.0.2
```

### 2.2 回到初始工作位置
```bash
source /opt/ros/noetic/setup.bash
source /home/ssui/franka_ros_ws/catkin_ws2/devel/setup.bash

roslaunch franka_example_controllers move_to_start.launch robot_ip:=172.16.0.2
```

---

## 3. 终端 2：Franka 控制电脑，启动遥操作服务和夹爪

打开控制机终端 2，运行：

```bash
source /opt/ros/noetic/setup.bash
source /home/ssui/franka_ros_ws/catkin_ws2/devel/setup.bash

cd /home/ssui/project/embodied_midterm/controller_lerobot
python3 Teleop_dataset_recorder.py --sample-hz 15
```

---

## 4. 终端 3：Windows 服务器，启动录制端

打开 Windows GPU 服务器终端（PowerShell 或 CMD），进入工程根目录：

```cmd
cd /d C:\Users\74727\Desktop\project\VLA_franka
```

### 4.1 开始录制新数据集（首次录制，不要加 `--resume`）

#### 推荐方式（一键批处理入口）：
```powershell
scripts\launch_record.bat `
  --host 10.197.16.43 `
  --port 8766 `
  --action-space droid_joint_delta `
  --task "pick and place the red cube" `
  --repo-id local/franka_red_cube `
  --root "C:\Users\74727\Desktop\project\VLA_franka\dataset\teleop_pick_cube_15hz_002" `
  --front-serial 254322072252 `
  --wrist-serial 348122070854 `
  --fps 15 `
  --no-preview
```

#### 等价 Python 完整命令：
```powershell
C:\Users\74727\miniconda3\envs\lerobot\python.exe src\franka_teleop\record_teleop.py `
  --host 10.197.16.43 `
  --port 8766 `
  --action-space droid_joint_delta `
  --task "pick and place the red cube" `
  --repo-id local/franka_red_cube `
  --root "C:\Users\74727\Desktop\project\VLA_franka\dataset\teleop_pick_cube_15hz_002" `
  --front-serial 254322072252 `
  --wrist-serial 348122070854 `
  --fps 15 `
  --no-preview
```

---

### 4.2 追加录制到已有数据集（后续追加，必须加 `--resume`）

```powershell
scripts\launch_record.bat `
  --host 10.197.16.43 `
  --port 8766 `
  --action-space droid_joint_delta `
  --task "pick and place the red cube" `
  --repo-id local/franka_red_cube `
  --root "C:\Users\74727\Desktop\project\VLA_franka\dataset\teleop_pick_cube_15hz_002" `
  --front-serial 254322072252 `
  --wrist-serial 348122070854 `
  --fps 15 `
  --resume `
  --no-preview
```

> **提示**：如果录制时需要实时预览相机画面，可将 `--no-preview` 替换为 `--preview`。

---

## 5. 键盘操作规范与流程

所有键盘操作均在 **Franka 的终端 2** 进行：

### 5.1 按键功能速查表

| 按键 / 操作 | 功能说明 |
|---|---|
| **Touch UP** | 开启 / 关闭遥操作跟随 |
| **Touch DOWN** | 开启 / 关闭夹爪开合 |
| **`s`** | 开始当前 episode 录制 |
| **`e`** | 保存当前 episode |
| **`d`** | 丢弃当前 episode（动作失败或碰撞时按此键） |
| **`p`** | 打印当前网络连接与录制状态 |
| **`q`** | 退出录制程序 |

### 5.2 实际标准操作顺序
1. 启动 ROS 阻抗控制器 (终端 1)
2. 启动 `Teleop_dataset_recorder.py` (终端 2)
3. 启动 Windows 录制器 (终端 3：`scripts\launch_record.bat ...`)
4. 确认 Windows 终端打印显示与 Franka 建立 TCP 连接成功
5. Touch 手柄按 **UP** 开启遥操作跟随
6. 键盘按 **`s`** 开始录制当前轨迹
7. 控制机械臂完成抓取与放置示范动作
8. 键盘按 **`e`** 保存该轨迹

> **重要验证依据**：GPU 端终端必须看到打印：
> ```text
> Episode X started
> Episode X saved: ... frames
> ```
> 只有看到 `Episode X saved` 后，该段数据才真正安全写入磁盘。
