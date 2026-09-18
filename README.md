# franka-vla: Franka Panda 具身智能 VLA 部署工程

本项目面向 Franka Panda 机械臂具身操作任务，基于 Hugging Face **LeRobot** 规范实现端到端数据采集、Pi0.5 / Gemma LoRA 具身策略微调、三维视觉避障安全滤波与实时 RTC 闭环部署。

---

## 1. 整体目录架构 (Industrial src-layout)

```text
franka-vla/
├── src/                                  <-- 【功能性代码核心统一目录】
│   ├── franka_teleop/                    <-- 机械臂驱动、RTC闭环服务、Pi0.5引擎、正逆运动学
│   ├── obstacle_avoidance/               <-- 3D包络感知、Franka胶囊碰撞体、实时安全滤波
│   └── vlm_planning/                     <-- 大模型任务规划、目标定位与三维空间坐标映射
│
├── tests/                                <-- 【测试与标定统一目录】
│   ├── camera_alignment/                 <-- 相机复位与手眼对齐子模块 (ChArUco 6-DoF + Web服务)
│   │   ├── align_front_camera.py
│   │   ├── launch_align_camera.bat
│   │   ├── query_d435_extrinsics.py
│   │   ├── extract_camera_pairs.py
│   │   ├── tag_charuco_4x4_printable.png
│   │   └── README.md
│   ├── check_cameras.py                  <-- 双摄硬件抽样体检与序列号排查
│   ├── check_safety_guards.py            <-- 机械臂软硬件限位与滤波自检
│   ├── check_datasets.py                 <-- LeRobot 数据集规范合规检查
│   ├── test_kinematics_and_rtc.py        <-- 运动学、零空间自稳与闭环控制单测 (10项全通)
│   └── test_differentiable_fk.py         <-- 可微正向运动学 (DFK) 梯度求导单测
│
├── scripts/                              <-- 【用户统一操作入口层 (CLI & Launchers)】
│   ├── run_agent.bat                     <-- 实机推理主入口 (绑定 GPU 1, 默认 Step 1500 权重)
│   ├── train_cartesian_lora_gpu0.bat     <-- DFK LoRA 训练入口 (绑定 GPU 0, 姿态锁死防滑移)
│   ├── train_red_cube_lora_gpu0.bat      <-- 纯关节 LoRA 训练入口 (绑定 GPU 0)
│   ├── live_camera.bat                   <-- 双摄持续监控入口 (OpenCV 窗口 + HTTP Web 推流)
│   ├── check_cameras.bat                 <-- 双摄硬件快速体检 (采样 15 帧核验)
│   ├── align_camera.bat                  <-- 相机复位与手眼对齐入口 (转发至 tests/camera_alignment)
│   ├── launch_record.bat                 <-- 真机遥操作录制
│   └── list_ckpts.py                     <-- 检查点查询工具
│
├── outputs/                              <-- 训练权重、测试日志、提取切片 (已被 .gitignore 保护)
├── dataset/                              <-- 官方格式轨迹数据集 (已被 .gitignore 保护)
├── docs/                                 <-- 架构与安全文档 (docs/ARCHITECTURE.md, docs/SAFETY_RULES.md)
├── quick_start/                          <-- 分步操作指南
└── pyproject.toml                        <-- 规范包配置 (where = ["src"])
```

---

## 2. 用户操作统一入口 (`scripts/`)

日常所有操作均可通过 `scripts\` 下的批处理一键执行，无需手动切换深层路径或配置复杂的 Conda/CUDA 环境变量：

| 场景 | 推荐命令 | 作用说明 | 硬件资源 |
| :--- | :--- | :--- | :---: |
| **实机推理 (推荐首选)** | `scripts\run_agent.bat --raw --checkpoint cartesian_1000` | 运行 DFK 笛卡尔 LoRA (Tilt=0.5°)，无生硬后处理，端到端执行抓取 | GPU 1 (RTX 5090) |
| **实机推理 (安全滤波)** | `scripts\run_agent.bat --rtc --checkpoint cartesian_1000` | 开启零空间姿态自稳与实时速度平滑滤波 | GPU 1 (RTX 5090) |
| **推理虚跑自检** | `scripts\run_agent.bat --mock --checkpoint cartesian_1000` | GPU 1 模型加载自测，不连机械臂与摄像头 | GPU 1 (RTX 5090) |
| **模型微调训练** | `scripts\train_cartesian_lora_gpu0.bat` | GPU 0 训练带 DFK 笛卡尔空间损失与垂直向下倾角约束的 LoRA | GPU 0 (独立运行) |
| **双摄实时监控** | `scripts\live_camera.bat --web` | 打开双目相机实时流，浏览器访问 `http://localhost:5000` 查看 | CPU / USB |
| **双摄硬件体检** | `scripts\check_cameras.bat` | 快速捕获 15 帧排查掉帧、色彩与设备序列号 | CPU / USB |
| **相机复位与标定** | `scripts\align_camera.bat` | 运行 ChArUco 6-DoF 相机快速复位与半透明叠图 (Web: `8088`) | CPU / RealSense |
| **真机数据录制** | `scripts\launch_record.bat` | 实机遥操数据同步录制入口 | Franka + RealSense |
| **权重清单查询** | `python scripts\list_ckpts.py` | 打印当前服务器所有训练完成的 Checkpoints 路径与步数 | 本地查询 |

---

## 3. 核心功能模块划分说明

1. **`src/franka_teleop/`**：
   - 机器人底层硬件驱动、TCP 通信协议包、实时 RTC (Real-Time Chunking) 连续闭环控制服务、Pi0.5 模型引擎与可微正向运动学（DFK）。
2. **`src/obstacle_avoidance/`**：
   - 视觉空间安全包络提取（Perception Envelope）、机械臂 9 胶囊体模型（Franka Capsule Model）与障碍物最小距离解析计算引擎。
3. **`src/vlm_planning/`**：
   - 多步骤大模型任务规划器（Qwen / RoboBrain）与桌面二维/三维坐标投影映射。
4. **`tests/camera_alignment/`**：
   - 第三视角相机防碰撞偏移 6-DoF 快速复位系统、ChArUco 标定板高清图纸及数据集双摄同步切片提取工具。

---

## 4. 快速分步教程

详细分步操作手册请参阅 **[quick_start/ 教程目录](quick_start/)**：
- [01. 离线环境与安全自检](quick_start/01_offline_sanity_checks.md)
- [02. 遥操作数据采集全流程](quick_start/02_teleop_data_collection.md)
- [03. 实机闭环 VLA 部署四步法](quick_start/03_live_vla_deployment.md)
- [04. 模型训练与微调](quick_start/04_model_training.md)
