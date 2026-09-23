# franka-vla: Franka Panda 具身智能 VLA 部署工程

本项目面向 Franka Panda 机械臂具身操作任务，基于 Hugging Face **LeRobot** 规范实现端到端数据采集、Pi0.5 具身策略微调、三维视觉避障安全滤波与实时闭环部署。

---

## 1. 整体目录架构 (Industrial src-layout)

```text
franka-vla/
├── src/                                  <-- 【功能性代码核心统一目录】
│   ├── franka_teleop/                    <-- 机械臂驱动、RTC闭环服务、同步执行、Pi0.5引擎、正逆运动学
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
├── scripts/                              <-- 【用户统一操作入口层 (一键批处理)】
│   ├── python/                           <-- 【底层核心 Python 脚本模块】
│   │   ├── sync_vla_agent.py             <-- 实机同步推理核心 (Stop-and-Go)
│   │   ├── async_rtc_vla_agent.py        <-- 实机异步 RTC 推理核心
│   │   ├── train_pi05_lora.py            <-- Pi0.5 LoRA 微调训练引擎
│   │   ├── vla_inference_server.py       <-- 常驻高性能推理服务
│   │   └── live_camera_stream.py         <-- 双摄持续实时流与 Web 推流
│   ├── run_agent.bat                     <-- 实机推理主入口 (默认执行同步模式)
│   ├── run_sync_agent.bat                <-- 实机同步抓取推理 (Stop-and-Go，推荐首选)
│   ├── run_async_agent.bat               <-- 实机异步 RTC 推理 (滚动时域平滑衔接)
│   ├── launch_server.bat                 <-- 独立推理服务启动入口
│   ├── train_pure_flow_50k_gpu1.bat      <-- 50k 步纯净流匹配训练 (GPU 1)
│   ├── train_crop169_20k_gpu1.bat        <-- 20k 步视场裁剪微调 (GPU 1)
│   ├── launch_record.bat                 <-- 真机遥操作数据同步录制
│   ├── live_camera.bat                   <-- 双摄持续监控入口 (OpenCV 窗口 + HTTP Web:8080)
│   ├── check_cameras.bat                 <-- 双摄硬件快速体检 (采样 15 帧核验)
│   ├── align_camera.bat                  <-- 相机复位与手眼对齐入口 (转发至 tests/camera_alignment)
│   └── list_ckpts.bat                    <-- 检查点清单一键查询
│
├── outputs/                              <-- 训练权重、测试日志、提取切片 (已被 .gitignore 保护)
├── dataset/                              <-- 官方格式轨迹数据集 (已被 .gitignore 保护)
├── docs/                                 <-- 架构与安全文档 (docs/ARCHITECTURE.md, docs/SAFETY_RULES.md)
├── quick_start/                          <-- 分步操作指南
└── pyproject.toml                        <-- 规范包配置 (where = ["src"])
```

---

## 2. 核心功能模块划分说明

1. **`src/franka_teleop/`**：
   - 机器人底层硬件驱动、TCP 通信协议包、实时 RTC (Real-Time Chunking) 连续闭环控制服务、Stop-and-Go 同步执行引擎、Pi0.5 模型推理与正逆运动学。
2. **`src/obstacle_avoidance/`**：
   - 视觉空间安全包络提取（Perception Envelope）、机械臂 9 胶囊体模型（Franka Capsule Model）与桌面安全防撞限制引擎。
3. **`src/vlm_planning/`**：
   - 多步骤大模型任务规划器（Qwen / RoboBrain）与桌面二维/三维空间坐标映射。
4. **`tests/camera_alignment/`**：
   - 第三视角相机防碰撞偏移 6-DoF 快速复位系统、ChArUco 标定板高清图纸及数据集双摄同步切片提取工具。

---

## 3. 快速分步教程

详细分步操作手册请参阅 **[quick_start/ 教程目录](quick_start/)**：
- [01. 离线环境与安全自检](quick_start/01_offline_sanity_checks.md)
- [02. 遥操作数据采集全流程](quick_start/02_teleop_data_collection.md)
- [03. 实机闭环 VLA 部署四步法](quick_start/03_live_vla_deployment.md)
- [04. 模型训练与微调](quick_start/04_model_training.md)
