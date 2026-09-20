# franka-vla 快速使用分步教程索引 (quick_start/)

欢迎使用 **franka-vla**！为了方便日常研发调试与实机操作，我们将完整流程拆解为独立的模块化分步教程：

---

## 教程导航

1. **[01. 离线环境与安全自检 (01_offline_sanity_checks.md)](01_offline_sanity_checks.md)**
   - 检查预训练权重与分词器模型资产完整性 (`tests\check_assets.py`)
   - 运行 Franka 软限位与安全门禁单元测试 (`tests\check_safety_guards.py`)
   - 运行运动学、零空间自稳与 RTC 闭环全套单测 (`tests\test_kinematics_and_rtc.py`，10 项全通)
   - 运行双目相机硬件抽样体检与持续实时流 (`scripts\check_cameras.bat` / `scripts\live_camera.bat`)
   - 执行真值轨迹对比评测 (`tests\eval_offline_jointpos.py`) 与 Agent 离线自测 (`scripts\run_agent.bat --mock`)

2. **[02. 遥操作数据采集全流程 (02_teleop_data_collection.md)](02_teleop_data_collection.md)**
   - Franka 控制电脑与 Windows GPU 服务器三终端协同方案
   - ROS 笛卡尔阻抗底层与 Touch 手柄遥操作启动
   - 统一录制批处理入口 (`scripts\launch_record.bat`)
   - LeRobot 格式数据录制（新数据集 vs 增量续录 `--resume`）
   - 键盘操作规范与 Episode 保存核验

3. **[03. 实机闭环 VLA 部署四步法 (03_live_vla_deployment.md)](03_live_vla_deployment.md)**
   - 启动 Franka 1kHz 关节速度底层控制器 (`joint_velocity_example_controller.launch`)
   - 启动 Linux 端闭环执行服务 (`closed_loop_franka_server.py --rtc / --sync`)
   - Windows GPU 端启动推理主入口 (`scripts\run_agent.bat`，默认加载 50k 纯净流匹配模型)
   - 高性能 HTTP REST 推理微服务与 Web 监控控制台 (`scripts\launch_server.bat 8088 pure_flow`)
   - 离线 Mock 自检 (`scripts\run_agent.bat --mock`) 与物理急停安全回车启动

4. **[04. 模型训练与微调 (04_model_training.md)](04_model_training.md)**
   - 严格硬件隔离：GPU 1（RTX 5090 32GB）物理单卡全量微调，严禁触碰 GPU 0
   - 100% 原生纯净关节流匹配（Pure Flow Matching），无状态人为加噪，无伪逆解笛卡尔损失
   - 一键 50,000 步正式训练启动入口 (`scripts\train_pure_flow_50k_gpu1.bat`)
   - W&B 全生命周期云端实时指标大屏同步
   - 训练产出 Checkpoints 结构化一键查询 (`scripts\list_ckpts.bat`)

5. **[05. 第三视角相机快速复位与手眼标定指引 (05_camera_alignment_and_calibration.md)](05_camera_alignment_and_calibration.md)**
   - 机械臂就位与相机复位彻底解耦架构说明
   - Franka ROS Launch 一键平滑就位 (`move_to_camera_calib.launch`)
   - 60mm ChArUco 标定板 6-DoF 亚像素位姿追踪与极简 HUD (`scripts\align_camera.bat`)
   - 局域网 Web 看板与半透明鬼影叠图 (`http://10.70.242.38:8088`)
