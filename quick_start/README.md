# VLA_franka 快速使用分步教程索引 (quick_start/)

欢迎使用 VLA_franka！为了方便查阅与操作，我们将完整流程拆解为独立的子教程：

---

## 教程导航

1. **[01. 离线环境与安全自检 (01_offline_sanity_checks.md)](01_offline_sanity_checks.md)**
   - 检查预训练权重与分词器资产完整性
   - 运行 Franka 软限位与安全门禁单元测试
   - 执行 GPU 1 单卡隔离影子推理测试（验证延迟与 0 NaN/Inf）

2. **[02. 遥操作数据采集全流程 (02_teleop_data_collection.md)](02_teleop_data_collection.md)**
   - Franka 控制电脑与 Windows GPU 服务器三终端协同方案
   - ROS 笛卡尔阻抗底层与 Touch 手柄遥操作启动
   - LeRobot 格式数据录制（新数据集 vs 增量续录）
   - 键盘操作规范与 Episode 保存核验

3. **[03. 实机闭环 VLA 部署四步法 (03_live_vla_deployment.md)](03_live_vla_deployment.md)**
   - 启动 Franka 1kHz 关节速度底层控制器
   - 启动闭环执行服务并配置镜像参数
   - Windows GPU 端启动实时推理 Agent（支持模型切换与任务变更）
   - 物理急停手持与回车安全闭环

4. **[04. LeRobot 官方模型训练与微调 (04_model_training.md)](04_model_training.md)**
   - 使用录制好的 LeRobot 数据集进行 Policy 微调
   - 官方 `train.py` 命令行调用与参数配置
   - GPU 1 物理隔离与显存控制

5. **[05. 第三视角相机快速复位与手眼标定指引 (05_camera_alignment_and_calibration.md)](05_camera_alignment_and_calibration.md)**
   - 机械臂就位与相机复位完全解耦架构说明
   - Franka ROS Launch 一键运动至标定位姿 (`move_to_camera_calib.launch`)
   - 60mm ChArUco 棋盘相间板 6-DoF 亚像素姿态追踪与极简 HUD (`align_camera.bat`)
   - 局域网 Web 看板实时指导与半透明鬼影叠图
