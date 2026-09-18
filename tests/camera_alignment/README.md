# 第三视角相机快速复位与手眼标定工具 (ChArUco 6-DoF)

针对 Franka 平台第三视角 RealSense D435I (S/N: 254322072252) 安装在活动支架上容易发生碰撞或微小偏移导致测试数据分布偏移 (OOD) 的问题，本模块提供基于 60mm ChArUco 标定板的高精度 6-DoF 复位与手眼标定支持。

---

## 目录文件说明

1. **`launch_align_camera.bat`**：一键启动脚本（同时启动本地 OpenCV 交互窗口与局域网 Web 界面 `http://10.70.242.38:8088`）。
2. **`align_front_camera.py`**：位姿解算内核（基于 ChArUco + PnP 求解 6-DoF 偏差，支持半透明鬼影叠图）。
3. **`tag_charuco_4x4_printable.png`**：300 DPI 高清 60mm 标定板图纸（边缘自带毫米校验标尺，100% 比例打印贴在操作台角落）。
4. **`query_d435_extrinsics.py`**：RealSense 相机出厂内参 (fx, fy, cx, cy) 及深度-彩色外参矩阵读取工具。
5. **`extract_camera_pairs.py`**：从真机遥操数据集提取第三视角与腕部相机帧同步成对图像，并自动生成对比时序长图（结果保存至 `outputs/extracted_camera_pairs/`）。

---

## 快速使用

```cmd
:: 1. 运行相机快速复位监控
cd /d C:\Users\74727\Desktop\project\VLA_franka\camera_alignment
launch_align_camera.bat

:: 2. 读取当前相机硬件内参与外参
C:\Users\74727\miniconda3\envs\lerobot\python.exe query_d435_extrinsics.py

:: 3. 提取数据集双目对齐对比图
C:\Users\74727\miniconda3\envs\lerobot\python.exe extract_camera_pairs.py
```
