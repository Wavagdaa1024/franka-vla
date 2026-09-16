# 实验室安全操作规范 (SAFETY_RULES.md)

1. **GPU 物理隔离**：
   - 本项目在 RTX 5090 双卡环境中运行。
   - 物理 GPU 0 严禁占用。
   - 所有脚本必须显式声明 `CUDA_VISIBLE_DEVICES=1`。

2. **机械臂运动边界**：
   - 最大关节速度 clamp：0.35 rad/s。
   - 桌面防护下界：$Z \ge 0.055$ m。
   - 步间跳跃限制：单步 $\Delta q \le 0.25$ rad。

3. **真机上电三要素**：
   - 急停开关必须握于操作者手中。
   - 控制机 `closed_loop_franka.py` 与 GPU 端连接成功后方可解锁上电。
   - 初次实验必须先执行 `--shadow` 影子运行验证动作方向。
