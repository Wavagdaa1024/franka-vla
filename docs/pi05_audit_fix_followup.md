# Pi0.5 第二轮修复说明

日期：2026-09-20。分支：fix/pi05-codex-audit。起点：df6d9bbba213e02e392d41c6629f36c9c921f699。
用户已授权修复和更新Git。本次未启动训练、GPU模型推理、摄像头或机器人，未改main，未改现有checkpoint。

## 已修复的行为

| 问题 | 修改后的行为 | 主要文件 |
|---|---|---|
| 裁剪按目录子串猜测 | 优先读checkpoint元数据；冲突拒绝；仅对两组已核实历史实验使用精确目录名兼容映射并告警；其他旧权重要求明确指定 | pi05_engine/contracts.py；sync/async/HTTP入口 |
| 入口精度不一致 | 三个部署入口都使用FP32投影，LoRA也保留FP32；rank/alpha统一解析 | sync_vla_agent.py、async_rtc_vla_agent.py、vla_inference_server.py |
| 异步无限复用旧帧 | 两个相机入口共用校验函数，过期、缺失、未来时间戳或双路时间差超限返回None | camera_frames.py；两个agent |
| 双路错时仅警告 | 0.5s年龄/0.08s双路时间差成为返回图像的硬条件；时间源统一monotonic | camera_frames.py |
| 等待分支重放chunk第0步 | worker绑定控制端本地请求时间和序号；所有接收路径共用年龄校验；超时预测不截断到最后一步强行执行；等待前停车 | closed_loop_franka.py |
| strict加载先写后检查、shape广播 | 写前检查key、精确shape、浮点/有限值及转换溢出；复制异常回滚；拒绝错误shape广播 | pi05_engine/lora.py |
| resume静默改契约/假恢复 | 默认strict要求标签、裁剪、精度、结构、数据集/划分及关键训练超参一致；缺权重、缺优化器/调度器等直接报错 | contracts.py、train_pi05_lora.py |
| 指标比较与来源含糊 | 每样本固定SHA256种子生成评估noise，不消耗训练RNG；W&B记录实际评估数和候选池总量；checkpoint记录代码revision、tracked dirty标记、alpha和noise策略 | train_pi05_lora.py |

所有路径以服务器 C:\Users\74727\Desktop\project\VLA_franka 为根；pi05_engine位于src/franka_teleop，agent和训练脚本位于scripts/python。

## 使用方式变化

新训练默认不裁剪，默认目录改为outputs/checkpoints/pi05_lora_v2_20k，不再用含crop169的名字描述none。scripts/train_crop169_20k_gpu1.bat明确传--image-crop 16_9，输出到独立的pi05_lora_crop169_v2_20k。已有checkpoint目录不能被新训练或迁移直接覆盖。

--resume-mode strict（默认）仅用于同训练契约续训；恢复优化器和scheduler，避免构造LambdaLR时改掉保存的当前学习率。检查的是训练契约一致性，不承诺位级重现：随机状态和sampler游标尚未序列化。

旧50k/旧crop169_20k缺元数据、使用旧标签或BF16时，应显式用--resume-mode migrate。该模式只加载权重，优化器、scheduler、步数重新开始；需要新输出目录，旧无crop元数据时必须显式--image-crop。记录中相同W&B ID会被拒绝复用。

scripts/resume_crop169_wandb_online.bat已改成明确的旧5k权重迁移入口：新目录pi05_lora_crop169_v2_migration_20k、新Run ID pi05_crop169_v2_migration_20k、从第1步开始训练新的20000步。它不再表示从5001继续旧实验。两个bat在开始时未跟踪，因直接属于修复范围现纳入版本控制；未运行。

同步、异步及HTTP默认auto读取新权重元数据。已核实历史目录pi05_lora_pure_flow_50k对应none，pi05_lora_crop169_20k对应16_9；重命名后的无元数据旧权重需要显式指定。旧BF16值扩展到FP32不会恢复过去训练中丢失的更新，推理计算精度也与旧部署有所不同，应作为新的部署版本记录。

HTTP加载保持权重、crop与模型身份一致；同结构LoRA热切换失败保留当前模型。为避免旧的全权重与LoRA混合残留，HTTP热切换入口现在只接受相同rank/alpha的LoRA；全权重请走独立进程的sync/async入口，不再由HTTP直接部分覆盖当前网络。该限制会明确报错，不静默退回base。

异步相机无效时显式发送hold标志，控制端立即停车等待。控制端根据本地请求发起时刻计算年龄，包含排队、传输、推理和等待时间，不相减两台机器的时钟。正常、阻塞接收、重复/过期响应走同一个激活逻辑。时序超出预测horizon时保持等待并重新取观测，不承诺任何延迟下仍连续运动。

## 测试与证据

- tests/test_audit_regressions.py新增19项CPU测试，全部通过。
- 原test_audit_fixes.py的01、03、04、05共4项轻量测试全部通过；05时间源随实现改为monotonic。
- 合计23/23通过。覆盖裁剪目录冲突/重命名、旧模型显式迁移、恢复契约、错误shape/NaN/key不改权重、copy异常回滚、FP32细小数值保留、双相机拒绝、HTTP成功/失败/结构变化、scheduler恢复、训练/评估夹爪命令切片、固定noise不扰动RNG。
- 虚拟时钟驱动真实run_rtc_loop：新chunk在阻塞等待中返回，激活从年龄对应的第8步开始，不从0重放；等待前已调用stop。纯内存假机械臂、无socket与真机。
- 四个CLI的--help解析成功；实际旧5k执行strict恢复检查，在数据集构建/GPU模型初始化前明确拒绝。
- 真实旧50k与crop169_20k权重通过新的CPU weights_only读取器，裁剪分别解析none和16_9，均296个张量。
- 变更Python文件AST语法检查通过；git diff --check通过。

可在服务器复现：
1. 显式设置CUDA_VISIBLE_DEVICES为空、PYTHONDONTWRITEBYTECODE=1；使用lerobot环境python -B。
2. python -B tests/test_audit_regressions.py
3. python -B tests/test_audit_fixes.py TestCodexAuditFixes.test_01_preprocessing_decoupling TestCodexAuditFixes.test_03_strict_checkpoint_loading TestCodexAuditFixes.test_04_fp32_precision_and_optimizer_updates TestCodexAuditFixes.test_05_camera_ttl_guard

没有运行原test02的全图像预加载，也没有运行完整GPU前向、训练或真机验证。上述测试不证明抓取成功率已经改善。

## 后续验证与部署边界

服务器Git代码已修复不等于Franka Linux控制机已经更新；本轮没有连接或修改控制机。应用异步修复需要后续受控部署相匹配的controller和agent。camera_frames.py是新增部署依赖，应随Windows源码一并更新。

双相机阈值校验基于主机接收时间，不等于曝光硬件同步；0.08s是否适合实际负载仍需采集日志验证。年龄补偿是状态请求年龄，不是完整的相机曝光—状态时间对齐。

最新旧20k仍不是这些修复的训练验收；旧夹爪标签不会被改脚本追溯修复。下一步先指定物理GPU做固定观测/固定noise离线预测检查，再按授权区分模型原始输出、控制命令和实机反馈。没有证据宣称视觉问题已排除或执行是唯一根因。新的固定noise评估口径也不能与旧随机noise单次指标直接当成严格同条件比较。

知识库截止日期：2025年12月。实现和测试结论基于本次实际源码与CPU运行。
