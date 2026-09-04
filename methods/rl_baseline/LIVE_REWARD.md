# 实时观察 500 epoch 训练

所有命令在项目根目录的 PowerShell 执行，使用现有 `.venv`。

终端一启动训练：

```powershell
.\.venv\Scripts\python.exe scripts/train_rl_epochs.py --member-epochs 400 --master-epochs 100 --steps-per-epoch 500 --output scripts/output/rl-500epochs
```

总计 **500 epoch × 500 步 = 250,000 个训练环境步**，另有验证和测试步。
400 + 100 是沿用现有两阶段训练方式的起始分配示例，不代表已验证的最优比例。
前 400 epoch 更新 Member，后 100 epoch 固定选出的 Member、更新 Master。
两个阶段的每个环境步都执行 Master 指挥、位置及区域归属更新、Member DAG 调度、仿真和奖励计算。
`--ppo-epochs` 默认 4，含义是每批数据的优化遍数，不是训练总 epoch 数。
环境配置及其他参数保留默认值。这里没有自动启动长时间训练。

终端二打开实时曲线窗口：

```powershell
.\.venv\Scripts\python.exe scripts/watch_rl_reward.py --run scripts/output/rl-500epochs --interval 2
```

每 2 秒检查一次 `epochs.csv`，每写入一个完整 epoch 就新增一个点。左图显示每步平均 reward，右图显示该 epoch 的 reward 总和；紫色虚线显示已有的验证结果。灰色竖虚线标记训练阶段切换。曲线显示原始记录，没有平滑，也不代表已经收敛。

一个 epoch 没跑完时不会生成该 epoch 的正式点；初始验证和定期验证也可能造成等待。当前步进度和即时 reward 在训练终端显示（默认每 25 步）。`epochs.csv` 当前在相应验证完成后写入，所以验证期间本 epoch 的点也会等待。

每次曲线更新自动保存 `RUN/live-reward.png`。窗口可在训练前打开，此时只等待日志，不创建训练目录或占用输出目录。关闭看图窗口不会停止训练；训练结束后窗口保持打开供查看。Windows 窗口使用 TkAgg，项目现有 Matplotlib 与本机 Python Tk 已可用，无需 TensorBoard。请从独立 PowerShell 终端运行以避免 IDE 的绘图窗口设置干扰。

只生成一次图片、不打开窗口：

```powershell
.\.venv\Scripts\python.exe scripts/watch_rl_reward.py --run scripts/output/rl-500epochs --once
```

首次完成的 epoch 出现前只显示等待，不生成图片。日志损坏会显示错误并重试；正在写入的末尾半行留待下次刷新读取。重新打开窗口会读取最新日志，反映训练器清理的未保存 epoch 记录。

中断后恢复到原计划的总轮数：

**Windows 恢复前先关闭看图窗口，等训练终端重新输出步进度后，再打开看图窗口。** 恢复可能原子替换日志，Windows 下若此时监控恰好正在读文件，会阻止替换；当前训练器没有对这一情况重试。本次坚持只新增文件，因此没有修改训练器。正常训练追加日志与监控并行已验证。

```powershell
.\.venv\Scripts\python.exe scripts/train_rl_epochs.py --resume scripts/output/rl-500epochs/latest.pt --member-epochs 400 --master-epochs 100
```

保存点在 epoch 边界，未保存完的 epoch 需要重跑；恢复参数表示目标总轮数，不是额外增加的轮数。
如果新输出目录已存在且包含文件，请换一个目录名；恢复已有实验则使用 `--resume`。
