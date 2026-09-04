# 每个 epoch 500 个完整环境步

这一入口在 `codex/global-delay-rl-baseline` 分支上以新增文件实现，保留上一版训练
入口、全部旧代码和已生成的实验结果。

## 先区分三个计数

| 名称 | 含义 | 新入口默认值 |
| --- | --- | ---: |
| 环境 step | Master 指挥飞行 → 关联/候选刷新 → Member 调度全部 DAG → 执行结算 | 一个完整时隙 |
| 训练 epoch | 重置一次场景，然后连续推进指定数量的环境 step | 500 步 |
| PPO epochs | 对已经采集的同一批样本重复优化多少遍 | 4 遍 |

上一版 `train_rl.py` 的 `--rollout-slots` 默认是 8；`--epochs=4` 是 PPO 内部优化遍数。
旧图的横轴是 PPO 更新/采样轮次，不能把旧数据直接说成每 epoch 500 步。

本入口 `--steps-per-epoch 500` 才是你要求的步数。一次 epoch 中默认有：

- 500 次 Master 联合指挥，每次覆盖当前 4 个 Master 的所属 Member。
- 500 批 DAG 生成，每批 100 个 DAG，每个 DAG 10 个子任务。
- 500 次全部 DAG 的放置与真实执行结算，共 50,000 个 DAG、500,000 个子任务。
- 500 个全局时延奖励。内部子任务放置不增加环境 step。

UAV 位置、所属区域和用户关联在这 500 步内连续演化。仅在下一个 epoch 开始时
重置物理位置。原环境仍在每步的截止时刻清空当期运行时和队列。

## 运行

当前项目 `.venv` 已安装可选 PyTorch。新机器仍按 `requirements-rl.txt` 安装。

```powershell
.\.venv\Scripts\python.exe scripts/train_rl_epochs.py --steps-per-epoch 500 --output scripts/output/rl-epochs-run01
```

新入口默认 Member 阶段 100 epochs，Master 阶段 30 epochs，因此是 **130 × 500 =
65,000 个训练环境步**，不含验证/测试。不是上一版总共 1040 个训练步的运行量。
按需要设置总 epoch 数，例如先运行每阶段 5 个：

```powershell
.\.venv\Scripts\python.exe scripts/train_rl_epochs.py --member-epochs 5 --master-epochs 5 --steps-per-epoch 500 --output scripts/output/rl-epochs-run02
```

`--eval-steps` 和 `--test-steps` 默认也为 500，使用与训练隔离的数据段；它们不计入
训练 epoch 的步数。`--eval-every` 默认 10，另在每阶段最后一个 epoch 后验证。

参数更新仍按上一版的稳定性方案分阶段，但每个阶段每步都执行两类策略：

| 阶段 | Master 每步做什么 | Member 每步做什么 | 更新谁 |
| --- | --- | --- | --- |
| Member | 当前冻结策略输出确定性飞行指令 | 采样并调度 DAG | Member |
| Master | 采样飞行指令 | 冻结策略确定性调度 DAG | Master |

“冻结”指停止参数更新，不是跳过该智能体的动作。这里没有同时联合更新两类策略。

## 内存、轨迹边界和更新次数

不能把 500 步的全部 500,000 个子任务观测无限堆在内存里。默认
`--update-every-steps 8`，在 Member 阶段每 8 个完整环境步更新一次 PPO，最后的
4 步也会更新一次。因此一个 500 步 Member epoch 有 `ceil(500/8)=63` 次更新。
这些更新不会重置位置、时隙时钟或环境步数。

Member 的任务放置仍以单时隙为子轨迹，内部奖励为 0，终局全局奖励作为该时隙
各动作的 return。当前环境无跨时隙队列/电量状态；冻结 Master 的输入不依赖
Member 的任务放置，所以这里沿用原有的单时隙 Member 回报定义。

Master 的轨迹只需要保存 500 × 4 条动作数据，可以在整个 epoch 后更新一次。
GAE 使用 gamma=1、lambda=0.95，仅第 500 步是真终端；不会每 8 步误作终端。
新的 Master critic 显式加入 epoch 剩余比例，以区分有限时域中不同剩余步数。
Actor 的输入仍然本地化，未额外获得全局 critic 状态。

## 奖励和日志

每步只优化原定义的全局平均截止截断 DAG 时延：

```text
r_t = -mean(所有 DAG 的相对完成时延，未完成按截止时长计) / 截止时长
epoch_return = sum(r_t), t=1..500
mean_reward = epoch_return / 500
```

当前截止时长是 1 秒。因此每步平均时延 0.5 秒，对应 `mean_reward=-0.5`，
500 步累计 `epoch_return=-250`。奖励越大（越接近 0），时延越小。

| 输出文件 | 含义 |
| --- | --- |
| steps.csv | 每一步的 epoch、步号、全局步号、时隙起止时间、Master 调用数、任务数、reward |
| epochs.csv | 每个完整 epoch 的实际步数、累计/平均 reward、PPO 更新次数和指标 |
| config.json | 原默认环境参数及新增训练设置 |
| member_final.pt | 冻结选择前、Member 阶段最后的模型 |
| member_best.pt | 验证集选出的 Member，用于后续冻结 |
| latest.pt | 最近完成的 epoch 状态、优化器、RNG 和 DAG 流位置 |
| final.pt | 最后训练状态 |
| best.pt | 验证集选出的完整策略 |
| summary.json | 完成 epoch 数、总环境步数及预留测试段结果 |

## 画 epoch–reward 图

```powershell
.\.venv\Scripts\python.exe scripts/plot_epoch_reward.py --runs scripts/output/rl-epochs-run01 --output scripts/output/rl-epochs-run01/reward-plot
```

图同时显示每步平均奖励与 epoch 累计奖励，输出 PNG/SVG/PDF、源数据 CSV 和图注。
不足 5 个 epoch 时只画真实点，不生成平滑的“收敛曲线”。多于 5 点时同时保留原始
细线和最近 5 次的移动平均。阶段切换用虚线标出。

也能绘制旧实验的 reward：

```powershell
.\.venv\Scripts\python.exe scripts/plot_epoch_reward.py --runs scripts/output/rl-baseline scripts/output/rl-baseline-seed21 scripts/output/rl-baseline-seed42 --output scripts/output/rl-reward-legacy
```

旧图会明确标注每轮 8 步，禁止与新日志混在同一张图里。图中 Member→Master
阶段会从随机 Member 策略切换为冻结的确定性策略，reward 可能发生跳变。

## 续训和评估

```powershell
.\.venv\Scripts\python.exe scripts/train_rl_epochs.py --resume scripts/output/rl-epochs-run01/latest.pt --member-epochs 100 --master-epochs 60
.\.venv\Scripts\python.exe scripts/evaluate_rl_epochs.py --checkpoint scripts/output/rl-epochs-run01/best.pt
```

续训只增加累计阶段 epoch 目标。进入 Master 阶段后不能再增加 Member epochs；
每 epoch 步数和预留数据分段不能在续训时改变。其余设置沿用 checkpoint。

中断后从最近完成的 epoch 边界重放。尚未保存 checkpoint 的残余步骤会移入
`*.uncheckpointed-*.csv`，防止正式日志重复计步。新旧入口的 critic 结构不同，
新 checkpoint 用本页的评估入口；旧 checkpoint 继续使用旧入口。

本次只实跑一个 Member epoch 和一个 Master epoch，每个均为 500 个完整训练步。
验证/测试用 4/16 步加速接口验收，未把这个短验收实验宣称为收敛实验。
