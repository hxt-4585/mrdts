# Master / Member 全局时延 PPO 基线

这一版只新增文件，通过现有 Simulator 和 ERS 接口运行。没有修改 `env/`、
`config/`、`methods/ers.py`、原测试、`pyproject.toml` 或 `uv.lock`。

## 安装和运行

在项目根目录执行。现有 `.venv` 已安装 CPU 版 PyTorch；换机器时执行：

```powershell
uv sync
uv pip install --python .venv/Scripts/python.exe -r requirements-rl.txt --index-url https://download.pytorch.org/whl/cpu
```

PyTorch 是独立可选依赖，随后直接调用虚拟环境 Python，避免再次同步原锁文件时移除它：

```powershell
.\.venv\Scripts\python.exe scripts\train_rl.py
```

默认训练参数：Member 100 次 PPO 更新，Master 30 次，每次采集 8 个完整时隙，
每 10 次更新评估一次。CPU 单线程、MLP 两层各 64 单元、RL 随机种子 7。
这些是新增训练参数；环境配置全部从原来的六个默认 TOML 文件读取。

快速检查完整流程（仍然是默认的 100 用户 / 每时隙 1000 子任务）：

```powershell
.\.venv\Scripts\python.exe scripts\train_rl.py --member-updates 2 --master-updates 1 --rollout-slots 2 --eval-slots 1 --test-slots 1 --eval-every 1 --output scripts/output/rl-smoke
```

评估验证集选出的策略：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_rl.py --checkpoint scripts/output/rl-baseline/best.pt
```

断点续训，更新次数表示累计目标，不是追加次数：

```powershell
.\.venv\Scripts\python.exe scripts\train_rl.py --resume scripts/output/rl-baseline/latest.pt --member-updates 100 --master-updates 60
```

续训沿用记录的网络、采样、优化和随机数设置，只改变累计更新目标。进入 Master
阶段后 Member 已冻结，不能通过续训重新增加 Member 更新次数。需要另开一个实验
目录重新训练。输出目录已有内容时，新训练会拒绝覆盖。

## 环境和动作

| 项目 | 本版行为 |
| --- | --- |
| 场景 | 默认 4 个区域、4 个 Master、12 个 Member、100 个地面用户 |
| 工作负载 | 每用户每时隙一个默认 10 节点 DAG，原独立随机种子不变 |
| Member 动作 | 0=任务自身 Ground，1..12=固定 ID 的 Member，13=全局 BS |
| Member 掩码 | 自身 Ground、当前所属区域全部 Member、全局 BS |
| Master 动作 | 每个 Master 为当前归属自己的 Member 输出二维归一化飞行动作 |
| 飞行变换 | 高斯潜变量经 tanh 到 [-1,1]，调用原环境的分量速度规则 |
| 跨区 | 每次飞行后刷新归属、用户关联和候选掩码，物理 Member ID 不变 |
| 规划 | 按每个 Member 的 ERS 顺序自回归分配，全部完成后统一提交 |
| 执行 | 原运行时处理实际输入传输、依赖、中继、队列、多核和截止结算 |

空区域处理是在新增 adapter 内完成的：如果联合飞行动作会使有用户的区域没有
Member，整个飞行批次被拒绝，并在原位置继续调度。全局地图越界仍由原环境逐 UAV
拒绝。两种事件都记录指标，不添加额外奖励惩罚。可行跨区动作照常执行。

初始化位置沿用默认生成器。Member 训练期间固定位置；Master 训练期间跨多个
时隙连续移动，每次 PPO 采样开始时重置位置。队列仍按原环境在时隙结束后释放。

## 唯一任务奖励

对时隙内第 d 个 DAG：

```text
delay[d] = completion_time[d] - slot_start  # 截止前完成
delay[d] = slot_duration                    # 截止仍未完成
J = mean(delay[d]) / slot_duration
reward = -J
```

这是**全局平均 DAG 截止截断时延**，所有 DAG 等权。它不是所有 DAG 完成时间的最大值，
也不是只统计成功 DAG 的平均时延。现有环境在 1 秒截止后结束，无法知道失败 DAG
最终需要多久，因此不能声称优化了这些任务的未截断真实完成时间。

- 不使用能耗、距离、负载均衡、迁移惩罚或额外失败惩罚。
- 失败率单独记录；超时 DAG 已以 1 秒进入时延目标。
- Member 的一个时隙规划是一个 episode。中间放置动作奖励为 0，最终全局奖励
  回传为每个动作的 Monte Carlo return；内部不折扣，不提前运行任务。
- Master 每实际时隙获得一次相同的全局奖励，有限时隙 episode 使用 gamma=1、
  lambda=0.95 的 GAE，episode 末端 value 为 0。
- 没有启用局部 makespan 增量 shaping，避免第一版引入估计偏差与边界校正错误。

## 策略和训练

两类独立策略，同类参数共享。Member 使用候选打分 MLP，所有候选复用权重。
固定 ID 槽位只用于掩码及动作映射，因此区域内 UAV 数量变化不需要重建网络。
Master 使用固定全局 Member 槽位以及归属掩码。Actor 输入只包含本地可获得的信息。

Member 的候选特征含任务计算量、输入量、rank、依赖、计算时间、输入/依赖路径
传输时间及**本 Member 已规划的工作量**。后者是局部代理，不能解释为实际全局
队列。其他 Member 的私有放置计划不会进入 actor。

训练采用 CTDE：独立 critic 可读取全局位置、归属、用户计数和工作负载统计；
全局状态单独传给 critic，不会传给 actor。虽然仿真在一个 Python 进程内批量采样，
两个 actor 的接口仍可按各自局部观测调用。此版本不包含真实多机通信服务。

先训练 Member，再选择验证集时延最低的 Member 并冻结，之后训练 Master。
Master 更新不反向传播到 Member。还没有同时联合更新两类策略。

Member 的本地候选有一个初值为 4 的可学习 logit 偏置。初次实跑发现近乎均匀的
随机放置会使默认负载几乎全部超时，奖励接近常数，因此采用这个本地偏好的
初始化来改善探索起点。所有合法候选仍具有非零采样概率，偏置随 PPO 更新。
初始确定性策略因此就是全本地；评估不能把这一初始能力算成学习收益。

PPO 参数：Member actor LR 3e-4，Master actor LR 1e-4，critic LR 1e-3，clip=0.2，
每轮 4 epochs，minibatch=256，梯度范数上限 0.5，target KL=0.02，优势标准化。
熵系数 0.001 是策略探索正则，不属于仿真任务奖励。Master 使用高斯潜变量的
log probability；固定动作变换的 Jacobian 在新旧策略比值中抵消，熵项也是潜变量熵。

每轮 PPO 都保留采样当时的观测、动作、归属掩码和旧 log probability，不使用后续
迁移后的掩码重算历史动作概率。空 Member 不生成虚假放置动作；空 Master 不进入
actor 更新。

## 数据隔离和输出

默认 DAG 随机流的前 4 个时隙用于验证、接下来的 16 个用于最终测试；训练从
第 21 个时隙开始。配置中的默认 seed=20 没有改动。验证/测试每次重新建立相同的
默认场景并重放对应数据段。它们是同一默认空间布局中的未见工作负载，不是跨地图
泛化测试。设置 `--eval-slots` 或 `--test-slots` 会相应调整预留数据段。

结果默认在 `scripts/output/rl-baseline/`，该目录由原 `.gitignore` 规则忽略：

| 文件 | 内容 |
| --- | --- |
| config.json | 实際读取的环境参数、训练参数、PyTorch 版本和目标 |
| metrics.csv | 每轮训练时延/失败率/本地比例、PPO loss/KL/熵、定期验证时延 |
| validation_baselines.json | 验证集上全本地和全关联 Member 两个基线 |
| member_best.pt | 验证集选出的 Member，供 Master 阶段冻结使用 |
| latest.pt | 最新训练参数、优化器、Torch RNG、DAG 流位置，用于续训 |
| final.pt | 最后一次更新后的模型，不一定是最佳策略 |
| best.pt | 验证集选出的策略，元数据标明是否启用 Master 飞行 |
| summary.json | 完成更新次数及独立测试段上的策略与两项基线 |

绘制日志中的真实曲线（单次或多种子都可以）：

```powershell
.\.venv\Scripts\python.exe scripts\plot_rl.py --runs scripts/output/rl-baseline scripts/output/rl-baseline-seed21 scripts/output/rl-baseline-seed42
```

导出 PNG/SVG/PDF、合并源数据 CSV 和图注。训练曲线明确区分原始值与最近 5 次
更新的移动平均；验证曲线使用确定性策略，不将它与随机采样回报混为一谈。

只使用 `latest.pt` 续训；`best.pt` 用于推理。评估脚本默认读取预留测试段，并检查
当前默认环境参数与 checkpoint 记录一致。

## 首版能证明什么

自动测试覆盖真实默认规模的执行时延、时隙原子提交、合法跨区、空区域联合拒绝、
动作掩码、局部观测隔离、PPO 更新、冻结、checkpoint、命令行训练/评估/续训。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

默认配置下，全本地执行是很强的基线：计算量的配置上限允许每个 Ground 在
1 秒内完成自己的 DAG，且不涉及通信。训练策略可能学成全本地。此时 UAV 位置
对任务时延不再有影响，Master 缺乏可优化信号；保持默认配置就需要接受这一结果。
不能把“代码能跑”或单次验证曲线稳定解释为一般性收敛保证。

实现 API 依据：[PyTorch distributions](https://docs.pytorch.org/docs/2.14/distributions.html)、
[PyTorch serialization](https://docs.pytorch.org/docs/2.14/notes/serialization.html)。
研究背景见项目 `docs/research/2026-09-04-rl-literature-and-framework.md`；本 README
描述的是按“先跑通、仅时延、只新增文件”实际落地的简化版本。
