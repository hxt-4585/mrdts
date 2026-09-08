# PPO 时延基线

迁移自恢复分支 `codex/global-delay-rl-baseline` 的 `c47059dee760e0095b7d135a856a114fa63e1a5e`。保留原学习逻辑，接入当前 Scene、随机流、ERS、放置契约和 Kbit 单位。

每时隙奖励为 `-mean(censored_DAG_delay) / H`，失败 DAG 按截止窗口 H 计入。epoch return 为所有真实时隙奖励之和。在时隙数、用户数和窗口固定时，最大化 return 等价于最小化整个 epoch 的 DAG 截断总时延。当前不增加能量、能量队列或违规惩罚。

默认先训练 Member 100 epoch，再冻结验证集选出的 Member，训练 Master 30 epoch。每 epoch 500 时隙，场景只在 epoch 开始重置，回合内连续移动。Member 每 8 时隙更新一次，Master 在完整 epoch 结束使用 gamma=1、lambda=0.95 的 GAE 更新。Member 阶段的 return 沿用旧版：每个调度决策共享所属时隙的最终全局 reward。

Member 使用共享的 23 维候选评分网络，初始化本地执行偏置为 4；Master 使用连续高斯潜变量、tanh 动作和稳定 Member ID 的 mask。actor 使用各自局部信息，critic 使用集中状态。非当前训练阶段的 actor 确定性执行，actor/critic 参数保持冻结。Master 每时隙均决策。

## 启动

在项目环境运行，或将相同参数填入 PyCharm 的 `experiments/train.py` 配置：

```powershell
uv run python -m experiments.train --config config/experiments/ppo_delay.toml

# 先做短程检查；这不是默认完整训练预算。
uv run python -m experiments.train --config config/experiments/ppo_delay.toml --member-epochs 30 --master-epochs 5 --slots 8 --eval-steps 8 --test-steps 16 --eval-every 5
```

此配置显式使用 CPU；可用 `--device cuda` 选择 GPU，设备不可用会报错。项目锁定的 CUDA 构建需要兼容驱动。

训练打印本次运行目录，下文 `<run>` 替换为该目录：

```powershell
# 自动读取该运行保存的配置；继续至指定的阶段总 epoch 数。
uv run python -m experiments.train --resume <run>/checkpoints/latest.pt --master-epochs 50

# 自动读取训练配置，默认在保留的 test episode 1 上运行一个回合。
uv run python -m experiments.run --checkpoint <run>/checkpoints/best.pt

uv run python -m visualization.training --run <run>
uv run python -m visualization.watch --run <run> --interval 2
```

续训只支持 `latest.pt` 的完整 epoch 边界，恢复网络、优化器、随机状态和模型选择状态，并备份、移除尚未提交到 checkpoint 的日志行。不支持中途时隙恢复。Member 进入 Master 阶段后不能继续增加 Member epoch。物理参数、seed、时隙数及学习设置必须一致；跨设备不保证逐位重现。若 checkpoint 被单独复制到其他目录，需提供原实验的 `--config`。

当前 PPO 独立评估只允许保留的一个测试回合（`--episodes 1 --episode-start 1`），可显式改变 `--slots`。验证 episode ID 为 0，最终测试为 1，训练从 2 开始，每个 epoch 使用下一 ID。评估入口拒绝混入验证或训练 ID；后续多测试回合实验需要先扩展划分协议。空间布局由同一 seed 固定，DAG 流按 ID 分开。不同方案配对评估必须使用相同 seed、物理配置、episode ID 和时隙数。验证/测试只是固定布局下的任务流划分，不代表对新布局的泛化验证。

## 输出

与 Random 共用路径 `results/<实验名>/runs/<方案_排序_飞行算法_调度算法>/seed_<seed>/<时间戳_ID>/`。PPO 使用 `ppo_delay_ers_ppo_ppo`，目录只写算法名称，不附加 Master/Member 角色：

```text
<run>/
  config.json, metadata.json     实际配置、代码与源基线版本、设备、划分
  metrics.csv, summary.json      公共逐时隙指标与汇总
  training/
    epochs.csv                  epoch return、mean reward、时延、阶段等
    updates.csv                 actor/value loss、KL、entropy、early stop
    validation.csv              固定验证集的确定性策略结果
    progress.json               完整 epoch 的训练进度
    initial_validation.json     未训练策略验证
    test.json                   best 模型的保留测试集结果
  checkpoints/
    latest.pt                   可续训的最新完整 epoch
    member_best.pt              验证集选出的 Member
    member_final.pt             切换 Master 前的 Member 末态（有 Master 阶段时）
    best.pt                     验证集选出的最终组合，用于推理
    final.pt                    训练末态
  figures/training/             可视化命令生成的图与 source.csv
```

`summary.json` 汇总该运行采集的训练时隙，最终模型表现查看 `training/test.json` 或独立评估运行。`experiments.aggregate` 只汇总已完成的评估运行，避免将探索训练与确定性评估混比。训练 reward 与验证 reward 分开显示；进入 Master 时 Member 从随机采样变成确定性执行，曲线可能发生跳变。验证总 return 只与同样时隙数的训练总 return 同图比较。

## 代码边界

`networks.py`、`observations.py`、`reward.py` 定义本方案；`method.py` 统一训练与推理执行；`rollout.py` 负责采样和阶段更新；`trainer.py` 负责训练生命周期；`settings.py`、`checkpoint.py`、`logs.py` 分别负责配置、状态保存和日志。

`methods/learning/` 的 PPO、GAE、Rollout 和 MLP 可供其他方法复用。`components/flight/ppo.py` 与 `components/scheduling/ppo.py` 是张量推理组件，由本方案提供观测；其上下文不同于 Random 的通用启发式上下文，工厂目前只允许 `ers / ppo_master / ppo_member` 的完整 PPO 组合。

旧版 KB 特征与当前 Kbit 不同，旧 checkpoint 不直接加载；迁移格式记录为 `kbit_23_v1`。实际验证记录见 `docs/research/2026-09-08-ppo-delay-migration-verification.md`。
