# 项目目录与实现约定

更新日期：2026-11-01。本文按当前代码整理；实现状态与文档不一致时，以代码为准。

本项目是个人科研仿真代码，以理解算法、修改方案和复现实验为主。默认配置、输入数据和内部调用顺序正确；不添加类型、形状、有限值、范围、调用顺序或组件兼容性的防御性校验，也不另建统一校验层。保留计算所需的飞行边界、候选节点、前驱依赖、完整执行结算等模型逻辑。

## 当前状态

环境已实现固定场景生成、epoch 重置、飞行与用户关联、DAG 生成、Master/Member 观测、事件调度、完整执行结算及能量虚拟队列更新。`env/step/reward.py` 已预留 `MasterReward.compute()` 和 `MemberReward.compute()`；公式尚未实现，调用会抛出 `NotImplementedError`，环境也尚未接入奖励调用或返回字段；不能把任务指标和能耗结算视为已经实现奖励。

`methods/random/run.py` 已提供 Random 基线评估入口，执行随机飞行、ERS 排序和逐次随机放置，输出时延、失败率与飞行违规统计。当前没有 PPO 或其他学习方法的训练入口。旧组件组合框架、统一实验入口及训练日志可视化工具已移除。

## 目录职责

| 位置 | 当前内容 |
|---|---|
| `env/env.py` | `MRDTSEnv` 场景生成、重置、分阶段 step、观测调用与能耗结算 |
| `env/step/observation.py` | Master 的 `(3M,)` 与 Member 的 `(6M+8,)` 观测及信息可见范围 |
| `env/step/reward.py` | Master 系统级时隙奖励（所有 Master 共享）和 Member 子任务级奖励框架，公式未实现、未接入 |
| `env/types.py`、`env/contracts.py` | 实体和任务标识、DAG 请求、放置决策及 step 返回结构 |
| `env/entities/` | 区域、用户、UAV 与基站状态及资源 |
| `env/workload/` | DAG 拓扑、计算量和数据量生成 |
| `env/communication/` | 通信速率、链路和路由 |
| `env/runtime/` | 事件推进、传输/计算队列、任务依赖与完整执行结算 |
| `env/settings/`、`config/*.toml` | 公共物理环境、工作负载和环境生命周期配置 |
| `methods/random/run.py` | Random 基线的评估、策略随机流和结果输出 |
| `methods/random/ers.py` | ERS 排序实现 |
| `methods/scheduling.py` | 基于底层 runtime 的在线策略调用循环 |
| `config/methods/random.toml` | Random 运行种子、回合数和输出目录，可由命令行覆盖 |
| `scripts/` | 环境检查、诊断可视化与调度回放；`ers.py` 仅重导出方法侧 ERS |
| `tests/` | 环境、方法及诊断脚本的回归验证 |
| `specs/` | 系统模型、参数与研究问题分析 |

M 为全网 Member UAV 数量，不包含 BS。环境提供观测编码并限制可见信息；方法负责任务排序、调用观测、选择飞行和执行位置动作，以及训练、网络、优化和评估。奖励框架位于环境目录，具体公式及事件结算接入尚待实现，不再沿用“观测和奖励必须放在方法目录”的旧约定。

每个方法优先在 `methods/<方法名>/` 中独立实现，允许不同方法之间复制相似逻辑。复用现有代码和数据结构，不要求建立组件注册表或通用训练框架。正式方法不导入 `tests/` 或 `scripts/`。

## 环境调用顺序

1. 用各配置对象构造 `MRDTSEnv(settings)`，调用 `generate_scene()` 生成场景。
2. 每个 epoch 调用 `reset(epoch_index=...)`，恢复 Member 初始位置、区域及数量，清零能量队列与步数，重建 DAG 随机流。
3. 读取 `observe_master(master_id)`，构造 `(M, 2)` 飞行动作，调用 `begin_step(flight_actions)`。调用方保证飞行后有用户的区域仍有 Member。
4. 根据返回的 DAG 请求和可决策 Member 选择前驱已完成的子任务，按需读取 `observe_member(task_key)`。同一时刻先完成全部观测和动作选择，再统一提交 `submit_schedule(placements)`。
5. 重复上一步，直到返回的可决策 Member 元组为空，再调用一次 `end_step()` 完成完整执行结算。
6. 根据 `EndStepResult.epoch_done` 判断是否结束本 epoch，否则继续下一步。返回结果包含 DAG 结果、逐 Member 能耗和能量队列快照，当前不包含奖励。

`methods/scheduling.py` 提供的是直接使用 `SchedulingRuntime` 的循环；Random 和回放脚本当前各自通过 `MRDTSEnv.submit_schedule()` 实现循环。两者都保持每个 owner Member 至多一个活动子任务、同刻动作统一提交的语义。

## 随机性与实验结果

`config/env.toml` 配置 `env_seed`、飞行时间、调度窗口长度及 `steps_per_epoch`。`generate_scene()` 从 `[env_seed, 0]` 派生区域和用户随机流；`reset(epoch_index=...)` 从 `[env_seed, epoch_index]` 派生 DAG 的 NumPy 与 Python 随机流。重置不重新生成区域和用户，同一 seed 与 epoch 可复现任务序列。

`Region`、`User` 和 `DAGGenerator` 仍接收外部随机流，由 `MRDTSEnv` 负责创建并传入。部分独立诊断和测试使用 `scripts.randomness.RandomStreams`（默认 seed 为 42）；环境自身不依赖该辅助模块。

Random 的运行 seed 覆盖环境 `env_seed`，方法侧另按 seed 和 episode 派生飞行与放置随机流。当前评估每个 episode 以相同空间种子重建场景，再按 episode 调用 reset；每回合步数读取 `steps_per_epoch`。比较方法时应记录一致的场景参数、任务负载和环境种子。

Random 默认将结果写入 `results/random/seed_<seed>/<时间戳与唯一后缀>/`，包括 `config.json`、`metadata.json`、`metrics.csv`、`dags.csv` 和成功完成后的 `summary.json`。DAG 时延为 `completion_time - slot_start`，允许超过名义时隙，当前不统计超时；现有 `truncated_delay_s` 字段对正常完成 DAG 同样保存完整时延。上述统计指标不是强化学习奖励。

## 接手与验证

先阅读本文和根 README，再查看目标方法目录或环境文件。修改前检查 `git status` 和当前分支，保留已有工作区修改。目录职责变化时同步本文；不为每次修改另建计划、交接或完成报告。

在项目根目录运行 `uv run python -B -m unittest discover -s tests`。基线评估、环境诊断与回放命令见[根 README](../README.md)。
