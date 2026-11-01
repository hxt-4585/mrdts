# MRDTS

多区域 UAV 的 DAG 子任务调度科研仿真项目。优先保证算法流程直观、便于阅读和修改；每种方法在一个目录内独立实现，允许不同方法之间重复代码。

当前已实现环境仿真、Master/Member 观测、Random 基线评估、诊断脚本和测试。`env/step/reward.py` 已预留 `MasterReward`（系统级时隙奖励，所有 Master 共享）和 `MemberReward`（子任务级）计算接口，公式尚未实现，奖励计算及返回接口尚未接入；当前没有学习方法的训练入口。方法位于 `methods/<方法名>/`，详见[项目目录说明](docs/PROJECT_STRUCTURE.md)。

## 目录

```text
env/                  区域、用户、UAV、通信、DAG、事件运行时与结算
config/               公共环境、工作负载及方法运行参数
methods/              Random 基线、ERS 排序和在线调度辅助循环
scripts/              环境检查、诊断可视化及少量辅助实现
tests/                环境、方法与诊断脚本测试
specs/                系统、场景参数和 DAG 模型定义
docs/                 项目结构与实现约定
```

## Random 基线评估

在项目根目录运行：

```powershell
uv run python methods/random/run.py
uv run python methods/random/run.py --episodes 1 --seed 42
```

运行参数默认读取 `config/methods/random.toml`，每回合步数读取 `config/env.toml` 的 `steps_per_epoch`（默认 500）。结果写入 `results/random/seed_<seed>/` 下的独立运行目录，包含配置、运行元数据、逐步/逐 DAG 指标及汇总。时延不含飞行阶段；失败 DAG 在截断平均时延中按调度窗口长度计入。

## 环境检查与可视化

```powershell
uv run python scripts/inspect_entities.py
uv run python scripts/visualize_region.py
uv run python scripts/visualize_user.py
uv run python scripts/visualize_uav.py
uv run python scripts/visualize_dag.py
uv run python scripts/visualize_scheduling.py
```

`MRDTSEnv` 按 `config/env.toml` 的 `env_seed` 生成固定场景，并按 epoch 派生 DAG 随机流；部分独立诊断和测试使用 `scripts/randomness.py`，默认 seed 为 42。ERS 实现在 `methods/random/ers.py`，`scripts/ers.py` 仅保留重导出入口。详见[调度回放说明](scripts/scheduling_replay.md)。

## 验证与模型资料

通信使用全网共享的连续带宽池，所有活动有向信道均分带宽，各发送设备的活动发送信道均分该设备总功率。假设理想正交、不同频带同时收发，忽略外部干扰、收发自干扰及重分配开销，不显式建模子载波。速率为 `b * log2(1 + p * g / (N0 * b))`。

`config/channel.toml` 的 `[bandwidth].total_mhz` 配置全网总带宽（默认 12 MHz）；`[propagation].noise_density_w_per_hz` 配置等效噪声谱密度（默认 `2.5e-20 W/Hz`，由旧噪声功率 `1e-13 W` 在 4 MHz 下换算）。原三类独立带宽和固定噪声功率字段已移除。设备总功率预算分别使用 `config/uav.toml` 的 `[member].transmit_power`、`[bs].transmit_power` 和 `config/user.toml` 的 `[population].transmit_power`。

每次活动传输变化时重新分配资源，先按旧速率和功率结算剩余数据与发送能耗，再更新完成时间预测。同一有向信道保持 FIFO，多跳仅在上一跳完成后激活。`TransferJob.predicted_finish_at` 表示预计完成，`finish_at` 仅在真实完成时填写；窗口截止不补算未发送的数据。窗口能耗统计仍只包含 Member UAV。ERS 使用独占资源参考时延排序，实际执行时长由运行时资源竞争决定。

```powershell
uv run python -B -m unittest discover -s tests
```

建模资料：[系统模型](specs/01_system_model.md)、[场景参数](specs/02_scenario_parameters.md)、[DAG 模型](specs/03_task_dag_model.md)。研究问题：[分布式 DAG 调度分析](specs/04_distribute_scenario_DAG_Task_scheduling.md)。
