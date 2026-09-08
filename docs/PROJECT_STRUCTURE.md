# 项目目录与后续开发说明

更新日期：2026-09-08。

本文用于新对话或新开发者接手项目时了解目录职责、方案组织方式和已确认的实验行为。运行命令与安装说明见[根 README](../README.md)。参数以当前 TOML 为准，代码实现状态以实际文件及注册表为准；历史计划不代表当前实现。

## 1. 项目目标与当前状态

MRDTS 是多区域 UAV 的 DAG 子任务调度仿真与方法比较项目。一个完整方法通常由三个模块构成：子任务排序、Master UAV 飞行策略、Member UAV 子任务调度。需要支持多个文献 baseline、自提方案，以及组件替换实验；方法可以使用强化学习、启发式或优化算法。

目前可运行的完整方法为 `random`（ERS + 随机飞行 + 随机调度）和 `ppo`（ERS + 连续 Master PPO + 共享 Member PPO）。后者已接入训练、续训和模型评估，只优化全局截断时延。`proposed` 仍为预留目录。

`baseline` 是对比方法的统称，不是只容纳一个方法的固定名称。每个正式方案在 `methods/solutions/` 中拥有自己的目录，与 `random/`、`proposed/` 同级。旧的 `baseline` 方案和 `scheduling_ablation` 示例配置已经移除。

## 2. 目录总览

```text
mrdts/
├── README.md                     安装、启动、配置和常用操作
├── pyproject.toml                Python 依赖与 uv 的 GPU torch 索引
├── uv.lock                       依赖锁文件
├── .python-version               项目 Python 版本
├── .gitignore                    本地环境、缓存和生成产物的忽略规则
├── config/                       所有业务配置
│   ├── region.toml               地图几何、区域划分参数
│   ├── user.toml                 用户分布、用户设备计算与发送参数
│   ├── uav.toml                  Master/Member、基站、飞行与计算参数
│   ├── dag.toml                  DAG 拓扑、计算量和数据量范围
│   ├── channel.toml              带宽、信道增益、噪声与传播参数
│   ├── scheduling.toml           每时隙的任务传输与计算截止窗口
│   ├── methods/                  每个完整方法采用的方案与组件配置
│   └── experiments/              实验总 seed、回合数、时隙数、输出等
├── env/                          方法无关的物理仿真
│   ├── entities/                 区域、用户、Master/Member UAV 和基站实体
│   ├── workload/                 DAG 拓扑与任务特征生成
│   ├── communication/            通信速率、链路与路由
│   ├── runtime/                  事件推进、传输/计算队列与时隙结算
│   └── settings/                 将公共 TOML 读取为类型化配置
├── methods/                      算法组件与完整方案
│   ├── components/               可被多个方案复用的三个模块
│   │   ├── ordering/             子任务排序，目前包含 ERS
│   │   ├── flight/               Master 给 Member 下发飞行动作
│   │   └── scheduling/           Member 为子任务选择执行节点
│   ├── solutions/                每个完整方案各占一个目录
│   │   ├── random/               已实现：ERS + 随机飞行 + 随机调度
│   │   ├── ppo/            已实现：观测、奖励、采样、分阶段 PPO 训练器
│   │   └── proposed/             预留：自提方案及其观测、奖励和训练器
│   └── learning/                 可复用的学习支持代码
│       ├── networks/             通用 MLP 与 ValueNetwork
│       ├── buffers/              PPO 批量 Rollout 数据容器
│       └── algorithms/           裁剪 PPO 与完整回合 GAE
├── experiments/                  正式评估、训练入口、记录和分析
├── data/
│   └── scenarios/                预留：可复用场景与任务实例输入
├── visualization/                独立训练绘图与实时监控，后续扩展分析图
├── results/                      正式实验输出，按实验与运行隔离
├── specs/                        系统、场景参数和 DAG 建模规范
├── docs/                         项目说明、研究资料和设计记录
│   ├── PROJECT_STRUCTURE.md      本文：后续对话的项目结构入口
│   ├── research/                 文献调研、框架讨论与研究笔记
│   └── superpowers/
│       ├── specs/                历史设计文档
│       └── plans/                历史实施计划及验证记录
├── tests/                        开发期间的自动化测试
└── scripts/                      开发期间的检查、可视化与诊断
    ├── templates/                开发用调度回放 HTML 模板
    └── diagnostics/              开发用诊断与回放材料
```

本地还可能出现 `.git/`（版本管理）、`.idea/`（PyCharm 配置）、`.venv/`（Python 环境）、`.uv-cache/`（依赖缓存）、`.claude/skills/`（本地工具技能）和 `__pycache__/`（Python 缓存）。这些不属于业务架构。`scripts/output/` 是开发脚本运行时可能生成的输出目录。

根目录 `handoff.md` 是被 Git 忽略的本地历史交接记录，不能据其中过时的路径或分支状态判断当前项目。

## 3. 各层职责与关键文件

### env：只维护仿真事实与执行规则

| 文件/目录 | 职责 |
|---|---|
| `env/simulator.py` | 组织 `begin_slot/end_slot`，应用飞行、刷新拓扑与用户关联、建立和结束当期运行时 |
| `env/types.py` | 统一实体类型、实体引用和任务标识 |
| `env/contracts.py` | DAG 请求和子任务放置决策等环境交互数据 |
| `env/entities/` | 生成区域和用户，初始化 UAV/基站，维护位置与物理资源 |
| `env/workload/dag_generator.py` | 接收外部随机流，生成 DAG 拓扑、节点和边的数据 |
| `env/communication/` | 计算通信能力和传输路径 |
| `env/runtime/` | 根据依赖关系、放置决策、队列和资源执行任务，统计完成/失败、时延及能耗 |
| `env/settings/` | 读取物理与工作负载参数，不包含模块 seed |

环境提供原始状态和执行结果。RL 的观测张量、状态编码、奖励、信息可见范围，以及优化方法的目标函数由完整方法组织，放在 `methods/solutions/<方案>/`。不要把这些固定到 `env` 中，否则会限制非 RL baseline。

### methods：区分组件与完整方案

| 文件/目录 | 职责 |
|---|---|
| `methods/contracts.py` | 排序、飞行、调度的接口和上下文，以及 Trainer 接口 |
| `methods/compose.py` | 通用组合流程、组件输入、决策完整性和合法性检查 |
| `methods/factory.py` | 注册排序/飞行/调度组件、完整方案和训练器，负责实例化 |
| `methods/components/ordering/ers.py` | ERS 排序算法；`adapter.py` 将其接入组件接口 |
| `methods/components/flight/` | `random` 随机飞行、`stationary` 原地停留、`ppo_master` 张量策略组件 |
| `methods/components/scheduling/` | `random` 随机合法节点、`local` 用户本地、`owner` 所属 Member、`ppo_member` 张量策略组件 |
| `methods/solutions/random/method.py` | Random 的完整方案组合及飞行可行性筛选 |
| `methods/solutions/proposed/` | 自提方案未来实现位置，目前仅说明文件 |
| `methods/solutions/ppo/` | 本方案观测、网络、奖励、统一执行与训练生命周期 |
| `methods/learning/` | 可复用的 PPO、GAE、Rollout、MLP 与 CPU/CUDA 检查 |

`local.py` 等组件文件的存在，不代表存在名为 Local 的完整 baseline。正式新增方案应同时具有完整方案实现、注册和配置。组件替换接口可以保留，但不要将未命名、未实现的组合描述成已有正式方案。

当前通用时隙流程是：应用飞行并刷新关联 → 对新 DAG 的子任务进行 ERS 排序 → 各 Member 选择子任务执行位置 → 合并所有 Member 决策一次提交 → 环境推进事件并结算。三个模块的分类顺序不等于实际调用顺序。

### experiments：统一启动与实验管理

| 文件 | 职责 |
|---|---|
| `experiments/run.py` | 非学习方法和已接入方法的正式评估入口，支持模块方式及直接运行 |
| `experiments/train.py` | 训练器入口；`--check` 检查配置和设备，实际训练依赖已注册 Trainer |
| `experiments/cli.py` | 解析共用命令行参数及覆盖项 |
| `experiments/config.py` | 定义实验配置、解析项目相对路径；`DEFAULT_CONFIG` 指定默认实验 |
| `experiments/randomness.py` | 根据唯一实验总 seed 创建各模块随机流 |
| `experiments/scene.py` | 加载场景参数、接收随机流并构建实体与 DAG 生成器 |
| `experiments/runner.py` | 按回合和时隙执行方法，管理运行状态和结果记录 |
| `experiments/artifacts.py` | 创建唯一结果目录，保存配置、代码版本和运行元数据 |
| `experiments/metrics.py` | 从环境结果计算方法无关的评价指标 |
| `experiments/aggregate.py` | 汇集完成运行的结果至比较 CSV |
| `experiments/plots.py` | 读取运行指标并绘图 |

正式算法、训练循环、评估流程、论文实验分析和结果均不放进 `tests/` 或 `scripts/`。这两个目录仅用于代码开发的测试与诊断，正式代码不依赖它们。

## 4. 已确认的回合、位置与随机规则

代码中使用 `episode` 表示回合，`slot` 表示每回合的一步/时隙。Random 是评估运行，没有训练 epoch 或参数更新。PPO 一个 epoch 对应一个完整物理回合，Member buffer 更新不重置场景；验证 ID 0、测试 ID 1、训练 ID 2 起。

1. 所有回合使用相同区域划分与用户位置，用户在回合内也不移动。
2. 每回合 UAV 重置到相同初始位置；Member UAV 在该回合内按策略连续移动，不继承上一回合的最终位置。
3. 当前代码通过相同空间随机流重建一致的区域与用户，因此位置一致；不要求复用同一个 Python 对象。
4. 每个用户每个时隙生成一个新 DAG。任务流随回合变化，各回合不会重新播放同一条随机序列；随机抽样不保证每个 DAG 在数学上都唯一。
5. 只有 `config/experiments/*.toml` 配置总 seed，命令行 `--seed` 可以覆盖本次运行。公共 TOML、环境配置类及方法组件均不设置独立 seed。
6. 实验层内部仍创建相互独立的随机流，以免策略多抽取一次随机数就改变后续 DAG。同一总 seed、配置与代码可复现整个实验；改变总 seed 也会改变初始场景。
7. 各时隙在 `config/scheduling.toml` 指定的调度窗口截止，当前值为 1 秒。截止时未完成全部子任务的 DAG 判为失败，剩余工作终止，不进入下一时隙。

回合数、每回合时隙数、用户数、DAG 节点数等可变参数应直接查看当前实验 TOML，本文不固定这些数值。

## 5. 配置、启动与切换方案

```text
config/experiments/random.toml
    └── method_config 引用 config/methods/random.toml
            ├── solution 选择已注册的完整方案
            └── components 选择排序、飞行、调度组件
```

当前默认训练入口（PPO，CPU）：

```powershell
uv run python -m experiments.train
```

选择某个已实现方案的实验配置：

```powershell
uv run python -m experiments.run --config config/experiments/random.toml
```

不传 `--config` 时读取 `experiments/config.py` 中的 `DEFAULT_CONFIG`。默认配置为 `config/experiments/ppo.toml`。PyCharm 使用项目 `.venv/Scripts/python.exe`，直接运行 `experiments/train.py` 启动 PPO 训练；`run.py` 评估 PPO 需指定 checkpoint，运行 Random 需显式选择其配置；需要切换方案时在运行参数中填写 `--config ...`。需要训练的方法通过 `train.py` 启动，PPO 已注册训练器，使用 `--config config/experiments/ppo.toml`。

GPU torch 的依赖来源由 `pyproject.toml` 和 `uv.lock` 管理。NumPy 环境仿真和当前 Random 在 CPU 执行，PyTorch GPU 支持用于学习方法；`train.py --check` 成功不代表实现了训练算法。

## 6. 输入数据与结果放在哪里

`data/scenarios/` 用于今后保存可复用输入场景和任务实例，目前只有说明文件，尚未接入读取固定数据集的实验流程。当前场景和任务由程序生成。

正式结果的路径结构为：

```text
results/<实验名>/
├── runs/<方案_排序_飞行_调度>/seed_<总seed>/<时间戳和唯一ID>/
│   ├── config.json               实验/方法配置与场景参数
│   ├── metadata.json             运行状态、代码版本、时间等
│   ├── metrics.csv               逐时隙指标
│   ├── summary.json              本次运行的汇总指标
│   └── figures/                  调用绘图入口后生成
└── analysis/                     调用汇总入口后生成比较 CSV
```

PPO 已在同一运行目录保存 `checkpoints/` 与 `training/`（epochs、updates、validation、progress、test），图输出到 `figures/training/`。模型选优只使用验证集；独立 PPO 评估只允许保留的测试 ID 1。汇总工具只收集已完成的评估运行，排除训练。新生成的 `config.json` 只在 `experiment.seed` 记录总 seed；旧结果保留当时的原始配置，不重写成新规则。

当前公共指标包括 DAG 完成/失败、成功 DAG 平均时延、失败按窗口长度计入的截断平均时延、计算/传输能耗，以及飞行违规、重采样和回退次数。计算/传输能耗包含失败任务在截止前的消耗；尚不包含 UAV 飞行能耗。

## 7. 后续改动应放在哪里

| 需求 | 修改位置 |
|---|---|
| 新增排序、飞行或调度算法 | 对应 `methods/components/` 子目录及 `methods/factory.py` |
| 新增文献 baseline 或自提完整方案 | `methods/solutions/<名称>/`、注册表、方法配置、实验配置 |
| 专用训练参数、配置解析、模型加载或测试集检查 | 方案目录中的 `solution.py` 及其调用的模块；公共入口只调用 `Solution` 接口 |
| 为 RL 方案定义观测、奖励、训练器或库适配器 | 对应 `methods/solutions/<名称>/` |
| 复用网络、经验缓存、学习更新逻辑 | `methods/learning/` 对应子目录 |
| 调整物理模型、通信规则、事件执行与结算 | `env/` 对应子模块，并同步 `specs/` |
| 修改用户数量、运行预算、总 seed 或默认方案 | 实验 TOML；默认配置路径在 `experiments/config.py` |
| 修改公共评价指标或结果格式 | `experiments/metrics.py`、`artifacts.py` 等 |
| 新增正式汇总与绘图 | 汇总放 `experiments/`，训练与后续分析绘图放 `visualization/`，输出到运行的 `figures/` |
| 自动化回归验证与临时诊断 | `tests/` 与 `scripts/` |

接手项目时，先阅读本文和根 README，再查看目标方案配置、`methods/factory.py` 及相关实现。修改前运行 `git status` 和 `git branch --show-current` 确认当前工作区，不依据历史交接记录推断分支或提交状态。目录职责或实验行为变化时同步本文；研究讨论、旧设计和旧结果应保留其历史语义。

新增方案的完整流程见 [方案接入说明](../methods/README.md)，公共方案接口为 `methods/solution.py`。`experiments/cli.py` 通过所选方案声明专用参数，`train.py` 和 `runner.py` 通过方案接口检查和启动训练/评估，不包含 PPO 分支。

PPO 的源码来源、分阶段策略、输出字段和续训约束见 [方案说明](../methods/solutions/ppo/README.md)，共享学习接口见 [learning 说明](../methods/learning/README.md)。PPO 推理组件接受本方案的张量上下文，与 Random 的启发式上下文不同；当前不允许直接混搭。
