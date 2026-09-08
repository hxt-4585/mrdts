# MRDTS

多区域 UAV 子任务调度仿真与方法比较。支持将子任务排序、Master 飞行策略、Member 子任务调度组合成完整方法。

后续对话或接手开发请先阅读[项目目录与开发说明](docs/PROJECT_STRUCTURE.md)，其中记录各目录职责、已实现与预留功能，以及固定位置和统一随机种子的实验规则。

## 安装与 GPU

```powershell
uv sync --locked
uv run python -m experiments.train --check
```

项目使用 Python 3.11。`pyproject.toml` 将 Windows/Linux 的 torch 显式绑定到官方 CUDA 12.6 索引，锁文件固定 `torch 2.10.0+cu126`；默认 PPO 配置使用 `cpu`；显式请求 GPU 失败时会报错，不会静默改用 CPU。
`--check` 实际执行矩阵运算和反向传播；不进行模型训练。显式 CPU 检查可用 `--device cpu --check`。
NumPy 仿真与启发式算法仍在 CPU 执行，GPU 用于 PyTorch 网络训练；PPO 默认配置显式使用 CPU。

配置依据：[uv 官方 PyTorch 集成说明](https://docs.astral.sh/uv/guides/integration/pytorch/)、[PyTorch 官方版本安装说明](https://pytorch.org/get-started/previous-versions/)。

## 两种启动方式

命令行：

```powershell
# 已迁移的 PPO 时延基线。
uv run python -m experiments.train --config config/experiments/ppo.toml

# 可立即运行的 Random 方案实验。
uv run python -m experiments.run --config config/experiments/random.toml
uv run python -m experiments.run --config config/experiments/random.toml --slots 2 --seed 7 --scheduling local
```

PyCharm：

1. 项目解释器选择本项目 `.venv/Scripts/python.exe`；先运行一次 `uv sync --locked`。
2. 打开 `experiments/train.py`，右键 Run，或点击文件底部 `if __name__ == "__main__"` 的运行按钮。
3. 检查默认 CPU 配置时填写 `--check`；检查 GPU 时填写 `--device cuda --check`。去掉 `--check` 即启动 PPO 训练。
4. 运行 Random 时打开 `experiments/run.py` 并填写 `--config config/experiments/random.toml`；评估 PPO 时填写 `--checkpoint <运行目录>/checkpoints/best.pt`。
5. 不传参数时读取 `config/experiments/ppo.toml`。可以直接修改此配置，也可在 Parameters 中填写 `--config config/experiments/xxx.toml`。

两种方式调用相同代码。默认配置、配置内部引用以及相对输出路径均以项目根目录解析，不要求 PyCharm 的 Working directory 恰好为项目根目录。

**当前训练能力：** `ppo` 已注册分阶段 PPO，奖励只优化 epoch 内的全局截断时延。默认 Member 100 epoch、Master 30 epoch，每 epoch 500 时隙。训练、续训、checkpoint 评估及日志说明见 [PPO 使用说明](methods/solutions/ppo/README.md)。无参数选择 PPO：`train.py` 启动训练，`run.py` 必须指定 checkpoint 才能评估。Random 无需训练，应显式选择其配置。

## 目录职责

```text
env/                              物理状态、约束、事件运行时与结算
methods/
  contracts.py                    三类组件与 Trainer 业务接口
  compose.py                      飞行→排序→批量放置→结算
  factory.py                      显式组件、完整方法和 Trainer 注册
  components/
    ordering/                     ERS 等排序组件
    flight/                       Master 飞行组件
    scheduling/                   Member 多 DAG 调度组件
  solutions/
    random/                       ERS + 随机飞行 + 随机调度
    ppo/                    已迁移的 PPO 时延基线
    proposed/                     自己的方法、观测、奖励、训练器落点
  learning/                       共享设备检查、网络、缓存和更新算法
config/
  *.toml                          现有公共仿真参数
  methods/                        完整方法和组件选择
  experiments/                    场景、种子、运行预算和训练参数
experiments/                      正式训练、评估、结果汇总与公共指标绘图
visualization/                    独立训练曲线、实时监控；与 results 同级
data/scenarios/                   可复用输入场景
results/                          每次运行的配置、指标、模型和图表
specs/                            建模与实验规范
docs/                             研究、设计与变更记录
tests/                            开发验证
scripts/                          开发检查与临时诊断
```

`methods/learning/` 已提供可复用的 PPO、GAE、Rollout、MLP 与设备工具；UAV 观测、动作规则和奖励保留在具体方案。

## 方法组合与扩展

`config/methods/random.toml`：

```toml
[method]
solution = "random"

[method.components]
ordering = "ers"
flight = "random"
scheduling = "random"
```

Random 组合：ordering=`ers`、flight=`random`、scheduling=`random`。另保留 stationary 飞行、owner/local 调度作为可替换组件，不再注册 baseline 完整方法。
实验 TOML 的 `[components]` 可以覆盖其中某一个组件。`--scheduling local` 提供同样的单项覆盖。

完整方法决定观测、信息可见范围与奖励/优化目标；`env` 只提供事实和物理执行。通用组件输入不是固定的 RL 张量。
飞行组件按原 Master 管辖的稳定 Member ID 返回动作；移动并更新用户关联后再排序和调度。每个 Member 处理自己全部 DAG，所有完整方案统一提交，再由仿真器计算真实完成时间。

现有 `ERS(runtime).plan(requests)` 保留算法数值与行为，入口迁移为 `methods.components.ordering.ers`；`priority_seq` 替代仿真契约和运行时中的 `ers_seq`。排序必须前驱先于后继。旧导入与字段名不保留兼容副本。

接入新方法：

1. 在对应 `components/` 目录实现算法，注册到 `methods/factory.py`；排序组件使用无参构造，飞行和调度工厂接收各自独立的 NumPy Generator。
2. 在 `solutions/<name>/` 实现完整方法与 `Solution` 适配器，将适配器加入 `methods/factory.py` 的 `SOLUTIONS`。方法可以使用 `CompositeMethod`，或定义自己的时隙流程。
3. 在方法目录实现观测/奖励。方法所需参数保留在方法配置字典中。
4. 需要学习时，实现 `Trainer.train(config, device)`，由适配器的 `create_trainer()` 返回。实验 `[training]` 参数通过 `config.training` 传入，训练器负责设备、模型和训练流程。
5. 训练器可复用 `experiments/artifacts.py` 存储配置和运行元数据，按需保存 checkpoints；冻结模块与重新训练的实验应使用不同配置和标识。

公共实验入口不判断方案名称。专用命令行参数、配置解析与模型检查由方案适配器提供；也可以用通用 `--training hidden=32` 覆盖配置。新增方案通常无需修改 `experiments/`，完整接口和例子见 [方案接入说明](methods/README.md)。

## 实验与结果

每次训练或评估创建唯一的 `results/<experiment>/runs/<method>/seed_<seed>/<UTC时间+ID>/`，包含：

- `config.json`：合并后的实验/方法配置，以及实际使用的每回合完整仿真参数。
- `metadata.json`：模式、代码版本、工作区状态、种子、开始结束时间及完成/失败状态。
- `metrics.csv`：逐时隙原始指标；失败后保留已写入的时隙。
- `summary.json`：DAG 总数、失败率、成功 DAG 平均时延、全部 DAG 截断平均时延、计算与发送能耗。

失败 DAG 在截断时延中按窗口长度计入，但不声称它已经在该时刻完成；全部失败时成功平均时延为 null。计算/发送能耗沿用当前物理计量，不包含尚未实现的 UAV 飞行能耗。
随机种子只在 `config/experiments/*.toml` 中设置，`--seed` 可覆盖本次运行的总种子。`experiments/randomness.py` 统一创建随机流，再注入区域、用户、DAG、随机飞行与随机调度组件。区域与用户的随机流不随回合变化，因此所有回合的区域划分和用户位置保持相同；每个回合重建相同初始场景，让 UAV 回到相同初始位置，回合内 Member UAV 连续移动。
DAG、飞行和调度的随机流由总 seed 与回合编号派生；DAG 在每个时隙继续随机生成，各回合不会重放同一组任务。各模块使用独立随机流，替换策略不会打乱 DAG 的随机序列。相同总 seed、场景参数和回合编号可用于不同方案的配对比较；修改总 seed 会改变初始场景和随机序列。
区域、用户和 DAG 的 TOML 与配置类均不含 seed，组件不会自行创建随机源。`Region`、`User` 要求传入 `rng`；`DAGGenerator` 要求传入 `rng` 与 `py_rng`；方法工厂要求传入飞行和调度的生成器。开发脚本也从默认实验配置创建随机流。新结果 `config.json` 只在 `experiment.seed` 保存总种子，场景参数不再包含模块种子；历史结果保留运行时的原始记录。PPO 使用 episode ID 0 验证、1 测试、2 起训练；后续多种子论文实验另行规定。

```powershell
uv run python -m experiments.aggregate results/random
# 将下方路径替换为 run.py 打印的本次运行目录。
uv run python -m experiments.plots <run_directory>
```

汇总输出 `<experiment>/analysis/comparison.csv`，每行保留一次已完成的独立评估，训练不参与混比；绘图输出该运行的 `figures/metrics.png`。所有生成产物默认不提交 Git。

## PPO 训练可视化

`visualization/` 与 `results/` 同级，只读取日志，不导入训练器。训练产物保存在各自运行的 `training/` 与 `checkpoints/`，图保存在 `figures/training/`。

```powershell
uv run python -m visualization.training --run <run_directory>
uv run python -m visualization.watch --run <run_directory>
```

支持 reward–epoch、actor/value loss、KL 与 entropy，离线导出 PNG/SVG/PDF 和源数据；实时监控可独立运行。后续时延、能量及能量队列图可继续放入该目录，见 [可视化说明](visualization/README.md)。

## 开发验证

```powershell
uv run python -B -m unittest discover -s tests
```

正式实验代码不导入 `tests/` 或 `scripts/`。现有开发可视化入口继续保留原用途。

## Random 飞行约束

随机飞行的两个归一化动作分量在 [-1, 1] 独立均匀采样；越界沿用环境的逐 Member 拒绝规则。造成有用户区域没有 Member 的联合提议重新采样，最多 128 次后保持原位。结果额外记录 `flight_resamples` 与 `flight_fallbacks`。随机调度在飞行后的合法候选节点中等概率选择。完整定义见 `methods/solutions/random/README.md`。
