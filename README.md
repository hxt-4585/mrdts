# MRDTS

多区域 UAV 的 DAG 子任务调度科研仿真项目。优先保证算法流程直观、便于阅读和修改；每种方法在一个目录内独立实现，允许不同方法之间重复代码。

旧组件组合框架、Random/PPO 实现、统一实验入口和训练日志可视化工具已移除。目前保留环境、诊断脚本和测试，没有完整方法的训练或评估入口。后续方法放在 `methods/<方法名>/`，详见[项目目录说明](docs/PROJECT_STRUCTURE.md)。

## 安装

使用 Python 3.11，在项目根目录执行：

```powershell
uv sync --locked
```

PyCharm 解释器选择 `.venv/Scripts/python.exe`。PyTorch 依赖与 CUDA 索引继续由 `pyproject.toml` 和 `uv.lock` 管理，供后续学习方法使用；当前环境仿真使用 NumPy。

## 目录

```text
env/                  区域、用户、UAV、通信、DAG、事件运行时与结算
config/*.toml         公共环境与工作负载参数
methods/              后续各方法的独立目录，当前仅保留说明
scripts/              环境检查、诊断可视化及少量辅助实现
tests/                环境与诊断脚本测试
specs/                系统、场景参数和 DAG 模型定义
docs/                 项目说明与文献调研
```

## 环境检查与可视化

```powershell
uv run python scripts/inspect_entities.py
uv run python scripts/visualize_region.py
uv run python scripts/visualize_user.py
uv run python scripts/visualize_uav.py
uv run python scripts/visualize_dag.py
uv run python scripts/visualize_scheduling.py
```

诊断与测试使用 `scripts/randomness.py`，默认 seed 为 42。`scripts/ers.py` 保留调度回放和环境测试所需的 ERS 排序实现，无需旧实验配置。详见[调度回放说明](scripts/scheduling_replay.md)。

## 验证与模型资料

```powershell
uv run python -B -m unittest discover -s tests
```

建模资料：[系统模型](specs/01_system_model.md)、[场景参数](specs/02_scenario_parameters.md)、[DAG 模型](specs/03_task_dag_model.md)。研究资料：[文献调研](docs/research/2026-09-04-rl-literature-and-framework.md)。
