# 环境代码结构

`env` 负责仿真，以及后续强化学习问题的定义；`methods` 负责排序、策略和学习算法。

## 当前已实现

```text
env/
├── simulator.py          # Simulator：拓扑、资源与时隙生命周期
├── contracts.py          # DAGRequest、PlacementDecision
├── types.py              # EntityKind、EntityRef、DirectedChannelKey、TaskKey
├── entities/             # Region、User、UAV、MasterUAV、MemberUAV
├── workload/             # DAG 数据结构与生成器
├── communication/        # 信道模型和路由规则
├── runtime/              # 事件推进、任务状态、计算/传输队列、结算
└── settings/             # 配置类；读取根目录 config/*.toml
```

依赖约束：

- `env` 不导入 `methods`。执行器接受公共数据契约，不负责选择策略。
- 基础标识仅在 `types.py` 定义，不在不同子包重复声明。
- 通信模型和路由不依赖运行时队列；队列归 `runtime` 管理。
- `Simulator` 协调生命周期；`SchedulingRuntime` 唯一管理时隙内调度状态。
- 各子包的 `__init__.py` 保持轻量，不集中导入所有实现。

## 导入迁移

本次为仓库内统一迁移，不保留旧平铺模块的兼容副本。已有外部脚本需同步更新导入；旧 Python 进程应重启。

| 旧入口 | 新入口 |
| --- | --- |
| `env.environment.Environment` | `env.simulator.Simulator` |
| `env.region`、`env.user`、`env.uav` | `env.entities` 下同名模块 |
| `env.dag_generator` | `env.workload.dag_generator` |
| `env.channel_model`、`env.routing` | `env.communication` 下同名模块 |
| `env.event_runtime`、`env.dag_runtime`、`env.task_runtime` | `env.runtime` 下同名模块 |
| `env.channel_queue`、`env.server_queue`、`env.slot_result` | `env.runtime` 下同名模块 |
| 队列/任务模块中的 `EntityKind`、`EntityRef`、`DirectedChannelKey`、`TaskKey` | `env.types` |
| `methods.contracts` 中的 `DAGRequest`、`PlacementDecision` | `env.contracts` |

```python
from env.simulator import Simulator
from env.entities.region import Region
from env.entities.uav import MasterUAV, MemberUAV
from env.entities.user import User
from env.communication.channel_model import ChannelModel
from env.contracts import DAGRequest, PlacementDecision
from env.types import EntityKind, EntityRef, TaskKey
from env.workload.dag_generator import DAGGenerator
```

`Simulator` 的构造参数、`begin_slot()`、`end_slot()` 等方法及其行为保持不变。现有脚本的启动命令不变，例如 `python scripts/visualize_scheduling.py`。

## 后续强化学习扩展（尚未实现）

后续在 `env/environment.py` 实现唯一公开的 `MRDTSEnv`，组合 `Simulator`，提供 `reset/step`。
状态、观测、动作、奖励及空间定义放在 `env/mdp/`，而不是顶层 `rl/`。
环境只维护一套真实仿真状态；神经网络、动作采样、轨迹缓存和 PPO 更新仍属于 `methods`。

本次不创建这些模块的空壳，不改变仿真参数、时隙推进、时延/能耗计算或 ERS 规则。

## 验证

在项目根目录运行：

```powershell
.venv/Scripts/python.exe -B -m unittest discover -s tests
```

`test_env_structure.py` 检查目录入口、公共契约和依赖方向；原有测试继续覆盖真实仿真行为。
