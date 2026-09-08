# 新增方案

公共实验入口只调用 `methods/solution.py` 定义的 `Solution` 接口。新增方案通常只需新增方案目录、方法/实验 TOML，并在 `factory.py` 的 `SOLUTIONS` 增加一项；新增决策组件时再注册对应组件。无需修改 `experiments/` 或已有算法。

## 最小非学习方案

如果使用现有 `CompositeMethod` 时隙流程，在 `solutions/greedy/solution.py` 中：

```python
from methods.solution import Solution

class GreedySolution(Solution):
    pass
```

在 `factory.py` 注册 `SOLUTIONS['greedy'] = 'methods.solutions.greedy.solution:GreedySolution'`，然后通过方法 TOML 的 `components` 选择已注册组件。若只是改变现有方案的组件组合，无需新建方案类。

## 学习或联合优化方案

继承 `Solution` 并按需要覆盖接口：

| 接口 | 职责 |
|---|---|
| `add_arguments(parser, *, training)` | 向命令行注册专用参数；可直接复用通用 `--training KEY=VALUE` |
| `resolve_config(config, args, *, training, restored)` | 返回解析后的配置；处理本方案的阶段预算、模型评估默认值。`restored` 表示配置来自模型旁的运行记录 |
| `validate_components(components)` | 检查方案允许的组件组合 |
| `validate_config(config, *, training)` | 在创建结果目录前检查专用配置、模型格式、环境匹配及评估约束 |
| `build_method(config, ordering, flight, scheduling, *, checkpoint, device)` | `config` 为方法配置字典，返回支持 `run_slot(scene, workload)` 的完整方法 |
| `create_trainer()` | 返回支持 `train(config, device)` 的训练器；此处的 `config` 为完整实验配置 |

默认接口不支持 checkpoint 或训练，默认方法使用 `CompositeMethod`。学习方案自行实现模型加载，不能仅覆盖训练器后依赖公共入口猜测模型格式。PPO 的完整例子在 `solutions/ppo/solution.py`。

方法参数保存在方法 TOML 的 `[method]` 中，可由 `build_method` 读取。训练参数保存在实验 TOML 的 `[training]` 中。命令行先读取实验配置识别方案，再加载该方案声明的参数，最后解析整条命令。`--config ... --help` 显示所选方案参数；仅 `--help` 显示默认方案参数。

```powershell
uv run python -m experiments.train --config config/experiments/ppo.toml --member-epochs 2 --master-epochs 1
uv run python -m experiments.train --config config/experiments/ppo.toml --training member_epochs=2 --training master_epochs=1
```

`--training` 值按 TOML 解析，支持数字、布尔值、数组和带引号的字符串；同名值后者覆盖前者，方案专用快捷参数优先于通用覆盖。合法参数名与取值仍由方案校验。

## 结果、恢复与绘图

所有方案沿用 `results/<实验名>/runs/<方案_排序_飞行_调度>/seed_<seed>/<运行ID>/`。组件显示名称可在 `COMPONENT_LABELS` 注册；学习组件同时加入 `LEARNED_COMPONENTS`，避免误用于 Random 的启发式上下文。

训练器使用公共 `create_run` 保存配置和元数据，自行管理训练日志及模型。使用自动配置恢复时，checkpoint 应放在 `<run>/checkpoints/`，完整实验配置保存在 `<run>/config.json` 的 `experiment` 字段；可选 `resolved_training` 保存实际训练参数。其他布局可以显式传入 `--config`。

评估入口统一记录公共指标与模型文件哈希，调用方案的检查及方法构造，不解释 checkpoint 内容。训练与评估模式分开保存，公共汇总只统计完成的评估运行。

训练曲线使用 `visualization/README.md` 的日志约定，阶段由日志识别，可选指标缺失时不绘制对应曲线。新的物理量或新的图表类型仍可以按需要扩展可视化。
