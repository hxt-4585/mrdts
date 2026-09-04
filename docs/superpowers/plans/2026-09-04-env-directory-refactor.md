# Environment Directory Refactor Implementation Plan

**Goal:** 按已确认的目录设计重组现有仿真代码，为后续统一的 RL 环境入口保留位置，不新增 RL 行为。

**Architecture:** 仿真实体、工作负载、通信与运行时分别归类到 env 子包。基础标识和仿真输入契约属于 env，methods 只依赖这些契约；现有 Environment 改名 Simulator，职责与方法签名不变。

**Tech Stack:** Python 3.11、NumPy、现有 unittest 测试；不新增依赖。

## Constraints

- 保留所有时延、能耗、ERS、路由、队列、跨域和时隙结算行为。
- env/settings 与 config 保持原位，配置路径不变。
- 仓库内调用方统一迁移，不保留旧平铺模块的兼容副本；外部脚本需要更新导入。
- 不创建未实现的 MRDTSEnv、reset/step、env/mdp 或 MAPPO 空壳。
- 测试继续平铺，避免本次目录任务顺带改变 unittest 发现和测试夹具路径。
- 历史设计/计划保留原始路径；当前 specs 与脚本说明同步更新。
- 在当前工作区分步执行；不推送远端。

## Task 1: Extract shared contracts

- [x] 确认工作区状态并运行 `.venv/Scripts/python.exe -B -m unittest discover -s tests`。
- [x] 新增 `tests/test_env_structure.py`，断言基础标识来自 env.types，仿真输入来自 env.contracts，env 不导入 methods。先确认测试失败。
- [x] 将 EntityKind、EntityRef、DirectedChannelKey、TaskKey 原样提取到 env/types.py；DAGRequest、PlacementDecision 原样提取到 env/contracts.py。
- [x] 更新所有 Python 调用方的上述符号导入，保留 methods/contracts.py 中方法协议及相关数据类。
- [x] 运行新增测试和完整测试。

## Task 2: Move modules and update consumers

迁移清单：

| 原文件 | 新文件 |
| --- | --- |
| env/region.py、user.py、uav.py | env/entities/ 下同名文件 |
| env/dag_generator.py | env/workload/dag_generator.py |
| env/channel_model.py、routing.py | env/communication/ 下同名文件 |
| env/event_runtime.py、dag_runtime.py、task_runtime.py | env/runtime/ 下同名文件 |
| env/channel_queue.py、server_queue.py、slot_result.py | env/runtime/ 下同名文件 |
| env/environment.py | env/simulator.py，Environment 改为 Simulator |
| tests/test_environment.py | tests/test_simulator.py |

- [x] 扩展结构测试，检查新包及 Simulator 入口，先确认测试失败。
- [x] 使用 apply_patch 迁移文件、建立有实际内容的包，并更新 env、methods、scripts、tests 的导入。
- [x] env/__init__.py 保持轻量，不提前导入仿真运行时；Simulator 使用明确的模块路径导入。
- [x] 新增 env/README.md 说明职责、导入迁移以及未来 mdp 边界；更新当前 specs 路径和类名。
- [x] 完整测试通过，正常/过载两组回放 JSON 摘要与迁移前一致。
- [x] 检查旧路径残留、导入顺序、git diff --check，审查最终 diff。

## Verification Results

- 迁移前：125 项测试通过；公共契约提取后：128 项通过；最终：130 项通过。
- 正常与 10 倍负载回放的完整 JSON SHA-256 均与迁移前完全一致。
- 53 个 Python 文件（包含提取后的公共类型/契约）排除导入、类名和说明调整后，代码主体一致。
- 全新进程先导入仿真不加载 methods；先导入 methods 再导入仿真也成功。
- 活跃 Python 代码无旧模块导入或 Environment 类名残留；git diff --check 通过。
- 修改保留在当前工作区，未提交、未推送。
