# Modular Methods and Experiment Entrypoints Implementation Plan

**Goal:** 在新分支建立可组合方法、正式实验目录、CUDA PyTorch 及命令行/PyCharm 共用入口。

**Architecture:** env 只管理仿真事实；methods/components 提供三类决策；solutions 定义完整方法与可选训练器；experiments 负责执行、评价和结果存储。

**Tech Stack:** Python 3.11、NumPy、unittest、uv、PyTorch CUDA 12.6。

## Constraints

- 当前分支 codex/modular-experiments；保留已有未跟踪研究笔记和诊断结果。
- 不改动现有物理模型、排序数值、队列规则和截止结算。
- 正式训练、运行、评价、绘图代码不依赖 tests/scripts。
- 未确定的 RL 算法不伪装成已经实现；训练器通过显式接口接入。
- 默认训练设备 cuda，不静默降级 CPU。
- 入口按项目路径读取默认配置和输出结果，不依赖启动工作目录。

## Tasks

- [x] 创建分支并运行原有 130 项测试。
- [x] 迁移 methods/ers.py 到 components/ordering，统一 priority_seq 命名并更新仓库调用方。
- [x] 在 methods/contracts.py 定义排序、飞行、Member 批量调度和 Trainer 协议；在 compose.py 按飞行、排序、批量放置、结算组织一个时隙。
- [x] 提供 stationary、local/owner 调度作为可运行的非学习参照，factory.py 解析组件组合。
- [x] experiments/config.py 解析实验和方法 TOML；scene.py 构建正式场景；runner.py 执行评估并保存完整配置、版本、原始逐时隙指标。
- [x] experiments/train.py 与 run.py 支持 python -m 和 PyCharm 直接运行，训练入口支持设备检查和后续 Trainer 接入。
- [x] 创建 results、data/scenarios 及算法保留目录的用途说明；添加汇总与绘图入口。
- [x] 配置 CUDA 12.6 PyTorch 显式索引、更新 uv.lock、同步并执行真实 CUDA 运算。
- [x] 验证两种启动方式、组件替换、种子可复现、输出不覆盖、未实现训练器的清晰错误以及全套回归。
- [x] 更新 README 和 env/specs 职责说明，检查 diff 与分支。

## Verification commands

```powershell
.venv/Scripts/python.exe -B -m unittest discover -s tests
uv sync --locked
uv run python -m experiments.run --slots 2
uv run python -m experiments.train --check
.venv/Scripts/python.exe experiments/train.py --check
git diff --check
```


## Verified results

- Branch: `codex/modular-experiments`; changes remain in the working tree, not committed or pushed.
- Full unittest suite: 137 tests passed.
- `uv sync --locked` and `uv lock --check` passed.
- Installed `torch 2.10.0+cu126`; CUDA 12.6 on NVIDIA GeForce RTX 4060 Ti.
- Module entry and direct file entry from outside the project both passed GPU matrix multiplication and backward checks.
- Baseline evaluation, unique output folders, repeatable seeds, component replacement, incomplete-placement rejection, aggregation and plotting verified.
- ERS AST unchanged after moving; physical runtime AST unchanged except generic priority names, documentation and validation wording.
- Production code does not import tests/scripts; git diff --check passed.
- Concrete RL training algorithms remain intentionally unimplemented. The training entry dispatches registered Trainers and reports a clear error when none exists; --check only validates configuration and device.
