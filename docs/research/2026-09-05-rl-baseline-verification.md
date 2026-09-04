# 全局时延 RL 基线：实现与实跑记录

日期：2026-09-05。分支：`codex/global-delay-rl-baseline`。

已完成仅新增文件的 Master/Member PPO 基线，使用默认环境配置。3 个 RL 随机种子
均完成 100 次 Member 更新和 30 次 Master 更新，并成功重载 checkpoint 评估。
训练回报与测试结果均有改善；这些实验不构成两层策略已充分收敛或全局最优的证明。

## 修改边界

新增 `methods/rl_baseline/`、训练/评估/绘图脚本、可选依赖文件、13 项测试和文档。
没有修改原有 `env/`、`config/`、ERS、测试、`pyproject.toml`、`uv.lock` 等跟踪文件。
CPU PyTorch 2.14.0 已装入项目 `.venv`。所有实验输出位于原规则已忽略的
`scripts/output/`。此前的文献调研文件没有包含在本次代码提交中。

环境保持 4 个 Master、12 个 Member、100 个用户，每时隙 100 个 10 节点 DAG。
原环境、用户和 DAG 的种子及所有物理参数都未更改。

## 奖励与实现

唯一任务奖励为负的全局平均 DAG 时延 / 时隙调度长度。未完成 DAG 按截止时长
1 秒计入，因此这是截止截断时延；不是所有 DAG 的最大完成时间，也不是只平均
成功任务。没有能耗、距离或额外失败惩罚。

Member 按本地 ERS 顺序逐子任务放置，整批原子提交至真实运行时；中间动作奖励
为 0，终局全局奖励形成无折扣 Monte Carlo return。Master 在 Member 冻结后训练，
使用实际时隙奖励和有限 episode 的 GAE。Actor 只读取本地输入，critic 集中训练。

初次均匀探索试验几乎全部超时，奖励接近常数。最终实现将 Member 的可学习本地
logit 偏置初始化为 4，所有合法候选仍保留正概率。这使初始确定性策略就是全本地，
不能把该初始表现算作学习收益。偏置参与后续 PPO 更新，未改变环境或奖励。

## 验证条件

- Python 3.11.15，PyTorch 2.14.0+cpu，每个训练进程 1 个 Torch 线程。
- RL 种子 7、21、42；每次 PPO 更新采集 8 个完整时隙。
- 每 10 次更新验证一次；默认 DAG 随机流前 4 个时隙用于验证，随后 16 个用于测试，
  训练从再后面的时隙开始。所有种子使用相同环境布局与数据分段。
- 测试每个策略覆盖 1600 个 DAG；这是同一默认布局内的未见工作负载评估，
  不是不同地图或自然发生大量迁移后的泛化评估。
- 每次完整训练加评估约 7.4–9.0 分钟；这是本机实测，不是其他设备的耗时承诺。

## 实测结果

| RL 种子 | 前 10 次 Member 更新平均时延 | 后 10 次平均时延 | 确定性测试时延 | 测试失败率 | 测试本地执行比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.586561 s | 0.488551 s | 0.492415 s | 0% | 91.49% |
| 21 | 0.577001 s | 0.495982 s | 0.547081 s | 0% | 99.37% |
| 42 | 0.572287 s | 0.462063 s | 0.450078 s | 0% | 83.75% |

同一测试段的全本地基线为 **0.552581 s、失败率 0%**；全卸载到关联 Member 的
基线为 **0.904347 s、失败率 49%**。三个策略相对全本地时延约降低 10.89%、1.00%、
18.55%。未按测试表现挑一个种子来代表全部实验。

前后窗口是随机采样策略的训练指标；测试列使用验证集选择的 checkpoint 和确定性
动作，两者不能直接当成同一条曲线。三个种子的学习速度和最终效果仍有明显差异。

Master 的独立增益很小：种子 7 的验证时延从 Master 初始化时的约 0.485731 s
到最优约 0.485557 s；另外两组最佳 Master 仍是初始化时的策略。因此当前证据主要
支持 Member 学习有效，不能宣称 Master 已学会有效的飞行协同。

本轮默认场景日志没有跨区事件。实际飞行跨区、归属刷新、候选变化和空区域联合
拒绝已由构造边界状态的集成测试覆盖，但迁移场景中的策略收敛尚未验证。

## 测试和审查

运行 `.venv/Scripts/python.exe -m unittest discover -s tests -q`：
**143 项测试全部通过**，包括原有 130 项与新增 13 项。新增覆盖：

- 默认规模全本地时延精确对齐、失败计入奖励、批次原子提交。
- 真正飞行跨区、ID 保持、掩码刷新、空区域联合拒绝。
- 不读取其他 Member 私有计划、训练/验证/测试随机流隔离。
- 合法动作概率、可训练初始化偏置、Master 归属概率掩码、GAE 终止边界。
- 真实 PPO 参数更新、Member 冻结、checkpoint 往返。
- 命令行训练、评估及断点续训。

独立代码审查未发现需修复的重要问题；另行运行了 5 项策略与训练测试。
三个最终 `best.pt` 都由独立评估命令重新加载，输出与各自 summary 一致。

## 使用与产物

完整说明：[README](E:/AppData/PycharmProjects/mrdts/methods/rl_baseline/README.md)。

```powershell
# 新实验使用一个尚未占用的目录。
.\.venv\Scripts\python.exe scripts\train_rl.py --output scripts/output/rl-run-01

# 重放已验证的默认种子 7 策略。
.\.venv\Scripts\python.exe scripts\evaluate_rl.py --checkpoint scripts/output/rl-baseline/best.pt

# 保留 Member 冻结阶段，继续训练 Master 到累计 60 次更新。
.\.venv\Scripts\python.exe scripts\train_rl.py --resume scripts/output/rl-baseline/latest.pt --member-updates 100 --master-updates 60
```

三个实验目录分别为 `scripts/output/rl-baseline`、`scripts/output/rl-baseline-seed21`、
`scripts/output/rl-baseline-seed42`，包含解析后的配置、CSV、模型、summary 和重载评估 JSON。

[训练曲线](E:/AppData/PycharmProjects/mrdts/scripts/output/rl-training-curves/training-curves.png)
另提供 SVG/PDF、源数据 CSV 和图注。曲线以原始日志为依据，粗线为最近 5 次更新
移动平均，淡线为未平滑时延；无置信区间或统计显著性声明。Python 导出结果已进行
可视检查，未发现标签裁切或重叠。
