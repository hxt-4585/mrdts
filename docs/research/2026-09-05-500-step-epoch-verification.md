# 500 步 epoch 验收记录

用户要求：提供 epoch–reward 图；每个 epoch 500 步，每步都由 Master 指挥和 DAG
任务调度构成。本次保持只新增文件，原有入口及旧实验数据保留。

新增入口：`scripts/train_rl_epochs.py`。默认 `--steps-per-epoch 500`，PPO 内部遍数
另用 `--ppo-epochs 4`。epoch 内 UAV 位置和时隙时间连续演化；两阶段的每一步都调用
Master 和 Member，仅参数更新对象不同。

实跑命令：

```powershell
.\.venv\Scripts\python.exe scripts/train_rl_epochs.py --member-epochs 1 --master-epochs 1 --steps-per-epoch 500 --eval-steps 4 --test-steps 16 --output scripts/output/rl-epochs-500-verified
```

环境参数全部沿用默认。此次验证/测试分别取 4/16 步以加速接口验收；两个训练
epoch 都是完整 500 步。新入口的验证/测试默认仍各 500 步。

| epoch | 参数更新阶段 | 环境步数 | Master 联合调用数 | 子任务调度数 | 累计 reward | 每步平均 reward |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Member | 500 | 500 | 500000 | -272.442110 | -0.544884 |
| 2 | Master | 500 | 500 | 500000 | -274.832543 | -0.549665 |

逐行审计 `steps.csv` 确认：

- 全局步号严格为 1–1000，无遗漏、无重复。
- 各 epoch 内步号严格为 1–500，时隙起点为 0–499。
- 每步 1 次联合 Master 调用、100 个 DAG、1000 个子任务。
- 每个 epoch 的 500 个 reward 之和与 `epochs.csv` 中 `epoch_return` 一致。
- Member epoch 内更新 63 次 PPO（62 批 × 8 步 + 最后一批 4 步），不中断物理轨迹。
- Master epoch 在完整 500 步后更新 1 次 PPO；Member 参数保持冻结。

完整测试：`python -m unittest discover -s tests -q`，**148 项通过**。
新测试额外覆盖默认 500 步、分批更新时位置/时间连续、每步两类策略都执行、
冻结、critic 时间输入、checkpoint、CLI 续训及中断残余日志去重、旧/新 reward 图数据。
独立审查未发现需修复的重要问题，并独立运行了 4 项相关小测试。

`best.pt` 已通过新的评估入口重载，预留的 16 步平均 reward 为 -0.552581，
失败率 0%，与 summary 一致。这是两个 epoch 的流程验收，不是收敛实验。
Master 阶段切换到冻结的确定性 Member，不能将两阶段 reward 的差异直接解释为
同一策略变好或变差。

[新 500 步 reward 图](E:/AppData/PycharmProjects/mrdts/scripts/output/rl-epochs-500-verified/reward-plot/epoch-reward.png)
包含两个实际 epoch 点，未补造收敛曲线。
[旧实验 reward 图](E:/AppData/PycharmProjects/mrdts/scripts/output/rl-reward-legacy/epoch-reward.png)
包含此前 3 个种子，标题标明每轮 8 个环境步。两种数据没有混在同一图中。
图均提供 PNG/SVG/PDF、源数据 CSV 和图注；导出后已目视检查标签和布局。

运行、绘图、续训及步数解释见
[EPOCHS.md](E:/AppData/PycharmProjects/mrdts/methods/rl_baseline/EPOCHS.md)。
