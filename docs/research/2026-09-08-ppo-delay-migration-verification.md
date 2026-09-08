# PPO 时延基线迁移验证

日期：2026-09-08。源基线为恢复分支 `codex/global-delay-rl-baseline` 的 `c47059dee760e0095b7d135a856a114fa63e1a5e`，目标工作区基于 `17b256c`。运行环境 Python 3.11、PyTorch 2.10.0+cu126、CPU；本机驱动不能运行该 CUDA 构建，所以所有实际训练验证显式使用 CPU。

## 算法与接口

从源提交直接读取网络与 PPO 代码，对相同 seed、32 条合成样本执行初始化和一次 PPO 更新。MemberActor、MasterActor、ValueNetwork 初始化逐位一致，PPO actor/critic 更新后的所有参数逐位一致，loss 报告完全相同。一次复验记录：actor loss `-0.0024357401562156156`、value loss `0.2382056973874569`、KL `1.0516494512557983e-05`、entropy `0.7170266658067703`，未提前停止。

迁移有意改变的部分为当前 Scene/RandomStreams、ERS `priority_seq` 契约、1000 bits/Kbit 的物理单位、输出目录和显式设备。上述数值对照验证学习公式，不意味着新旧仿真单位下的训练曲线应相同。

自动化验证命令：

```powershell
.venv/Scripts/python.exe -B -m unittest discover -s tests -q
```

当前 164 项全部通过，覆盖原项目回归、动作 mask、本地先验、Kbit 输入成本、完整任务计划、回合内连续移动、Member buffer 刷新、阶段冻结、GAE 真实终止、失败奖励、结果层级、模型加载评估、任务流划分、续训和绘图。续训测试比较完整训练与分段恢复的四个网络所有参数及逐时隙 CSV，结果逐位/逐字节一致；同时模拟未提交日志、已覆盖但未提交的最佳模型、阶段切换未提交验证行，确认恢复一致且原日志留有备份。

独立代码审查发现多回合评估可能走入训练 ID，已添加测试集范围检查和回归用例。当前 PPO 独立评估只允许测试 ID 1；验证 ID 0、训练 ID 2 起。多测试回合协议尚未扩展。

## 100 用户短程训练

100 用户、每 DAG 10 节点、seed 42，使用当前真实仿真。命令：

```powershell
.venv/Scripts/python.exe -B -m experiments.train --config config/experiments/ppo_delay.toml --member-epochs 30 --master-epochs 5 --slots 8 --eval-steps 8 --test-steps 16 --eval-every 5 --device cpu
```

运行目录：`results/ppo_delay/runs/ppo_delay_ers_ppo_ppo/seed_42/20260907T173334_390991Z_0fedc62e/`。

| 固定验证集（ID 0，8 时隙） | 截断平均时延 / 秒 |
|---|---:|
| 未训练确定性策略 | 0.5501994014 |
| Member 第 30 epoch 后 | 0.4519957095 |
| Master 第 5 epoch 后 | 0.4520217776 |

验证选优时延比未训练下降约 17.8%。短程 Master 阶段没有进一步改善验证结果，因此 best checkpoint 保留该阶段初始的组合。训练采样 reward 与确定性验证 reward 不同，切换阶段会改变动作采样方式，不把曲线跳变解释成训练崩溃，也不把短程下降声称为长期收敛。

在独立保留任务流（ID 1，16 时隙，共 1600 DAG）上做配对比较：

| 方法 | 截断平均时延 / 秒 | 失败 DAG |
|---|---:|---:|
| PPO 验证选优模型 | 0.4504597411 | 0 |
| ERS + 原地停留 + 本地执行 | 0.5500116855 | 0 |
| Random（ERS + 随机飞行 + 随机调度） | 0.4838021887 | 10 |

训练目录的 `training/test.json` 与独立加载 checkpoint 的评估一致。配对运行详情保存在该目录的 `training/paired_check.json`，独立评估运行现已并入 `results/ppo_delay/runs/`。这里仅为单 seed、固定布局、短任务流的工程验证，不能用来宣称多种子优势或新布局泛化。

## 长回合与产物检查

补充真实 500 时隙回合检查，使用 Member 1 epoch、Master 1 epoch，共 1000 训练时隙；验证长度 4、测试长度 16。命令：

```powershell
.venv/Scripts/python.exe -B -m experiments.train --config config/experiments/ppo_delay.toml --member-epochs 1 --master-epochs 1 --slots 500 --eval-steps 4 --test-steps 16 --eval-every 1 --device cpu
```

运行目录：`results/ppo_delay/runs/ppo_delay_ers_ppo_ppo/seed_42/20260908T020039_495610Z_82a519e8/`。运行正常完成：1000 条连续逐时隙记录，各阶段 slot_start 均为 0–499，训练任务流 ID 分别为 2、3；Member 63 次更新、Master 1 次更新，所有 loss/KL/entropy 有限。Master 阶段的 Member actor/critic 与选出的 Member checkpoint 逐位一致，Master 参数确实发生更新。

| 指标 | 结果 |
|---|---:|
| 初始验证截断平均时延（4 时隙） | 0.5537078729 秒 |
| 选优模型验证截断平均时延（同 4 时隙） | 0.2540760314 秒 |
| Master 末态验证截断平均时延 | 0.2550702358 秒 |
| 选优模型保留测试时延（16 时隙） | 0.2543532322 秒 |
| Member epoch return（500 时隙） | -191.3614593308 |
| Master epoch return（500 时隙） | -131.8989342644 |

该运行同样由验证集保留 Master 阶段初始组合，末态不覆盖更好的模型。短验证流上的下降只证明实现能够学习，不证明稳定收敛。长回合产物也成功离线绘图；默认完整预算的 100 + 30 epoch 尚未执行。

离线绘图和实时监控的无界面模式均在短程真实日志上执行成功：

```powershell
.venv/Scripts/python.exe -B -m visualization.training --run <run>
.venv/Scripts/python.exe -B -m visualization.watch --run <run> --once
```

生成 reward/epoch return/时延、actor/value loss、KL/entropy 的 PNG/SVG/PDF 及 `source.csv`，并检查实际 reward 与 loss 图布局。保留原始点，5 点完整窗口后开始平滑，Member/Master 分段，验证数据单独显示；不同长度的总 return 不混比。Value loss 使用 symlog（绝对值 0.0001 以内为线性），保留两个阶段不同量级的变化。交互窗口模式实现了刷新，但本次未执行人工 GUI 操作验证。

从项目外目录直接运行 `experiments/train.py --resume ... --check --device cpu` 也通过，确认 PyCharm 类入口与项目相对路径解析。随后实际执行不带 `--config` 的 `--resume latest.pt --device cpu`，自动恢复已完成的短程运行并重现保留测试 reward `-0.4504597411`。最终代码、使用说明和验证记录保存在开发分支；大量运行产物按现有规则不提交 Git。

结果目录命名按后续要求简化为 `ppo_delay_ers_ppo_ppo`；原 `ppo_delay_check` 的配对运行已并入同一实验。移动保留原指标和 checkpoint，JSON 中的活动路径同步更新，metadata 记录原位置。Checkpoint 内的历史保存位置仍保留为来源记录，续训以传入的最新文件位置为准。
