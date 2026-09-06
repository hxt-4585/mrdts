# 自己提出的完整方法

此目录为已确认的方案保留位置，当前没有实现或注册 RL 训练算法。

- `method.py`：选择三个组件，构造各阶段业务输入。
- `observation.py`：方法自己的 Master/Member 观测、集中 critic 状态及可见范围。
- `reward.py`：方法自己的奖励和归一化规则。
- `trainer.py`：实现 `train(config, device)`；网络和输入必须移动到传入的设备。
- 如需第三方 RL 库的 reset/step 协议，在此增加 `adapter.py` 包装 Simulator。

组件实现位于 `methods/components/`。在 `methods/factory.py` 注册完整方法及 Trainer；参数分别保存在方法 TOML 和实验 `[training]` 中。
Trainer 可以调用 `experiments/artifacts.py` 记录训练配置、状态与产物；训练不能把 baseline 的 evaluate 当成参数更新。

方法可以覆盖通用阶段流程，不要求所有文献联合优化算法都强行拆成独立组件。排序须覆盖全部子任务、前驱先于后继；放置须使用合法节点；全部 Member 方案在推进物理事件前统一提交。
