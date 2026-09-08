# 可复用学习工具

这里不依赖 UAV、DAG、场景生成或某个方案的奖励。后续方法可直接复用：

- `networks/mlp.py`：MLP 构造与 ValueNetwork。
- `buffers/rollout.py`：PPO 批量样本，保存采样时的观测、mask、动作、旧 log probability、critic 输入、旧 value 和 return。
- `algorithms/ppo.py`：裁剪 PPO 更新、独立 actor/critic 优化器、梯度裁剪、KL 提前停止，以及完整回合的 GAE。
- `device.py`：显式 CPU/CUDA 检查。

使用 PPO 的 actor 需提供 `evaluate(observations, masks, actions)`，返回每条样本的 log probability 与 entropy；critic 输入为 `critic_observations`，输出每条样本一个 value。actor 与 critic 应放在同一设备。PPO 接收采样时保存的 NumPy 批量数据，再转换到 actor 所在设备。

`Rollout` 是一次更新的数据容器，不负责环境采样或长期经验回放。`gae_returns` 接受完整回合，末步必须是真实终止，不支持将任意 buffer 截断当成终止。需要截断自举的其他方案应显式扩展接口。

Member 候选特征、Master 动作维度、动作 mask 规则、奖励、阶段冻结和训练/验证划分均在 `solutions/ppo/`。其他方案可以复用以上工具，自行实现 actor、采样流程和 Trainer；无需照搬 PPO Delay 的业务定义。
