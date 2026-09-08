# 共享缓存

`rollout.py` 提供 PPO 更新的数据容器，保存采样时的观测、mask、动作、旧概率、critic 输入、value 和 return。方案负责保持候选 ID 与动作列的一致性；这里不生成场景，也不决定何时更新。
