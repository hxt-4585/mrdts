# 方案边界重构实施计划

没有用户的明确允许，不可以提交（git commit）。

**Goal:** 新增方案只需方案文件、配置和注册项，公共实验入口不包含算法分支；ppo_delay 更名为 ppo。

**Architecture:** 使用 methods/solution.py 的 Solution 接口承接参数扩展、配置解析、校验、方法构造与训练器构造。factory 仅做显式注册和通用组件检查。PPO 专用行为放入 solutions/ppo/solution.py。CLI 先读取配置识别方案，再加载该方案参数；通用 --training KEY=VALUE 支持 TOML 类型覆盖。

**Tech Stack:** Python 3.11、现有 PyTorch、unittest、matplotlib，不增加依赖。

## 约束

- 保持物理环境、奖励、采样、网络和 PPO 更新行为不变。
- 保留现有 Member/Master 命令行参数；方案通过接口声明参数。
- 配置、源码、结果当前路径统一为 ppo；历史来源分支名不改写。
- 旧 checkpoint 仅在读取后的内存中规范化方案签名，不修改原始二进制。
- Random 保持原评估行为；检查失败必须发生在创建结果目录之前。

## 实施与验证

- [x] 添加扩展性测试：在 SOLUTIONS 临时注册测试方案，验证同一 CLI/train/evaluate 支持其专用参数、检查和 checkpoint，无需更改框架代码。
- [x] 新增 Solution 默认接口和 Random/PPO 适配；修改 factory、cli、train、runner 调用统一接口，保留通用 config/artifacts。
- [x] 重命名 solutions/ppo、两个配置文件及源码引用，增加旧签名兼容测试；运行 PPO 和 Random 实验回归。
- [x] 可视化阶段动态识别，允许可选指标；验证非 PPO 日志可绘图及原有图回归。
- [x] 迁移现有 results/ppo_delay 到 results/ppo，更新当前路径引用，保留历史来源；校验文件数和 checkpoint 哈希未变。
- [x] 更新使用文档和扩展指南；完整 unittest、真实旧模型评估/续训检查、git diff --check。记录结果，不提交。

验证结果：最终完整 unittest 174 项通过（36.935 秒）；真实旧模型评估及临时副本恢复值与历史记录一致；10 个原始模型哈希未变；git diff --check 通过。所有变更未提交。
