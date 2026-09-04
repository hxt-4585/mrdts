# MRDTS 环境设计文档

本目录记录 MRDTS（Multi-Region Distributed Task Scheduling）强化学习环境的建模假设、参数设置与后续实现方案。

代码目录与导入迁移说明见 [环境代码结构](../env/README.md)。现有仿真入口为 `env.simulator.Simulator`，Gym 风格交互接口尚未实现。

## 文档列表

| 编号 | 文件                             | 内容                                         | 状态   |
| ---: | -------------------------------- | -------------------------------------------- | ------ |
|   01 | `01_system_model.md`             | 系统实体、区域划分、通信与计算关系、时隙流程 | 已完成 |
|   02 | `02_scenario_parameters.md`      | 仿真区域、设备数量、UAV、BS、通信与能耗参数  | 待编写 |
|   03 | `03_task_dag_model.md`           | DAG 任务生成规则、拓扑参数与子任务参数       | 已完成 |
|   04 | `04_network_compute_energy.md`   | 通信、计算、时延与能耗模型                   | 待编写 |
|   05 | `05_information_observation.md`  | Master 与 Member UAV 的观测信息设计          | 待编写 |
|   06 | `06_action_space_constraints.md` | 飞行动作、卸载动作及其约束                   | 待编写 |
|   07 | `07_reward_design.md`            | 分层多智能体奖励函数设计                     | 待编写 |
|   08 | `08_rl_architecture.md`          | 异构多智能体强化学习架构                     | 待编写 |
|   09 | `09_environment_transition.md`   | 环境 `reset`、`step` 与状态转移流程          | 待编写 |
|   10 | `10_experiment_protocol.md`      | 实验场景、对比方法与评价指标                 | 待编写 |
|   11 | `11_implementation_plan.md`      | 代码模块划分与开发顺序                       | 待编写 |
|   12 | `12_decision_log.md`             | 关键建模决策及变更记录                       | 待编写 |
