# ERS 时延排序设计

用户已确认本阶段只考虑时延，逐核心平均计算成本，按合法中继路径计算平均依赖通信成本，自身输入上传不进入 rank。模块使用 `methods/ers.py`。

## 成本和排序

- 每个 DAG 的候选设备为自身 Ground、当前 owner 所在区域的全部 Member、唯一全局 BS。
- 每个核心等权。平均计算时间为 `sum(C_i / f_pk) / sum(K_p)`，一个任务只使用一个核心。
- 复用运行时 RoutePlanner 和信道速率计算；路径每比特时间为各跳 `1 / R_ab` 之和。
- 依赖边平均时间为 `D_ij * sum(K_p * K_q * path_s_per_bit[p,q]) / sum(K_p)^2`，包含同设备成本为零的组合。
- 原始输入和边数据单位为十进制 Kbit，转换为 `Kbit * 1000` bit。输入上传只进入实际执行时间。
- `rank[i] = compute[i] + max(comm[i,j] + rank[j])`。使用迭代拓扑计算，拒绝环和无效成本；同 rank 时仍保持前驱优先。
- 同一 owner 的全部 DAG 合并排序，保留 `(owner, user, dag, node)` 身份。不同 owner 的列表以 `(-rank, owner, user, dag, topological_position)` 稳定合并，供共享 BS 和信道使用；不改变各 owner 内部次序。
- 一次规划内缓存各有向跳及用户的路径矩阵。重新规划时重新读取当前拓扑，避免跨时隙或飞行后使用过期成本。

## 接口和执行

- `DAGRequest(dag_id, dag, owner_member, ground_device)` 表示待排序 DAG。
- `ERS(runtime).plan(requests)` 返回全局顺序、各 Member 顺序、rank 和各 DAG 的平均成本，供外部卸载策略选择设备。
- `SchedulingRuntime.candidate_execution_nodes(owner, ground)` 与运行时 placement 校验共享区域、关联和 BS 规则。
- `SchedulingRuntime.transfer_duration_s(hop, bits)` 供 ERS 和真实传输共同使用。
- `SchedulingRuntime.submit_dags(requests, placements, epoch_start)` 接收以 TaskKey 为键的 PlacementDecision，整批预备成功后按全局 ERS 入队并统一启动。序号允许有间隔，按相对次序压缩并追加到已提交批次后。
- `submit_dag` 保留单 DAG 便捷入口。多 DAG 统一排序必须一次批量提交，已经启动的旧批次不重新排序。
- 不改变物理设备级 FIFO、多核派发、逐跳中继、信道队首阻塞、时隙截止结算和现有能耗记录。
- 批量提交先验证并准备全部任务及传输，失败时不留下部分任务、队列或序号占用。

## 验证

手算 1/2/4 核、20/100 Mbps 场景验证平均依赖通信为 0.017142857 秒；验证非对称链路、同设备零成本、输入量不影响 rank、同 owner 多 DAG 交错顺序、重复身份和环被拒绝、长链无需递归、更新拓扑不沿用旧成本。运行时验证批量顺序在真实传输和计算中生效、错误批次不污染状态、原始输入和依赖均真实传输。十时隙集成和事件回放改用实际 ERS 成本与批量提交。

## 调用示例

在 `Environment.begin_slot(...)` 后读取 `runtime = environment.runtime`。调用方提供当前时隙的 DAG、所属 Ground 和 owner；下例将全部任务放到 BS，仅用于展示接口，不是卸载优化策略。

```python
from methods.contracts import DAGRequest, PlacementDecision
from methods.ers import ERS

requests = [
    DAGRequest(dag_id=0, dag=dag_a, owner_member=owner, ground_device=ground),
    DAGRequest(dag_id=1, dag=dag_b, owner_member=owner, ground_device=ground),
]
plan = ERS(runtime).plan(requests)
placements = {
    key: PlacementDecision(runtime.global_bs, seq)
    for seq, key in enumerate(plan.order)
}
runtime.submit_dags(requests, placements, epoch_start=runtime.now)
result = environment.end_slot()
```

`plan.member_orders[owner]` 是该 Member 的合并队列。`plan.costs[request.key]` 包含 `candidates`、`path_seconds_per_bit`、`average_compute_s`、`average_edge_comm_s`；`plan.ranks[key]` 是秒单位的向上 rank。外部策略应沿 `plan.order` 生成位置决策，再整批提交。批次内序号须唯一且前驱先于后继；不能逐 DAG 重置序号后分别提交来替代统一批次。

输入和依赖数据量可以为零；零数据不创建信道作业，依赖仍等待前驱计算完成。每个任务计算量为有限正数。
