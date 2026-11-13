# 单时隙区域事件回放

## 范围与设计

在 `scripts/` 实现场景、只读记录提取和 HTML 展示，在 `tests/` 验证记录；场景调用 `methods/random/ers.py` 的 ERS 和环境分阶段调度接口，不覆盖项目配置文件。

- 使用现有默认区域划分，脚本内覆盖为每区域 1 用户、2 架 Member、1 个四节点菱形 DAG（`0 → {1, 2} → 3`）。Master 按现有初始化流程生成，但不参与任务执行。
- 两架 Member 在脚本初始化时选取区域内不同位置，便于辨认；调用 `begin_step` 时飞行动作为零。时间轴从飞行阶段之后开始。
- 预设完整卸载位置为 owner、任务自身 Ground、同区域另一 Member、全局 BS。排序调用 `ERS(runtime).plan(requests)`，使用当前区域候选核心与真实中继链路的平均时延；自身输入上传不进入 rank，但完整参与运行时执行。
- `begin_step()` 登记所有区域的 DAG。脚本按 `plan.order` 组织完整位置字典，在窗口起点调用一次 `submit_schedule()`。计算与传输按就绪时间排队，同刻争用同一资源时按提交顺序派发。页面按区域筛选，共享 BS 的核心时间线同时标出其他区域的任务。
- `end_step()` 运行至所有任务完成。脚本从保留的运行时读取逐任务、逐跳的实际时间记录。超时 DAG 的详情展示实际完成时刻和超时状态。
- 输出独立、离线可打开的 HTML 和完整 JSON。支持播放、拖动、跳转前后事件、区域切换；展示拓扑、DAG 状态、信道与核心时间线、当前队首和事件列表。

## 实施与验证

- [x] 验证小场景规模、路由、事件记录及过载后的完整执行。
- [x] 实现 `scripts/visualize_scheduling.py`，在现有运行时执行场景并输出 JSON。
- [x] 实现 `scripts/templates/scheduling_replay.html`，将 JSON 嵌入离线回放页。
- [x] 运行默认和过载示例，检查实际页面与交互，补充使用说明。

## 使用

在项目根目录运行：

```powershell
uv run python scripts/visualize_scheduling.py
```

用浏览器打开生成的 `scripts/output/scheduling_replay.html`。所有数据和界面资源均内嵌，无需联网、启动服务或新增依赖。同目录的 `scheduling_replay.json` 保存逐任务、逐跳、各 DAG 结算和事件记录。

JSON 每个任务另含 `rank_s`、`average_compute_s`、`average_edge_comm_s`，可核对 ERS 成本和实际执行顺序。平均成本不包含排队；实际传输与计算记录包含排队和源数据等待。

切换初始区域或生成超时示例：

```powershell
uv run python scripts/visualize_scheduling.py --region 2
uv run python scripts/visualize_scheduling.py --load-scale 100 --output scripts/output/scheduling_replay_overload.html
```

`--load-scale` 同时缩放原始输入、中间结果和 CPU cycles，保持拓扑、硬件和链路参数不变。默认各节点输入为 `[120, 70, 100, 80] Kbit`，计算量为 `[800, 90, 1600, 1800] M cycles`，四条依赖边结果为 `[35, 60, 40, 55] Kbit`。这些仅是脚本演示参数；不使用默认随机 DAG 负载，也不修改配置文件。

默认每区域只有一个 DAG，各节点分散在四个执行位置；可能没有计算排队。这是该场景的真实结果，不为展示而虚构排队。信道队列仍会等待源数据或前序传输，页面的“当前信道队首”可以查看。

## 已验证

- 回放和 ERS 的回归检查位于 `tests/`，运行 `uv run python -B -m unittest discover -s tests` 获取当前结果；本节其余条目保留历史手动验证记录。
- 默认 4 个 DAG 全部完成；100 倍负载下 4 个 DAG 全部截止失败。
- 浏览器检查区域切换、节点高亮、共享 BS 核心记录、逐事件前进、播放、时间拖动、重置和过载截止显示，未发现脚本错误。
- `scripts/output/` 已由项目现有 `.gitignore` 忽略；生成的 HTML/JSON 不纳入 Git，可通过上述命令重新生成。
