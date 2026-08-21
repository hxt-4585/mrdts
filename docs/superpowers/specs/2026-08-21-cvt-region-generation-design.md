# CVT 区域生成设计

## 目标

以紧凑、连通的正方形网格区域取代随机前沿扩张产生的尖刺和狭窄通道。每个区域必须完整覆盖地图、满足面积下限，并可由独立种子复现。

## 方法

生成器以最远点采样初始化种子。每次 Lloyd 迭代都在全部网格上构造离散加权 Voronoi（power diagram）：网格 `x` 归属最小化 `||x - seed_r||² - weight_r` 的区域。内部权重迭代使实际面积逼近由 `area_imbalance` 给出的目标面积；随后将每个种子移至所属网格质心。

连续 power cell 是半平面的交集，因此是凸集；这避免了一个区域从中间截断另一个区域。离散结果仍由现有四邻接校验检查。紧凑度通过 `perimeter² / area` 的回归测试限制，以检测细长尖刺。

该设计基于 Du、Faber、Gunzburger 对 centroidal Voronoi tessellation 与 Lloyd 算法的综述（SIAM Review, 1999, DOI: 10.1137/S0036144599352836）。

## 配置

- `lloyd_iterations`：种子向质心收敛的次数；越大，区域越紧凑。
- `capacity_iterations`：每一轮调整 power 权重以满足目标面积的次数。
- `area_imbalance`：通过区域目标面积而非不受控的边界生长控制面积差异。

`shape_smoothness` 被移除，因为随机前沿扩张已不再是生成过程的一部分。若以后需要不规则边界，应在 CVT 基线之上加入保持四邻接连通和最小通道宽度的受限边界交换，而不是恢复随机扩张。
