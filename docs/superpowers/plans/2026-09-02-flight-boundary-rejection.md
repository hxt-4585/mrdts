+# Flight Boundary Rejection — Historical Decision

Member 飞行动作只有在候选位置处于地图边界内时才生效；拒绝动作保持当前位置并记录结果。

实体行为位于 `env/entities/uav.py`，随机方案的可行性筛选位于 `methods/solutions/random/method.py`。
