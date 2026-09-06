"""使现有 ERS 实现适配时隙组件接口。"""

from methods.components.ordering.ers import ERS


class ERSOrdering:
    def plan(self, runtime, requests):
        # 每个时隙使用新的 runtime，不保留上个时隙的拓扑与成本缓存。
        return ERS(runtime).plan(requests)
