"""DAG 输入和中间结果的确定性传输路径。"""

from dataclasses import dataclass

from env.types import DirectedChannelKey, EntityKind, EntityRef


@dataclass(frozen=True)
class RoutePlan:
    """一份数据在实体间依次经过的有向链路。"""

    hops: tuple[DirectedChannelKey, ...]


class RoutePlanner:
    """仅实现题设规定的输入中继与前驱结果直达规则。"""

    def input_route(
        self, ground_device: EntityRef, owner_member: EntityRef, execution_node: EntityRef
    ) -> RoutePlan:
        if execution_node.kind is EntityKind.GROUND_DEVICE:
            return RoutePlan(())
        first_hop = DirectedChannelKey(ground_device, owner_member)
        if execution_node == owner_member:
            return RoutePlan((first_hop,))
        return RoutePlan((first_hop, DirectedChannelKey(owner_member, execution_node)))

    def predecessor_route(
        self,
        source_node: EntityRef,
        target_node: EntityRef,
        ground_device: EntityRef,
        owner_member: EntityRef,
    ) -> RoutePlan:
        if source_node == target_node:
            return RoutePlan(())
        if source_node == ground_device:
            return self.input_route(ground_device, owner_member, target_node)
        if target_node == ground_device:
            final_hop = DirectedChannelKey(owner_member, ground_device)
            if source_node == owner_member:
                return RoutePlan((final_hop,))
            return RoutePlan((DirectedChannelKey(source_node, owner_member), final_hop))
        return RoutePlan((DirectedChannelKey(source_node, target_node),))
