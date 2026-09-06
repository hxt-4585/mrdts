"""全部子任务在所属地面设备执行。"""

from methods.contracts import SchedulingContext
from env.types import EntityKind, EntityRef


class LocalScheduling:
    def schedule(self, context: SchedulingContext):
        return {key: EntityRef(EntityKind.GROUND_DEVICE, key.user_id) for key in context.order}
