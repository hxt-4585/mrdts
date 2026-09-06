"""全部子任务卸载到关联的 Member UAV。"""

from methods.contracts import SchedulingContext


class OwnerScheduling:
    def schedule(self, context: SchedulingContext):
        return {key: context.owner for key in context.order}
